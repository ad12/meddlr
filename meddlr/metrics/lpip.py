from typing import Sequence

import torch

from meddlr.metrics.metric import Metric
from meddlr.ops import complex as cplx
from meddlr.utils import env

if env.package_available("lpips"):
    from lpips import LPIPS as _LPIPS


# TODO: Refactor SSFD Class to extract shared logic into parent class FeatureMetric
class LPIPS(Metric):
    """
    Learned Perceptual Image Patch Similarity.

    LPIPS evaluates the feature distance between a pair of images from features extracted
    from a pre-trained neural network [1]. LPIPS has been shown to correspond well to
    perceived image quality on natural images.

    References:
    ..  [1] R. Zhang, P. Isola, A. A. Efros, E. Shechtman, O. Wang.
        The Unreasonable Effectiveness of Deep Features as a Perceptual Metric.
        In CVPR, 2018 http://arxiv.org/abs/1801.03924
    """

    def __init__(
        self,
        net_type: str = "alex",
        mode: str = "grayscale",
        lpips: bool = True,
        pretrained: bool = True,
        channel_names: Sequence[str] = None,
        reduction="none",
        compute_on_step: bool = False,
        dist_sync_on_step: bool = False,
        process_group: bool = None,
        dist_sync_fn: bool = None,
    ):
        """
        Args:
        net_type (str): The pre-trained network to use for extracting features. One of:
            * ``'alex'``: Alex-Net w/ feature extraction layers 'relu1' through 'relu5'
            * ``'vgg'``: VGG-16 w/ feature extration layers ['relu1_2', 'relu2_2',
                       'relu3_3', 'relu4_3', 'relu5_3']
            * ``'squeeze'``: Squeeze-Net w/ feature extration layers 'relu1' through 'relu7'
        mode (str): Determines how to interpret the channel dimension of the inputs. One of:
            * ``'grayscale'``: Each channel corresponds to a distinct grayscale input image.
            * ``'rgb'``: The 3 channel dimensions correspond to a single rgb image.
                         Exception will be thrown if channel dimension != 3 dtype data is complex.
        lpips (bool): This flag determines if a linear layer is used on top of the
                      extracted features.
            * ``True``: linear layers on top of base/trunk network.
            * ``False``: no linear layers; each layer is averaged together.
        pretrained (bool): This flag controls the linear layers, which are only in
                           effect when lpips=True above.
            * ``True``: linear layers are calibrated with human perceptual judgments.
            * ``False``: linear layers are randomly initialized.
        """

        if not env.package_available("lpips"):
            raise ModuleNotFoundError(
                "LPIPS metric requires that lpips is installed."
                "Either install as `pip install meddlr[metrics]` or `pip install lpips`."
            )

        super().__init__(
            channel_names=channel_names,
            units="",
            reduction=reduction,
            compute_on_step=compute_on_step,
            dist_sync_on_step=dist_sync_on_step,
            process_group=process_group,
            dist_sync_fn=dist_sync_fn,
        )

        valid_net_type = ("vgg", "alex", "squeeze")
        if net_type not in valid_net_type:
            raise ValueError(
                f"Invalid `net_type` ('{net_type}'). Expected one of {valid_net_type}."
            )

        valid_modes = ("grayscale", "rgb")
        if mode not in valid_modes:
            raise ValueError(f"Invalid `mode` ('{mode}'). Expected one of {valid_modes}.")

        self.net = NoTrainLpips(net=net_type, lpips=lpips, verbose=False)
        self.mode = mode

    def func(self, preds: torch.Tensor, targets: torch.Tensor):

        if self.mode == "grayscale":
            loss_shape = (targets.shape[0], targets.shape[1])
        elif self.mode == "rgb":
            if targets.shape[1] != 3:
                raise ValueError(
                    f"Channel dimension must have size 3 for rgb mode,\
                    but got tensor of shape {targets.shape}."
                )

            is_complex = cplx.is_complex(targets) or cplx.is_complex_as_real(targets)
            if is_complex:
                raise TypeError(
                    f"Data type must be real when mode is {self.mode},\
                    but got data type {targets.dtype}"
                )

            loss_shape = (targets.shape[0], 1)

        preds = self.preprocess_lpips(preds)
        targets = self.preprocess_lpips(targets)

        loss = self.net(preds, targets).squeeze()
        loss = loss.view(loss_shape)

        return loss

    def preprocess_lpips(self, img: torch.Tensor) -> torch.Tensor:
        """
        Preprocess image per LPIPS implementation.

        Converts images to magnitude images if complex and normalizes between [-1, 1].
        If self.mode is 'grayscale', then each channel dimension will be replicated 3 times.

        Args:
            img (torch.Tensor): Tensor to preprocess.

        Returns:
            img (torch.Tensor): Preprocessed tensor.
        """

        is_complex = cplx.is_complex(img) or cplx.is_complex_as_real(img)
        if is_complex:
            img = cplx.abs(img)

        if self.mode == "grayscale":
            # normalize each image independently (channel dim. represents different images)
            shape = (img.shape[0], img.shape[1], -1)
            img_min = torch.amin(img.reshape(shape), dim=-1, keepdim=True).unsqueeze(-1)
            img_max = torch.amax(img.reshape(shape), dim=-1, keepdim=True).unsqueeze(-1)
            img = 2 * (img - img_min) / (img_max - img_min) - 1

            img = img.reshape(img.shape[0] * img.shape[1], 1, img.shape[2], img.shape[3])
            img = img.repeat(1, 3, 1, 1)
        elif self.mode == "rgb":
            # normalize each image independently (channel dim. represents the same image)
            shape = (img.shape[0], -1)
            img_min = (
                torch.amin(img.reshape(shape), dim=-1, keepdim=True).unsqueeze(-1).unsqueeze(-1)
            )
            img_max = (
                torch.amax(img.reshape(shape), dim=-1, keepdim=True).unsqueeze(-1).unsqueeze(-1)
            )
            img = 2 * (img - img_min) / (img_max - img_min) - 1

        return img


if env.package_available("lpips"):

    class NoTrainLpips(_LPIPS):
        def train(self, mode: bool) -> "NoTrainLpips":
            """the network should not be able to be switched away from evaluation mode.
            Implementation adapted from torchmetrics LPIPS."""
            return super().train(False)

else:
    NoTrainLpips = None
