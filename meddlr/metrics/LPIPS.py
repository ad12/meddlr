from typing import Sequence

import torch
from lpips import LPIPS as _LPIPS

from meddlr.metrics.metric import Metric
from meddlr.ops import complex as cplx


class LPIPS(Metric):
    """
    Learned Perceptual Image Patch Similarity. LPIPS evaluates the feature distance between a
    pair of images from features extracted from a pre-trained neural network [1]. LPIPS has been
    shown to correspond well to perceived image quality on natural images.


    Attributes:
        net_type (str): ['alex','vgg','squeeze'] - pre-trained network to extract features from.
        mode (str):
            This flag determines how to interpret the channel dimension of the inputs.
            ['grayscale']: Each channel corresponds to a distinct grayscale input image.
            ['rgb']: The 3 channel dimensions correspond to a single rgb image.
                     Exception will be thrown if channel dimension != 3.
        lpips (bool):
            [True] use linear layers on top of base/trunk network
            [False] means no linear layers; each layer is averaged together
        pretrained (bool):
            This flag controls the linear layers, which are only in effect when lpips=True above
            [True] means linear layers are calibrated with human perceptual judgments
            [False] means linear layers are randomly initialized


    References:
    ..  [1] R. Zhang, P. Isola, A. A. Efros, E. Shechtman, O. Wang.
        The Unreasonable Effectiveness of Deep Features as a Perceptual Metric.
        In CVPR, 2018 http://arxiv.org/abs/1801.03924
    """

    def __init__(
        self,
        net_type: str = "vgg",
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

        super().__init__(
            channel_names=channel_names,
            units="",
            reduction=reduction,
            compute_on_step=compute_on_step,
            dist_sync_on_step=dist_sync_on_step,
            process_group=process_group,
            dist_sync_fn=dist_sync_fn,
        )
        # choose parameters to be default but flexible in build dictionary config
        valid_net_type = ("vgg", "alex", "squeeze")
        if net_type not in valid_net_type:
            raise ValueError(
                f"Argument `net_type` must be one \
                            of {valid_net_type}, but got {net_type}."
            )

        valid_modes = ("grayscale", "rgb")
        if mode not in valid_modes:
            raise ValueError(
                f"Argument `mode` must be one \
                            of {valid_modes}, but got {mode}."
            )

        self.net = NoTrainLpips(net=net_type, lpips=lpips, verbose=False)
        self.mode = mode

    def func(self, preds, targets) -> torch.Tensor:

        if self.mode == "grayscale":
            loss_shape = (targets.shape[0], targets.shape[1])

        elif self.mode == "rgb":
            if targets.shape[1] != 3:
                raise ValueError(
                    f"Channel dimension must have size 3 for rgb mode,\
                    but got tensor of shape {targets.shape}."
                )

            loss_shape = (targets.shape[0], 1)

        preds = self.preprocess_lpips(preds)
        targets = self.preprocess_lpips(targets)

        loss = self.net(preds, targets).squeeze()
        loss = loss.view(loss_shape)

        return loss

    def preprocess_lpips(self, img):
        """Preprocess image per LPIPS implementation.

        Converts images to magnitude images, normalizes between [-1, 1], and reshapes greyscale
        inputs to have a channel dimension of 3 for input to LPIPS network

        Args:
            img (torch.Tensor): Tensor to preprocess

        Returns:
            Preprocessed tensor
        """

        is_complex = cplx.is_complex(img) or cplx.is_complex_as_real(img)
        abs_func = cplx.abs if is_complex else torch.abs

        img = abs_func(img)
        shape = (img.shape[0], img.shape[1], -1)

        img_min = torch.amin(img.view(shape), dim=-1, keepdim=True).unsqueeze(-1)
        img_max = torch.amax(img.view(shape), dim=-1, keepdim=True).unsqueeze(-1)

        img = 2 * (img - img_min) / (img_max - img_min) - 1

        if self.mode == "grayscale":
            img = img.view(img.shape[0] * img.shape[1], 1, img.shape[2], img.shape[3])
            img = img.repeat(1, 3, 1, 1)

        elif self.mode == "rgb":
            img = img

        return img


class NoTrainLpips(_LPIPS):
    def train(self, mode: bool) -> "NoTrainLpips":
        """the network should not be able to be switched away from evaluation mode."""
        return super().train(False)
