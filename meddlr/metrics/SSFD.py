"""
Self-Supervised
"""

from typing import Sequence

import torch
from torch import nn

from meddlr.metrics.functional.image import mse
from meddlr.metrics.metric import Metric
from meddlr.ops import complex as cplx
from meddlr.utils import env

SSFD_HUGGINGFACE_URL = "https://huggingface.co/philadamson93/SSFD/resolve/main/model.ckpt"


class SSFD(Metric):
    """
    Self-Supervised Feature Distance. SSFD evaluates the feature distance between a
    pair of images from features extracted from a pre-trained neural network trained on
    a self-supervision task with fastMRI datasets [1]. SSFD has been shown to correspond
    well to Radiologist Reader Scores of accelerated  MR reconstructions.


    Attributes:
        mode (str):
            This flag determines how to interpret the channel dimension of the inputs.
            ['grayscale']: Each channel corresponds to a distinct grayscale input image.
            ['rgb']: The 3 channel dimensions correspond to a single rgb image.
                     Exception will be thrown if channel dimension != 3.


    References:
    ..  [1] Adamson, Philip M., et al.
        SSFD: Self-Supervised Feature Distance as an MR Image Reconstruction Quality Metric."
        NeurIPS 2021 Workshop on Deep Learning and Inverse Problems. 2021.
        https://openreview.net/forum?id=dgMvTzf6M_3
    """

    def __init__(
        self,
        mode: str = "grayscale",
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

        valid_modes = ("grayscale", "rgb")
        if mode not in valid_modes:
            raise ValueError(
                f"Argument `mode` must be one \
                            of {valid_modes}, but got {mode}."
            )

        self.mode = mode

        path_manager = env.get_path_manager()
        file_path = path_manager.get_local_path(SSFD_HUGGINGFACE_URL, force=False)

        self.net = SSFD_Encoder(in_channels=1, out_channels=320)
        self.net.load_state_dict(torch.load(file_path))
        self.net.eval()

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

        preds = self.preprocess_ssfd(preds)
        targets = self.preprocess_ssfd(targets)

        features = {}

        def get_features(name):
            def hook(model, input, output):
                features[name] = output.detach()

            return hook

        self.net.encoder3.conv_block.conv2.register_forward_hook(
            get_features("encoder3.conv_block.conv2")
        )  # set default to this but make flexible
        _ = self.net(targets)
        target_features = features["encoder3.conv_block.conv2"]

        _ = self.net(preds)
        pred_features = features["encoder3.conv_block.conv2"]

        loss = torch.mean(mse(target_features, pred_features), dim=1)
        loss = loss.view(loss_shape)

        return loss

    def preprocess_ssfd(self, img):
        """Preprocess image for SSFD model input. Converts to a magnitude scan and
        normalizes to zero-mean, unit variance.

        Args:
            img (torch.Tensor): Tensor to preprocess

        Returns:
            Preprocessed tensor
        """

        is_complex = cplx.is_complex(img) or cplx.is_complex_as_real(img)
        abs_func = cplx.abs if is_complex else torch.abs
        img = abs_func(img)

        if self.mode == "grayscale":
            img = img.view(img.shape[0] * img.shape[1], 1, img.shape[2], img.shape[3])

        elif self.mode == "rgb":
            img = torch.mean(img, axis=1, keepdim=True)

        shape = (img.shape[0], img.shape[1], -1)

        img_mean = torch.mean(img.view(shape), dim=-1, keepdim=True).unsqueeze(-1)
        img_std = torch.std(img.view(shape), dim=-1, keepdim=True).unsqueeze(-1)
        # import pdb; pdb.set_trace()
        img = (img - img_mean) / img_std

        return img


class SSFD_Encoder(nn.Module):
    """Pytorch architecture of pre-trained SSFD network"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.in_ch = in_channels
        self.out_ch = out_channels

        self.firstblock = DoubleConv(1, 20)
        self.encoder1 = EncoderBlock(20, 40)
        self.encoder2 = EncoderBlock(40, 80)
        self.encoder3 = EncoderBlock(80, 160)
        self.encoder4 = EncoderBlock(160, 320)

    def forward(self, xin):
        xdown1 = self.firstblock(xin)
        xdown2 = self.encoder1(xdown1)
        xdown3 = self.encoder2(xdown2)
        xdown4 = self.encoder3(xdown3)
        xdown5 = self.encoder4(xdown4)

        return xdown5


class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()

        self.in_ch = in_ch
        self.out_ch = out_ch

        self.conv1 = nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=out_ch, out_channels=out_ch, kernel_size=3, padding=1)
        self.GN = nn.GroupNorm(num_groups=10, num_channels=out_ch)
        self.ReLu = nn.ReLU(inplace=True)
        self.droupout = nn.Dropout(p=0.1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.ReLu(x)
        x = self.conv2(x)
        x = self.ReLu(x)
        x = self.GN(x)

        return x


class EncoderBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.MP = nn.MaxPool2d(2)
        self.conv_block = DoubleConv(in_ch, out_ch)

    def forward(self, x):
        x = self.MP(x)
        x = self.conv_block(x)

        return x
