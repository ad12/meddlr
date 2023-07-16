from typing import Sequence

import torch
from torch import nn

from meddlr.metrics.functional.image import mse
from meddlr.metrics.metric import Metric
from meddlr.ops import complex as cplx
from meddlr.utils import env

# Pre-trained model for computing SSFD. This is a UNet model trained on a self-supervsed
# in-painting pretext task on the fastMRI knee dataset.
SSFD_HUGGINGFACE_URL = "https://huggingface.co/philadamson93/SSFD/resolve/main/model.ckpt"


# TODO: Refactor SSFD Class to extract shared logic into parent class FeatureMetric
class SSFD(Metric):
    """
    Self-Supervised Feature Distance. SSFD evaluates the feature distance between a
    pair of images from features extracted from a pre-trained neural network trained on
    a self-supervision task with fastMRI datasets [1]. SSFD has been shown to correspond
    well to Radiologist Reader Scores of accelerated  MR reconstructions.

    References:
    ..  [1] Adamson, Philip M., et al.
        SSFD: Self-Supervised Feature Distance as an MR Image Reconstruction Quality Metric."
        NeurIPS 2021 Workshop on Deep Learning and Inverse Problems. 2021.
        https://openreview.net/forum?id=dgMvTzf6M_3
    """

    is_differentiable = True
    higher_is_better = False

    def __init__(
        self,
        mode: str = "grayscale",
        layer_names: Sequence[str] = ("block4_relu2",),
        channel_names: Sequence[str] = None,
        reduction="none",
        compute_on_step: bool = False,
        dist_sync_on_step: bool = False,
        process_group: bool = None,
        dist_sync_fn: bool = None,
    ):
        """
        Args:
            mode (str): Determines how to interpret the channel dimension of the inputs. One of:
                * ``'grayscale'``: Each channel corresponds to a distinct grayscale input image.
                * ``'rgb'``: The 3 channel dimensions correspond to a single rgb image.
                             Exception will be thrown if channel dimension != 3 or dtype is complex
            layer_names (Sequence[str]):
                A list of strings specifying the layers to extract features from. Any of:
                ['block1_relu2', 'block2_relu2', 'block3_relu2', 'block4_relu2', 'block5_relu2']
                SSFD from each layer will be summed if multiple layers are specified.
        """

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
            raise ValueError(f"Invalid `mode` ('{mode}'). Expected one of {valid_modes}.")

        self.mode = mode
        self.layer_names = layer_names

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

            is_complex = cplx.is_complex(targets) or cplx.is_complex_as_real(targets)
            if is_complex:
                raise TypeError(
                    f"Data type must be real when mode is {self.mode},\
                    but got data type {targets.dtype}"
                )

            loss_shape = (targets.shape[0], 1)

        preds = self.preprocess_ssfd(preds)
        targets = self.preprocess_ssfd(targets)

        target_features = self.net(targets)
        pred_features = self.net(preds)

        loss = 0
        for layer in self.layer_names:
            loss += torch.mean(mse(target_features[layer], pred_features[layer]), dim=1)
        loss = loss.view(loss_shape)

        return loss

    def preprocess_ssfd(self, img: torch.Tensor) -> torch.Tensor:
        """
        Preprocess image for SSFD model input.

        Converts to a magnitude scan if complex and normalizes to zero-mean, unit variance.
        If self.mode is 'rgb', then the image will be averaged over the channel dimension.

        Args:
            img (torch.Tensor): Tensor to preprocess.

        Returns:
            img (torch.Tensor): Preprocessed tensor.
        """

        is_complex = cplx.is_complex(img) or cplx.is_complex_as_real(img)
        if is_complex:
            img = cplx.abs(img)

        if self.mode == "grayscale":
            img = img.reshape(img.shape[0] * img.shape[1], 1, img.shape[2], img.shape[3])
        elif self.mode == "rgb":
            img = torch.mean(img, axis=1, keepdim=True)

        shape = (img.shape[0], img.shape[1], -1)

        img_mean = torch.mean(img.reshape(shape), dim=-1, keepdim=True).unsqueeze(-1)
        img_std = torch.std(img.reshape(shape), dim=-1, keepdim=True).unsqueeze(-1)
        img = (img - img_mean) / img_std

        return img


class SSFD_Encoder(nn.Module):
    """
    Pytorch architecture of pre-trained SSFD network.

    This is a UNet architecture with 5 layers and 2 convolutional blocks per layer.
    Modifications to this architecture are not supported as SSFD loads weights from
    a pre-trained network.

    """

    def __init__(self, in_channels: int, out_channels: int):
        """
        Args:
            in_chans (int): Number of channels in the input.
            out_chans (int): Number of channels in the output.
        """

        super().__init__()
        self.in_ch = in_channels
        self.out_ch = out_channels

        self.firstblock = DoubleConv(1, 20)
        self.encoder1 = EncoderBlock(20, 40)
        self.encoder2 = EncoderBlock(40, 80)
        self.encoder3 = EncoderBlock(80, 160)
        self.encoder4 = EncoderBlock(160, 320)

    def forward(self, xin: torch.Tensor):
        """
        Args:
            xin (torch.Tensor): Input tensor of shape [batch_size, self.in_ch, height, width].
        Returns:
            (Dict[str,torch.Tensor]): Dictionary with keys specified by self.layer_names
            and values of the corresponding intermedaite layer output tensors.
        """

        int_layers = {}
        xdown1, int_layers["block1_relu2"] = self.firstblock(xin)
        xdown2, int_layers["block2_relu2"] = self.encoder1(xdown1)
        xdown3, int_layers["block3_relu2"] = self.encoder2(xdown2)
        xdown4, int_layers["block4_relu2"] = self.encoder3(xdown3)
        _, int_layers["block5_relu2"] = self.encoder4(xdown4)

        return int_layers


class DoubleConv(nn.Module):
    """
    A Convolutional Block that consists of two convolution layers each followed by
    group normalization, ReLu activation and dropout.
    """

    def __init__(self, in_ch: int, out_ch: int):
        """
        Args:
            in_ch (int): Number of channels in the input.
            out_ch (int): Number of channels in the output.
        """
        super().__init__()

        self.in_ch = in_ch
        self.out_ch = out_ch

        self.conv1 = nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=out_ch, out_channels=out_ch, kernel_size=3, padding=1)
        self.GN = nn.GroupNorm(num_groups=10, num_channels=out_ch)
        self.ReLu = nn.ReLU(inplace=True)
        self.droupout = nn.Dropout(p=0.1)

    def forward(self, x: torch.Tensor):
        """
        Args:
            x (torch.Tensor): Input tensor of shape [batch_size, self.in_ch, height, width].
        Returns:
            x (torch.Tensor): Output tensor of shape [batch_size, self.out_ch, height, width].
            int_layer (torch.Tensor): Intermediate output tensor after the second ReLu activation
                                      of shape [batch_size, self.out_ch, height, width]
        """
        x = self.conv1(x)
        x = self.ReLu(x)
        x = self.conv2(x)
        x = self.ReLu(x)
        int_layer = x  # intermediate layer to extract features from for SSFD
        x = self.GN(x)

        return x, int_layer


class EncoderBlock(nn.Module):
    """
    A Convolutional Block that consists of Max Pooling followed by two convolution layers
    each followed by group normalization, ReLu activation and dropout.
    """

    def __init__(self, in_ch: int, out_ch: int):
        """
        Args:
            in_ch (int): Number of channels in the input.
            out_ch (int): Number of channels in the output.
        """
        super().__init__()
        self.MP = nn.MaxPool2d(2)
        self.conv_block = DoubleConv(in_ch, out_ch)

    def forward(self, x: torch.Tensor):
        """
        Args:
            x (torch.Tensor): Input tensor of shape [batch_size, self.in_ch, height, width].
        Returns:
            x (torch.Tensor): Output tensor of shape [batch_size, self.out_ch, height, width].
            int_layer (torch.Tensor): Intermediate output tensor from self.conv_block
                                      of shape [batch_size, self.out_ch, height, width]
        """
        x = self.MP(x)
        x, int_layer = self.conv_block(x)

        return x, int_layer
