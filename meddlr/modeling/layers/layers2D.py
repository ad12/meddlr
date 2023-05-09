"""Implementation of 2D base layers.
"""

from typing import Tuple, Union

import torch
from torch import nn

from meddlr.modeling.layers.conv import ConvWS2d
from meddlr.modeling.layers.scale import Scale2d


class ConvBlock(nn.Module):
    """
    A 2D Convolutional Block that consists of Norm -> ReLU -> Dropout -> Conv

    Based on implementation described by:
        K He, et al. "Identity Mappings in Deep Residual Networks" arXiv:1603.05027
    """

    def __init__(
        self,
        in_chans: int,
        out_chans: int,
        kernel_size: Union[int, Tuple[int, int]],
        drop_prob: float,
        act_type: str = "relu",
        norm_type: str = "none",
        norm_affine: bool = False,
        order: Tuple[str, str, str, str] = ("norm", "act", "drop", "conv"),
        bias: bool = True,
    ):
        """
        Args:
            in_chans: Number of channels in the input.
            out_chans: Number of channels in the output.
            kernel_size: Convolution kernel size.
            drop_prob: Dropout probability.
            act_type: Activation type.
            norm_type: Normalization type.
            norm_affine: Whether to learn affine parameters for normalization.
            order: Order of operations in the block.
            bias: Whether to use bias in the convolution.
        """
        super().__init__()

        self.in_chans = in_chans
        self.out_chans = out_chans
        self.drop_prob = drop_prob

        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)
        else:
            assert len(kernel_size) == 2
        if not all(k % 2 == 1 for k in kernel_size):
            raise ValueError("Kernel size must be odd - got {}".format(kernel_size))

        padding = tuple(k // 2 for k in kernel_size)

        # Define choices for each layer in ConvBlock
        conv_idx = order.index([x for x in order if "conv" in x][0])
        conv_after_norm = "norm" in order and conv_idx > order.index("norm")
        norm_channels = in_chans if conv_after_norm else out_chans
        normalizations = dict(
            none=lambda: nn.Identity(),
            instance=lambda: nn.InstanceNorm2d(norm_channels, affine=norm_affine),
            batch=lambda: nn.BatchNorm2d(norm_channels, affine=norm_affine),
            group=lambda: nn.GroupNorm(norm_channels // 8, norm_channels, affine=norm_affine),
        )
        activations = {"relu": lambda: nn.ReLU(), "leaky_relu": lambda: nn.LeakyReLU()}

        if norm_type not in normalizations:
            raise ValueError(
                f"Unknown norm_type '{norm_type}'. Must be one of {normalizations.keys()}"
            )

        layer_dict = {
            "conv": lambda: nn.Conv2d(
                in_chans,
                out_chans,
                kernel_size=kernel_size,
                padding=padding,
                bias=bias,
            ),
            "conv+ws": lambda: ConvWS2d(
                in_chans,
                out_chans,
                kernel_size=kernel_size,
                padding=padding,
                bias=bias,
            ),
            "drop": lambda: nn.Dropout2d(p=drop_prob),
            "act": activations[act_type],
            "norm": normalizations[norm_type],
            "scale": lambda **kwargs: Scale2d(**kwargs),
        }

        layers = []
        for lyr in order:
            if isinstance(lyr, dict):
                lyr = lyr.copy()
                name = lyr.pop("name")
                layers.append(layer_dict[name](**lyr))
                continue

            layers.append(layer_dict[lyr]())

        # Define forward pass
        self.layers = nn.Sequential(*layers)

    def forward(self, input):
        """
        Args:
            input (torch.Tensor): Input tensor of shape ``(B,C_{in},D,H,W)``.

        Returns:
            (torch.Tensor): Output tensor of shape ``(B,C_{in},D,H,W)``.
        """
        return self.layers(input)


class ResBlock(nn.Module):
    """
    A ResNet block that consists of two convolutional layers
    followed by a residual connection.
    """

    def __init__(
        self,
        in_chans,
        out_chans,
        kernel_size,
        drop_prob,
        act_type: str = "relu",
        norm_type: str = "none",
        norm_affine: bool = False,
        order: Tuple[str, str, str, str] = ("norm", "act", "drop", "conv"),
        bias: bool = True,
        num_conv_blocks: int = 2,
    ):
        """
        Args:
            in_chans: Number of channels in the input (``C_{in}``).
            out_chans: Number of channels in the output (``C_{out}``).
            drop_prob: Dropout probability.
        """
        super().__init__()

        conv_block_kwargs = dict(
            kernel_size=kernel_size,
            drop_prob=drop_prob,
            act_type=act_type,
            norm_type=norm_type,
            norm_affine=norm_affine,
            order=order,
            bias=bias,
        )

        channels = [(in_chans, out_chans)] + [(out_chans, out_chans)] * (num_conv_blocks - 1)
        self.layers = nn.Sequential(
            *[ConvBlock(in_ch, out_ch, **conv_block_kwargs) for (in_ch, out_ch) in channels]
        )

        if in_chans != out_chans:
            self.resample = nn.Conv2d(in_chans, out_chans, kernel_size=1, bias=bias)
        else:
            self.resample = nn.Identity()

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input (torch.Tensor): Input tensor of shape ``(B,C_{in},D,H,W)``.

        Returns:
            (torch.Tensor): Output tensor of shape ``(B,C_{out},D,H,W)``.
        """

        # To have a residual connection, number of inputs must be equal to outputs
        shortcut = self.resample(input)

        return self.layers(input) + shortcut
