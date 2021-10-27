"""Implementation of 2D base layers.
"""

from typing import Tuple, Union

from torch import nn


def _get_same_padding(kernel_size: Union[int, Tuple[int, int]]):
    if isinstance(kernel_size, int):
        kernel_size = (kernel_size, kernel_size)
    else:
        assert len(kernel_size) == 2
    if not all(k % 2 == 1 for k in kernel_size):
        raise ValueError("Kernel size must be odd - got {}".format(kernel_size))
    padding = tuple(k // 2 for k in kernel_size)

    return padding


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
    ):
        """
        Args:
            in_chans (int): Number of channels in the input.
            out_chans (int): Number of channels in the output.
            drop_prob (float): Dropout probability.
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
        conv_after_norm = "norm" in order and order.index("conv") > order.index("norm")
        norm_channels = in_chans if conv_after_norm else out_chans
        normalizations = nn.ModuleDict(
            [
                ["none", nn.Identity()],
                ["instance", nn.InstanceNorm2d(norm_channels, affine=norm_affine)],
                ["batch", nn.BatchNorm2d(norm_channels, affine=norm_affine)],
            ]
        )
        activations = nn.ModuleDict([["relu", nn.ReLU()], ["leaky_relu", nn.LeakyReLU()]])
        dropout = nn.Dropout2d(p=drop_prob)
        convolution = nn.Conv2d(in_chans, out_chans, kernel_size=kernel_size, padding=padding)

        layer_dict = {
            "conv": convolution,
            "drop": dropout,
            "act": activations[act_type],
            "norm": normalizations[norm_type] if norm_type in normalizations else nn.Identity(),
        }
        layers = [layer_dict[lyr] for lyr in order]

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
    ):
        """
        Args:
            in_chans (int): Number of channels in the input.
            out_chans (int): Number of channels in the output.
            drop_prob (float): Dropout probability.
        """
        super().__init__()

        self.layers = nn.Sequential(
            ConvBlock(
                in_chans, out_chans, kernel_size, drop_prob, act_type, norm_type, norm_affine, order
            ),  # noqa
            ConvBlock(
                out_chans,
                out_chans,
                kernel_size,
                drop_prob,
                act_type,
                norm_type,
                norm_affine,
                order,
            ),  # noqa
        )

        if in_chans != out_chans:
            self.resample = nn.Conv2d(in_chans, out_chans, kernel_size=1)
        else:
            self.resample = nn.Identity()

    def forward(self, input):
        """
        Args:
            input (torch.Tensor): Input tensor of shape ``(B,C_{in},D,H,W)``.

        Returns:
            (torch.Tensor): Output tensor of shape ``(B,C_{in},D,H,W)``.
        """

        # To have a residual connection, number of inputs must be equal to outputs
        shortcut = self.resample(input)

        return self.layers(input) + shortcut


class ResNet(nn.Module):
    """
    Prototype for 3D ResNet architecture
    """

    def __init__(
        self,
        num_resblocks,
        in_chans,
        chans,
        kernel_size,
        drop_prob,
        circular_pad=False,
        act_type: str = "relu",
        norm_type: str = "none",
        norm_affine: bool = False,
        order: Tuple[str, str, str, str] = ("norm", "act", "drop", "conv"),
    ):
        """ """
        super().__init__()

        if circular_pad:
            raise NotImplementedError(
                "Circular padding is not available. "
                "It is retained in the init to be used in the future."
            )
        self.circular_pad = circular_pad
        self.pad_size = 2 * num_resblocks + 1

        resblock_params = {
            "act_type": act_type,
            "norm_type": norm_type,
            "norm_affine": norm_affine,
            "order": order,
            "kernel_size": kernel_size,
            "drop_prob": drop_prob,
        }
        # Declare ResBlock layers
        self.res_blocks = nn.ModuleList([ResBlock(in_chans, chans, **resblock_params)])
        for _ in range(num_resblocks - 1):
            self.res_blocks += [ResBlock(chans, chans, **resblock_params)]

        # Declare final conv layer (down-sample to original in_chans)
        padding = _get_same_padding(kernel_size)
        self.final_layer = nn.Conv2d(chans, in_chans, kernel_size=kernel_size, padding=padding)

    def forward(self, input):
        """
        Args:
            input (torch.Tensor): Input tensor of shape ``(B,C_{in},D,H,W)``.

        Returns:
            (torch.Tensor): Output tensor of shape ``(B,C_{in},D,H,W)``.
        """

        # orig_shape = input.shape
        # if self.circular_pad:
        #     input = nn.functional.pad(
        #         input, 2 * (self.pad_size,) + (0, 0), mode="circular"
        #     )

        # Perform forward pass through the network
        output = input
        for res_block in self.res_blocks:
            output = res_block(output)
        output = self.final_layer(output) + input

        # return center_crop(output, orig_shape)

        return output
