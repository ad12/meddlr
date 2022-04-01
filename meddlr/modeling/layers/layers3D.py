"""
Implementations of different CNNs.
"""

from torch import nn

from meddlr.utils.transforms import center_crop


class SeparableConv3d(nn.Module):
    """
    A separable 3D convolutional operator.
    """

    def __init__(self, in_chans, out_chans, kernel_size, spatial_chans=None, act_type="relu"):
        """
        Args:
            in_chans (int): Number of channels in the input.
            out_chans (int): Number of channels in the output.
            kernel_size (int): Size of kernel (repeated for all three dimensions).
        """
        super().__init__()

        sp_kernel_size = (1, kernel_size, kernel_size)
        sp_pad_size = (0, 1, 1)
        t_kernel_size = (kernel_size, 1, 1)
        t_pad_size = (1, 0, 0)

        if spatial_chans is None:
            # Force number of spatial features, such that the total number of
            # parameters is the same as a nn.Conv3D(in_chans, out_chans)
            spatial_chans = (kernel_size**3) * in_chans * out_chans
            spatial_chans /= (kernel_size**2) * in_chans + kernel_size * out_chans
            spatial_chans = int(spatial_chans)

        # Define each layer in SeparableConv3d block
        spatial_conv = nn.Conv3d(
            in_chans, spatial_chans, kernel_size=sp_kernel_size, padding=sp_pad_size
        )
        temporal_conv = nn.Conv3d(
            spatial_chans, out_chans, kernel_size=t_kernel_size, padding=t_pad_size
        )

        # Define choices for intermediate activation layer
        activations = nn.ModuleDict([["none", nn.Identity()]["relu", nn.ReLU()]])

        # Define the forward pass
        self.layers = nn.Sequential(spatial_conv, activations[act_type], temporal_conv)

    def forward(self, input):
        """
        Args:
            input (torch.Tensor): Input tensor of shape ``(B,C_{in},D,H,W)``.

        Returns:
            (torch.Tensor): Output tensor of shape ``(B,C_{in},D,H,W)``.
        """
        return self.layers(input)


class ConvBlock(nn.Module):
    """
    A 3D Convolutional Block that consists of Norm -> ReLU -> Dropout -> Conv

    Based on implementation described by:
        K He, et al. "Identity Mappings in Deep Residual Networks" arXiv:1603.05027
    """

    def __init__(
        self,
        in_chans,
        out_chans,
        kernel_size,
        drop_prob,
        conv_type="conv3d",
        act_type="relu",
        norm_type="none",
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

        # Define choices for each layer in ConvBlock
        normalizations = nn.ModuleDict(
            [
                ["none", nn.Identity()],
                ["instance", nn.InstanceNorm3d(in_chans, affine=False)],
                ["batch", nn.BatchNorm3d(in_chans, affine=False)],
            ]
        )
        activations = nn.ModuleDict([["relu", nn.ReLU()], ["leaky_relu", nn.LeakyReLU()]])
        dropout = nn.Dropout3d(p=drop_prob, inplace=True)

        # Note: don't use ModuleDict here. Otherwise, the parameters for the un-selected
        # convolution type will still be initialized and added to model.parameters()
        if conv_type == "conv3d":
            convolution = nn.Conv3d(in_chans, out_chans, kernel_size=kernel_size, padding=1)
        else:
            convolution = SeparableConv3d(in_chans, out_chans, kernel_size=kernel_size, padding=1)

        # Define forward pass
        self.layers = nn.Sequential(
            normalizations[norm_type], activations[act_type], dropout, convolution
        )

    def forward(self, input):
        """
        Args:
            input (torch.Tensor): Input tensor of shape ``(B,C_{in},D,H,W)``.

        Returns:
            (torch.Tensor): Output tensor of shape ``(B,C_{in},D,H,W)``.
        """
        return self.layers(input)

    def __repr__(self):
        return (
            f"ConvBlock3D(in_chans={self.in_chans}, out_chans={self.out_chans}, "
            f"drop_prob={self.drop_prob})"
        )


class ResBlock(nn.Module):
    """
    A ResNet block that consists of two convolutional layers followed by a residual connection.
    """

    def __init__(self, in_chans, out_chans, kernel_size, drop_prob):
        """
        Args:
            in_chans (int): Number of channels in the input.
            out_chans (int): Number of channels in the output.
            drop_prob (float): Dropout probability.
        """
        super().__init__()

        self.layers = nn.Sequential(
            ConvBlock(in_chans, out_chans, kernel_size, drop_prob),
            ConvBlock(out_chans, out_chans, kernel_size, drop_prob),
        )

        if in_chans != out_chans:
            self.resample = nn.Conv3d(in_chans, out_chans, kernel_size=1)
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

    def __init__(self, num_resblocks, in_chans, chans, kernel_size, drop_prob, circular_pad=True):
        """ """
        super().__init__()

        self.circular_pad = circular_pad
        self.pad_size = 2 * num_resblocks + 1

        # Declare ResBlock layers
        self.res_blocks = nn.ModuleList([ResBlock(in_chans, chans, kernel_size, drop_prob)])
        for _ in range(num_resblocks - 1):
            self.res_blocks += [ResBlock(chans, chans, kernel_size, drop_prob)]

        # Declare final conv layer (down-sample to original in_chans)
        self.final_layer = nn.Conv3d(chans, in_chans, kernel_size=kernel_size, padding=1)

    def forward(self, input):
        """
        Args:
            input (torch.Tensor): Input tensor of shape ``(B,C_{in},D,H,W)``.

        Returns:
            (torch.Tensor): Output tensor of shape ``(B,C_{in},D,H,W)``.
        """

        orig_shape = input.shape
        if self.circular_pad:
            input = nn.functional.pad(input, (0, 0, 0, 0) + 2 * (self.pad_size,), mode="circular")
            # input = nn.functional.pad(input, 4*(self.pad_size,) + (0,0), mode='replicate')

        # Perform forward pass through the network
        output = input
        for res_block in self.res_blocks:
            output = res_block(output)
        output = self.final_layer(output) + input

        return center_crop(output, orig_shape)
