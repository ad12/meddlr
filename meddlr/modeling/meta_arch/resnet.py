from typing import Any, Dict, Sequence, Tuple, Union

from torch import nn

from meddlr.config.config import CfgNode, configurable
from meddlr.modeling.layers.layers2D import ResBlock
from meddlr.modeling.meta_arch.build import META_ARCH_REGISTRY


@META_ARCH_REGISTRY.register()
class ResNetModel(nn.Module):
    """
    A fully convolutional residual network.

    A ResNet model that consists of a series of ResBlocks.
    Each ResBlock consists of a series of ConvBlocks.
    """

    @configurable
    def __init__(
        self,
        num_blocks: int,
        in_channels: int,
        channels: int,
        *,
        kernel_size: Union[int, Tuple[int, int]] = 3,
        dropout: float = 0.0,
        circular_pad: bool = False,
        act_type: str = "relu",
        norm_type: str = "none",
        norm_affine: bool = False,
        order: Tuple[str, str, str, str] = ("norm", "act", "drop", "conv"),
        pre_conv: bool = False,
        post_conv: bool = False,
        bias: bool = True,
        num_conv_blocks: int = 2,
    ):
        """
        Args:
            num_blocks: Number of ResBlocks to use.
            in_channels: Number of channels in the input (``C_{in}``).
            chans: Number of output channels of the first convolutional layer
            kernel_size: Kernel size of the convolutional layers.
            drop_prob: Dropout probability.
            circular_pad: Whether to use circular padding.
            act_type: Type of activation to use.
            norm_type: Type of normalization to use.
            norm_affine: Whether to learn affine parameters in normalization.
            order: Order of operations in the conv block.
            pre_conv: Whether to use a convolutional layer before the ResBlocks.
            post_conv: Whether to use a convolutional layer after the ResBlocks.
            bias: Whether to use a bias in the convolutional layers.
            num_conv_blocks: Number of ConvBlocks to use in each ResBlock.
        """
        super().__init__()

        if circular_pad:
            raise NotImplementedError(
                "Circular padding is not available. "
                "It is retained in the init to be used in the future."
            )
        self.circular_pad = circular_pad
        self.pad_size = 2 * num_blocks + 1
        padding = _get_same_padding(kernel_size)

        self.pre_conv = None
        if pre_conv:
            self.pre_conv = nn.Conv2d(
                in_channels, channels, kernel_size=kernel_size, bias=bias, padding=padding
            )

        resblock_params = {
            "act_type": act_type,
            "norm_type": norm_type,
            "norm_affine": norm_affine,
            "order": order,
            "kernel_size": kernel_size,
            "drop_prob": dropout,
            "bias": bias,
            "num_conv_blocks": num_conv_blocks,
        }
        # Declare ResBlock layers
        self.res_blocks: Sequence[ResBlock] = nn.ModuleList(
            [ResBlock(channels if pre_conv else in_channels, channels, **resblock_params)]
        )
        for _ in range(num_blocks - 1):
            self.res_blocks += [ResBlock(channels, channels, **resblock_params)]

        # Declare final conv layer (down-sample to original in_chans)
        # TODO: Rename this to pre_sum_conv
        self.final_layer = nn.Conv2d(
            channels, in_channels, kernel_size=kernel_size, padding=padding, bias=bias
        )

        self.post_conv = None
        if post_conv:
            self.post_conv = nn.Conv2d(
                in_channels, in_channels, kernel_size=kernel_size, bias=bias, padding=padding
            )

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

        output = input

        if self.pre_conv:
            output = self.pre_conv(output)

        for res_block in self.res_blocks:
            output = res_block(output)
        output = self.final_layer(output) + input

        if self.post_conv:
            output = self.post_conv(output)

        # return center_crop(output, orig_shape)

        return output

    @classmethod
    def from_config(cls, cfg: CfgNode, **kwargs) -> Dict[str, Any]:
        kernel_size = cfg.MODEL.RESNET.KERNEL_SIZE
        if len(kernel_size) == 1:
            kernel_size = kernel_size[0]
        out = {
            "num_blocks": cfg.MODEL.RESNET.NUM_BLOCKS,
            "in_channels": cfg.MODEL.RESNET.IN_CHANNELS,
            "channels": cfg.MODEL.RESNET.CHANNELS,
            "kernel_size": kernel_size,
            "dropout": cfg.MODEL.RESNET.DROPOUT,
            "circular_pad": cfg.MODEL.RESNET.PADDING == "circular",
            "act_type": cfg.MODEL.RESNET.CONV_BLOCK.ACTIVATION,
            "norm_type": cfg.MODEL.RESNET.CONV_BLOCK.NORM,
            "norm_affine": cfg.MODEL.RESNET.CONV_BLOCK.NORM_AFFINE,
            "order": cfg.MODEL.RESNET.CONV_BLOCK.ORDER,
            "num_conv_blocks": cfg.MODEL.RESNET.CONV_BLOCK.NUM_BLOCKS,
            "pre_conv": cfg.MODEL.RESNET.PRE_CONV,
            "post_conv": cfg.MODEL.RESNET.POST_CONV,
            "bias": cfg.MODEL.RESNET.BIAS,
        }
        out.update(kwargs)
        return out


def _get_same_padding(kernel_size: Union[int, Tuple[int, int]]):
    if isinstance(kernel_size, int):
        kernel_size = (kernel_size, kernel_size)
    else:
        assert len(kernel_size) == 2
    if not all(k % 2 == 1 for k in kernel_size):
        raise ValueError("Kernel size must be odd - got {}".format(kernel_size))
    padding = tuple(k // 2 for k in kernel_size)

    return padding
