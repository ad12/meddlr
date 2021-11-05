from typing import Sequence, Tuple, Union

import torch
from torch import nn

from meddlr.config.config import configurable
from meddlr.modeling.blocks import SimpleConvBlockNd
from meddlr.modeling.layers.build import get_layer_kind, get_layer_type
from meddlr.modeling.meta_arch import META_ARCH_REGISTRY


@META_ARCH_REGISTRY.register()
class GeneralizedUNet(nn.Module):
    _VERSION = 1

    @configurable
    def __init__(
        self,
        dimensions,
        in_channels,
        out_channels,
        channels: Sequence[int],
        strides: Sequence[int] = 1,
        kernel_size: Union[Sequence[int], int] = 3,
        up_kernel_size: Union[Sequence[int], int] = None,
        dropout: float = 0.0,
        block_order: Tuple[str, ...] = ("conv", "relu", "conv", "relu", "batchnorm", "dropout"),
    ):
        super().__init__()

        channels = list(channels)
        depth = len(channels)
        assert depth >= 2  # must have at least 2 blocks for U-Net

        kernel_size = self._arg_to_seq(kernel_size, depth)
        up_kernel_size = (
            self._arg_to_seq(up_kernel_size, depth - 1) if up_kernel_size else kernel_size[:-1]
        )
        strides = self._arg_to_seq(strides, depth)
        pool_type = get_layer_type("maxpool", dimension=dimensions)

        block_order_names: Sequence[str] = [x if isinstance(x, str) else x[0] for x in block_order]
        act_idx = [i for i, x in enumerate(block_order_names) if get_layer_kind(x) == "act"][0]
        norm_idx = [i for i, x in enumerate(block_order_names) if get_layer_kind(x) == "norm"][0]
        up_block_order = ("convtranspose", block_order[act_idx], block_order[norm_idx])

        # Down blocks + Bottleneck
        down_blocks = {}
        pool_blocks = {}
        for i, inc, outc, ks, s in zip(
            range(depth), [in_channels] + channels[:-1], channels, kernel_size, strides
        ):
            d_block = SimpleConvBlockNd(
                inc,
                outc,
                kernel_size=ks,
                dimension=dimensions,
                stride=s,
                dropout=dropout,
                order=block_order,
            )
            down_blocks[f"block{i}"] = d_block
            if i < depth - 1:
                pool_blocks[f"block{i}"] = pool_type(2)
        self.down_blocks = nn.ModuleDict(down_blocks)
        self.pool_blocks = nn.ModuleDict(pool_blocks)

        # Up blocks - Up conv + concat + Block
        up_blocks = {}
        for i, inc, outc, ks, s in zip(
            range(depth - 2, -1, -1),
            channels[::-1],
            channels[::-1][1:],
            kernel_size[::-1],
            strides[::-1],
        ):
            conv_t = SimpleConvBlockNd(
                inc,
                outc,
                dimension=dimensions,
                kernel_size=2,
                stride=2,
                order=up_block_order,
                padding=None,
            )
            block = SimpleConvBlockNd(
                outc * 2,
                outc,
                kernel_size=ks,
                dimension=dimensions,
                stride=s,
                dropout=dropout,
                order=block_order,
            )
            block = nn.ModuleList([conv_t, block])
            up_blocks[f"block{i}"] = block
        self.up_blocks = nn.ModuleDict(up_blocks)

        self.output_block = get_layer_type("conv", dimension=dimensions)(
            channels[0],
            out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
        )

    @property
    def bottleneck(self):
        """Easy access property for bottleneck layer."""
        return self.down_blocks[list(self.down_blocks.keys())[-1]]

    @property
    def depth(self):
        return len(self.down_blocks)

    def forward(self, x):
        down_blocks = self.down_blocks
        pool_blocks = self.pool_blocks
        up_blocks = self.up_blocks

        skip_connections = []
        for i in range(len(down_blocks)):
            x = self.down_blocks[f"block{i}"](x)
            if i < len(down_blocks) - 1:
                skip_connections.append(x)
                x = pool_blocks[f"block{i}"](x)

        for i, sc in zip(range(self.depth - 2, -1, -1), skip_connections[::-1]):
            upsample, conv_block = tuple(up_blocks[f"block{i}"])
            x = torch.cat([upsample(x), sc], dim=1)
            x = conv_block(x)

        return self.output_block(x)

    @staticmethod
    def _arg_to_seq(arg, num: int):
        """Converts arguments that are supposed to be sequences of a particular length."""
        if isinstance(arg, Sequence) and len(arg) != num:
            raise ValueError(f"Got `arg` {arg}. Expected len {num}")
        if isinstance(arg, Sequence) and len(arg) == 1:
            arg = arg[0]
        if not isinstance(arg, Sequence):
            arg = (arg,) * num
        return arg

    @classmethod
    def from_config(cls, cfg, **kwargs):
        # Returns kwargs to be passed to __init__
        if "MODEL" in cfg:
            cfg = cfg.MODEL.UNET
        elif "UNET" in cfg:
            cfg = cfg.UNET
        in_channels = cfg.get("IN_CHANNELS", None)
        out_channels = cfg.get("OUT_CHANNELS", None)
        dimensions = kwargs.get("dimensions", 2)
        num_pool_layers = cfg.NUM_POOL_LAYERS
        num_channels = tuple(cfg.CHANNELS * (2 ** i) for i in range(num_pool_layers + 1))

        init_args = {
            "dimensions": dimensions,
            "in_channels": in_channels,
            "out_channels": out_channels,
            "channels": num_channels,
            "dropout": cfg.DROPOUT,
        }
        block_order = cfg.get("BLOCK_ORDER", None)
        if block_order is not None:
            init_args["block_order"] = block_order
        init_args.update(**kwargs)
        return init_args
