import inspect
from abc import ABC, abstractmethod
from typing import Tuple, Union

import torch
import torch.nn as nn

from meddlr.modeling.blocks.conv_blocks import SimpleConvBlockNd

__all__ = [
    "ResBlockNd",
    "ResBlock2d",
    "ResBlock3d",
    "ConcatBlockNd",
    "ConcatBlock2d",
    "ConcatBlock3d",
]


class _SimpleFuseBlockNd(nn.Module, ABC):
    """Series of :class:`SimpleConvBlockNd` with residual connection."""

    # Assumes order is the last argument in SimpleConvBlockNd
    _DEFAULT_CONV_BLOCK_ORDER = inspect.getfullargspec(SimpleConvBlockNd).defaults[-1]

    def __init__(
        self,
        in_channels: int,
        kernel_size: Union[int, Tuple[int, ...]],
        n_blocks: int,
        dimension: int,
        connect_before: str = None,
        **kwargs,
    ):
        """
        Args:
            in_channels (int): Number of channels in the input.
            kernel_size (`int(s)`): Convolution kernel size.
            n_blocks (int): Number of conv blocks.
            dimension (int): Integer specifying the dimension of convolution.
            connect_before (str, optional): Layer to add residual connection before in conv block.
                For example, if `n_blocks=1`, conv block `order=("conv", "batchnorm", "relu")`,
                and `connect_before="relu"`, residual block will look like below. If `None`,
                residual connection will be made after full conv block.
                    x -> Conv -> BatchNorm -> + -> ReLU
                    |                         ^
                    |                         |
                    --------------------------
            kwargs: `SimpleConvBlockNd` arguments. `in_channels` and `out_channels` required.
        """
        super().__init__()
        self.n_blocks = n_blocks

        # Determine order for connecting before.
        order = kwargs.pop("order", self._DEFAULT_CONV_BLOCK_ORDER)
        if connect_before:
            if connect_before not in order:
                raise ValueError(f"Layer {connect_before} not in conv block `order` ({order})")
            layer_idx = order.index(connect_before)
            if layer_idx == 0:
                raise ValueError(
                    f"Layer {connect_before} occurs first in conv block `order` ({order}). "
                    f"Reduce n_block by 1, set `connect_before=None`, and add `SimpleConvBlockNd` "
                    f"after this residual block."
                )
            standard_order = [order] * (self.n_blocks - 1)
            split_order = [order[:layer_idx], order[layer_idx:]]
            conv_orders = standard_order + split_order
        else:
            conv_orders = [order] * self.n_blocks
        self.connect_before = connect_before

        # Build conv blocks
        self.blocks = nn.ModuleDict(
            {
                f"block_{i+1}": SimpleConvBlockNd(
                    in_channels=in_channels,
                    out_channels=in_channels,
                    kernel_size=kernel_size,
                    dimension=dimension,
                    order=order,
                    **kwargs,
                )
                for i, order in enumerate(conv_orders)
            }
        )
        assert (self.connect_before and len(self.blocks) == self.n_blocks + 1) or (
            not self.connect_before and len(self.blocks) == self.n_blocks
        )

    def forward(self, x):
        out = x
        for i in range(self.n_blocks):
            out = self.blocks[f"block_{i+1}"](out)

        out = self.fuse(x, out)

        # Handle any remaining layers when connect_before is specified
        if self.connect_before:
            out = self.blocks[f"block_{self.n_blocks + 1}"](out)

        return out

    @abstractmethod
    def fuse(self, x, y):
        """Fuse two tensors"""
        pass


class ResBlockNd(_SimpleFuseBlockNd):
    """Residual block.

    This block adds a residual connection to the the :cls:`SimpleConvBlockNd` block.
    The order of the layers follows the same order used by :cls:`SimpleConvBlockNd`
    and can be manually configured using the ``order`` argument.

    Args:
        in_channels (int): Number of channels in the input.
        kernel_size (int(s)): Convolution kernel size.
        n_blocks (int): Number of conv blocks.
        dimension (int): Integer specifying the dimension of convolution.
        connect_before (str, optional): Layer to add residual connection before in conv block.
            For example, if `n_blocks=1`, conv block `order=("conv", "batchnorm", "relu")`,
            and `connect_before="relu"`, residual block will look like below. If `None`,
            residual connection will be made after full conv block.
                x -> Conv -> BatchNorm -> + -> ReLU
                |                         ^
                |                         |
                --------------------------
        kwargs: `SimpleConvBlockNd` arguments. `in_channels` and `out_channels` required.
    """

    def __init__(
        self,
        in_channels: int,
        kernel_size: Union[int, Tuple[int, ...]],
        n_blocks: int,
        dimension: int,
        connect_before: str = None,
        **kwargs,
    ):
        super().__init__(in_channels, kernel_size, n_blocks, dimension, connect_before, **kwargs)

    def fuse(self, x, y):
        return x + y


class ResBlock2d(ResBlockNd):
    """2D implementation of :cls:`ResBlockNd`."""

    def __init__(
        self,
        in_channels: int,
        kernel_size: Union[int, Tuple[int, ...]],
        n_blocks: int,
        connect_before: str = None,
        **kwargs,
    ):
        super().__init__(in_channels, kernel_size, n_blocks, 2, connect_before, **kwargs)


class ResBlock3d(ResBlockNd):
    """3D implementation of :class:`ResBlockNd`."""

    def __init__(
        self,
        in_channels: int,
        kernel_size: Union[int, Tuple[int, ...]],
        n_blocks: int,
        connect_before: str = None,
        **kwargs,
    ):
        super().__init__(in_channels, kernel_size, n_blocks, 3, connect_before, **kwargs)


class ConcatBlockNd(_SimpleFuseBlockNd):
    """

    Args:
        in_channels (int): Number of channels in the input.
        kernel_size (`int(s)`): Convolution kernel size.
        n_blocks (int): Number of conv blocks.
        dimension (int): Integer specifying the dimension of convolution.
        connect_before (str, optional): Layer to add residual connection before in conv block.
            For example, if `n_blocks=1`, conv block `order=("conv", "batchnorm", "relu")`,
            and `connect_before="relu"`, residual block will look like below. If `None`,
            residual connection will be made after full conv block.
                x -> Conv -> BatchNorm -> [concat] -> ReLU
                |                            ^
                |                            |
                ------------------------------
        kwargs: `SimpleConvBlockNd` arguments. `in_channels` and `out_channels` required.
    """

    def __init__(
        self,
        in_channels: int,
        kernel_size: Union[int, Tuple[int, ...]],
        n_blocks: int,
        dimension: int,
        connect_before: str = None,
        **kwargs,
    ):
        super().__init__(in_channels, kernel_size, n_blocks, dimension, connect_before, **kwargs)

    def fuse(self, x, y):
        return torch.cat([x, y], dim=1)


class ConcatBlock2d(ConcatBlockNd):
    """2D implementation of :class:`ConcatBlockNd`."""

    def __init__(
        self,
        in_channels: int,
        kernel_size: Union[int, Tuple[int, ...]],
        n_blocks: int,
        connect_before: str = None,
        **kwargs,
    ):
        super().__init__(in_channels, kernel_size, n_blocks, 2, connect_before, **kwargs)


class ConcatBlock3d(ConcatBlockNd):
    """3D implementation of :class:`ConcatBlockNd`."""

    def __init__(
        self,
        in_channels: int,
        kernel_size: Union[int, Tuple[int, ...]],
        n_blocks: int,
        connect_before: str = None,
        **kwargs,
    ):
        super().__init__(in_channels, kernel_size, n_blocks, 3, connect_before, **kwargs)
