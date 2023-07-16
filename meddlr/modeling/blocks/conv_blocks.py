import inspect
from typing import List, Sequence, Tuple, Union

import torch.nn as nn

from meddlr.modeling.layers.build import LayerInfo, LayerInfoRawType, build_layer_info_from_seq

__all__ = [
    "SimpleConvBlockNd",
    "SimpleConvBlock2d",
    "SimpleConvBlock3d",
]


class SimpleConvBlockNd(nn.Sequential):
    """A convolutional block supporting normalization, conv, activation, and dropout.

    The first conv layer will change the number of channels from
    ``in_channels`` to ``out_channels``.

    The order of layers can be specified by certain keywords:
        * "conv": Convolution layer
        * "convws": Convolution + Weight Standardization layer
        * "batchnorm"/"bn": Batch Normalization
        * "instancenorm": Instance Normalization
        * "groupnorm": Group Normalization
        * "relu": ReLU
        * "dropout": Dropout

    Args:
        in_channels: Number of channels in the input.
        out_channels: Number of channels in the output.
        kernel_size: Convolution kernel size.
        dimension: Integer specifying the dimension of convolution.
        dropout: Dropout probability.
        order: Order of layers in the convolution block.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int, ...]],
        dimension: int,
        stride: Union[int, Tuple[int, ...]] = 1,
        dropout: float = 0.0,
        padding: Union[str, int, Tuple[int, ...]] = "same",
        order: Sequence[Union[LayerInfoRawType, LayerInfo]] = (
            "conv",
            "batchnorm",
            "relu",
            "dropout",
        ),
    ):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.dropout_prob = dropout

        if isinstance(kernel_size, int):
            kernel_size = (kernel_size,) * dimension
        else:
            assert len(kernel_size) == dimension

        if not padding:
            padding = 0
        if padding == "same":
            if not all(k % 2 == 1 for k in kernel_size):
                raise ValueError(f"Kernel sizes must be odd - got {kernel_size}")
            padding = tuple(k // 2 for k in kernel_size)
        elif not isinstance(padding, int) and not isinstance(padding, Tuple):  # TODO: Improve check
            raise ValueError(f"Invalid value for padding '{padding}'")

        order: List[LayerInfo] = build_layer_info_from_seq(order, dimension=dimension)

        layers = []
        running_num_channels = in_channels
        for layer_info in order:
            name, layer_cls, kind = layer_info.name, layer_info.ltype, layer_info.kind
            if kind == "conv":
                layer = layer_info.build(
                    running_num_channels,
                    out_channels,
                    kernel_size=kernel_size,
                    padding=padding,
                    stride=stride,
                )
                running_num_channels = out_channels
            elif kind == "norm":
                sig = inspect.signature(layer_cls)
                lyr_kwargs = {}
                for kwarg_name in ("num_channels", "num_features"):
                    if kwarg_name not in layer_info.init_kwargs and kwarg_name in sig.parameters:
                        lyr_kwargs[kwarg_name] = running_num_channels
                layer = layer_info.build(**lyr_kwargs)
            elif kind == "dropout":
                layer = layer_info.build(dropout)
            elif kind == "act":
                layer = layer_info.build()
            else:
                raise ValueError(f"Layer {name} (kind: {kind}) not supported")
            layers.append(layer)

        # Define forward pass
        super().__init__(*layers)


class SimpleConvBlock2d(SimpleConvBlockNd):
    """2D implementation of :class:`SimpleConvBlockNd`.

    Args:
        in_channels (int): Number of channels in the input.
        out_channels (int): Number of channels in the output.
        kernel_size (`int(s)`): Convolution kernel size.
        dropout (float, optional): Dropout probability.
        order (:obj:`str(s)`, optional): Order of layers in the convolution block. Note layers
            can be repeated.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int, int]],
        stride: Union[int, Tuple[int, int]] = 1,
        dropout: float = 0.0,
        order: Tuple[str, ...] = ("conv", "batchnorm", "relu", "dropout"),
        padding: Union[str, int, Tuple[int, ...]] = "same",
    ):
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            dimension=2,
            stride=stride,
            dropout=dropout,
            order=order,
            padding=padding,
        )


class SimpleConvBlock3d(SimpleConvBlockNd):
    """3D implementation of :class:`SimpleConvBlockNd`.

    Args:
        in_channels (int): Number of channels in the input.
        out_channels (int): Number of channels in the output.
        kernel_size (`int(s)`): Convolution kernel size.
        dropout (float, optional): Dropout probability.
        order (:obj:`str(s)`, optional): Order of layers in the convolution block. Note layers
            can be repeated.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int, int, int]],
        stride: Union[int, Tuple[int, int]] = 1,
        dropout: float = 0.0,
        order: Tuple[str, ...] = ("conv", "batchnorm", "relu", "dropout"),
        padding: Union[str, int, Tuple[int, ...]] = "same",
    ):
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            dimension=3,
            stride=stride,
            dropout=dropout,
            order=order,
            padding=padding,
        )
