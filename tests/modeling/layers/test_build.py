from typing import Any, Dict, Optional

import numpy as np
import pytest
import torch
from torch import nn

from meddlr.modeling import layers
from meddlr.modeling.layers.build import CUSTOM_LAYERS_REGISTRY, LayerInfo, get_layer_type


def test_pt_layers_type():
    assert issubclass(get_layer_type("conv1d"), nn.Conv1d)
    assert issubclass(get_layer_type("conv", 1), nn.Conv1d)
    assert issubclass(get_layer_type("conv2d"), nn.Conv2d)
    assert issubclass(get_layer_type("conv", 2), nn.Conv2d)
    assert issubclass(get_layer_type("conv3d"), nn.Conv3d)
    assert issubclass(get_layer_type("conv", 3), nn.Conv3d)

    assert issubclass(get_layer_type("convtranspose1d"), nn.ConvTranspose1d)
    assert issubclass(get_layer_type("convtranspose", 1), nn.ConvTranspose1d)
    assert issubclass(get_layer_type("convtranspose2d"), nn.ConvTranspose2d)
    assert issubclass(get_layer_type("convtranspose", 2), nn.ConvTranspose2d)
    assert issubclass(get_layer_type("convtranspose3d"), nn.ConvTranspose3d)
    assert issubclass(get_layer_type("convtranspose", 3), nn.ConvTranspose3d)

    assert issubclass(get_layer_type("batchnorm1d"), nn.BatchNorm1d)
    assert issubclass(get_layer_type("batchnorm", 1), nn.BatchNorm1d)
    assert issubclass(get_layer_type("batchnorm2d"), nn.BatchNorm2d)
    assert issubclass(get_layer_type("batchnorm", 2), nn.BatchNorm2d)
    assert issubclass(get_layer_type("batchnorm3d"), nn.BatchNorm3d)
    assert issubclass(get_layer_type("batchnorm", 3), nn.BatchNorm3d)

    assert issubclass(get_layer_type("syncbatchnorm"), nn.SyncBatchNorm)
    assert issubclass(get_layer_type("syncbatchnorm", 1), nn.SyncBatchNorm)
    assert issubclass(get_layer_type("syncbatchnorm", 2), nn.SyncBatchNorm)
    assert issubclass(get_layer_type("syncbatchnorm", 3), nn.SyncBatchNorm)

    assert issubclass(get_layer_type("groupnorm"), nn.GroupNorm)
    assert issubclass(get_layer_type("groupnorm", 1), nn.GroupNorm)
    assert issubclass(get_layer_type("groupnorm", 2), nn.GroupNorm)
    assert issubclass(get_layer_type("groupnorm", 3), nn.GroupNorm)

    assert issubclass(get_layer_type("instancenorm1d"), nn.InstanceNorm1d)
    assert issubclass(get_layer_type("instancenorm", 1), nn.InstanceNorm1d)
    assert issubclass(get_layer_type("instancenorm2d"), nn.InstanceNorm2d)
    assert issubclass(get_layer_type("instancenorm", 2), nn.InstanceNorm2d)
    assert issubclass(get_layer_type("instancenorm3d"), nn.InstanceNorm3d)
    assert issubclass(get_layer_type("instancenorm", 3), nn.InstanceNorm3d)

    assert issubclass(get_layer_type("layernorm"), nn.LayerNorm)
    assert issubclass(get_layer_type("layernorm", 1), nn.LayerNorm)
    assert issubclass(get_layer_type("layernorm", 2), nn.LayerNorm)
    assert issubclass(get_layer_type("layernorm", 3), nn.LayerNorm)

    assert issubclass(get_layer_type("dropout1d"), nn.Dropout)
    assert issubclass(get_layer_type("dropout", 1), nn.Dropout)
    assert issubclass(get_layer_type("dropout2d"), nn.Dropout2d)
    assert issubclass(get_layer_type("dropout", 2), nn.Dropout2d)
    assert issubclass(get_layer_type("dropout3d"), nn.Dropout3d)
    assert issubclass(get_layer_type("dropout", 3), nn.Dropout3d)

    assert issubclass(get_layer_type("maxpool1d"), nn.MaxPool1d)
    assert issubclass(get_layer_type("maxpool", 1), nn.MaxPool1d)
    assert issubclass(get_layer_type("maxpool2d"), nn.MaxPool2d)
    assert issubclass(get_layer_type("maxpool", 2), nn.MaxPool2d)
    assert issubclass(get_layer_type("maxpool3d"), nn.MaxPool3d)
    assert issubclass(get_layer_type("maxpool", 3), nn.MaxPool3d)

    assert issubclass(get_layer_type("maxunpool1d"), nn.MaxUnpool1d)
    assert issubclass(get_layer_type("maxunpool", 1), nn.MaxUnpool1d)
    assert issubclass(get_layer_type("maxunpool2d"), nn.MaxUnpool2d)
    assert issubclass(get_layer_type("maxunpool", 2), nn.MaxUnpool2d)
    assert issubclass(get_layer_type("maxunpool3d"), nn.MaxUnpool3d)
    assert issubclass(get_layer_type("maxunpool", 3), nn.MaxUnpool3d)

    assert issubclass(get_layer_type("avgpool1d"), nn.AvgPool1d)
    assert issubclass(get_layer_type("avgpool", 1), nn.AvgPool1d)
    assert issubclass(get_layer_type("avgpool2d"), nn.AvgPool2d)
    assert issubclass(get_layer_type("avgpool", 2), nn.AvgPool2d)
    assert issubclass(get_layer_type("avgpool3d"), nn.AvgPool3d)
    assert issubclass(get_layer_type("avgpool", 3), nn.AvgPool3d)


def test_custom_layers_type():
    assert issubclass(get_layer_type("GaussianBlur"), layers.GaussianBlur)
    assert issubclass(get_layer_type("gaussianblur"), layers.GaussianBlur)

    assert issubclass(get_layer_type("convws", 2), layers.ConvWS2d)
    assert issubclass(get_layer_type("convws2d"), layers.ConvWS2d)
    assert issubclass(get_layer_type("convws", 3), layers.ConvWS3d)
    assert issubclass(get_layer_type("convws3d"), layers.ConvWS3d)


def test_custom_layer_conflicting_names():
    """Verify that lowercasing custom layers does not cause layer overlap."""
    custom_layer_names = {x.lower(): x for x in CUSTOM_LAYERS_REGISTRY._obj_map}
    assert len(custom_layer_names) == len(CUSTOM_LAYERS_REGISTRY._obj_map)


@pytest.mark.parametrize(
    "name,dimension,init_kwargs,expected_ltype,expected_lkind",
    [
        # Convolutional layers.
        ["conv", 1, None, nn.Conv1d, "conv"],
        ["conv", 2, None, nn.Conv2d, "conv"],
        ["conv", 3, None, nn.Conv3d, "conv"],
        ["convtranspose", 1, None, nn.ConvTranspose1d, "conv"],
        ["convws", 2, None, layers.ConvWS2d, "conv"],
        ["conv1d", None, None, nn.Conv1d, "conv"],
        ["conv", 1, {}, nn.Conv1d, "conv"],
        # Normalization layers.
        ["batchnorm", 1, None, nn.BatchNorm1d, "norm"],
        ["batchnorm", 2, None, nn.BatchNorm2d, "norm"],
        ["batchnorm", 3, None, nn.BatchNorm3d, "norm"],
        ["groupnorm", None, None, nn.GroupNorm, "norm"],
        ["groupnorm", 1, None, nn.GroupNorm, "norm"],
        ["instancenorm", 1, None, nn.InstanceNorm1d, "norm"],
        ["instancenorm", 2, None, nn.InstanceNorm2d, "norm"],
        ["instancenorm", 3, None, nn.InstanceNorm3d, "norm"],
        ["layernorm", None, None, nn.LayerNorm, "norm"],
        ["layernorm", 1, None, nn.LayerNorm, "norm"],
        # Activation layers.
        ["relu", None, None, nn.ReLU, "act"],
        ["leakyrelu", None, None, nn.LeakyReLU, "act"],
        ["leakyrelu", None, {"negative_slope": 0.2, "inplace": True}, nn.LeakyReLU, "act"],
        # Dropout layers.
        ["dropout", 1, None, nn.Dropout, "dropout"],
        ["dropout", 2, None, nn.Dropout2d, "dropout"],
        ["dropout", 3, None, nn.Dropout3d, "dropout"],
        # Pooling layers.
        # TODO: Add a pool layer kind.
        ["maxpool", 1, None, nn.MaxPool1d, "unknown"],
        ["maxpool", 2, None, nn.MaxPool2d, "unknown"],
        ["maxpool", 3, None, nn.MaxPool3d, "unknown"],
        ["maxunpool", 1, None, nn.MaxUnpool1d, "unknown"],
        ["maxunpool", 2, None, nn.MaxUnpool2d, "unknown"],
        ["maxunpool", 3, None, nn.MaxUnpool3d, "unknown"],
        ["avgpool", 1, None, nn.AvgPool1d, "unknown"],
        ["avgpool", 2, None, nn.AvgPool2d, "unknown"],
        ["avgpool", 3, None, nn.AvgPool3d, "unknown"],
    ],
)
def test_layer_info_properties(
    name: str,
    dimension: str,
    init_kwargs: Optional[Dict[str, Any]],
    expected_ltype: type,
    expected_lkind: str,
):
    """Test that the layer info properties are correct."""
    layer_info = LayerInfo(name=name, dimension=dimension, init_kwargs=init_kwargs)
    assert layer_info.ltype == expected_ltype
    assert layer_info.kind == expected_lkind
    if init_kwargs:
        assert layer_info.init_kwargs == init_kwargs


@pytest.mark.parametrize(
    "layer_info,expected_layer_info",
    [
        # Success cases.
        ["conv1d", LayerInfo(name="conv1d")],
        [
            ("conv1d", {"kernel_size": 3, "stride": 2}),
            LayerInfo(name="conv1d", init_kwargs={"kernel_size": 3, "stride": 2}),
        ],
        [
            ("conv1d", ("kernel_size", 3, "stride", 2)),
            LayerInfo(name="conv1d", init_kwargs={"kernel_size": 3, "stride": 2}),
        ],
        [
            ("conv1d", (("kernel_size", 3), ("stride", 2))),
            LayerInfo(name="conv1d", init_kwargs={"kernel_size": 3, "stride": 2}),
        ],
        [
            ["conv1d", {"kernel_size": 3, "stride": 2}],
            LayerInfo(name="conv1d", init_kwargs={"kernel_size": 3, "stride": 2}),
        ],
        [
            ["conv1d", ["kernel_size", 3, "stride", 2]],
            LayerInfo(name="conv1d", init_kwargs={"kernel_size": 3, "stride": 2}),
        ],
        [
            ["conv1d", [["kernel_size", 3], ["stride", 2]]],
            LayerInfo(name="conv1d", init_kwargs={"kernel_size": 3, "stride": 2}),
        ],
        [
            {"conv1d": {"kernel_size": 3, "stride": 2}},
            LayerInfo(name="conv1d", init_kwargs={"kernel_size": 3, "stride": 2}),
        ],
        [
            {"conv1d": ["kernel_size", 3, "stride", 2]},
            LayerInfo(name="conv1d", init_kwargs={"kernel_size": 3, "stride": 2}),
        ],
        [
            {"conv1d": [["kernel_size", 3], ["stride", 2]]},
            LayerInfo(name="conv1d", init_kwargs={"kernel_size": 3, "stride": 2}),
        ],
        # Failure cases.
        [{"conv1d": {"kernel_size": 3}, "conv2d": {"kernel_size": 3}}, None],
        [("conv1d", ("kernel_size", 3, ("stride", 2))), None],
        [("conv1d", ("kernel_size", 3, "stride")), None],
    ],
)
def test_layer_info_format(layer_info, expected_layer_info: Optional[LayerInfo]):
    """Test formatting LayerInfo from raw types (typically from parsed yaml files)."""
    if expected_layer_info is None:
        with pytest.raises(ValueError):
            obj = LayerInfo.format(layer_info)
    else:
        obj = LayerInfo.format(layer_info)
        assert obj == expected_layer_info


@torch.no_grad()
@pytest.mark.parametrize(
    "layer_info,expected_layer",
    [
        [
            LayerInfo(name="leakyrelu", init_kwargs=dict(negative_slope=0.2)),
            nn.LeakyReLU(negative_slope=0.2),
        ],
        [
            LayerInfo(
                name="conv",
                dimension=1,
                init_kwargs=dict(in_channels=2, out_channels=4, kernel_size=3),
            ),
            nn.Conv1d(in_channels=2, out_channels=4, kernel_size=3),
        ],
        [
            LayerInfo(
                name="conv1d",
                init_kwargs=dict(in_channels=2, out_channels=4, kernel_size=3),
            ),
            nn.Conv1d(in_channels=2, out_channels=4, kernel_size=3),
        ],
        [
            LayerInfo(
                name="conv2d",
                init_kwargs=dict(in_channels=2, out_channels=4, kernel_size=3),
            ),
            nn.Conv2d(in_channels=2, out_channels=4, kernel_size=3),
        ],
    ],
)
def test_layer_info_build(layer_info: LayerInfo, expected_layer: nn.Module):
    """Test that the layer builds correctly."""

    def tensor_to_shape(x):
        if isinstance(x, torch.Tensor):
            return x.shape
        if isinstance(x, dict):
            return type(x)({k: tensor_to_shape(v) for k, v in x.items()})
        if isinstance(x, (list, tuple)):
            return type(x)([tensor_to_shape(v) for v in x])
        return x

    layer = layer_info.build()
    assert type(layer) == type(expected_layer)  # noqa: E721

    layer_shape = tensor_to_shape(layer.__dict__)
    expected = tensor_to_shape(expected_layer.__dict__)
    np.testing.assert_equal(layer_shape, expected)
