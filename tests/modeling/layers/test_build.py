import unittest

from torch import nn

from ss_recon.modeling import layers
from ss_recon.modeling.layers.build import CUSTOM_LAYERS_REGISTRY, get_layer_type


class TestGetLayerType(unittest.TestCase):
    def test_pt_layers(self):
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

    def test_custom_layers(self):
        assert issubclass(get_layer_type("GaussianBlur"), layers.GaussianBlur)
        assert issubclass(get_layer_type("gaussianblur"), layers.GaussianBlur)

        assert issubclass(get_layer_type("convws", 2), layers.ConvWS2d)
        assert issubclass(get_layer_type("convws2d"), layers.ConvWS2d)
        assert issubclass(get_layer_type("convws", 3), layers.ConvWS3d)
        assert issubclass(get_layer_type("convws3d"), layers.ConvWS3d)


class TestCustomLayersRegistry(unittest.TestCase):
    def test_lowercasing(self):
        """Verify that lowercasing custom layers does not cause layer overlap."""
        custom_layer_names = {x.lower(): x for x in CUSTOM_LAYERS_REGISTRY._obj_map}
        assert len(custom_layer_names) == len(CUSTOM_LAYERS_REGISTRY._obj_map)
