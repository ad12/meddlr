import torch

from meddlr.modeling.meta_arch import GeneralizedUNet
from meddlr.modeling.meta_arch.build import _get_model_layers, initialize_model


def test_initialize_model():
    model = GeneralizedUNet(
        dimensions=2,
        in_channels=1,
        out_channels=4,
        channels=(4, 8, 16),
        block_order=("conv", "relu", "conv", "relu", "batchnorm", "dropout"),
    )
    initialize_model(
        model, initializers={"kind": "conv", "patterns": ".*bias", "initializers": "zeros_"}
    )

    layers_by_kind = _get_model_layers(model, by_kind=True)
    for layer in layers_by_kind["conv"]:
        assert torch.all(layer.bias == 0)
