from typing import Any, Dict

import monai.networks.nets as monai_nets
import pytest
import torch

from meddlr.config import CfgNode as CN
from meddlr.config import get_cfg
from meddlr.modeling.meta_arch import GeneralizedUNet
from meddlr.modeling.meta_arch.build import _get_model_layers, build_model, initialize_model


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


@pytest.mark.parametrize(
    "meta_arch,build_kwargs",
    [
        (
            "VNet",
            {
                "in_channels": 2,
                "out_channels": 2,
                "spatial_dims": 2,
                "dropout_dim": 2,
                "dropout_prob": 0.0,
            },
        ),
        (
            "UNet",
            {
                "in_channels": 2,
                "out_channels": 2,
                "spatial_dims": 2,
                "channels": [2, 4, 8],
                "strides": [1, 1, 1],
            },
        ),
    ],
)
def test_build_monai_model(meta_arch: str, build_kwargs: Dict[str, Any]):
    cfg = get_cfg().defrost()
    cfg.MODEL.META_ARCHITECTURE = f"monai/{meta_arch}"
    cfg.set_recursive(f"MODEL.MONAI.{meta_arch}", CN(build_kwargs))
    cfg = cfg.freeze()
    monai_type = getattr(monai_nets, meta_arch)

    model = build_model(cfg)
    assert isinstance(model, monai_type)

    # TODO: Intialize both the built model and the MONAI model with
    # the same seed. Then, check that the weights are the same.
