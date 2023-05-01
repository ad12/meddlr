import pytest
from torch import nn

from meddlr.config.config import get_cfg
from meddlr.modeling.layers.build import get_layer_kind
from meddlr.modeling.meta_arch.build import build_model


@pytest.mark.parametrize("pre_conv", [True, False])
@pytest.mark.parametrize("post_conv", [True, False])
def test_from_config(pre_conv, post_conv):
    cfg = get_cfg()
    cfg.MODEL.META_ARCHITECTURE = "ResNetModel"
    cfg.MODEL.RESNET.PRE_CONV = pre_conv
    cfg.MODEL.RESNET.POST_CONV = post_conv
    cfg.MODEL.RESNET.CONV_BLOCK.ORDER = ("conv", "act", "conv", "act")

    model = build_model(cfg)

    assert isinstance(model, nn.Module)
    assert len(model.res_blocks) == cfg.MODEL.RESNET.NUM_BLOCKS
    if cfg.MODEL.RESNET.PRE_CONV:
        assert isinstance(model.pre_conv, nn.Conv2d)
    if cfg.MODEL.RESNET.POST_CONV:
        assert isinstance(model.post_conv, nn.Conv2d)

    for res_block in model.res_blocks:
        assert len(res_block.layers) == cfg.MODEL.RESNET.CONV_BLOCK.NUM_BLOCKS
        for conv_block in res_block.layers:
            for i, layer in enumerate(conv_block.layers):
                assert get_layer_kind(type(layer)) == cfg.MODEL.RESNET.CONV_BLOCK.ORDER[i]
