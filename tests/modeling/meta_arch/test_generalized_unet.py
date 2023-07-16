import unittest

import numpy as np
import torch
from torch import nn

from meddlr.config import get_cfg
from meddlr.modeling.meta_arch import GeneralizedUNet


class TestGeneralizedUnet(unittest.TestCase):
    def test_build(self):
        cfg = get_cfg()
        cfg.MODEL.META_ARCH = "GeneralizedUnet"

        model = GeneralizedUNet(cfg, dimensions=2)
        assert model.depth == 5

    def test_forward(self):
        model = GeneralizedUNet(
            dimensions=2,
            in_channels=1,
            out_channels=2,
            channels=(32, 64, 128),
            block_order=("conv", "relu", "conv", "relu", "batchnorm", "dropout"),
        )

        x = torch.randn(1, 1, 128, 128)
        out = model(x)
        assert out.shape == (1, 2, 128, 128)

    def test_build_with_order(self):
        block_order = (
            "conv",
            "instancenorm",
            ("leakyrelu", {"negative_slope": 0.1, "inplace": True}),
            "conv",
            "instancenorm",
            ("leakyrelu", {"negative_slope": 0.2, "inplace": False}),
        )
        expected_types = (
            nn.Conv2d,
            nn.InstanceNorm2d,
            nn.LeakyReLU,
            nn.Conv2d,
            nn.InstanceNorm2d,
            nn.LeakyReLU,
        )
        expected_up_types = (nn.ConvTranspose2d, nn.InstanceNorm2d, nn.LeakyReLU)

        model = GeneralizedUNet(
            dimensions=2, in_channels=1, out_channels=1, channels=(32, 64), block_order=block_order
        )

        for block in model.down_blocks.values():
            np.testing.assert_equal(tuple(type(x) for x in block), expected_types)
            assert block[2].negative_slope == 0.1
            assert block[2].inplace
            assert block[5].negative_slope == 0.2
            assert not block[5].inplace

        for blocks in model.up_blocks.values():
            conv_t = blocks[0]  # the transpose block
            np.testing.assert_equal(tuple(type(x) for x in conv_t), expected_up_types)
            assert conv_t[2].negative_slope == 0.1
            assert conv_t[2].inplace
