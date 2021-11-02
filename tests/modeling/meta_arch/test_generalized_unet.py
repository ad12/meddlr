import unittest

import torch

from ss_recon.config import get_cfg
from ss_recon.modeling.meta_arch import GeneralizedUNet


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
