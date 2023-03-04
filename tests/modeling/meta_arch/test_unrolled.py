import unittest

import torch
from torch import nn

from meddlr.modeling.meta_arch.unrolled import GeneralizedUnrolledCNN

from ...transforms.mock import MockCounter, generate_mock_mri_data


class TestGeneralizedUnrolledCNN(unittest.TestCase):
    def test_build_shared_weights(self):
        """
        If the model is sharing weights, the same regularization block
        should be called multiple times.
        """
        reg = nn.Sequential(nn.Conv2d(2, 2, 3, padding=1), nn.ReLU())
        reg = MockCounter(reg)

        num_steps = 3
        unrolled = GeneralizedUnrolledCNN(blocks=reg, num_grad_steps=num_steps)
        unrolled = unrolled.eval()

        kspace, maps, target = generate_mock_mri_data()
        with torch.no_grad():
            _ = unrolled({"kspace": kspace, "maps": maps})

        assert reg.call_count("__call__") == num_steps
