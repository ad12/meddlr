import unittest
from unittest.mock import MagicMock

import torch
from torch import nn

from meddlr.modeling.meta_arch.unrolled import GeneralizedUnrolledCNN

from ...transforms.mock import generate_mock_mri_data


class TestGeneralizedUnrolledCNN(unittest.TestCase):
    def test_build_shared_weights(self):
        reg = nn.Sequential(nn.Conv2d(2, 2, 3), nn.ReLU())
        mock = MagicMock()
        reg.__call__ = mock

        num_steps = 3
        unrolled = GeneralizedUnrolledCNN(blocks=reg, num_grad_steps=num_steps)
        unrolled = unrolled.eval()

        kspace, maps, target = generate_mock_mri_data()
        with torch.no_grad():
            _ = unrolled({"kspace": kspace, "maps": maps})

        assert mock.call_count == num_steps
