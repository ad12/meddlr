import unittest

import torch

from meddlr.modeling.layers.gauss import GaussianBlur, get_gaussian_kernel


class TestGaussianBlur(unittest.TestCase):
    def test_kernel_normalized(self):
        for dim in [1, 2, 3]:
            kernel_size = (3,) * dim
            sigma = torch.rand(dim).tolist()
            kernel = get_gaussian_kernel(kernel_size, sigma)
            assert torch.allclose(kernel.sum(), torch.Tensor([1.0]))

    def test_filter(self):
        s = 5
        x = torch.zeros(s, s, s)
        x[s // 2, s // 2, s // 2] = 1.0
        x = x.unsqueeze(-1).unsqueeze(-1)
        gauss = GaussianBlur((3, 3, 3), (1.0, 1.0, 1.0))
        y = gauss(x)  # noqa: F841
