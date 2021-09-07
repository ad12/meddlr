import unittest

import torch
import torchvision.transforms.functional as tvf

from ss_recon.transforms.base.spatial import AffineTransform, FlipTransform, Rot90Transform


class TestAffineTransform(unittest.TestCase):
    def test_func(self):
        angle, translate, scale = 12, [5, 5], 2.0
        x = torch.randn(1, 3, 50, 50)

        expected_out = tvf.affine(
            x, angle=angle, translate=translate, scale=scale, shear=[0.0, 0.0]
        )

        tfm = AffineTransform(angle=angle, translate=translate, scale=scale)
        out = tfm.apply_image(x)

        assert torch.all(out == expected_out)


class TestFlipTransform(unittest.TestCase):
    def test_func(self):
        x = torch.randn(1, 3, 50, 50)

        dims = (-1, -2)
        expected_out = torch.flip(x, dims=dims)
        tfm = FlipTransform(dims)
        out = tfm.apply_image(x)
        assert torch.all(out == expected_out)

        dims = -1
        expected_out = torch.flip(x, dims=(dims,))
        tfm = FlipTransform(dims)
        out = tfm.apply_image(x)
        assert torch.all(out == expected_out)

        dims = (-2,)
        expected_out = torch.flip(x, dims=dims)
        tfm = FlipTransform(dims)
        out = tfm.apply_image(x)
        assert torch.all(out == expected_out)

    def test_inverse(self):
        x = torch.randn(1, 3, 50, 50)

        dims = (-1, -2)
        tfm = FlipTransform(dims)
        out = tfm.inverse().apply_image(tfm.apply_image(x))
        assert torch.all(out == x)


class TestRot90Transform(unittest.TestCase):
    def test_func(self):
        x = torch.randn(1, 3, 50, 50)
        k = 1
        dims = (-1, -2)

        expected_out = torch.rot90(x, k=1, dims=dims)
        tfm = Rot90Transform(k, dims)
        out = tfm.apply_image(x)
        assert torch.all(out == expected_out)

    def test_inverse(self):
        x = torch.randn(1, 3, 50, 50)
        k = 1
        dims = (-1, -2)

        tfm = Rot90Transform(k, dims)
        out = tfm.inverse().apply_image(tfm.apply_image(x))
        assert torch.all(out == x)
