import unittest

import torch

import meddlr.ops as oF
import meddlr.utils.transforms as T
from meddlr.utils import env


class TestFFTReproducibility(unittest.TestCase):
    """
    Test reproducibility between meddlr.ops and meddlr.utils.transforms.

    TODO: Delete when ``meddlr/utils/transforms.py`` is to be removed.
    """

    @unittest.skipIf(env.pt_version() >= [1, 8], "torch.fft not supported in torch>=1.8")
    def test_fft2(self):
        g = torch.Generator().manual_seed(1)

        # Complex tensor as real tensor.
        x = torch.rand(4, 3, 3, 8, 2, generator=g)  # B x H x W x #coils x 2
        X = T.fft2(x)
        X2 = oF.fft2c(x, channels_last=True)
        assert torch.allclose(X2, X)

        x = torch.rand(4, 3, 3, 5, 8, 2, generator=g)  # B x H x W x D x #coils x 2
        X = T.fft2(x)
        X2 = oF.fft2c(x, channels_last=True)
        assert torch.allclose(X2, X)

        # Complex tensor.
        x = torch.rand(4, 3, 3, 8, generator=g, dtype=torch.complex64)  # B x H x W x #coils
        X = T.fft2(x)
        X2 = oF.fft2c(x, channels_last=True)
        assert torch.allclose(X2, X)

        x = torch.rand(4, 3, 3, 5, 8, generator=g, dtype=torch.complex64)  # B x H x W x D x #coils
        X = T.fft2(x)
        X2 = oF.fft2c(x, channels_last=True)
        assert torch.allclose(X2, X)

    @unittest.skipIf(env.pt_version() >= [1, 8], "torch.fft not supported in torch>=1.8")
    def test_ifft2(self):
        g = torch.Generator().manual_seed(1)

        # Complex tensor as real tensor.
        x = torch.rand(4, 3, 3, 8, 2, generator=g)  # B x H x W x #coils x 2
        X = T.fft2(x)
        xhat = T.ifft2(X)
        xhat2 = oF.ifft2c(X, channels_last=True)
        assert torch.allclose(xhat2, xhat)

        x = torch.rand(4, 3, 3, 5, 8, 2, generator=g)  # B x H x W x D x #coils x 2
        X = T.fft2(x)
        xhat = T.ifft2(X)
        xhat2 = oF.ifft2c(X, channels_last=True)
        assert torch.allclose(xhat2, xhat)

        # Complex tensor.
        x = torch.rand(4, 3, 3, 8, generator=g, dtype=torch.complex64)  # B x H x W x #coils
        X = T.fft2(x)
        xhat = T.ifft2(X)
        xhat2 = oF.ifft2c(X, channels_last=True)
        assert torch.allclose(xhat2, xhat)

        x = torch.rand(4, 3, 3, 5, 8, generator=g, dtype=torch.complex64)  # B x H x W x D x #coils
        X = T.fft2(x)
        xhat = T.ifft2(X)
        xhat2 = oF.ifft2c(X, channels_last=True)
        assert torch.allclose(xhat2, xhat)


class TestFFTOp(unittest.TestCase):
    def test_fft2_cplx_tensors(self):
        """Test fft2 with PyTorch>=1.7 complex tensor support."""
        g = torch.Generator().manual_seed(1)
        x = torch.rand(4, 3, 3, 8, 2, generator=g)  # B x H x W x #coils x 2
        X = oF.fft2c(x)
        X2 = oF.fft2c(torch.view_as_complex(x))  # B x H x W x #coils

        assert torch.allclose(torch.view_as_real(X2), X)

    def test_ifft2_cplx_tensors(self):
        """Test ifft2 with PyTorch>=1.7 complex tensor support."""
        g = torch.Generator().manual_seed(1)
        x = torch.rand(4, 3, 3, 8, 2, generator=g)  # B x H x W x #coils x 2

        X = oF.fft2c(x)
        X2 = torch.view_as_complex(X)  # B x H x W x #coils

        xhat = oF.ifft2c(X)
        xhat2 = oF.ifft2c(X2)
        assert torch.allclose(torch.view_as_real(xhat2), xhat)
        # torch.allclose causes some issues with the comparison below
        # on certain machines. However, the maximum deviation between
        # the two tensors is the same across machines where torch.allclose
        # works and doesn't work. We compare the maximum of the difference
        # instead to get around this issue.
        assert torch.max(x - xhat) < 5e-7
