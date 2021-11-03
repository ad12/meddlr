import unittest

import torch

import meddlr.utils.transforms as T
from meddlr.utils import env


class TestFFT(unittest.TestCase):
    @unittest.skipIf(env.pt_version() >= [1, 8], "torch.fft not supported in torch>=1.8")
    def test_fft2_cplx_tensors(self):
        """Test fft2 with PyTorch>=1.7 complex tensor support."""
        g = torch.Generator().manual_seed(1)
        x = torch.rand(4, 3, 3, 8, 2, generator=g)  # B x H x W x #coils x 2
        X = T.fft2(x)
        X2 = T.fft2(torch.view_as_complex(x))  # B x H x W x #coils

        assert torch.allclose(torch.view_as_real(X2), X)

    @unittest.skipIf(env.pt_version() >= [1, 8], "torch.fft not supported in torch>=1.8")
    def test_ifft2_cplx_tensors(self):
        """Test ifft2 with PyTorch>=1.7 complex tensor support."""
        g = torch.Generator().manual_seed(1)
        x = torch.rand(4, 3, 3, 8, 2, generator=g)  # B x H x W x #coils x 2

        X = T.fft2(x)
        X2 = torch.view_as_complex(X)  # B x H x W x #coils

        xhat = T.ifft2(X)
        xhat2 = T.ifft2(X2)
        assert torch.allclose(torch.view_as_real(xhat2), xhat)
        # torch.allclose causes some issues with the comparison below
        # on certain machines. However, the maximum deviation between
        # the two tensors is the same across machines where torch.allclose
        # works and doesn't work. We compare the maximum of the difference
        # instead to get around this issue.
        assert torch.max(x - xhat) < 1e-7


if __name__ == "__main__":
    unittest.main()
