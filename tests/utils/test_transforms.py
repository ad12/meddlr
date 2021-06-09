import unittest

import torch

import ss_recon.utils.transforms as T


class TestFFT(unittest.TestCase):
    def test_fft2_cplx_tensors(self):
        """Test fft2 with PyTorch>=1.7 complex tensor support."""
        g = torch.Generator().manual_seed(1)
        x = torch.rand(4, 3, 3, 8, 2, generator=g)  # B x H x W x #coils x 2
        X = T.fft2(x)
        X2 = T.fft2(torch.view_as_complex(x))  # B x H x W x #coils

        assert torch.allclose(torch.view_as_real(X2), X)

    def test_ifft2_cplx_tensors(self):
        """Test ifft2 with PyTorch>=1.7 complex tensor support."""
        g = torch.Generator().manual_seed(1)
        x = torch.rand(4, 3, 3, 8, 2, generator=g)  # B x H x W x #coils x 2

        X = T.fft2(x)
        X2 = torch.view_as_complex(X)  # B x H x W x #coils

        xhat = T.ifft2(X)
        xhat2 = T.ifft2(X2)
        assert torch.allclose(x, xhat)
        assert torch.allclose(torch.view_as_real(xhat2), xhat)


if __name__ == "__main__":
    unittest.main()
