import unittest

import torch

import ss_recon.ops.complex as cplx
import ss_recon.utils.transforms as T
from ss_recon.forward import SenseModel


class TestSenseModel(unittest.TestCase):
    def test_reproducibility(self):
        ky = 20
        kz = 20
        nc = 8
        nm = 5
        bsz = 8
        scale = 1.0

        # Complex tensor.
        kspace = torch.view_as_complex(torch.randn(bsz, ky, kz, nc, 2)) * scale
        maps = torch.view_as_complex(torch.randn(bsz, ky, kz, nc, nm, 2))
        maps = maps / cplx.rss(maps, dim=-2).unsqueeze(-2)

        A = T.SenseModel(maps)
        A_new = SenseModel(maps)
        expected = A(kspace, adjoint=True)
        out = A_new(kspace, adjoint=True)
        assert torch.allclose(out, expected)
        assert torch.allclose(A_new(out, adjoint=False), A(expected, adjoint=False))

        # With mask.
        mask = cplx.get_mask(kspace)
        A = T.SenseModel(maps, weights=mask)
        A_new = SenseModel(maps, weights=mask)
        expected = A(kspace, adjoint=True)
        out = A_new(kspace, adjoint=True)
        assert torch.allclose(out, expected)
        assert torch.allclose(A_new(out, adjoint=False), A(expected, adjoint=False))

        # Complex tensor as real view.
        kspace = torch.view_as_real(kspace)
        maps = torch.view_as_real(maps)

        A = T.SenseModel(maps)
        A_new = SenseModel(maps)
        expected = A(kspace, adjoint=True)
        out = A_new(kspace, adjoint=True)
        assert torch.allclose(out, expected)
        assert torch.allclose(A_new(out, adjoint=False), A(expected, adjoint=False))
