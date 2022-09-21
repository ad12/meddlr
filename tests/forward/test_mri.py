import unittest

import torch

import meddlr.ops.complex as cplx
import meddlr.utils.transforms as T
from meddlr.forward.mri import SenseModel, hard_data_consistency
from meddlr.utils import env

from ..transforms.mock import generate_mock_mri_data


class TestSenseModel(unittest.TestCase):
    @unittest.skipIf(env.pt_version() >= [1, 8], "Old SENSE model requires torch.fft module")
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

    def test_multichannel(self):
        """Test multi-channel inputs."""
        ky = 20
        kz = 20
        nc = 8
        num_channels = 3
        nm = 1
        bsz = 5

        kspace = torch.randn(bsz, ky, kz, nc, num_channels, dtype=torch.complex64)
        maps = torch.rand(bsz, ky, kz, nc, nm, dtype=torch.complex64)
        maps = maps / cplx.rss(maps, dim=-2).unsqueeze(-2)

        A = SenseModel(maps)

        expected = []
        for c in range(num_channels):
            expected.append(A(kspace[..., c], adjoint=True))
        expected = torch.cat(expected, dim=-1)
        out_image = A(kspace, adjoint=True)
        assert torch.allclose(out_image, expected, atol=1e-5)

        expected = []
        for c in range(num_channels):
            expected.append(A(out_image[..., c : c + 1], adjoint=False))
        expected = torch.stack(expected, dim=-1)
        out_kspace = A(out_image, adjoint=False)
        # both clauses required for CI to pass on python 3.7 - torch.allclose does not work
        assert torch.allclose(out_kspace, expected, atol=1e-5)


def test_hard_data_consistency_trivial():
    ky = 20
    kz = 20
    nc = 8

    kspace, maps, target = generate_mock_mri_data(ky=ky, kz=kz, nc=nc, nm=1, bsz=1)
    mask = torch.ones_like(kspace)
    recon = hard_data_consistency(target, kspace, mask=mask, maps=maps)
    assert torch.allclose(recon, target, atol=1e-5)
