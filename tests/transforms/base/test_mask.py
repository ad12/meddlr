import unittest

import numpy as np
import torch

import ss_recon.utils.complex_utils as cplx
from ss_recon.transforms.base import KspaceMaskTransform

from ..mock import generate_mock_mri_data


class TestKspaceMaskTransform(unittest.TestCase):
    def test_basic(self):
        """Test basic functionality."""
        rho = 0.1
        ky = 20
        kz = 20

        kspace, _, _ = generate_mock_mri_data(ky=ky, kz=kz, rand_func="rand")
        kspace = kspace.permute(0, 3, 1, 2)  # B x Nc x H x W

        tfm = KspaceMaskTransform(rho=rho, seed=40)
        masked_kspace = tfm.apply_kspace(kspace)
        mask = cplx.get_mask(masked_kspace)
        assert np.allclose(torch.sum(mask) / np.prod(mask.shape), 1 - rho)

    def test_calib_size(self):
        rho = 0.1
        ky = 20
        kz = 20
        csize = 4

        kspace, _, _ = generate_mock_mri_data(ky=ky, kz=kz, rand_func="rand")
        kspace = kspace.permute(0, 3, 1, 2)  # B x Nc x H x W

        tfm = KspaceMaskTransform(rho=rho, seed=40, calib_size=(csize, csize))
        masked_kspace = tfm.apply_kspace(kspace)
        mask = cplx.get_mask(masked_kspace)
        sl = (
            Ellipsis,
            slice(ky // 2 - csize // 2, ky // 2 + csize // 2),
            slice(kz // 2 - csize // 2, kz // 2 + csize // 2),
        )
        assert torch.all(mask[sl] == 1)

        # base = mask.clone()
        # base[sl] = 0
        # base = base[0, 0]
        # assert np.allclose(torch.sum(base) / (ky * kz - csize*csize), 1-rho)

    def test_per_example(self):
        rho = 0.1

        kspace, _, _ = generate_mock_mri_data(bsz=10, rand_func="rand")
        kspace = kspace.permute(0, 3, 1, 2)  # B x Nc x H x W

        tfm = KspaceMaskTransform(rho=rho, per_example=True, seed=40)
        masked_kspace = tfm.apply_kspace(kspace)
        mask = cplx.get_mask(masked_kspace)

        # All masks should not be equal
        assert not torch.all(mask == mask[0:1])

        mask = mask.view((mask.shape[-1], -1))
        assert np.allclose(torch.sum(mask, -1) / mask.shape[-1], 1 - rho)
