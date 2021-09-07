import torch

import ss_recon.utils.complex_utils as cplx
import ss_recon.utils.transforms as T


def generate_mock_mri_data(ky=20, kz=20, nc=8, nm=1, scale=1.0):
    kspace = torch.view_as_complex(torch.randn(1, ky, kz, nc, 2)) * scale
    maps = torch.view_as_complex(torch.randn(1, ky, kz, nc, nm, 2))
    maps = maps / cplx.rss(maps, dim=-2).unsqueeze(-2)
    A = T.SenseModel(maps)
    target = A(kspace, adjoint=True)
    return kspace, maps, target
