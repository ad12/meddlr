import torch

import ss_recon.ops.functional.complex as cplx
import ss_recon.utils.transforms as T
from ss_recon.transforms.tf_scheduler import SchedulableMixin


def generate_mock_mri_data(ky=20, kz=20, nc=8, nm=1, bsz=1, scale=1.0, rand_func="randn"):
    func = getattr(torch, rand_func)
    kspace = torch.view_as_complex(func(bsz, ky, kz, nc, 2)) * scale
    maps = torch.view_as_complex(func(bsz, ky, kz, nc, nm, 2))
    maps = maps / cplx.rss(maps, dim=-2).unsqueeze(-2)
    A = T.SenseModel(maps)
    target = A(kspace, adjoint=True)
    return kspace, maps, target


class MockSchedulable(SchedulableMixin):
    def __init__(self, a=0.5, b=(0.2, 1.0)) -> None:
        self._params = {"a": a, "b": b}
        self._param_kinds = {}
        self._schedulers = []


class MockIterTracker:
    def __init__(self, start=0) -> None:
        self._iter = start

    def step(self, num=1):
        self._iter += num

    def get_iter(self):
        return self._iter
