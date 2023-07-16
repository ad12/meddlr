import inspect
from collections import defaultdict
from typing import Any, Generic, TypeVar

import torch
from wrapt import ObjectProxy

import meddlr.ops.complex as cplx
from meddlr.forward import SenseModel
from meddlr.transforms.tf_scheduler import SchedulableMixin


def generate_mock_mri_data(
    ky=20, kz=20, nc=8, nm=1, bsz=1, scale=1.0, rand_func="randn", as_dict: bool = False
):
    func = getattr(torch, rand_func)
    kspace = torch.view_as_complex(func(bsz, ky, kz, nc, 2)) * scale
    maps = torch.view_as_complex(func(bsz, ky, kz, nc, nm, 2))
    maps = maps / cplx.rss(maps, dim=-2).unsqueeze(-2)
    A = SenseModel(maps)
    target = A(kspace, adjoint=True)
    if as_dict:
        return {"kspace": kspace, "maps": maps, "target": target}
    else:
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


T = TypeVar("T")


class MockCounter(Generic[T], ObjectProxy):
    def __init__(self, wrapped: T):
        super().__init__(wrapped)

        # Number of times the forward method is called.
        self._self_call_count = defaultdict(int)
        self._self_access_count = defaultdict(int)

    def call_count(self, key: str) -> int:
        return self._self_call_count[key]

    def access_count(self, key: str) -> int:
        return self._self_access_count[key]

    def __getattr__(self, __name: str) -> Any:
        attr = getattr(self.__wrapped__, __name)
        self._self_access_count[__name] += 1

        if inspect.ismethod(attr):

            def _wrapper(*args, **kwargs):
                self._self_call_count[__name] += 1
                return attr(*args, **kwargs)

            return _wrapper

        return attr

    def __call__(self, *args, **kwargs):
        self._self_call_count["__call__"] += 1
        return self.__wrapped__(*args, **kwargs)
