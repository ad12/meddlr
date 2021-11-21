from typing import Sequence, Tuple, Union

import torch

from meddlr.transforms.base.mask import KspaceMaskTransform
from meddlr.transforms.build import TRANSFORM_REGISTRY
from meddlr.transforms.transform import NoOpTransform
from meddlr.transforms.transform_gen import TransformGen

__all__ = ["RandomKspaceMask"]


@TRANSFORM_REGISTRY.register()
class RandomKspaceMask(TransformGen):
    """A model that generates random masks."""

    _base_transform = KspaceMaskTransform
    _param_names = ["rhos"]

    def __init__(
        self,
        p: float = 0.0,
        rhos: Union[float, Tuple[float, float]] = None,
        kind: str = "uniform",
        std_scale: float = None,
        per_example=False,
        calib_size=None,
    ):
        if isinstance(rhos, Sequence) and len(rhos) == 1:
            rhos = rhos[0]

        params = locals()
        params = {k: params[k] for k in list(self._param_names) + ["p"]}
        super().__init__(params=params, p=p)

        self.kind = kind
        self.per_example = per_example
        self.calib_size = calib_size
        self.std_scale = std_scale

    def get_transform(self, input: torch.Tensor):
        params = self._get_param_values(use_schedulers=True)
        rhos = params["rhos"]

        if self._rand() >= params["p"]:
            return NoOpTransform()

        if isinstance(rhos, Sequence):
            rho = self._rand_range(*rhos)
        else:
            rho = rhos

        gen = self._generator
        if gen is None or gen.device != input.device:
            gen = torch.Generator(device=input.device).manual_seed(int(self._rand() * 1e10))
        return KspaceMaskTransform(
            rho=rho,
            kind=self.kind,
            per_example=self.per_example,
            calib_size=self.calib_size,
            generator=gen,
            std_scale=self.std_scale,
        )
