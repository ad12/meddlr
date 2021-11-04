from typing import Sequence, Tuple

import torch

from meddlr.transforms.base.noise import NoiseTransform
from meddlr.transforms.build import TRANSFORM_REGISTRY
from meddlr.transforms.transform import NoOpTransform
from meddlr.transforms.transform_gen import TransformGen

__all__ = ["RandomNoise"]


@TRANSFORM_REGISTRY.register()
class RandomNoise(TransformGen):
    """A model that adds additive white noise."""

    _base_transform = NoiseTransform
    _param_names = ["std_devs", "rhos"]

    def __init__(
        self,
        p: float = 0.0,
        std_devs: Tuple[float, float] = None,
        rhos: Tuple[float, float] = None,
        use_mask: bool = True,
    ):
        std_devs = (std_devs, std_devs) if not isinstance(std_devs, Sequence) else std_devs
        if rhos is not None:
            rhos = (rhos, rhos) if not isinstance(rhos, Sequence) else rhos

        params = locals()
        params = {k: params[k] for k in list(self._param_names) + ["p"]}
        self.use_mask = use_mask
        super().__init__(params=params, p=p)

    def get_transform(self, input: torch.Tensor):
        params = self._get_param_values(use_schedulers=True)
        std_devs = params["std_devs"]
        rho = params["rhos"]

        if self._rand() >= params["p"]:
            return NoOpTransform()

        std_dev = self._rand_range(*std_devs)
        if rho is not None:
            rho = self._rand_range(*rho)

        gen = self._generator
        if gen is None or gen.device != input.device:
            gen = torch.Generator(device=input.device).manual_seed(int(self._rand() * 1e10))
        return NoiseTransform(std_dev=std_dev, use_mask=self.use_mask, rho=rho, generator=gen)
