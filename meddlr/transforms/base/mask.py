from typing import Sequence, Tuple

import torch

from meddlr.ops import complex as cplx
from meddlr.transforms.build import TRANSFORM_REGISTRY
from meddlr.transforms.transform import Transform


@TRANSFORM_REGISTRY.register()
class KspaceMaskTransform(Transform):
    """A model that masks kspace."""

    def __init__(
        self,
        rho: float,
        kind="uniform",
        std_scale=None,
        per_example=False,
        seed: int = None,
        calib_size=None,
        generator: torch.Generator = None,
    ):
        if seed is None and generator is None:
            raise ValueError("One of `seed` or `generator` must be specified.")
        if kind not in ("uniform",):
            raise ValueError(f"Unknown kspace mask kind={kind}")
        self.rho = rho
        self.kind = kind
        self.seed = seed
        self.calib_size = calib_size
        self.per_example = per_example
        self.std_scale = std_scale

        gen_state = None
        if generator is not None:
            gen_state = generator.get_state()
        self._generator_state = gen_state

    def apply_kspace(self, kspace: torch.Tensor):
        return self._subsample(kspace)

    def _generator(self, data):
        seed = self.seed

        g = torch.Generator(device=data.device)
        if seed is None:
            g.set_state(self._generator_state)
        else:
            g = g.manual_seed(seed)
        return g

    def _subsample(self, data: torch.Tensor):
        g = self._generator(data)

        func_and_kwargs = {
            "uniform": (
                _uniform_mask,
                {"rho": self.rho, "mask": True, "calib_size": self.calib_size, "generator": g},
            ),
            "gaussian": (
                _gaussian_mask,
                {
                    "rho": self.rho,
                    "mask": True,
                    "std_scale": self.std_scale,
                    "calib_size": self.calib_size,
                    "generator": g,
                },
            ),
        }
        func, kwargs = func_and_kwargs[self.kind]

        if self.per_example:
            mask = torch.cat([func(data[i : i + 1], **kwargs) for i in range(len(data))], dim=0)
        else:
            mask = func(data, **kwargs)

        return mask * data

    def _eq_attrs(self) -> Tuple[str]:
        return ("std_dev", "use_mask", "rho", "seed", "_generator_state")


def _uniform_mask(kspace, rho, mask=True, calib_size=None, generator=None):
    """
    Args:
        rho (float): Fraction of samples to drop.
    """
    kspace = kspace[:, 0:1, ...]
    shape = kspace.shape

    if mask is True:
        orig_mask = cplx.get_mask(kspace)
    else:
        orig_mask = mask
    mask = orig_mask.clone()

    calib_region = None
    if calib_size is not None:
        if not isinstance(calib_size, Sequence):
            calib_size = (calib_size,) * (kspace.ndim - 2)
        center = tuple(s // 2 for s in kspace.shape[2:])[-len(calib_size) :]
        calib_region = tuple(slice(s - cs // 2, s + cs // 2) for s, cs in zip(center, calib_size))
        calib_region = (Ellipsis,) + calib_region
        mask[calib_region] = 0

    mask = mask.view(-1) if mask.is_contiguous() else mask.reshape(-1)

    # TODO: this doesnt work if the matrix is > 2*24 in size.
    # TODO: make this a bit more optimized
    num_valid = torch.sum(mask)
    weights = mask / num_valid
    num_samples = int(rho * num_valid)
    samples = torch.multinomial(weights, num_samples, replacement=False, generator=generator)
    mask[samples] = 0

    mask = mask.view(shape)
    if calib_region:
        mask[calib_region] = orig_mask[calib_region]
    return mask


def _gaussian_mask(kspace, rho, std_scale, mask=True, calib_size=None, generator=None):
    """
    Args:
        rho (float): Fraction of samples to drop.

    Note:
        Currently this creates the same mask for all the examples.
        Make sure to use per_example=True in :cls:`KspaceMaskTransform`.
    """
    kspace = kspace[:, 0:1, ...]
    shape = kspace.shape

    if mask is True:
        orig_mask = cplx.get_mask(kspace)
    else:
        orig_mask = mask
    mask = orig_mask.clone()

    spatial_shape = kspace.shape[2:]
    center = tuple(s // 2 for s in spatial_shape)

    calib_region = None
    if calib_size is not None:
        if not isinstance(calib_size, Sequence):
            calib_size = (calib_size,) * (kspace.ndim - 2)
        calib_region = tuple(
            slice(s - cs // 2, s + cs // 2) for s, cs in zip(center[-len(calib_size) :], calib_size)
        )
        calib_region = (Ellipsis,) + calib_region
        mask[calib_region] = 0

    num_valid = torch.sum(mask)
    num_samples = int(rho * num_valid)

    temp_mask = torch.zeros_like(mask)
    count = 0
    while count <= num_samples:
        idxs = [
            torch.round(torch.normal(c, (s - 1) / std_scale)).type(torch.long)
            for c, s in zip(center, spatial_shape)
        ]
        if any(i < 0 for i in idxs) or any(i >= s for i, s in zip(idxs, spatial_shape)):
            continue
        if mask[idxs] == 0 or temp_mask[idxs] == 1:
            continue
        temp_mask[idxs] = 1
        count += 1

    # TODO: this doesnt work if the matrix is > 2*24 in size.
    # TODO: make this a bit more optimized
    mask = mask - temp_mask

    mask = mask.view(shape)
    if calib_region:
        mask[calib_region] = orig_mask[calib_region]
    return mask


# def _multivariate_normal_weights(shape, loc=None):
#     if loc is None:
#         loc = [s // 2 for s in shape]
#     grids = torch.meshgrid(*[torch.arange(s) for s in shape])
