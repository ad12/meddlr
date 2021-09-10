from numbers import Number
from typing import Dict, Sequence, Tuple, Union

import torch

from ss_recon.transforms.base.spatial import AffineTransform, FlipTransform, Rot90Transform
from ss_recon.transforms.build import TRANSFORM_REGISTRY
from ss_recon.transforms.transform import NoOpTransform
from ss_recon.transforms.transform_gen import TransformGen

__all__ = ["RandomAffine", "RandomFlip", "RandomRot90"]


@TRANSFORM_REGISTRY.register()
class RandomAffine(TransformGen):
    _base_transform = AffineTransform
    _param_names = ("angle", "translate", "scale", "shear")

    def __init__(
        self,
        p: Union[float, Dict[str, float]] = 0.0,
        angle: Union[float, Tuple[float, float]] = None,
        translate: Union[float, Tuple[float, float], Sequence[Tuple[float, float]]] = None,
        scale: Union[float, Tuple[float, float]] = None,
        shear: Union[float, Tuple[float, float], Sequence[Tuple[float, float]]] = None,
    ):
        if isinstance(p, Number):
            p = {n: p for n in self._param_names}
        else:
            assert isinstance(p, dict)
            p = p.copy().update({k: 0.0 for k in self._param_names if k not in p})
        params = locals()
        params = {k: params[k] for k in list(self._param_names) + ["p"]}
        self.fill = None
        self.resample = None
        super().__init__(params=params, p=p)

    def _get_params(self, shape):
        ndim = len(shape)
        params = self._get_param_values()

        p = params["p"]
        param_angle = params["angle"]
        param_translate = params["translate"]
        param_scale = params["scale"]
        param_shear = params["shear"]

        if isinstance(param_angle, Number):
            param_angle = (-param_angle, param_angle)
        if isinstance(param_translate, Number):
            param_translate = (-param_translate, param_translate)
        if isinstance(param_scale, Number):
            param_scale = tuple(sorted([1.0 / param_scale, param_scale]))
        if isinstance(param_shear, Number):
            param_shear = (-param_shear, param_shear)

        param_translate = _duplicate_ndim(param_translate, ndim)
        param_shear = _duplicate_ndim(param_shear, ndim)

        angle, translate, scale, shear = None, None, None, None

        if param_angle is not None and self._rand() < p["angle"]:
            angle = self._rand_range(*param_angle)
        if param_translate is not None and self._rand() < p["translate"]:
            translate = [int(self._rand_range(*x) * s) for x, s in zip(param_translate, shape)]
        if param_scale is not None and self._rand() < p["scale"]:
            scale = self._rand_range(*param_scale)
        if param_shear is not None and self._rand() < p["shear"]:
            shear = [self._rand_range(*x) for x in param_shear]

        return angle, translate, scale, shear

    def get_transform(self, image):
        spatial_shape = _get_spatial_shape(image)

        out = self._get_params(spatial_shape)
        if all(x is None for x in out):
            return NoOpTransform()

        angle, translate, scale, shear = out
        return AffineTransform(
            angle=angle,
            translate=translate,
            scale=scale,
            shear=shear,
        )


@TRANSFORM_REGISTRY.register()
class RandomFlip(TransformGen):
    _base_transform = FlipTransform

    def __init__(self, dims=None, ndim=None, p: Union[float, Dict[int, float]] = 0.0) -> None:
        if dims is None and ndim is None:
            raise ValueError("Either `dims` or `ndim` must be specified")
        if all(x is not None for x in (dims, ndim)):
            raise ValueError("Only one of `dims` or `ndim` can be specified.")
        if isinstance(dims, int):
            dims = (dims,)
        self.dims = dims
        self.ndim = ndim
        super().__init__(p=p)

    def get_transform(self, input):
        params = self._get_param_values(use_schedulers=True)
        p = params["p"]
        if self.dims is not None:
            dims = tuple(d for d in self.dims if self._rand() < p)
        else:
            if isinstance(p, Dict):
                dims = tuple(k for k, v in p.items() if self._rand() < v)
            else:
                dims = tuple(d for d in range(-self.ndim, 0) if self._rand() < p)

        return FlipTransform(dims) if dims else NoOpTransform()


@TRANSFORM_REGISTRY.register()
class RandomRot90(TransformGen):
    _base_transform = Rot90Transform

    def __init__(self, ks=None, p=0.0) -> None:
        self.ks = ks if ks is not None else list(range(1, 4))
        super().__init__(p=p)

    def get_transform(self, input):
        params = self._get_param_values(use_schedulers=True)
        if self._rand() >= params["p"]:
            return NoOpTransform()
        k = self.ks[torch.randperm(len(self.ks))[0].item()]
        return Rot90Transform(k=k, dims=(-1, -2))


def _duplicate_ndim(param, ndim):
    if param is None:
        return None

    if isinstance(param, Sequence) and isinstance(param[0], Sequence):
        return [[x if len(x) > 1 else (-x[0], x[0]) for x in y] for y in param]

    if isinstance(param, Sequence):
        param = (-param[0], param[0]) if len(param) == 1 else param
    else:
        param = (-param, param)
    return [param] * ndim


def _get_spatial_shape(x):
    start = 2  # first 2 channels are B, C
    end = x.ndim
    return tuple(range(start, end))
