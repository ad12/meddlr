from numbers import Number
from typing import Dict, Sequence, Tuple, Union

import torch

from meddlr.transforms.base.spatial import (
    AffineTransform,
    FlipTransform,
    Rot90Transform,
    TranslationTransform,
)
from meddlr.transforms.build import TRANSFORM_REGISTRY
from meddlr.transforms.param_kind import ParamKind
from meddlr.transforms.transform import NoOpTransform
from meddlr.transforms.transform_gen import TransformGen

__all__ = ["RandomAffine", "RandomFlip", "RandomRot90", "RandomTranslation"]

SPATIAL_RANGE_OR_VAL = Union[float, Sequence[float], Sequence[Tuple[float, float]]]


@TRANSFORM_REGISTRY.register()
class RandomAffine(TransformGen):
    _base_transform = AffineTransform
    _param_names = ("angle", "translate", "scale", "shear")

    def __init__(
        self,
        p: Union[float, Dict[str, float]] = 0.0,
        angle: Union[float, Tuple[float, float]] = None,
        translate: SPATIAL_RANGE_OR_VAL = None,
        scale: Union[float, Tuple[float, float]] = None,
        shear: SPATIAL_RANGE_OR_VAL = None,
        pad_like=None,
    ):
        if isinstance(p, Number):
            p = {n: p for n in self._param_names}
        else:
            assert isinstance(p, dict)
            unknown_keys = set(p.keys()) - set(self._param_names)
            if len(unknown_keys):
                raise ValueError(f"Unknown keys for `p`: {unknown_keys}")
            p = p.copy()
            p.update({k: 0.0 for k in self._param_names if k not in p})
        params = locals()
        params = {k: params[k] for k in list(self._param_names)}
        self.pad_like = pad_like
        super().__init__(
            params=params,
            p=p,
            param_kinds={"translate": ParamKind.MULTI_ARG, "shear": ParamKind.MULTI_ARG},
        )

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
            param_translate = ((-param_translate, param_translate),)
        if isinstance(param_scale, Number):
            param_scale = tuple(sorted([1.0 - param_scale, 1.0 + param_scale]))
        if isinstance(param_shear, Number):
            param_shear = ((-param_shear, param_shear),)

        param_translate = self._format_param(param_translate, ParamKind.MULTI_ARG, ndim)
        param_shear = self._format_param(param_shear, ParamKind.MULTI_ARG, ndim)

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
        # Affine only supports 2D spatial transforms
        spatial_shape = image.shape[-2:]

        out = self._get_params(spatial_shape)
        if all(x is None for x in out):
            return NoOpTransform()

        angle, translate, scale, shear = out
        return AffineTransform(
            angle=angle, translate=translate, scale=scale, shear=shear, pad_like=self.pad_like
        )


@TRANSFORM_REGISTRY.register()
class RandomTranslation(TransformGen):
    _base_transform = TranslationTransform

    def __init__(
        self,
        p: Union[float, Dict[str, float]] = 0.0,
        translate: SPATIAL_RANGE_OR_VAL = None,
        pad_mode=None,
        pad_value=0,
        ndim=2,
    ):
        params = {"translate": translate}
        self.pad_mode = pad_mode
        self.pad_value = pad_value
        self.ndim = ndim
        super().__init__(params=params, p=p, param_kinds={"translate": ParamKind.MULTI_ARG})

    def get_transform(self, image):
        shape = image.shape[-self.ndim :]
        ndim = len(shape)

        params = self._get_param_values(use_schedulers=True)
        p = params["p"]
        param_translate = params["translate"]
        translate = self._format_param(param_translate, ParamKind.MULTI_ARG, ndim)

        if self._rand() >= p:
            return NoOpTransform()
        translate = [int(self._rand_range(*x) * s) for x, s in zip(translate, shape)]
        return TranslationTransform(translate, pad_mode=self.pad_mode, pad_value=self.pad_value)


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
