import inspect
import pprint
from numbers import Number
from typing import Any, Dict, Mapping, Sequence, Union

import torch

from ss_recon.config import CfgNode
from ss_recon.transforms.mixins import DeviceMixin, TransformCacheMixin
from ss_recon.transforms.param_kind import ParamKind
from ss_recon.transforms.tf_scheduler import SchedulableMixin
from ss_recon.transforms.transform import Transform

__all__ = ["TransformGen", "RandomTransformChoice"]


class TransformGen(DeviceMixin, SchedulableMixin, TransformCacheMixin):
    """
    TransformGen takes an array of type float as input.
    It creates a :class:`Transform` based on the given image, sometimes with
    randomness. The transform can then be used to transform images
    or other data (boxes, points, annotations, etc.) associated with it.
    The assumption made in this class is that the image itself is sufficient to
    instantiate a transform. When this assumption is not true, you need to
    create the transforms by your own.
    A list of `TransformGen` can be applied with :func:`apply_transform_gens`.
    """

    def __init__(
        self,
        params: Dict[str, Any] = None,
        p: float = 0.0,
        param_kinds: Dict[str, Any] = None,
    ) -> None:
        from ss_recon.transforms.tf_scheduler import TFScheduler

        self._params = {}

        if params is None:
            params = {}
        if param_kinds is None:
            param_kinds = {}
        params.update({"p": p})
        self._set_attributes(params)
        self._param_kinds = param_kinds
        self._schedulers: Sequence[TFScheduler] = []

        self._generator = None
        self._device = "cpu"

    def _set_attributes(self, params=None, **kwargs):
        if params is None:
            params = {}
        params.update(kwargs)
        if params:
            self._params.update(
                {k: v for k, v in params.items() if k != "self" and not k.startswith("_")}
            )

    def get_transform(self, input):
        raise NotImplementedError

    def reset(self):
        self._reset_transform()

    def __getattr__(self, name):
        if name in self._params:
            return self._params[name]
        raise AttributeError(f"Attribute '{name}' does not exist in class {type(self)}")

    def _get_param_values(self, use_schedulers=False):
        if not use_schedulers:
            return self._params

        params = self._params.copy()
        for s in self._schedulers:
            params.update(s.get_params())
        return params

    def _rand(self) -> float:
        return torch.rand(1, generator=self._generator).cpu().item()

    def _rand_choice(self, n=None, probs=None) -> int:
        if probs is None:
            probs = torch.ones(n) / n
        return torch.multinomial(probs, 1).cpu().item()

    def _rand_range(self, low, high, size: int = None):
        """
        Uniform float random number between low and high.
        """
        if size is None:
            size = 1

        if low > high:
            high, low = low, high

        if high - low == 0:
            val = low
        else:
            val = (low + (high - low) * torch.rand(size, generator=self._generator)).cpu().item()
        return val

    def _format_param(self, val, kind: ParamKind, ndim=None):
        if kind == ParamKind.MULTI_ARG:
            if isinstance(val, Number):
                return ((-val, val),) * ndim
            elif isinstance(val, (list, tuple)):
                out = []
                for v in val:
                    if isinstance(v, (list, tuple)):
                        out.append(v)
                    elif isinstance(v, Number):
                        out.append((-v, v))
                    else:
                        raise ValueError(f"Type {type(val)} not supported - val={val}")
                return type(val)(out)
        return val

    def seed(self, value: int):
        self._generator = torch.Generator(device=self._device).manual_seed(value)
        return self

    def __repr__(self):
        """
        Produce something like:
        "MyTransformGen(field1={self.field1}, field2={self.field2})"
        """
        try:
            sig = inspect.signature(self.__init__)
            classname = type(self).__name__
            argstr = []
            for name, param in sig.parameters.items():
                assert (
                    param.kind != param.VAR_POSITIONAL and param.kind != param.VAR_KEYWORD
                ), "The default __repr__ doesn't support *args or **kwargs"
                assert hasattr(self, name), (
                    "Attribute {} not found! "
                    "Default __repr__ only works if attributes match "
                    "the constructor.".format(name)
                )
                attr = getattr(self, name)
                default = param.default
                if default is attr:
                    continue
                argstr.append("{}={}".format(name, pprint.pformat(attr)))
            return "{}({})".format(classname, ", ".join(argstr))
        except AssertionError:
            return super().__repr__()

    def __str__(self) -> str:
        return self.__repr__()


class RandomTransformChoice(TransformGen):
    def __init__(
        self, tfms_or_gens: Sequence[Union[Transform, TransformGen]], tfm_ps="uniform", p=0.0
    ) -> None:
        self.tfms_or_gens = tfms_or_gens

        N = len(tfms_or_gens)
        if tfm_ps == "uniform":
            tfm_ps = torch.ones(N) / N
        else:
            tfm_ps = torch.as_tensor(tfm_ps)
        assert torch.allclose(torch.sum(tfm_ps), 1.0)
        self.tfm_ps = tfm_ps

        super().__init__(p=p)

    def get_transform(self, input):
        return self.tfms_or_gens[self._rand_choice(probs=self.tfm_ps)]

    def seed(self, value: int):
        self._generator = torch.Generator(device=self._device).manual_seed(value)
        for g in self.tfms_or_gens:
            if isinstance(g, TransformGen):
                g.seed(value)
        return self

    def __repr__(self):
        classname = type(self).__name__
        argstr = ",\n\t".join(
            "{} - p={:0.2f}".format(t, p) for t, p in zip(self.tfms_or_gens, self.tfm_ps)
        )
        return "{}(\n\t{}\n\t)".format(classname, ", ".join(argstr))

    @classmethod
    def from_dict(cls, cfg: CfgNode, init_kwargs: Mapping[str, Any], **kwargs):
        from ss_recon.transforms.build import build_transforms

        init_kwargs = init_kwargs.copy()
        tfms_or_gens = init_kwargs.pop("tfms_or_gens")
        tfms_or_gens = build_transforms(cfg, tfms_or_gens, **kwargs)

        return cls(tfms_or_gens, **init_kwargs)
