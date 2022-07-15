import inspect
import pprint
from numbers import Number
from typing import Any, Dict, Sequence

import torch

from meddlr.transforms.mixins import DeviceMixin, TransformCacheMixin
from meddlr.transforms.param_kind import ParamKind
from meddlr.transforms.tf_scheduler import SchedulableMixin
from meddlr.transforms.transform import Transform

__all__ = ["TransformGen"]


class TransformGen(DeviceMixin, SchedulableMixin, TransformCacheMixin):
    """The base class for transform generators.

    A transform generator creates a :class:`Transform` based on the given image, sometimes with
    randomness. The transform can then be used to transform images
    or other data (boxes, points, annotations, etc.) associated with it.

    The assumption made in this class is that the image itself is sufficient to
    instantiate a transform. When this assumption is not true, you need to
    create the transforms by your own.

    A list of :cls:`TransformGen` can be applied with :func:`apply_transform_gens`.
    """

    def __init__(
        self,
        params: Dict[str, Any] = None,
        p: float = 0.0,
        param_kinds: Dict[str, ParamKind] = None,
    ) -> None:
        """
        Args:
            params: A dictionary of parameter names to values that are used for initialization.
                These parameters are typically schedulable, which means a scheduler
                can be used to control the parameter.
            p: The probability of applying this transform.
            param_kinds: A dictionary of parameter names to kinds.
        """
        # Avoid circular import.
        from meddlr.transforms.tf_scheduler import TFScheduler

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

    def _set_attributes(self, params: Dict[str, Any] = None, **kwargs):
        if params is None:
            params = {}
        params.update(kwargs)
        if params:
            self._params.update(
                {k: v for k, v in params.items() if k != "self" and not k.startswith("_")}
            )

    def get_transform(self, input) -> Transform:
        """Returns a transform based on the given input."""
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
        """Uniform sample between [0, 1) using ``self._generator``.

        Returns:
            float: The sample between [0, 1).
        """
        return torch.rand(1, generator=self._generator).cpu().item()

    def _rand_choice(self, n=None, probs: torch.Tensor = None) -> int:
        """Chooses random integer between [0, n-1].

        Args:
            n (int): Number of choices. This is required if ``probs``
                is not specified.
            probs (torch.Tensor): The probability tensor.

        Returns:
            int: The index of the selected choice.
        """
        device = "cpu" if self._generator is None else self._generator.device
        if probs is None:
            probs = torch.ones(n, device=device) / n
        return torch.multinomial(probs.to(device), 1, generator=self._generator).cpu().item()

    def _rand_range(self, low, high, size: int = None):
        """Uniform float random number between [low, high).

        Args:
            low (number-like): The lower bound.
            high (number-like): The upper bound.
            size (int): Number of samples to draw in the range.

        Returns:
            float: A uniformly sampled number in range [low, high).
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
                if len(val) == 1:
                    val = val * ndim
                for v in val:
                    if isinstance(v, (list, tuple)):
                        out.append(v)
                    elif isinstance(v, Number):
                        out.append((-v, v))
                    else:
                        raise ValueError(f"Type {type(val)} not supported - val={val}")
                return type(val)(out)
        return val

    def seed(self, value: int) -> "TransformGen":
        """Sets the seed for the random number generator.

        Args:
            value (int): The seed value.

        Returns:
            TransformGen: self.

        Note:
            This operation is done in-place.
        """
        self._generator = torch.Generator(device=self._device).manual_seed(value)
        return self

    def __repr__(self) -> str:
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
