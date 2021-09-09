import logging
import weakref
from bisect import bisect_right
from numbers import Number
from typing import Dict, List, Sequence, Tuple, Union

import numpy as np

from ss_recon.utils.events import get_event_storage


class TFScheduler:
    def __init__(self, tfm, params: List[str] = None):
        self.tfm: SchedulableMixin = weakref.proxy(tfm)
        self._params = []
        self._register_parameters(params)

    def _parameter_names(self) -> List[str]:
        return self._params if self._params is not None else list(self.tfm.params().keys())

    def _register_parameters(self, names: Union[str, Sequence[str]]):
        if isinstance(names, str):
            names = [names]
        unknown_params = set(names) - set(self.tfm._params.keys())
        if len(unknown_params) > 0:
            raise ValueError(
                f"Unknown parameters for transform {self.tfm.__class__.__name__}: {unknown_params}"
            )
        params = self._parameter_names()
        params = [n for n in names if n not in params]
        self._params.extend(params)

    def _unregister_parameters(self, names: Union[str, Sequence[str]]):
        if isinstance(names, str):
            names = [names]
        params = self._parameter_names()
        params = [x for x in params if x not in names]
        self._params = params

    def get_params(self):
        raise NotImplementedError

    def get_iteration(self):
        return get_event_storage().iter

    def _repr_args(self) -> List[str]:
        return ["tfm", "_params"]

    def __repr__(self) -> str:
        args = self._repr_args()
        args = {k: getattr(self, k) for k in args}
        args_str = "\n  ".join(
            [f"{k}={v.__repr__() if hasattr(v, '__repr__') else v}," for k, v in args.items()]
        )
        return "{}(\n  {}\n)".format(type(self).__name__, args_str)

    def __str__(self) -> str:
        return self.__repr__()


class SchedulableMixin:
    _params: Dict[str, Union[Number, Tuple[Number, Number]]]
    _schedulers: List[TFScheduler]

    def base_params(self):
        return self._params

    def validate_schedulers(self):
        schedulers = self._schedulers

        # No parameters should overlap between schedulers.
        all_params = [set(scheduler._params) for scheduler in schedulers]
        for i in range(len(all_params)):
            for j in range(i + 1, len(all_params)):
                if len(all_params[i] & all_params[j]) > 0:
                    raise ValueError(f"Parameters overlapping between schedulers {i} and {j}")

        # All schedulers should be tied to this schedulable.
        if any(s.tfm != self.__weakref__ for s in self._schedulers):
            raise ValueError(
                "Schedulers can only be configured for the transform it is registered to"
            )

    def register_schedulers(
        self, schedulers: Sequence[TFScheduler], overwrite_params: bool = False
    ):
        if overwrite_params:
            new_params = [name for s in schedulers for name in s._parameter_names()]
            for s in self._schedulers:
                s._unregister_parameters([p for p in new_params if p in s._parameter_names()])

        self._schedulers.extend(schedulers)
        self.validate_schedulers()


class WarmupTF(TFScheduler):
    def __init__(
        self,
        tfm: SchedulableMixin,
        warmup_iters: int,
        warmup_method: str = "linear",
        delay_iters: int = 0,
        gamma: float = 1.0,
        params=None,
    ):
        _logger = logging.getLogger(__name__)

        if warmup_method not in ("linear", "exp"):
            raise ValueError(
                f"`warmup_method` must be one of (None, 'linear', 'exp'). Got '{warmup_method}'"
            )
        if gamma < 0:
            raise ValueError("gamma must be >=0")
        if warmup_iters == 0 and delay_iters == 0:
            _logger.warning(
                "No warmup or delay time specified. " "This will functionally be a no-op scheduler."
            )

        self.delay_iters = delay_iters
        self.warmup_iters = warmup_iters
        self.warmup_method = warmup_method
        self.gamma = gamma

        super().__init__(tfm, params)

    def get_params(self):
        names = self._parameter_names()
        params = {}
        for pname in names:
            params[pname] = self._compute_value(self.tfm._params[pname])
        return params

    def _compute_value(self, value):
        is_number = isinstance(value, Number)
        if is_number:
            value = (0, value)
        else:
            assert isinstance(value, Sequence) and len(value) == 2

        alpha = _get_warmup_factor_at_iter(
            self.warmup_method,
            self.get_iteration(),
            self.warmup_iters,
            self.delay_iters,
            self.gamma,
        )
        lb, ub = tuple(value)
        ub = lb + alpha * (ub - lb)
        return ub if is_number else (lb, ub)

    def _repr_args(self) -> List[str]:
        base = super()._repr_args()
        base.extend(["warmup_iters", "warmup_method", "delay_iters", "gamma"])
        return base


class WarmupMultiStepTF(TFScheduler):
    def __init__(
        self,
        tfm: SchedulableMixin,
        warmup_milestones: Sequence[int],
        warmup_method: str = "linear",
        gamma: float = 1.0,
        params=None,
    ):
        if warmup_method not in (None, "linear", "exp"):
            raise ValueError(
                f"`warmup_method` must be one of (None, 'linear', 'exp'). Got '{warmup_method}'"
            )
        if gamma < 0:
            raise ValueError("gamma must be >=0")

        self.warmup_milestones = warmup_milestones
        self.warmup_method = warmup_method
        self.gamma = gamma

        super().__init__(tfm, params)

    def get_params(self):
        names = self._parameter_names()
        params = {}
        for pname in names:
            params[pname] = self._compute_value(self.tfm._params[pname])
        return params

    def _compute_value(self, value):
        t = self.get_iteration()
        step = bisect_right(self.warmup_milestones, t)
        total_steps = len(self.warmup_milestones)

        is_number = isinstance(value, Number)
        if is_number:
            value = (0, value)
        else:
            assert isinstance(value, Sequence) and len(value) == 2

        alpha = _get_warmup_factor_at_iter(
            self.warmup_method,
            step,
            total_steps,
            0,
            self.gamma,
        )
        alpha = (step > 0) * alpha

        lb, ub = tuple(value)
        ub = lb + alpha * (ub - lb)
        return ub if is_number else (lb, ub)

    def _repr_args(self) -> List[str]:
        base = super()._repr_args()
        base.extend(["warmup_milestones", "warmup_method", "delay_iters", "gamma"])
        return base


def _get_warmup_factor_at_iter(
    method: str, iter: int, warmup_iters: int, delay_iters: int, gamma: float = None
) -> float:
    """
    TODO: Merge this with the lr scheduler logic

    Args:
        method (str): warmup method; either "constant" or "linear".
        iter (int): iteration at which to calculate the warmup factor.
        warmup_iters (int): the number of warmup iterations.
        warmup_factor (float): the base warmup factor (the meaning changes
            according to the method used).

    Returns:
        float: the effective warmup factor at the given iteration.
    """
    if iter >= warmup_iters:
        return 1.0
    if iter < delay_iters:
        return 0.0
    assert warmup_iters > delay_iters

    if method == "constant":
        return 1.0 / (warmup_iters - delay_iters)
    elif method == "linear":
        return min((iter - delay_iters) / (warmup_iters - delay_iters), 1.0)
    elif method == "exp":
        assert gamma is not None
        tau = (warmup_iters - delay_iters) / gamma
        return min((1 - np.exp(-(iter - delay_iters) / tau)) / (1 - np.exp(-gamma)), 1.0)
    else:
        raise ValueError("Unknown warmup method: {}".format(method))
