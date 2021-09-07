from numbers import Number
from typing import Dict, List, Sequence, Tuple, Union

import numpy as np

from ss_recon.utils.events import get_event_storage


class TFScheduler:
    def __init__(self, tfm, params: List[str] = None):
        self.tfm = tfm
        self._params = params

    def _parameter_names(self) -> List[str]:
        return self._params if self._params is not None else list(self.tfm.params().keys())

    def _unregister_parameters(self, names: Union[str, Sequence[str]]):
        if isinstance(names, str):
            name = [names]
        params = self._parameter_names()
        params = [x for x in params if x not in name]
        self._params = params

    def get_params(self):
        raise NotImplementedError

    def get_iteration(self):
        return get_event_storage().iter


class ScheduleableMixin:
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
        if any(s.tfm is not self for s in self._schedulers):
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
        tfm: ScheduleableMixin,
        params=None,
        warmup_method: str = None,
        delay_iters: int = 0,
        warmup_iters: int = 0,
        gamma: float = 1.0,
    ):
        if warmup_method not in (None, "linear", "exp"):
            raise ValueError(
                f"`warmup_method` must be one of (None, 'linear', 'exp'). Got '{warmup_method}'"
            )
        if gamma < 0:
            raise ValueError("gamma must be >=0")

        self.delay_iters = delay_iters
        self.warmup_iters = warmup_iters
        self.warmup_method = warmup_method
        self.gamma = gamma

        super().__init__(tfm, params)

    def get_params(self):
        names = self._parameter_names()
        params = {}
        for pname in names:
            params[pname] = self._compute_value(getattr(self.tfm, pname))

    def _compute_value(self, value):
        d = self.delay_iters
        T = self.warmup_iters
        t = self.get_iteration()

        is_number = isinstance(value, Number)
        if is_number:
            value = (0, value)
        else:
            assert isinstance(value, Sequence) and len(value) == 2

        lb, ub = tuple(value)
        if self.warmup_method == "linear":
            ub = min((t - d) / (T - d), 1.0) * (ub - lb) + lb
        elif self.warmup_method == "exp":
            tau = (T - d) / self.gamma
            delta = (ub - lb) * (1 - np.exp(-(t - d) / tau)) / (1 - np.exp(-self.gamma))
            ub = lb + (t > T) * delta

        return ub if is_number else (lb, ub)
