import logging
import multiprocessing as mp
import sys
import weakref
from bisect import bisect_right
from numbers import Number
from typing import Dict, List, Mapping, Sequence, Tuple, Union

import numpy as np

from ss_recon.transforms.param_kind import ParamKind
from ss_recon.utils.events import get_event_storage


class TFScheduler:
    def __init__(self, tfm, params: List[str] = None, iter_fn=None):
        self.tfm: SchedulableMixin = weakref.proxy(tfm)
        self._params = []
        self._register_parameters(params)
        self._iter_fn = iter_fn
        self._step = 0

    def _parameter_names(self) -> List[str]:
        return self._params if self._params is not None else list(self.tfm.params().keys())

    def _get_tfm_keys(self):
        def _dfs_pname(params, prefix=""):
            out = []
            if isinstance(params, Mapping):
                for k, v in params.items():
                    out.extend(_dfs_pname(v, f"{prefix}.{k}"))
            else:
                out.append(prefix)
            return out

        out = _dfs_pname(self.tfm._params)
        return [x[1:] if x.startswith(".") else x for x in out]

    def _register_parameters(self, names: Union[str, Sequence[str]]):
        if isinstance(names, str):
            names = [names]
        names = [x for x in self._get_tfm_keys() if any(x.startswith(n) for n in names)]
        unknown_params = set(names) - set(self._get_tfm_keys())
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
        names = [x for x in self._get_tfm_keys() if any(x.startswith(n) for n in names)]
        params = self._parameter_names()
        params = [x for x in params if x not in names]
        self._params = params

    def _get_tfm_param_val(self, pname: str):
        pname = _parse_pname(pname)
        if isinstance(pname, str):
            return self.tfm._params[pname]

        pval = self.tfm._params
        for pn in pname:
            pval = pval[pn]
        return pval

    def _fill_param_dict(self, params: Dict, pname: str, pval):
        pname = _parse_pname(pname)
        if isinstance(pname, str):
            params[pname] = pval
            return

        for idx, pn in enumerate(pname):
            if pn not in params:
                params[pn] = pval if idx == len(pname) - 1 else {}
            params = params[pn]

    def get_params(self):
        names = self._parameter_names()
        params = {}
        for pname in names:
            kind = self.tfm._param_kinds.get(pname, ParamKind.SINGLE_ARG)
            self._fill_param_dict(
                params, pname, self._compute_value(self._get_tfm_param_val(pname), kind)
            )
        return params

    def _compute_value(self, value, param_kind: ParamKind):
        raise NotImplementedError

    def get_iteration(self):
        if _is_main_process():
            return get_event_storage().iter

        # The EventStorage object is not always synchronized betweeen different
        # forks. Therefore, directly accessing get_event_storage().iter may not
        # give the correct iteration estimate. `self._step` keeps track of changes
        # in iterations for each worker based on the computation from `self._iter_fn`.
        # `self._step` is and should always be 0 on the main process. In other words,
        # it is only updated on each worker and is reset to 0 everytime states are
        # synchronized.
        # Therefore, the true iteration count can be estimated by adding the
        # iteration number from EventStorage with the estimated elapsed iteration
        # between synchronization periods given by `self._iter_fn(self._step)`.
        base_iter = get_event_storage().iter
        delta_iter = self._iter_fn(self._step)
        return base_iter + delta_iter

    def step(self, n=1):
        if _is_main_process():
            return get_event_storage().iter

        self._step += n

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
    _param_kinds: Dict[str, ParamKind]
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

    def _compute_value(self, value, kind: ParamKind):
        if kind == ParamKind.MULTI_ARG and isinstance(value, (list, tuple)):
            return [self._compute_value(v, kind.SINGLE_ARG) for v in value]

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

    def _compute_value(self, value, kind: ParamKind):
        if kind == ParamKind.MULTI_ARG and isinstance(value, (list, tuple)):
            return [self._compute_value(v, kind.SINGLE_ARG) for v in value]

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
        base.extend(["warmup_milestones", "warmup_method", "gamma"])
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


def _parse_pname(pname: str):
    if "." not in pname:
        return pname
    return pname.split(".")


def _is_main_process():
    py_version = tuple(sys.version_info[0:2])
    return (py_version < (3, 8) and mp.current_process().name == "MainProcess") or (
        py_version >= (3, 8) and mp.parent_process() is None
    )
