import logging
import weakref
from bisect import bisect_right
from numbers import Number
from typing import Any, Callable, Dict, List, Mapping, Sequence, Tuple, Union

import numpy as np

from meddlr.transforms.param_kind import ParamKind
from meddlr.utils import env
from meddlr.utils.events import get_event_storage


class TFScheduler:
    """Abstract class that helps schedule parameters for different transforms.

    This class helps with scheduling parameters for classes which have schedulable
    parameters (:cls:`SchedulableMixin`). This may be helpful when estabilishing
    some curricula for augmentation parameters.

    Each :cls:`TFScheduler` (or scheduler) operates on some set of base parameters
    (``base_params``) available in the :cls:`SchedulableMixin` (or schedulable)
    class. These base parameters define the maximum values or operating ranges of
    the parameters that are to be used by the schedulable. See :cls:`SchedulableMixin`
    for more details.

    Scheduling requires access to the current iteration (or epoch) count.
    However, when these schedulers are used across different workers,
    the synchronized iteration count is not available. We build around this
    issue by maintaining a global and local iteration count, where the total
    iteration is approximately equal to the sum of the two. When worker states
    are synchronized with the main thread, the global iteration count is updated
    and the local count is reset to 0.

    Such a scheduling system requires a local iteration counter that is initialized and
    kept at 0 on the main thread. This state is computed from ``self._step``.
    When a process is forked, ``self._step == 0``. The state is routinely updated on
    each worker until the worker syncs with the state on the main thread. Upon sync,
    ``self._step`` is reset to 0, but the global iteration counter is updated. Thus,
    the sum ``global_iter + iter_fn(self._step)`` approximately provides the current
    iteration. See :meth:`get_iteration` for more detail.

    Attributes:
        tfm (weakref.proxy[ScheduleableMixin]): A weak reference to input ``tfm``.

    Notation:
        - ``t``: The estimated iteration.

    Note:
        Functionality has not been tested with MacOS or Windows OS. It may work only with
        forking (default on Unix) not spawning (default on Windows and MacOS).
    """

    def __init__(
        self, tfm, params: Union[str, List[str], Tuple[str]] = None, iter_fn: Callable = None
    ):
        """
        Args:
            tfm (SchedulableMixin): An instance of :cls:`SchedulableMixin`.
                Typically a transform.
            params (List[str]): Parameter name(s) for ``tfm`` that should be scheduled.
                These parameters should be accessible from ``tfm._params``.
            iter_fn (Callable): A function that returns the current iteration.
                This function should take in one argument, ``self._step``.
                Only used when the scheduler is being used across multiple workers
                (e.g. data loading).
        """
        if not env.is_main_process():
            raise RuntimeError(f"{type(self)} must be constructed on the main thread.")

        self.tfm: SchedulableMixin = weakref.proxy(tfm)
        self._params = []
        self._register_parameters(params)
        self._iter_fn = iter_fn
        self._step = 0

    def get_params(self) -> Dict[str, Any]:
        """Returns dictionary of ``self.tfm`` parameters after scheduling.

        Returns:
            Dict[str, Any]: The parameters after being passed through the scheduler.
        """
        names = self._parameter_names()
        params = {}
        for pname in names:
            kind = self.tfm._param_kinds.get(pname, ParamKind.SINGLE_ARG)
            self._fill_param_dict(
                params, pname, self._compute_value(self._get_tfm_param_val(pname), kind)
            )
        return params

    def get_iteration(self) -> int:
        """Returns the estimated iteration.

        Returns:
            int: The estimated iteration.
        """
        if env.is_main_process():
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

    def step(self, n: int = 1):
        """Take ``n`` step(s).

        This function is used to maintain the step count
        used to compute the current iteration number.
        ``n`` should be configured to be compatible with ``self._iter_fn``.
        If you are using :func:`build_iter_func`, ``n`` should correspond
        to the number of examples in the batch.

        This function does not do anything if ``self`` is executing on the
        main thread.

        Args:
            n (int): The number of steps/examples to increase internal
                step count by.
        """
        if env.is_main_process():
            return

        self._step += n

    def _compute_value(self, value, param_kind: ParamKind):
        raise NotImplementedError

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
        # TODO (arjundd): names=None does not work.
        if isinstance(names, str):
            names = [names]

        tfm_keys = self._get_tfm_keys()
        unmatched_names = [n for n in names if not any(x.startswith(n) for x in tfm_keys)]
        names = [x for x in tfm_keys if any(x.startswith(n) for n in names)]
        unknown_params = (set(names) - set(tfm_keys)) | set(unmatched_names)
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

    def _repr_args(self) -> List[str]:  # pragma: no cover
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
    """Mixin for any class that should be compatible with :cls:`TFScheduler`."""

    _params: Dict[str, Union[Number, Tuple[Number, Number]]]
    _param_kinds: Dict[str, ParamKind]
    _schedulers: List[TFScheduler]

    def base_params(self):
        """The base that defines the maximum value or range for different parameters.

        Currently, this only supports numeric parameter types.

        Returns:
            Dict[str, Any]: The maximum value or range of different parameters.
        """
        return self._params

    def validate_schedulers(self) -> None:
        """Verifies that schedulers are valid.

        A valid set of schedulers must fulfill at least the following criteria:
            1. Any two different schedulers should not operate on any of the same paramters.
            2. All schedulers must have a weak reference to this schedulable.

        Raises:
            ValueError: If parameters overlap between any two schedulers
                or schedulers do not have a weakref to this schedulable.
        """
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
    ) -> None:
        """Register scheduler(s).

        Args:
            schedulers (Sequence[TFScheduler]): The sequence of schedulers to
                register. These schedulers will be used to operate on certain
                parameters.
            overwrite_params (bool, optional): If ``True``, more recently added
                schedulers will have exclusive priority to manage parameter values.
                For example, if schedulers ``A`` and ``B`` were to be added in that
                order, and they were both configured to operate on some parameter
                ``param1``, then ``B``'s scheduling would be used, as it is the most
                recent scheduler.
        """
        if isinstance(schedulers, TFScheduler):
            schedulers = [schedulers]
        if overwrite_params:
            new_params = [name for s in schedulers for name in s._parameter_names()]
            for s in self._schedulers:
                s._unregister_parameters([p for p in new_params if p in s._parameter_names()])

        self._schedulers.extend(schedulers)
        self.validate_schedulers()

    def schedulers(self) -> List[TFScheduler]:
        return self._schedulers


class WarmupTF(TFScheduler):
    """A warmup scheduler.

    In warmup, a numeric value is scaled between some lower (:math:`b_l`)
    and upper (:math:`b_u`) bound over some number of iterations (i.e.
    the ``warmup_iters``). If ``t >= warmup_iters``, the value is clamped
    at the maximum value.

    If the base parameter is a scalar value, that value is assumed to be the
    upper bound with a default lower bound of ``0``. If the parameter is a
    tuple, it is assumed to be the semi-open range corresponding to
    [:math:`b_l`, :math:`b_u`).

    For notation, see cls:`TFScheduler`.
    """

    def __init__(
        self,
        tfm: SchedulableMixin,
        warmup_iters: int,
        warmup_method: str = "linear",
        delay_iters: int = 0,
        gamma: float = 1.0,
        params=None,
    ):
        """
        Args:
            tfm: See :cls:`TFScheduler`.
            warmup_iters (int): Number of iterations to warmup.
            warmup_method (str, optional): One of [``'linear'``, ``'exp'``].
            delay_iters (int, optional): Number of iterations to delay scheduling by.
            gamma (float, optional): Scaling factor for time constant in exponential (``'exp'``)
                scheduling.
            params: See :cls:`TFScheduler`.
        """
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
    """Multi-step warmup scheduler.

    This function works exactly like :cls:`WarmupTF` with the difference that
    all updates (or scheduling steps) are taken at discrete step times (or
    *milestones*). After the final milestone, the value is clamped at the upper
    bound for that parameter.

    This class is useful for epoch-level scheduling.
    """

    def __init__(
        self,
        tfm: SchedulableMixin,
        warmup_milestones: Sequence[int],
        warmup_method: str = "linear",
        gamma: float = 1.0,
        params=None,
    ):
        """
        Args:
            tfm: See :cls:`TFScheduler`.
            warmup_milestones (Sequence[int]): Discrete iterations at which to take a
                step on the scheduler.
            warmup_method (str, optional): One of [``'linear'``, ``'exp'``].
            gamma (float, optional): Scaling factor for time constant in exponential (``'exp'``)
                scheduling.
            params: See :cls:`TFScheduler`.
        """
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

        alpha = _get_warmup_factor_at_iter(self.warmup_method, step, total_steps, 0, self.gamma)
        alpha = (step > 0) * alpha

        lb, ub = tuple(value)
        ub = lb + alpha * (ub - lb)
        return ub if is_number else (lb, ub)

    def _repr_args(self) -> List[str]:  # pragma: no cover
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
