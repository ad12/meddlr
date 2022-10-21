import inspect
import itertools
from typing import Any, Callable, Dict, List, Optional, Sequence, Union

import pandas as pd
import torch
from torchmetrics.metric import Metric as _Metric
from torchmetrics.utilities import reduce
from torchmetrics.utilities.data import _flatten
from torchmetrics.utilities.distributed import gather_all_tensors

from meddlr.utils import comm

__all__ = ["Metric"]


class Metric(_Metric):
    """Interface for new metrics.

    A metric should be implemented as a callable with explicitly defined
    arguments. In other words, metrics should not have `**kwargs` or `**args`
    options in the `__call__` method.

    While not explicitly constrained to the return type, metrics typically
    return float value(s). The number of values returned corresponds to the
    number of categories.

    This class is opinionated in that it computes metrics for each (example, channel)
    pair. This means that outputs of ``compute`` are not scalars, but rather tensors
    of shape ``(B, C)``. Note, this opinion may be relaxed in the future.

    * metrics should have different name() for different functionality.
    * `category_dim` duck type if metric can process multiple categories at
        once.

    To compute metrics:

    .. code-block:: python

        metric = Metric()
        results = metric(...)
    """

    def __init__(
        self,
        channel_names: Sequence[str] = None,
        units: str = None,
        reduction="none",
        compute_on_step: bool = False,
        dist_sync_on_step: bool = False,
        process_group: bool = None,
        dist_sync_fn: bool = None,
    ):
        self.units = units
        super().__init__(
            compute_on_step=compute_on_step,
            dist_sync_on_step=dist_sync_on_step,
            process_group=process_group,
            dist_sync_fn=dist_sync_fn,
        )

        self.reduction = reduction
        self.channel_names = channel_names
        self._update_kwargs_aliases = {}

        # Identifiers for the examples that are seen.
        self.add_state("ids", default=[], dist_reduce_fx=lambda x: list(itertools.chain(x)))
        self.add_state("values", default=[], dist_reduce_fx="cat")

    def func(self, preds, targets, *args, **kwargs) -> torch.Tensor:
        """Computes metrics for each element in the batch.

        Returns:
            torch.Tensor: A torch Tensor with first dimension being
                batch dimension (``Bx...``).
        """
        raise NotImplementedError

    def update(self, preds, targets, *args, ids=None, **kwargs):
        assert preds.shape == targets.shape

        values: torch.Tensor = self.func(preds, targets, *args, **kwargs)
        self.values.append(values)
        self._add_ids(ids=ids, num_samples=len(values))

    def _generate_ids(self, num_samples):
        id_start = sum(len(x) for x in self.values)
        rank = comm.get_rank()
        ids = [f"{rank}-{id_start + idx}" for idx in range(num_samples)]
        return ids

    def _add_ids(self, ids, num_samples):
        if ids is None:
            ids = self._generate_ids(num_samples)
        self.ids.extend(ids)

    def compute(self, reduction=None):
        if reduction is None:
            reduction = self.reduction
        return reduce(torch.cat(self.values), reduction)

    def to_pandas(self, sync_dist: bool = True) -> pd.DataFrame:
        return pd.DataFrame.from_dict(self.to_dict(sync_dist=sync_dist, device="cpu"))

    def to_dict(self, sync_dist: bool = True, device=None):
        if sync_dist:
            with self.sync_context():
                data = self._to_dict(device=device)
        else:
            data = self._to_dict(device=device)
        return data

    def _sync_dist(
        self,
        dist_sync_fn: Callable = gather_all_tensors,
        process_group: Optional[Any] = None,
        **kwargs,
    ) -> None:  # pragma: no cover
        """Includes synchronizing ids, which is not a tensor object.

        torchmetrics only synchronizes tensors. This method extends the synchronization
        to `ids`, which is a non-tensor object.
        """
        super()._sync_dist(dist_sync_fn=dist_sync_fn, process_group=process_group, **kwargs)

        input_dict = {"ids": self.ids}
        output_dict = {
            k: comm.all_gather(v, group=process_group or self.process_group)
            for k, v in input_dict.items()
        }

        for attr in output_dict.keys():
            reduction_fn = self._reductions[attr]
            if isinstance(output_dict[attr][0], list):
                output_dict[attr] = _flatten(output_dict[attr])
            if not (callable(reduction_fn) or reduction_fn is None):
                raise TypeError("reduction_fn must be callable or None")

            reduced = (
                reduction_fn(output_dict[attr]) if reduction_fn is not None else output_dict[attr]
            )
            setattr(self, attr, reduced)

    def _to_dict(self, device=None) -> Dict[str, Any]:
        if _is_empty(self.values):
            return {"id": self.ids}

        values = torch.cat(self.values) if isinstance(self.values, list) else self.values
        if device is not None:
            values = values.to(device)
        channel_names = (
            self.channel_names
            if self.channel_names
            else [f"channel_{idx}" for idx in range(values.shape[1])]
        )
        data = {"id": self.ids}
        data.update({name: values[:, idx] for idx, name in enumerate(channel_names)})
        return data

    def _filter_kwargs(self, **kwargs: Any) -> Dict[str, Any]:
        """filter kwargs such that they match the update signature of the metric"""
        if self._update_kwargs_aliases:
            filtered_kwargs = {k: v for k, v in kwargs.items()}
            aliases = {}
            for alias in self._update_kwargs_aliases:
                if alias not in kwargs or self._update_kwargs_aliases[alias] in aliases:
                    continue
                aliases[self._update_kwargs_aliases[alias]] = kwargs.pop(alias)
            filtered_kwargs.update(aliases)
        else:
            filtered_kwargs = kwargs

        # Use filtering from torch 0.6.0 where kwargs are preserved and passed along.
        filtered_kwargs = _filter_kwargs(self._update_signature, **filtered_kwargs)
        return filtered_kwargs

    def register_update_aliases(self, **kwargs):
        """Register aliases for keyword arguments when calling update."""
        # filter all parameters based on update signature except those of
        # type VAR_POSITIONAL (*args) and VAR_KEYWORD (**kwargs)
        _params = (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD)
        _sign_params = self._update_signature.parameters
        supported_kwargs = tuple(
            k for k in _sign_params.keys() if _sign_params[k].kind not in _params
        )
        unsupported_kwargs = [v for v in kwargs.values() if v not in supported_kwargs]
        if len(unsupported_kwargs) > 0:
            raise ValueError(
                f"Found unsupported kwargs '{unsupported_kwargs}'. "
                f"Supported keyword arguments include:{supported_kwargs}"
            )
        aliases = {k: v for k, v in kwargs.items()}
        self._update_kwargs_aliases.update(aliases)

    def name(self):
        return type(self).__name__

    def display_name(self):
        """Name to use for pretty printing and display purposes."""
        name = self.name()
        return "{} ({})".format(name, self.units) if self.units else name


def _filter_kwargs(sig, **kwargs: Any) -> Dict[str, Any]:
    # filter all parameters based on update signature except those of
    # type VAR_POSITIONAL (*args) and VAR_KEYWORD (**kwargs)
    _params = (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD)
    _sign_params = sig.parameters
    filtered_kwargs = {
        k: v
        for k, v in kwargs.items()
        if (k in _sign_params.keys() and _sign_params[k].kind not in _params)
    }

    # if no kwargs filtered, return al kwargs as default
    if not filtered_kwargs:
        filtered_kwargs = kwargs
    return filtered_kwargs


def _is_empty(x: Optional[Union[List[torch.Tensor], torch.Tensor]]):  # pragma: no cover
    if isinstance(x, list):
        return len(x) == 0
    else:
        return x is None or x.numel() == 0
