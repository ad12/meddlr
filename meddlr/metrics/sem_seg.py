from typing import Sequence

import torch

import meddlr.metrics.functional as mF
from meddlr.metrics.metric import Metric

__all__ = ["DSC", "CV", "VOE", "ASSD"]


class DSC(Metric):
    def __init__(
        self,
        channel_names: Sequence[str] = None,
        reduction="none",
        compute_on_step: bool = False,
        dist_sync_on_step: bool = False,
        process_group: bool = None,
        dist_sync_fn: bool = None,
    ):
        super().__init__(
            channel_names=channel_names,
            units="",
            reduction=reduction,
            compute_on_step=compute_on_step,
            dist_sync_on_step=dist_sync_on_step,
            process_group=process_group,
            dist_sync_fn=dist_sync_fn,
        )

    def func(self, preds, targets) -> torch.Tensor:
        return mF.dice_score(y_pred=preds, y_true=targets)


class CV(Metric):
    def __init__(
        self,
        channel_names: Sequence[str] = None,
        reduction="none",
        compute_on_step: bool = False,
        dist_sync_on_step: bool = False,
        process_group: bool = None,
        dist_sync_fn: bool = None,
    ):
        super().__init__(
            channel_names=channel_names,
            units="",
            reduction=reduction,
            compute_on_step=compute_on_step,
            dist_sync_on_step=dist_sync_on_step,
            process_group=process_group,
            dist_sync_fn=dist_sync_fn,
        )

    def func(self, preds, targets) -> torch.Tensor:
        return mF.coefficient_variation(y_pred=preds, y_true=targets)


class VOE(Metric):
    def __init__(
        self,
        channel_names: Sequence[str] = None,
        reduction="none",
        compute_on_step: bool = False,
        dist_sync_on_step: bool = False,
        process_group: bool = None,
        dist_sync_fn: bool = None,
    ):
        super().__init__(
            channel_names=channel_names,
            units="",
            reduction=reduction,
            compute_on_step=compute_on_step,
            dist_sync_on_step=dist_sync_on_step,
            process_group=process_group,
            dist_sync_fn=dist_sync_fn,
        )

    def func(self, preds, targets) -> torch.Tensor:
        return mF.volumetric_overlap_error(y_pred=preds, y_true=targets)


class ASSD(Metric):
    def __init__(
        self,
        connectivity: int = 1,
        channel_names: Sequence[str] = None,
        reduction="none",
        compute_on_step: bool = False,
        dist_sync_on_step: bool = False,
        process_group: bool = None,
        dist_sync_fn: bool = None,
    ):
        super().__init__(
            channel_names=channel_names,
            units="",
            reduction=reduction,
            compute_on_step=compute_on_step,
            dist_sync_on_step=dist_sync_on_step,
            process_group=process_group,
            dist_sync_fn=dist_sync_fn,
        )
        self.connectivity = connectivity

    def func(self, preds, targets, spacing=None) -> torch.Tensor:
        return mF.assd(
            y_pred=preds, y_true=targets, spacing=spacing, connectivity=self.connectivity
        )

    def update(self, preds, targets, spacing=None, ids=None):
        return super().update(preds, targets, spacing=spacing, ids=ids)
