from typing import Sequence

import torch

import meddlr.metrics.functional as mF
from meddlr.metrics.metric import Metric

__all__ = ["DSC", "CV", "VOE", "ASSD"]


class DSC(Metric):
    """Dice score coefficient.

    Attributes:
        channel_names (Sequence[str]): Category names corresponding to the channels.
    """

    is_differentiable = True
    higher_is_better = True

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


Dice = DSC


class CV(Metric):
    """Coefficient of variation.

    Attributes:
        channel_names (Sequence[str]): Category names corresponding to the channels.
    """

    is_differentiable = True
    higher_is_better = False

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
    """Volumetric overlap error.

    Attributes:
        channel_names (Sequence[str]): Category names corresponding to the channels.
    """

    is_differentiable = True
    higher_is_better = False

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
    """Average symmetric surface distance.

    Attributes:
        connectivity (int): The neighbourhood/connectivity considered when determining
            the surface of the binary objects.
        channel_names (Sequence[str]): Category names corresponding to the channels.

    Note:
        This metric is not differentiable.
    """

    is_differentiable = False
    higher_is_better = False

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
        """
        Args:
            connectivity (int): The neighbourhood/connectivity considered when determining
                the surface of the binary objects. If in doubt, leave it as it is.
            channel_names (Sequence[str]): Category names corresponding to the channels.
        """
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
