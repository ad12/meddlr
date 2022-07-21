from typing import Sequence

import torch
from lpips import LPIPS as _LPIPS

from meddlr.metrics.metric import Metric
from meddlr.ops import complex as cplx


class NoTrainLpips(_LPIPS):
    def train(self, mode: bool) -> "NoTrainLpips":
        """the network should not be able to be switched away from evaluation mode."""
        return super().train(False)


class LPIPS(Metric):
    def __init__(
        self,
        net_type: str = "vgg",
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
        # choose parameters to be default but flexible in build dictionary config
        valid_net_type = ("vgg", "alex", "squeeze")
        if net_type not in valid_net_type:
            raise ValueError(
                f"Argument `net_type` must be one \
                            of {valid_net_type}, but got {net_type}."
            )

        self.net = NoTrainLpips(net=net_type, lpips=False, verbose=False)

    def func(self, preds, targets) -> torch.Tensor:
        if targets.shape[1] != 1:
            raise ValueError(
                f"LPIPS input tensor must have 1 channel, \
                            but got input tensor with shape {preds.shape}"
            )

        shape = (targets.shape[0], targets.shape[1])
        preds = self.preprocess_lpips(preds)
        targets = self.preprocess_lpips(targets)

        loss = self.net(preds, targets)
        loss = loss.view(shape)

        return loss

    def preprocess_lpips(self, img):
        """Preprocess image per LPIPS implementation.

        Converts to a magnitude scan, normalizes between -1 and 1, and reshape to tensor of shape
        Bx3xHxW where the channel dimension is repeated

        Args:
            img (torch.Tensor): Tensor to preprocess of shape Bx1xHxW

        Returns:
            Preprocessed tensor shape Bx3xHxW
        """
        # mode here as well...
        is_complex = cplx.is_complex(img) or cplx.is_complex_as_real(img)
        abs_func = cplx.abs if is_complex else torch.abs

        img = abs_func(img)
        shape = (img.shape[0], img.shape[1], -1)

        img_min = torch.min(img.view(shape), dim=-1)[:, :, None, None]
        img_max = torch.max(img.view(shape), dim=-1)[:, :, None, None]

        img = 2 * (img - img_min) / (img_max - img_min) - 1
        img = img.repeat(1, 3, 1, 1)

        return img
