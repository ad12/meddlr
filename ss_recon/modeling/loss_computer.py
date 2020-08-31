from abc import ABC, abstractmethod

import torch
from fvcore.common.registry import Registry

from ss_recon.data.transforms.transform import build_normalizer
from ss_recon.evaluation.metrics import compute_nrmse
from ss_recon.utils import complex_utils as cplx


LOSS_COMPUTER_REGISTRY = Registry("LOSS_COMPUTER")  # noqa F401 isort:skip
LOSS_COMPUTER_REGISTRY.__doc__ = """
Registry for loss computers.

The registered object will be called with `obj(cfg)`
and expected to return a LossComputer object.
"""

EPS = 1e-11
IMAGE_LOSSES = ["l1", "l2", "psnr", "nrmse"]


def build_loss_computer(cfg, name):
    return LOSS_COMPUTER_REGISTRY.get(name)(cfg)


class LossComputer(ABC):
    def __init__(self, cfg):
        self._normalizer = build_normalizer(cfg)

    @abstractmethod
    def __call__(self, input, output):
        pass

    def _get_metrics(self, target: torch.Tensor, output: torch.Tensor, loss_name):
        # Compute metrics
        abs_error = cplx.abs(output - target)
        l1 = torch.mean(abs_error)
        N = target.shape[0]

        abs_error = abs_error.view(N, -1)
        tgt_mag = cplx.abs(target).view(N, -1)
        l2 = torch.sqrt(torch.mean(abs_error ** 2, dim=1))
        psnr = 20 * torch.log10(tgt_mag.max(dim=1)[0] / (l2 + EPS))
        nrmse = l2 / torch.sqrt(torch.mean(tgt_mag ** 2, dim=1))

        metrics_dict = {"l1": l1, "l2": l2.mean(), "psnr": psnr.mean(), "nrmse": nrmse.mean()}
        loss = metrics_dict[loss_name]
        metrics_dict["loss"] = loss

        return metrics_dict


@LOSS_COMPUTER_REGISTRY.register()
class BasicLossComputer(LossComputer):
    def __init__(self, cfg):
        super().__init__(cfg)
        loss_name = cfg.MODEL.RECON_LOSS.NAME
        assert loss_name in IMAGE_LOSSES
        self.loss = loss_name
        self.renormalize_data = cfg.MODEL.RECON_LOSS.RENORMALIZE_DATA

    def __call__(self, input, output):
        pred: torch.Tensor = output["pred"]
        target = output["target"].to(pred.device)

        if self.renormalize_data:
            normalized = self._normalizer.undo(
                image=pred, target=target, mean=input["mean"], std=input["std"]
            )
            output = normalized["image"]
            target = normalized["target"]
        else:
            output = pred

        metrics_dict = self._get_metrics(target, output, self.loss)
        return metrics_dict


@LOSS_COMPUTER_REGISTRY.register()
class N2RLossComputer(LossComputer):
    def __init__(self, cfg):
        super().__init__(cfg)
        recon_loss = cfg.MODEL.RECON_LOSS.NAME
        consistency_loss = cfg.MODEL.CONSISTENCY.LOSS_NAME

        assert recon_loss in IMAGE_LOSSES
        assert consistency_loss in IMAGE_LOSSES

        self.recon_loss = recon_loss
        self.consistency_loss = consistency_loss
        self.renormalize_data = cfg.MODEL.RECON_LOSS.RENORMALIZE_DATA
        self.consistency_weight = cfg.MODEL.CONSISTENCY.LOSS_WEIGHT
        # self.use_robust = cfg.MODEL.LOSS.USE_ROBUST
        # self.beta = cfg.MODEL.LOSS.BETA
        # self.robust_step_size = cfg.MODEL.LOSS.ROBUST_STEP_SIZE

    def _compute_metrics(self, output, loss):
        """Computes image metrics on prediction and target data.
        """
        if output is None or len(output) == 0:
            return {
                k: torch.Tensor([0.0]).detach() for k in ["l1", "l2", "psnr", "loss"]
            }

        pred: torch.Tensor = output["pred"]
        target = output["target"].to(pred.device)
        if self.renormalize_data:
            normalized = self._normalizer.undo(
                image=pred, target=target, mean=input["mean"], std=input["std"]
            )
            output = normalized["image"]
            target = normalized["target"]
        else:
            output = pred

        # Compute metrics
        metrics_dict = self._get_metrics(target, output, loss)
        return metrics_dict

    # def compute_robust_loss(self, group_loss):
    #     if torch.is_grad_enabled():  # update adv_probs if in training mode
    #         adjusted_loss = group_loss
    #         if self.do_adj:
    #             adjusted_loss += self.loss_adjustment
    #         logit_step = self.robust_step_size * adjusted_loss.data
    #         if self.stable:
    #             self.adv_probs_logits = self.adv_probs_logits + logit_step
    #         else:
    #             self.adv_probs = self.adv_probs * torch.exp(logit_step)
    #             self.adv_probs = self.adv_probs / self.adv_probs.sum()
    #
    #     if self.stable:
    #         adv_probs = torch.softmax(self.adv_probs_logits, dim=-1)
    #     else:
    #         adv_probs = self.adv_probs
    #     robust_loss = group_loss @ adv_probs
    #     return robust_loss, adv_probs

    def __call__(self, input, output):
        output_recon = output.get("recon", None)
        output_consistency = output.get("consistency", None)

        loss = 0
        metrics_recon = {
            "recon_{}".format(k): v
            for k, v in self._compute_metrics(output_recon, self.recon_loss).items()
        }
        if output_recon is not None:
            loss += metrics_recon["recon_loss"]

        metrics_consistency = {
            "cons_{}".format(k): v
            for k, v in self._compute_metrics(output_consistency, self.consistency_loss).items() # noqa
        }
        if output_consistency is not None:
            loss += self.consistency_weight * metrics_consistency["cons_loss"]

        metrics_consistency.update(metrics_recon)
        metrics = metrics_consistency

        metrics["loss"] = loss
        return metrics
