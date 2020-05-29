import torch
from fvcore.common.registry import Registry

from ss_recon.utils import complex_utils as cplx


LOSS_COMPUTER_REGISTRY = Registry("LOSS_COMPUTER")  # noqa F401 isort:skip
LOSS_COMPUTER_REGISTRY.__doc__ = """
Registry for loss computers.

The registered object will be called with `obj(cfg)`
and expected to return a LossComputer object.
"""


def build_loss_computer(cfg, name):
    return LOSS_COMPUTER_REGISTRY.get(name)(cfg)


@LOSS_COMPUTER_REGISTRY.register()
class BasicLossComputer(object):
    def __init__(self, cfg):
        loss_name = cfg.MODEL.RECON_LOSS.NAME
        assert loss_name in ["l1", "l2", "psnr"]
        self.loss = loss_name
        self.renormalize_data = cfg.MODEL.RECON_LOSS.RENORMALIZE_DATA

    def __call__(self, input, output):
        pred: torch.Tensor = output["pred"]
        target = output["target"].to(pred.device)

        if self.renormalize_data:
            mean = input["mean"].to(pred.device)
            std = input["std"].to(pred.device)
            output = pred * std + mean
            target = target * std + mean
        else:
            output = pred

        # Compute metrics
        abs_error = cplx.abs(output - target)
        l1 = torch.mean(abs_error)
        l2 = torch.sqrt(torch.mean(abs_error ** 2))
        psnr = 20 * torch.log10(cplx.abs(target).max() / l2)

        metrics_dict = {"l1": l1, "l2": l2, "psnr": psnr}
        loss = metrics_dict[self.loss]
        metrics_dict["loss"] = loss

        return metrics_dict


@LOSS_COMPUTER_REGISTRY.register()
class N2RLossComputer(object):
    def __init__(self, cfg):
        recon_loss = cfg.MODEL.RECON_LOSS.NAME
        consistency_loss = cfg.MODEL.CONSISTENCY.LOSS_NAME

        assert recon_loss in ["l1", "l2", "psnr"]
        assert consistency_loss in ["l1", "l2", "psnr"]

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
                k: torch.Tensor([0.0]) for k in ["l1", "l2", "psnr"]
            }

        pred: torch.Tensor = output["pred"]
        target = output["target"].to(pred.device)
        if self.renormalize_data:
            mean = output["mean"].to(pred.device)
            std = output["std"].to(pred.device)
            output = pred * std + mean
            target = target * std + mean
        else:
            output = pred

        # Compute metrics
        abs_error = cplx.abs(output - target)
        l1 = torch.mean(abs_error)
        l2 = torch.sqrt(torch.mean(abs_error ** 2))
        psnr = 20 * torch.log10(cplx.abs(target).max() / l2)

        metrics_dict = {"l1": l1, "l2": l2, "psnr": psnr}
        metrics_dict["loss"] = metrics_dict[loss]

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
        output_consistency = output("consistency", None)

        metrics_recon = {
            "recon_{}".format(k): v
            for k, v in self._compute_metrics(output_recon, self.recon_loss)
        }

        metrics_consistency = {
            "cons_{}".format(k): v
            for k, v in self._compute_metrics(output_consistency, self.consistency_loss)  # noqa
        }

        metrics_consistency.update(metrics_recon)
        metrics = metrics_consistency

        metrics["loss"] = metrics["recon_loss"] + self.consistency_weight * metrics["cons_loss"]  # noqa
        return metrics

