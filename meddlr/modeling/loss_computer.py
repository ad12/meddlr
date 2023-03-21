from abc import ABC, abstractmethod
from typing import Callable, Optional

import torch
from fvcore.common.registry import Registry

import meddlr.metrics.functional as mF
import meddlr.ops as oF
from meddlr.data.transforms.transform import build_normalizer
from meddlr.forward.mri import SenseModel
from meddlr.metrics.build import build_metrics
from meddlr.metrics.metric import Metric
from meddlr.ops import complex as cplx

LOSS_COMPUTER_REGISTRY = Registry("LOSS_COMPUTER")  # noqa F401 isort:skip
LOSS_COMPUTER_REGISTRY.__doc__ = """
Registry for loss computers.

The registered object will be called with `obj(cfg)`
and expected to return a LossComputer object.
"""

EPS = 1e-11
IMAGE_LOSSES = [
    "l1",
    "l2",
    "psnr",
    "nrmse",
    "mag_l1",
    "perp_loss",
    "ssim_loss",
    "ssim_phase_loss",
    "ssim_mag_phase_loss",
]
KSPACE_LOSSES = [
    "k_l1",
    "k_l1_sum",
    "k_l1_normalized",
    "k_l1_sum_normalized",
    "k_l1_l2_sum_normalized",
]
# Add option for signal model corrected loss computation.
# TODO (arjundd): Clean up.
KSPACE_LOSSES += [f"signal_model_corr_{loss}" for loss in KSPACE_LOSSES]


def build_loss_computer(cfg, name, **kwargs):
    return LOSS_COMPUTER_REGISTRY.get(name)(cfg, **kwargs)


class LossComputer(ABC):
    def __init__(self, cfg, loss_func: Callable = None):
        self._normalizer = build_normalizer(cfg)
        self.loss_func = loss_func

    @abstractmethod
    def __call__(self, input, output):
        pass

    def _get_metrics(
        self,
        target: torch.Tensor,
        output: torch.Tensor,
        loss_name: str = None,
        signal_model: Optional[SenseModel] = None,
    ):
        if self.loss is not None:
            is_same_loss = isinstance(self.loss, str) and loss_name == self.loss
            assert loss_name is None or is_same_loss

        # Compute metrics
        if loss_name == "mag_l1":
            abs_error = torch.abs(output - target)
            abs_mag_error = abs_error
        else:
            abs_error = cplx.abs(output - target)
            abs_mag_error = torch.abs(cplx.abs(output) - cplx.abs(target))
        l1 = torch.mean(abs_error)
        mag_l1 = torch.mean(abs_mag_error)
        N = target.shape[0]

        abs_error = abs_error.view(N, -1)
        if loss_name == "mag_l1":
            tgt_mag = torch.abs(target).view(N, -1)
        else:
            tgt_mag = cplx.abs(target).view(N, -1)
        l2 = torch.sqrt(torch.mean(abs_error**2, dim=1))
        psnr = 20 * torch.log10(tgt_mag.max(dim=1)[0] / (l2 + EPS))
        nrmse = l2 / torch.sqrt(torch.mean(tgt_mag**2, dim=1))

        metrics_dict = {
            "l1": l1,
            "l2": l2.mean(),
            "psnr": psnr.mean(),
            "nrmse": nrmse.mean(),
            "mag_l1": mag_l1,
            "ssim_wang": mF.ssim(
                cplx.channels_first(output).contiguous(),
                cplx.channels_first(target).contiguous(),
                method="wang",
            ).mean(),
            "ssim_wang_phase": mF.ssim(
                cplx.channels_first(output).contiguous(),
                cplx.channels_first(target).contiguous(),
                method="wang",
                im_type="phase",
            ).mean(),
        }
        if loss_name == "perp_loss":
            metrics_dict.update(perp_loss(output, target))
        if loss_name == "ssim_loss":
            metrics_dict["ssim_loss"] = 1.0 - metrics_dict["ssim_wang"]
        if loss_name == "ssim_phase_loss":
            metrics_dict["ssim_phase_loss"] = 1.0 - metrics_dict["ssim_wang_phase"]
        if loss_name == "ssim_mag_phase_loss":
            avg_ssim = (metrics_dict["ssim_wang"] + metrics_dict["ssim_wang_phase"]) / 2
            metrics_dict["ssim_mag_phase_loss"] = 1.0 - avg_ssim

        if loss_name in KSPACE_LOSSES:
            if loss_name.startswith("signal_model_corr_"):
                assert signal_model is not None
                loss_name = loss_name.split("signal_model_corr_")[1]
                output = signal_model(output)
                target = signal_model(target)
            else:
                target = oF.fft2c(target, channels_last=True)
                output = oF.fft2c(output, channels_last=True)
            abs_error = cplx.abs(target - output)
            if loss_name == "k_l1":
                metrics_dict["loss"] = torch.mean(abs_error)
            elif loss_name == "k_l1_sum":
                metrics_dict["loss"] = torch.sum(abs_error)
            elif loss_name == "k_l1_normalized":
                metrics_dict["loss"] = torch.mean(abs_error / (cplx.abs(target) + EPS))
            elif loss_name == "k_l1_sum_normalized":
                metrics_dict["loss"] = torch.sum(abs_error) / torch.sum(cplx.abs(target))
            elif loss_name == "k_l1_l2_sum_normalized":
                kl1_norm = torch.sum(abs_error) / torch.sum(cplx.abs(target))
                kl2_norm = torch.sqrt(torch.sum(abs_error**2)) / torch.sqrt(
                    torch.sum(cplx.abs(target) ** 2)
                )  # noqa: E501
                metrics_dict["loss"] = 0.5 * kl1_norm + 0.5 * kl2_norm
            else:
                assert False  # should not reach here
        elif self.loss_func is not None:
            output = cplx.channels_first(output)
            target = cplx.channels_first(target)
            metrics_dict["loss"] = self.loss_func(output, target)
        else:
            loss = metrics_dict[loss_name]
            metrics_dict["loss"] = loss

        return metrics_dict


@LOSS_COMPUTER_REGISTRY.register()
class BasicLossComputer(LossComputer):
    def __init__(self, cfg):
        loss_name = cfg.MODEL.RECON_LOSS.NAME
        if loss_name in IMAGE_LOSSES or loss_name in KSPACE_LOSSES:
            loss_func = None
        else:
            loss_func: Metric = list(build_metrics([loss_name]).values(copy_state=False))[0]
            loss_func = loss_func.to(cfg.MODEL.DEVICE)
            loss_name = None
            if not loss_func.is_differentiable:
                raise ValueError("Loss function must be differentiable")
            if loss_func.higher_is_better:
                raise ValueError(
                    "Loss function must be lower is better. "
                    "We do not currently support higher_is_better losses."
                )
        self.loss = loss_name
        self.renormalize_data = cfg.MODEL.RECON_LOSS.RENORMALIZE_DATA

        super().__init__(cfg, loss_func=loss_func)

    def __call__(self, input, output):
        pred: torch.Tensor = output["pred"]
        target = output["target"].to(pred.device)
        signal_model = output.get("signal_model")

        if self.renormalize_data:
            normalization_args = {k: input.get(k, output.get(k, None)) for k in ["mean", "std"]}
            normalized = self._normalizer.undo(
                image=pred,
                target=target,
                mean=normalization_args["mean"],
                std=normalization_args["std"],
            )
            output = normalized["image"]
            target = normalized["target"]
        else:
            output = pred

        metrics_dict = self._get_metrics(target, output, self.loss, signal_model=signal_model)
        return metrics_dict


@LOSS_COMPUTER_REGISTRY.register()
class N2RLossComputer(LossComputer):
    def __init__(self, cfg):
        super().__init__(cfg)
        recon_loss = cfg.MODEL.RECON_LOSS.NAME
        consistency_loss = cfg.MODEL.CONSISTENCY.LOSS_NAME
        latent_loss = cfg.MODEL.CONSISTENCY.LATENT_LOSS_NAME

        assert recon_loss in IMAGE_LOSSES or recon_loss in KSPACE_LOSSES
        assert consistency_loss in IMAGE_LOSSES or consistency_loss in KSPACE_LOSSES

        self.loss = None
        self.recon_loss = recon_loss
        self.consistency_loss = consistency_loss
        self.latent_loss = latent_loss
        self.renormalize_data = cfg.MODEL.RECON_LOSS.RENORMALIZE_DATA
        self.consistency_weight = cfg.MODEL.CONSISTENCY.LOSS_WEIGHT
        self.latent_weight = cfg.MODEL.CONSISTENCY.LATENT_LOSS_WEIGHT
        self.use_latent = cfg.MODEL.CONSISTENCY.USE_LATENT
        self.use_consistency = cfg.MODEL.CONSISTENCY.USE_CONSISTENCY
        self.num_latent_layers = cfg.MODEL.CONSISTENCY.NUM_LATENT_LAYERS
        self.latent_keys = ["E4", "E3", "D3", "E2", "D2", "E1", "D1"]
        # self.use_robust = cfg.MODEL.LOSS.USE_ROBUST
        # self.beta = cfg.MODEL.LOSS.BETA
        # self.robust_step_size = cfg.MODEL.LOSS.ROBUST_STEP_SIZE

    def _compute_metrics(self, input, output, loss):
        """Computes image metrics on prediction and target data."""
        if output is None or len(output) == 0:
            return {k: torch.Tensor([0.0]).detach() for k in ["l1", "l2", "psnr", "loss"]}

        pred: torch.Tensor = output["pred"]
        target = output["target"].to(pred.device)
        signal_model = output.get("signal_model")
        if self.renormalize_data:
            normalized = self._normalizer.undo(
                image=pred, target=target, mean=input["mean"], std=input["std"]
            )
            output = normalized["image"]
            target = normalized["target"]
        else:
            output = pred

        # Compute metrics
        metrics_dict = self._get_metrics(target, output, loss, signal_model=signal_model)
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
            for k, v in self._compute_metrics(
                input.get("supervised", None), output_recon, self.recon_loss
            ).items()
        }
        if output_recon is not None:
            loss += metrics_recon["recon_loss"]

        metrics_consistency = {
            "cons_{}".format(k): v
            for k, v in self._compute_metrics(
                input.get("unsupervised", None), output_consistency, self.consistency_loss
            ).items()  # noqa
        }
        if output_consistency is not None and self.use_consistency:
            loss += self.consistency_weight * metrics_consistency["cons_loss"]

        if self.use_latent:
            num_losses = self.num_latent_layers * 2 - 1
            all_metrics_latent = []
            for i in range(num_losses):
                output_latent = {}
                output_latent["target"] = output_recon["latent"][self.latent_keys[i]]
                output_latent["pred"] = output_consistency["latent"][self.latent_keys[i]]

                metrics_latent = {
                    "latent_" + self.latent_keys[i] + "_{}".format(k): v
                    for k, v in self._compute_metrics(
                        None, output_latent, self.latent_loss
                    ).items()  # noqa
                }

                all_metrics_latent.append(metrics_latent)
                loss += (
                    self.latent_weight * metrics_latent["latent_" + self.latent_keys[i] + "_loss"]
                )

        metrics = {}
        if output_consistency is not None and self.use_consistency:
            metrics.update(metrics_consistency)
        if output_recon is not None:
            metrics.update(metrics_recon)
        if self.use_latent:
            for i in range(num_losses):
                metrics.update(all_metrics_latent[i])

        metrics["loss"] = loss
        return metrics


def perp_loss(yhat, y):
    """Implementation of the perpendicular loss.

    Args:
        yhat: Predicted reconstruction. Must be complex.
        y: Target reconstruction. Must be complex.

    Returns:
        Dict[str, scalar]:

    References:
        Terpstra, et al. "Rethinking complex image reconstruction:
        ⟂-loss for improved complex image reconstruction with deep learning."
        International Society of Magnetic Resonance in Medicine Annual Meeting
        2021.
    """
    if cplx.is_complex(yhat):
        yhat = torch.view_as_real(yhat)
    if cplx.is_complex(y):
        y = torch.view_as_real(y)

    P = torch.abs(yhat[..., 0] * y[..., 1] - yhat[..., 1] * y[..., 0]) / cplx.abs(y)
    l1 = torch.abs(cplx.abs(y) - cplx.abs(yhat))

    return {"p_perp_loss": torch.mean(P), "perp_loss": torch.mean(P + l1)}
