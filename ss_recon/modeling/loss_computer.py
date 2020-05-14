import torch
from ss_recon.utils import complex_utils as cplx


class BasicLossComputer(object):
    def __init__(self, cfg):
        loss_name = cfg.LOSS.NAME
        assert loss_name in ["l1", "l2", "psnr"]
        self.loss = loss_name

    def __call__(self, output_dict):
        pred: torch.Tensor = output_dict["pred"]
        mean = output_dict["mean"].to(pred.device)
        std = output_dict["std"].to(pred.device)
        target = output_dict["target"].to(pred.device)

        output = pred * std + mean
        target = target * std + mean

        # Compute metrics
        abs_error = cplx.abs(output - target)
        l1 = torch.mean(abs_error)
        l2 = torch.sqrt(torch.mean(abs_error ** 2))
        psnr = 20 * torch.log10(cplx.abs(target).max() / l2)

        metrics_dict = {
            "l1": l1,
            "l2": l2,
            "psnr": psnr,
        }
        loss = metrics_dict[self.loss]
        metrics_dict["loss"] = loss

        return metrics_dict
