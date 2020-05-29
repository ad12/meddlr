import torch
from torch import nn

from ss_recon.utils import complex_utils as cplx
from .build import META_ARCH_REGISTRY
from .unrolled import GeneralizedUnrolledCNN


@META_ARCH_REGISTRY.register()
class N2RModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.unrolled = GeneralizedUnrolledCNN(cfg)
        self.unrolled.vis_period = -1

        self.vis_period = cfg.VIS_PERIOD
        # Keep gradient for base images in transform.
        self.use_base_grad = False

    def augment(self, inputs):
        """Noise augmentation module.
        TODO: Perform the augmentation here.
        """
        kspace = inputs["kspace"].clone()
        mask = cplx.get_mask(kspace)

        # Replace line below with augmentation.
        aug_kspace = kspace 

        inputs = {k: v.clone() for k, v in inputs.items() if k != "kspace"}
        inputs["kspace"] = aug_kspace
        return inputs

    def forward(self, inputs):
        if not self.training:
            assert "unsupervised" not in inputs, (
                "unsupervised inputs should not be provided in eval mode"
            )
            inputs = inputs.get("supervised", inputs)
            return self.unrolled(inputs)

        inputs_supervised = inputs.get("supervised", None)
        inputs_unsupervised = inputs.get("unsupervised", None)
        if inputs_supervised is None and inputs_unsupervised is None:
            raise ValueError("Examples not formatted in the proper way")
        output_dict = {}

        # Recon
        if inputs_supervised is not None:
            output_dict["recon"] = self.unrolled(
                inputs_supervised, return_pp=True
            )

        # Consistency.
        if inputs_unsupervised is not None:
            inputs_us_aug = self.augment(inputs_unsupervised)
            with torch.no_grad():
                pred_base = self.unrolled(inputs_unsupervised)["pred"]
            pred_aug = self.unrolled(inputs_us_aug, return_pp=True)
            if "target" in pred_aug:
                del pred_aug["target"]
            pred_aug["target"] = pred_base.detach()
            output_dict["consistency"] = pred_aug

        return output_dict
