import torch
from torch import nn
import torchvision.utils as tv_utils

from ss_recon.utils import complex_utils as cplx
from ss_recon.utils.events import get_event_storage

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

        noise_std_dev = cfg.MODEL.CONSISTENCY.AUG.NOISE.STD_DEV
        assert len(noise_std_dev) == 1, (
            "Noise std dev currently only supports one value."
        )
        self.noise_std_dev = noise_std_dev[0]

    def augment(self, inputs):
        """Noise augmentation module.
        TODO: Perform the augmentation here.
        """
        kspace = inputs["kspace"].clone()
        mask = cplx.get_mask(kspace)

        noise_std = self.noise_std_dev
        noise = noise_std * torch.randn(inputs['kspace'].size())
        masked_noise = noise * mask
        aug_kspace = kspace + masked_noise

        inputs = {k: v.clone() for k, v in inputs.items() if k != "kspace"}
        inputs["kspace"] = aug_kspace
        return inputs

    def visualize_aug_training(self, kspace, kspace_aug, preds, preds_aug):
        """Visualize training of augmented data.

        Args:
            kspace: The base kspace.
            kspace_aug: The augmented kspace.
            pred: Reconstruction of base kspace. Shape: NxHxWx2.
            pred_aug: Reconstruction of augmented kspace. Shape: NxHxWx2.
        """
        storage = get_event_storage()

        with torch.no_grad():
            # calc mask for first coil only
            kspace = kspace.cpu()[0, ..., 0, :].unsqueeze(0)
            kspace_aug = kspace_aug.cpu()[0, ..., 0, :].unsqueeze(0)
            preds = preds.cpu()[0, ...].unsqueeze(0)
            preds_aug = preds_aug.cpu()[0, ...].unsqueeze(0)
            # zfs = zfs.cpu()[0, ...].unsqueeze(0)

            all_images = torch.cat([preds, preds_aug], dim=2)
            all_kspace = torch.cat([kspace, kspace_aug], dim=2)

            imgs_to_write = {
                "phases": cplx.angle(all_images),
                "images": cplx.abs(all_images),
                "errors": cplx.abs(preds_aug - preds),
                "masks": cplx.get_mask(kspace),
                "kspace": cplx.abs(all_kspace),
            }

            for name, data in imgs_to_write.items():
                data = data.squeeze(-1).unsqueeze(1)
                data = tv_utils.make_grid(
                    data, nrow=1, padding=1, normalize=True, scale_each=True,
                )
                storage.put_image(
                    "train_aug/{}".format(name), data.numpy(), data_format="CHW"
                )

    def forward(self, inputs):
        if not self.training:
            assert "unsupervised" not in inputs, (
                "unsupervised inputs should not be provided in eval mode"
            )
            inputs = inputs.get("supervised", inputs)
            return self.unrolled(inputs)

        vis_training = False
        if self.training and self.vis_period > 0:
            storage = get_event_storage()
            if storage.iter % self.vis_period == 0:
                vis_training = True

        inputs_supervised = inputs.get("supervised", None)
        inputs_unsupervised = inputs.get("unsupervised", None)
        if inputs_supervised is None and inputs_unsupervised is None:
            raise ValueError("Examples not formatted in the proper way")
        output_dict = {}

        # Recon
        if inputs_supervised is not None:
            output_dict["recon"] = self.unrolled(
                inputs_supervised, return_pp=True, vis_training=vis_training,
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
            if vis_training:
                self.visualize_aug_training(
                    inputs_unsupervised["kspace"],
                    inputs_us_aug["kspace"],
                    pred_aug["target"],
                    pred_base,
                )

        return output_dict
