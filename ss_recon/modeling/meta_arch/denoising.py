"""Unrolled Compressed Sensing (2D).

This file contains an implementation of the Unrolled Compressed Sensing
framework by CM Sandino, JY Cheng, et al. See paper below for more details.

It is also based heavily on the codebase below:

https://github.com/MRSRL/dl-cs

Implementation is based on:
    CM Sandino, JY Cheng, et al. "Compressed Sensing: From Research to
    Clinical Practice with Deep Neural Networks" IEEE Signal Processing
    Magazine, 2020.
"""
import itertools

import torch
import torchvision.utils as tv_utils
from torch import nn

import ss_recon.utils.complex_utils as cplx
from ss_recon.data.transforms.noise import NoiseModel
from ss_recon.modeling.meta_arch.build import META_ARCH_REGISTRY, build_model
from ss_recon.utils.events import get_event_storage
from ss_recon.utils.transforms import SenseModel

__all__ = ["DenoisingModel"]


@META_ARCH_REGISTRY.register()
class DenoisingModel(nn.Module):
    def __init__(self, cfg):
        """
        Args:
            params (dict): Dictionary containing network parameters
        """
        super().__init__()
        self.device = torch.device(cfg.MODEL.DEVICE)

        model_cfg = cfg.clone()
        model_cfg.defrost()
        model_cfg.MODEL.META_ARCHITECTURE = cfg.MODEL.DENOISING.META_ARCHITECTURE
        model_cfg.freeze()
        self.model = build_model(model_cfg)

        # Visualization done by this model
        if hasattr(self.model, "vis_period"):
            self.model.vis_period = -1
        self.vis_period = cfg.VIS_PERIOD

        noise_cfg = cfg.clone()
        noise_cfg.defrost()
        noise_cfg.MODEL.CONSISTENCY.AUG.NOISE.STD_DEV = cfg.MODEL.DENOISING.NOISE.STD_DEV
        noise_cfg.freeze()
        self.noiser = NoiseModel.from_cfg(noise_cfg)

        # TODO: Move to config at some point
        # If fully sampled kspace is available, perform denoising on the fully sampled kspace.
        # If False, denoising will be performed on the undersampled kspace.
        self.add_noise_fully_sampled = True

    def augment(self, kspace):
        """Noise augmentation module.
        TODO: Perform the augmentation here.
        """
        kspace = kspace.detach().clone()
        return self.noiser(kspace, clone=False)

    def visualize_training(self, kspace, zfs, targets, preds):
        """A function used to visualize reconstructions.

        Args:
            targets: NxHxWx2 tensors of target images.
            preds: NxHxWx2 tensors of predictions.
        """
        storage = get_event_storage()

        with torch.no_grad():
            if cplx.is_complex(kspace):
                kspace = torch.view_as_real(kspace)
            kspace = kspace[0, ..., 0, :].unsqueeze(0).cpu()  # calc mask for first coil only
            targets = targets[0, ...].unsqueeze(0).cpu()
            preds = preds[0, ...].unsqueeze(0).cpu()
            zfs = zfs[0, ...].unsqueeze(0).cpu()

            all_images = torch.cat([zfs, preds, targets], dim=2)

            imgs_to_write = {
                "phases": cplx.angle(all_images),
                "images": cplx.abs(all_images),
                "errors": cplx.abs(preds - targets),
                "masks": cplx.get_mask(kspace),
            }

            for name, data in imgs_to_write.items():
                data = data.squeeze(-1).unsqueeze(1)
                data = tv_utils.make_grid(
                    data,
                    nrow=1,
                    padding=1,
                    normalize=True,
                    scale_each=True,
                )
                storage.put_image("train/{}".format(name), data.numpy(), data_format="CHW")

    def forward(self, inputs, return_pp=False, vis_training=False):
        """
        TODO: condense into list of dataset dicts.
        Args:
            inputs: Standard ss_recon module input dictionary
                * "kspace": Kspace. If fully sampled, and want to simulate
                    undersampled kspace, provide "mask" argument.
                * "maps": Sensitivity maps
                * "target" (optional): Target image (typically fully sampled).
                * "mask" (optional): Undersampling mask to apply.
                * "signal_model" (optional): The signal model. If provided,
                    "maps" will not be used to estimate the signal model.
                    Use with caution.
            return_pp (bool, optional): If `True`, return post-processing
                parameters "mean", "std", and "norm" if included in the input.
            vis_training (bool, optional): If `True`, force visualize training
                on this pass. Can only be `True` if model is in training mode.

        Returns:
            Dict: A standard ss_recon output dict
                * "pred": The reconstructed image
                * "target" (optional): The target image.
                    Added if provided in the input.
                * "mean"/"std"/"norm" (optional): Pre-processing parameters.
                    Added if provided in the input.
                * "zf_image": The zero-filled image.
                    Added when model is in eval mode.
        """
        if vis_training and not self.training:
            raise ValueError("vis_training is only applicable in training mode.")

        if self.training and self.vis_period > 0:
            storage = get_event_storage()
            if storage.iter % self.vis_period == 0:
                vis_training = True

        if not any(k in inputs for k in ["supervised", "unsupervised"]):
            inputs = {"supervised": inputs}

        if "supervised" in inputs:
            sup_inputs = inputs["supervised"]
            if self.add_noise_fully_sampled:
                img = sup_inputs["target"]
                A = SenseModel(sup_inputs["maps"])
                sup_inputs["kspace"] = self.augment(A(img, adjoint=False))
            else:
                kspace = sup_inputs["target"]
                A = SenseModel(sup_inputs["maps"], weights=cplx.get_mask(kspace))
                sup_inputs["target"] = A(kspace, adjoint=True).detach()
                sup_inputs["kspace"] = self.augment(kspace)
        if "unsupervised" in inputs:
            unsup_inputs = inputs["unsupervised"]
            kspace = unsup_inputs["kspace"]
            A = SenseModel(unsup_inputs["maps"], weights=cplx.get_mask(kspace))
            unsup_inputs["target"] = A(kspace, adjoint=True).detach()
            unsup_inputs["kspace"] = self.augment(kspace)

        keys = [set(v.keys()) for v in inputs.values()]
        keys = keys[0].intersection(*keys)
        inputs = {k: [inputs[field][k] for field in inputs.keys()] for k in keys}
        inputs = {
            k: torch.cat(inputs[k], dim=0)
            if isinstance(inputs[k][0], torch.Tensor)
            else itertools.chain(inputs[k])
            for k in inputs.keys()
        }

        output_dict = self.model(
            inputs,
            return_pp=True,
            vis_training=vis_training,
        )

        return output_dict
