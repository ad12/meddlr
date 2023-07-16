import itertools

import torch
import torchvision.utils as tv_utils
from torch import nn

import meddlr.ops.complex as cplx
from meddlr.config.config import configurable
from meddlr.data.transforms.noise import NoiseModel
from meddlr.forward.mri import SenseModel
from meddlr.modeling.meta_arch.build import META_ARCH_REGISTRY, build_model
from meddlr.utils.events import get_event_storage
from meddlr.utils.general import move_to_device

__all__ = ["DenoisingModel"]


@META_ARCH_REGISTRY.register()
class DenoisingModel(nn.Module):
    """A denoising trainer."""

    @configurable
    def __init__(
        self,
        model,
        noiser: NoiseModel,
        use_fully_sampled_target=False,
        use_fully_sampled_target_eval: bool = None,
        vis_period: int = -1,
    ):
        """
        Args:
            model (nn.Module): The base model.
            noiser (NoiseModel): The additive noise model.
            use_fully_sampled_target (bool, optional): If ``True``,
                use fully sampled images as the target for denoising during training.
            use_fully_sampled_target_eval (bool, optional): If ``True``,
                use fully sampled images as the target for denoising during evaluation
                (includes validation). If ``None``, defaults to ``use_fully_sampled_target``.
        """
        super().__init__()
        self.model = model
        self.noiser = noiser

        # Default validation to use_fully_sampled_target_eval.
        if use_fully_sampled_target_eval is None:
            use_fully_sampled_target_eval = use_fully_sampled_target
        self.use_fully_sampled_target = use_fully_sampled_target
        self.use_fully_sampled_target_eval = use_fully_sampled_target_eval

        # Visualization done by this model
        if hasattr(self.model, "vis_period") and vis_period > 0:
            self.model.vis_period = -1
        self.vis_period = vis_period

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
                data = tv_utils.make_grid(data, nrow=1, padding=1, normalize=True, scale_each=True)
                storage.put_image("train/{}".format(name), data.numpy(), data_format="CHW")

    def forward(self, inputs, return_pp=False, vis_training=False):
        """
        TODO: condense into list of dataset dicts.
        Args:
            inputs: Standard meddlr module input dictionary
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
            Dict: A standard meddlr output dict
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

        device = next(self.parameters()).device
        inputs = move_to_device(inputs, device)

        if self.training and self.vis_period > 0:
            storage = get_event_storage()
            if storage.iter % self.vis_period == 0:
                vis_training = True

        if not any(k in inputs for k in ["supervised", "unsupervised"]):
            inputs = {"supervised": inputs}

        use_fully_sampled_target = (self.training and self.use_fully_sampled_target) or (
            not self.training and self.use_fully_sampled_target_eval
        )

        if "supervised" in inputs:
            sup_inputs = inputs["supervised"]
            if use_fully_sampled_target:
                img = sup_inputs["target"]
                A = SenseModel(sup_inputs["maps"])
                sup_inputs["kspace"] = self.augment(A(img, adjoint=False))
            else:
                kspace = sup_inputs["kspace"]
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

        output_dict = self.model(inputs, return_pp=True, vis_training=vis_training)

        return output_dict

    @classmethod
    def from_config(cls, cfg):
        device = torch.device(cfg.MODEL.DEVICE)

        model_cfg = cfg.clone()
        model_cfg.defrost()
        model_cfg.MODEL.META_ARCHITECTURE = cfg.MODEL.DENOISING.META_ARCHITECTURE
        model_cfg.freeze()
        model = build_model(model_cfg)

        noise_cfg = cfg.clone()
        noise_cfg.defrost()
        noise_cfg.MODEL.CONSISTENCY.AUG.NOISE.STD_DEV = cfg.MODEL.DENOISING.NOISE.STD_DEV
        noise_cfg.freeze()
        noiser = NoiseModel.from_cfg(noise_cfg, device=device)

        use_fully_sampled_target = cfg.MODEL.DENOISING.NOISE.USE_FULLY_SAMPLED_TARGET
        use_fully_sampled_target_eval = cfg.MODEL.DENOISING.NOISE.USE_FULLY_SAMPLED_TARGET_EVAL
        return {
            "model": model,
            "noiser": noiser,
            "use_fully_sampled_target": use_fully_sampled_target,
            "use_fully_sampled_target_eval": use_fully_sampled_target_eval,
        }
