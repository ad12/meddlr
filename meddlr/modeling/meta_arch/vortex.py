import logging

import torch
import torchvision.utils as tv_utils
from torch import nn

from meddlr.config.config import configurable
from meddlr.modeling.meta_arch.build import META_ARCH_REGISTRY, build_model
from meddlr.ops import complex as cplx
from meddlr.transforms.builtin.mri import MRIReconAugmentor
from meddlr.utils.events import get_event_storage
from meddlr.utils.general import move_to_device


@META_ARCH_REGISTRY.register()
class VortexModel(nn.Module):
    """VORTEX model.

    This is the generalized model implementation for augmentation-based consistency.
    It differs from :class:`N2RModel` and :class:`M2RModel` in some ways:

        1. **Generalizable augmentor**: :class:`MRIReconAugmentor` is used to
           perform augmentations.
        2. **Faster augmentations:** Augmentations are performed on the
           operating device (e.g. GPU) with large, but reproducible seeds.
        3. **Spatial augmentations**: Consistency with spatial augmentations
           are also supported. These augmentation are also used to transform
           the target image.

    Reference:
        A Desai, B Gunel, B Ozturkler, et al.
        VORTEX: Physics-Driven Data Augmentations Using Consistency Training
        for Robust Accelerated MRI Reconstruction.
        https://arxiv.org/abs/2111.02549.
    """

    _version = 1
    _aliases = ["A2RModel"]

    @configurable
    def __init__(
        self,
        model: nn.Module,
        augmentor: MRIReconAugmentor,
        use_supervised_consistency: bool = False,
        vis_period: int = -1,
    ):
        """
        Args:
            model (nn.Module): The base model.
            augmentor (MRIReconAugmentor): The augmentation module.
            use_supervised_consistency (bool, optional): If ``True``, use consistency
                with supervised examples too.
            vis_period (int, optional): The period over which to visualize images.
                If ``<=0``, it is ignored. Note if the ``model`` has a ``vis_period``
                attribute, it will be overridden so that this class handles visualization.
        """
        super().__init__()

        self.model = model
        self.augmentor = augmentor
        self.use_base_grad = False  # Keep gradient for base images in transform.
        self.use_supervised_consistency = use_supervised_consistency

        # Visualization done by this model
        if hasattr(model, "vis_period") and vis_period > 0:
            self.model.vis_period = -1
        self.vis_period = vis_period

    def augment(self, inputs, pred_base):
        inputs = move_to_device(inputs, device="cuda")
        pred_base = move_to_device(pred_base, device="cuda")
        kspace, maps = inputs["kspace"].clone(), inputs["maps"].clone()

        out, _, _ = self.augmentor(kspace, maps, pred_base, mask=True)

        inputs = {
            k: v.clone() if isinstance(v, torch.Tensor) else v
            for k, v in inputs.items()
            if k not in ("kspace", "maps")
        }
        inputs["kspace"] = out["kspace"]
        inputs["maps"] = out["maps"]
        aug_pred_base = out["target"]
        return inputs, aug_pred_base

    def visualize_aug_training(self, kspace, kspace_aug, preds, preds_base, target=None):
        """Visualize training of augmented data.

        Args:
            kspace: The base kspace.
            kspace_aug: The augmented kspace.
            preds: Reconstruction of augmented kspace. Shape: NxHxWx2.
            preds_base: Reconstruction of base kspace. Shape: NxHxWx2.
        """
        storage = get_event_storage()

        with torch.no_grad():
            # calc mask for first coil only
            if cplx.is_complex(kspace):
                kspace = torch.view_as_real(kspace)
            kspace = kspace.cpu()[0, ..., 0, :].unsqueeze(0)
            if cplx.is_complex(kspace_aug):
                kspace_aug = torch.view_as_real(kspace_aug)
            kspace_aug = kspace_aug.cpu()[0, ..., 0, :].unsqueeze(0)
            preds = preds.cpu()[0, ...].unsqueeze(0)
            preds_base = preds_base.cpu()[0, ...].unsqueeze(0)

            all_images = [preds, preds_base]
            errors = [cplx.abs(preds_base - preds)]
            if target is not None:
                target = target.cpu()[0, ...].unsqueeze(0)
                all_images.append(target)
                errors.append(cplx.abs(target - preds))

            all_images = torch.cat(all_images, dim=2)
            all_kspace = torch.cat([kspace, kspace_aug], dim=2)
            errors = torch.cat(errors, dim=2)

            imgs_to_write = {
                "phases": cplx.angle(all_images),
                "images": cplx.abs(all_images),
                "errors": errors,
                "masks": cplx.get_mask(kspace),
                "kspace": cplx.abs(all_kspace),
            }

            for name, data in imgs_to_write.items():
                data = data.squeeze(-1).unsqueeze(1)
                data = tv_utils.make_grid(data, nrow=1, padding=1, normalize=True, scale_each=True)
                storage.put_image("train_aug/{}".format(name), data.numpy(), data_format="CHW")

    def forward(self, inputs):
        if not self.training:
            assert (
                "unsupervised" not in inputs
            ), "unsupervised inputs should not be provided in eval mode"
            inputs = inputs.get("supervised", inputs)
            return self.model(inputs)

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
            output_dict["recon"] = self.model(
                inputs_supervised, return_pp=True, vis_training=vis_training
            )

        # Consistency.
        # kspace_aug = kspace + U \sigma \mathcal{N}
        # Loss = L(f(Ti(Te(kspace)), \theta), Te(f(kspace, \theta)))
        inputs_consistency = []
        if inputs_unsupervised is not None:
            inputs_consistency.append(inputs_unsupervised)
        if self.use_supervised_consistency and inputs_supervised is not None:
            inputs_consistency.append({k: v for k, v in inputs_supervised.items() if k != "target"})

        if len(inputs_consistency) > 0:
            if len(inputs_consistency) > 1:
                inputs_consistency = {
                    k: torch.cat([x[k] for x in inputs_consistency], dim=0)
                    for k in inputs_consistency[0].keys()
                }
            else:
                inputs_consistency = inputs_consistency[0]
            with torch.no_grad():
                pred_base = self.model(inputs_consistency)
                # Target only used for visualization purposes not for loss.
                target = inputs_unsupervised.get("target", None)
                pred_base = pred_base["pred"]
            inputs_consistency_aug, pred_base = self.augment(inputs_consistency, pred_base)
            pred_aug = self.model(inputs_consistency_aug, return_pp=True)
            if "target" in pred_aug:
                del pred_aug["target"]
            pred_aug["target"] = pred_base.detach()
            output_dict["consistency"] = pred_aug
            if vis_training:
                self.visualize_aug_training(
                    inputs_consistency["kspace"],
                    inputs_consistency_aug["kspace"],
                    pred_aug["pred"],
                    pred_base,
                    target=target,
                )

        return output_dict

    @classmethod
    def from_config(cls, cfg):
        _logger = logging.getLogger(__name__)

        model_cfg = cfg.clone()
        model_cfg.defrost()
        model_cfg.MODEL.META_ARCHITECTURE = cfg.MODEL.A2R.META_ARCHITECTURE
        model_cfg.freeze()
        model = build_model(model_cfg)

        augmentor = MRIReconAugmentor.from_cfg(
            cfg, aug_kind="consistency", device=cfg.MODEL.DEVICE, seed=cfg.SEED
        )
        _logger.info("Built augmentor:\n{}".format(str(augmentor.tfms_or_gens)))

        return {
            "model": model,
            "augmentor": augmentor,
            "use_supervised_consistency": cfg.MODEL.A2R.USE_SUPERVISED_CONSISTENCY,
            "vis_period": cfg.VIS_PERIOD,
        }
