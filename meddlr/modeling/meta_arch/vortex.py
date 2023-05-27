import logging
from typing import Dict, Optional

import torch
import torchvision.utils as tv_utils
from torch import nn

import meddlr.ops as oF
from meddlr.config.config import configurable
from meddlr.forward.mri import SenseModel
from meddlr.modeling.meta_arch import SSDUModel
from meddlr.modeling.meta_arch.build import META_ARCH_REGISTRY, build_model
from meddlr.ops import complex as cplx
from meddlr.transforms.builtin.mri import MRIReconAugmentor
from meddlr.utils.events import get_event_storage
from meddlr.utils.general import flatten_dict, move_to_device


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
        edge_dc: bool = False,
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
        self.edge_dc = edge_dc
        self._multicoil_image = True

        # Visualization done by this model
        if (
            not isinstance(self.model, SSDUModel)
            and hasattr(model, "vis_period")
            and vis_period > 0
        ):
            self.model.vis_period = -1
        self.vis_period = vis_period

    def augment(self, inputs: Dict[str, torch.Tensor], pred_base: torch.Tensor):
        """Apply augmentations to inputs and base image reconstruction.

        Args:
            inputs (Dict[str, torch.Tensor]): The inputs to augment.
            pred_base (torch.Tensor): The reconstruction of the base (i.e. non-augmented) image.

        Returns:
            Tuple[Dict[str, torch.Tensor], torch.Tensor]: The augmented inputs and pseudo-targets.
        """
        inputs = move_to_device(inputs, device="cuda")
        pred_base = move_to_device(pred_base, device="cuda")
        kspace, maps = inputs["kspace"].clone(), inputs["maps"].clone()

        out, _, _ = self.augmentor(kspace=kspace, maps=maps, target=pred_base, mask=True)

        inputs = {
            k: v.clone() if isinstance(v, torch.Tensor) else v
            for k, v in inputs.items()
            if k not in ("kspace", "maps")
        }
        inputs["kspace"] = out["kspace"]
        inputs["maps"] = out["maps"]
        aug_pred_base = out["target"]
        return inputs, aug_pred_base

    def log_augmentor_params(self):
        scheduler_params = self.augmentor.get_tfm_gen_params(scalars_only=True)
        if len(scheduler_params):
            storage = get_event_storage()
            scheduler_params = flatten_dict({"scheduler": scheduler_params})
            storage.put_scalars(**scheduler_params)

    def visualize_aug_training(
        self, kspace, kspace_aug, preds, preds_base, target=None, dc_mask=None
    ):
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
            if dc_mask is not None:
                # Take the mask for the first coil, it should be the same for all coils.
                imgs_to_write["dc_mask"] = dc_mask[0:1, ..., 0].cpu()

            for name, data in imgs_to_write.items():
                data = data.squeeze(-1).unsqueeze(1)
                data = tv_utils.make_grid(data, nrow=1, padding=1, normalize=True, scale_each=True)
                storage.put_image("train_aug/{}".format(name), data.numpy(), data_format="CHW")

    def _aggregate_consistency_inputs(
        self,
        inputs_supervised: Optional[Dict[str, torch.Tensor]] = None,
        inputs_unsupervised: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Dict[str, torch.Tensor]:
        """Aggregates consistency inputs into a single dictionary.

        Args:
            inputs_supervised: A dict of inputs, their metadata, and their ground truth references.
            inputs_unsupervised: A dict of inputs and their metadata.

        Returns:
            Dict[str, Dict[str, Tensor]]: A dictionary of base inputs and augmented inputs:
                - 'base': Inputs to be used to generate the pseudo-label (i.e. target)
                  for consistency optimization.
                - 'aug': Augmented inputs to use for consistency training.
        """
        inputs_consistency = []
        if inputs_unsupervised is not None:
            inputs_consistency.append(inputs_unsupervised)
        if self.use_supervised_consistency and inputs_supervised is not None:
            inputs_consistency.append({k: v for k, v in inputs_supervised.items() if k != "target"})

        if len(inputs_consistency) == 0:
            return {}  # No consistency training.

        if len(inputs_consistency) > 1:
            inputs_consistency = {
                k: torch.cat([x[k] for x in inputs_consistency], dim=0)
                for k in inputs_consistency[0].keys()
            }
        else:
            inputs_consistency = inputs_consistency[0]

        return inputs_consistency

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
        # Whether to use self-supervised via data undersampling (SSDU) for reconstruction.
        is_ssdu_enabled = isinstance(self.model, SSDUModel)
        output_dict = {}

        # Reconstruction (supervised).
        if is_ssdu_enabled:
            output_dict["recon"] = self.model(inputs)
        elif inputs_supervised is not None:
            output_dict["recon"] = self.model(
                inputs_supervised, return_pp=True, vis_training=vis_training
            )

        # Consistency.
        model = self.model.model if is_ssdu_enabled else self.model
        inputs_consistency = self._aggregate_consistency_inputs(
            inputs_supervised, inputs_unsupervised
        )
        inputs_consistency = move_to_device(inputs_consistency, device="cuda")
        if len(inputs_consistency) > 0:
            # Add the edge mask to the DC layers of the unrolled network.
            # TODO: Make this configurable.
            if self.edge_dc and "mask" not in inputs_consistency:
                inputs_consistency["mask"] = (
                    (cplx.get_mask(inputs_consistency["kspace"]) + inputs_consistency["edge_mask"])
                    .bool()
                    .to(torch.float32)
                )

            with torch.no_grad():
                pred_base = model(inputs_consistency)
                # Target only used for visualization purposes not for loss.
                target = inputs_unsupervised.get("target", None)
                pred_base_zf = pred_base["zf_image"]  # noqa: F841
                pred_base = pred_base["pred"]
                # Augment the inputs.
                inputs_consistency_aug, pred_base = self.augment(inputs_consistency, pred_base)

            pred_aug = model(inputs_consistency_aug, return_pp=True)

            if "target" in pred_aug:
                del pred_aug["target"]
            pred_aug["target"] = pred_base.detach()
            output_dict["consistency"] = pred_aug

            # Add DC loss between prediction and consistency inputs.
            # A = SenseModel(maps=inputs_consistency["maps"], weights=cplx.get_mask(inputs_consistency["kspace"]))  # noqa: E501
            # output_dict["dc"] = {
            #     "pred": A(pred_aug["pred"]),
            #     "target": inputs_consistency["kspace"],
            # }

            if vis_training:
                self.visualize_aug_training(
                    inputs_consistency["kspace"],
                    inputs_consistency_aug["kspace"],
                    pred_aug["pred"],
                    pred_base,
                    target=target,
                    dc_mask=inputs_consistency.get("mask", None),
                )

            # Convert to multicoil image.
            # This is needed for loss to be computed over each coil rather than
            # a coil combined k-space/image.
            # We do this after visualization to avoid visualizing multicoil images.
            if self._multicoil_image:
                with torch.no_grad():
                    pred_aug["target"] = _to_multicoil_image(
                        x=pred_aug["target"], maps=inputs_consistency["maps"]
                    )
                pred_aug["pred"] = _to_multicoil_image(
                    x=pred_aug["pred"], maps=inputs_consistency_aug["maps"]
                )

        # Log augmentor parameters.
        self.log_augmentor_params()

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
            "edge_dc": cfg.MODEL.A2R.EDGE_DC,
            "vis_period": cfg.VIS_PERIOD,
        }


def _to_multicoil_image(x, maps):
    """Convert image x into multicoil image for loss computer compatibility."""
    # Use signal model (SENSE) to get weighted kspace.
    A = SenseModel(maps=maps)  # no weights - we do not want to mask the data.
    return oF.ifft2c(A(x, adjoint=False), channels_last=True)
