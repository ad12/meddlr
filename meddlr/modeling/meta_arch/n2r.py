from typing import Dict, Optional

import torch
import torchvision.utils as tv_utils
from torch import nn

from meddlr.config.config import configurable
from meddlr.data.transforms.noise import NoiseModel
from meddlr.modeling.meta_arch.build import META_ARCH_REGISTRY, build_model
from meddlr.modeling.meta_arch.ssdu import SSDUModel
from meddlr.ops import complex as cplx
from meddlr.utils.events import get_event_storage
from meddlr.utils.general import nested_apply


@META_ARCH_REGISTRY.register()
class N2RModel(nn.Module):
    """Noise2Recon model.

    Reference:
        AD Desai, BM Ozturkler, CM Sandino, et al. Noise2Recon: A Semi-Supervised Framework
        for Joint MRI Reconstruction and Denoising. ArXiv 2021.
        https://arxiv.org/abs/2110.00075
    """

    _version = 3

    @configurable
    def __init__(
        self,
        model: nn.Module,
        noiser: NoiseModel,
        use_supervised_consistency: bool = False,
        vis_period: int = -1,
    ):
        """
        Args:
            model: The base model.
            noiser: The additive noise module.
            use_supervised_consistency: Whether to apply noise-based consistency
                to supervised examples.
            vis_period (int, optional): The period over which to visualize images.
                If ``<=0``, it is ignored. Note if the ``model`` has a ``vis_period``
                attribute, it will be overridden so that this class handles visualization.
        """
        super().__init__()
        self.model = model

        # Visualization done by this model.
        # If sub-model is SSDU, we allow SSDU to log images
        # for the recon pipeline.
        if (
            not isinstance(self.model, SSDUModel)
            and hasattr(self.model, "vis_period")
            and vis_period > 0
        ):
            self.model.vis_period = -1
        self.vis_period = vis_period

        # Whether to keep gradient for base images in transform.
        self.use_base_grad = False
        # Use supervised examples for consistency
        self.use_supervised_consistency = use_supervised_consistency
        self.noiser = noiser

    def augment(self, inputs):
        """Noise augmentation module for the consistency branch.

        Args:
            inputs (Dict[str, Any]): The input dictionary.
                It must contain a key ``'kspace'``, which traditionally
                corresponds to the undersampled kspace when performing
                augmentation for consistency.

        Returns:
            Dict[str, Any]: The input dictionary with the kspace polluted
                with additive masked complex Gaussian noise.
        """
        kspace = inputs["kspace"].clone()
        aug_kspace = self.noiser(kspace, clone=False)

        inputs = {
            k: nested_apply(v, lambda _v: _v.clone()) for k, v in inputs.items() if k != "kspace"
        }
        inputs["kspace"] = aug_kspace
        return inputs

    @torch.no_grad()
    def visualize_aug_training(self, kspace, kspace_aug, preds, preds_base, target=None):
        """Visualize training of augmented data.

        Args:
            kspace: The base kspace.
            kspace_aug: The augmented kspace.
            preds: Reconstruction of augmented kspace. Shape: NxHxWx2.
            preds_base: Reconstruction of base kspace. Shape: NxHxWx2.
        """
        storage = get_event_storage()

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

    def _format_consistency_inputs(
        self,
        inputs_supervised: Optional[Dict[str, torch.Tensor]] = None,
        inputs_unsupervised: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Dict[str, torch.Tensor]:
        """Generate base and augmented inputs to be used for consistency training.

        Args:
            inputs_supervised: A dict of inputs, their metadata, and their ground truth references.
            inputs_unsupervised: A dict of inputs and their metadata.

        Returns:
            Dict[str, Dict[str, Tensor]]: A dictionary of base inputs and augmented inputs:
                - 'base': Inputs to be used to generate the pseudo-label (i.e. target)
                  for consistency optimization.
                - 'aug': Noise augmented inputs to use for consistency training.
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

        # Augment the inputs.
        inputs_consistency_aug = self.augment(inputs_consistency)

        return {"base": inputs_consistency, "aug": inputs_consistency_aug}

    def forward(self, inputs):
        if not self.training:
            assert (
                "unsupervised" not in inputs
            ), "unsupervised inputs should not be provided in eval mode"
            inputs = inputs.get("supervised", inputs)
            return self.model(inputs)

        storage = get_event_storage()
        vis_training = self.training and self.vis_period > 0 and storage.iter % self.vis_period == 0

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

        # Consistency (unsupervised).
        # kspace_aug = kspace + U \sigma \mathcal{N}
        # Loss = L(f(kspace_aug, \theta), f(kspace, \theta))
        # If the model is an SSDU model, unpack it to use the internal model for consistency.
        model = self.model.model if is_ssdu_enabled else self.model
        consistency_inputs = self._format_consistency_inputs(inputs_supervised, inputs_unsupervised)
        if len(consistency_inputs) > 0:
            inputs_consistency = consistency_inputs["base"]
            inputs_consistency_aug = consistency_inputs["aug"]

            with torch.no_grad():
                pred_base = model(inputs_consistency)
                # Target only used for visualization purposes not for loss.
                target = inputs_consistency.get("target", None)
                pred_base = pred_base["pred"]

            pred_aug: Dict[str, torch.Tensor] = model(inputs_consistency_aug, return_pp=True)

            pred_aug.pop("target", None)
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

    def load_state_dict(self, state_dict, strict=True):  # pragma: no cover
        # TODO: Configure backwards compatibility
        if any(x.startswith("unrolled") for x in state_dict.keys()):
            raise ValueError(
                "`self.unrolled` was renamed to `self.model`. "
                "Backwards compatibility has not been configured."
            )
        return super().load_state_dict(state_dict, strict)

    @classmethod
    def from_config(cls, cfg):
        model_cfg = cfg.clone()
        model_cfg.defrost()
        model_cfg.MODEL.META_ARCHITECTURE = cfg.MODEL.N2R.META_ARCHITECTURE
        model_cfg.freeze()
        model = build_model(model_cfg)

        noiser = NoiseModel.from_cfg(cfg)

        return {
            "model": model,
            "noiser": noiser,
            "use_supervised_consistency": cfg.MODEL.N2R.USE_SUPERVISED_CONSISTENCY,
            "vis_period": cfg.VIS_PERIOD,
        }
