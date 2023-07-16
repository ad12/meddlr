import torch
import torchvision.utils as tv_utils
from torch import nn

from meddlr.config.config import configurable
from meddlr.data.transforms.noiseandmotion import NoiseAndMotionModel
from meddlr.modeling.meta_arch.build import META_ARCH_REGISTRY, build_model
from meddlr.ops import complex as cplx
from meddlr.utils.deprecated import deprecated
from meddlr.utils.events import get_event_storage


@deprecated(vremove="0.1.0", replacement="modeling.meta_arch.VortexModel")
@META_ARCH_REGISTRY.register()
class NM2RModel(nn.Module):
    """(Deprecated) An extension of Noise2Recon to noise and motion artifacts.

    See :class:`Noise2Recon` for a detailed description of the method.

    Deprecated:
        This class is deprecated. Use :class:`VortexModel` instead.

    Attributes:
        model (nn.Module): The model performing reconstruction.
        augmentor (NoiseandMotionModel): The augmentation module.
        vis_period (int): How often to visualize the training.
        use_base_grad (bool): Whether to keep gradient for base
            reconstruction in consistency training. Defaults to
            `False`.
        use_supervised_consistency (bool): Whether to use
            consistency training for supervised examples.
    """

    _version = 2

    @configurable
    def __init__(
        self,
        model: nn.Module,
        augmentor: NoiseAndMotionModel,
        use_supervised_consistency: bool = False,
        vis_period: int = -1,
    ):
        super().__init__()
        self.model = model

        # Visualization done by this model
        if hasattr(self.model, "vis_period") and vis_period > 0:
            self.model.vis_period = -1
        self.vis_period = vis_period

        # Keep gradient for base images in transform.
        self.use_base_grad = False
        # Use supervised examples for consistency
        self.use_supervised_consistency = use_supervised_consistency
        self.augmentor = augmentor

    def augment(self, inputs):
        """Noise + motion augmentation."""
        kspace = inputs["kspace"].clone()
        aug_kspace = self.augmentor(kspace, clone=False)

        inputs = {k: v.clone() for k, v in inputs.items() if k != "kspace"}
        inputs["kspace"] = aug_kspace
        return inputs

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
            inputs_consistency_aug = self.augment(inputs_consistency)
            with torch.no_grad():
                pred_base = self.model(inputs_consistency)
                # Target only used for visualization purposes not for loss.
                target = inputs_unsupervised.get("target", None)
                pred_base = pred_base["pred"]
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

    def load_state_dict(self, state_dict, strict=True):
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
        model_cfg.MODEL.META_ARCHITECTURE = cfg.MODEL.NM2R.META_ARCHITECTURE
        model_cfg.freeze()
        model = build_model(model_cfg)

        augmentor = NoiseAndMotionModel.from_cfg(cfg)

        return {
            "model": model,
            "augmentor": augmentor,
            "use_supervised_consistency": cfg.MODEL.NM2R.USE_SUPERVISED_CONSISTENCY,
            "vis_period": cfg.VIS_PERIOD,
        }
