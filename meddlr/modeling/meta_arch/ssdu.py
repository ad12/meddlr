import warnings

import torch
import torchvision.utils as tv_utils
from torch import nn

import meddlr.ops as oF
from meddlr.config.config import configurable
from meddlr.forward.mri import SenseModel
from meddlr.modeling.meta_arch.build import META_ARCH_REGISTRY, build_model
from meddlr.ops import complex as cplx
from meddlr.transforms.base.mask import KspaceMaskTransform
from meddlr.transforms.gen.mask import RandomKspaceMask
from meddlr.utils.events import get_event_storage
from meddlr.utils.general import move_to_device


@META_ARCH_REGISTRY.register()
class SSDUModel(nn.Module):
    """Self-supervised learning via data undersampling.

    This model is the relaxed form of the SSDU model that can be
    used to train with both supervised and unsupervised data.

    The mask used to acquire the data (:math:`\Omega`) is partitioned
    into train mask for the zero-filled image (:math:`\Theta`) and a
    mask for the loss (:math:`\Lambda`).

    Reference:
        B Yaman, SAH Hosseini, S Moeller, et al. Self-supervised
        learning of physics-guided reconstruction neural networks
        without fully sampled reference data.
        https://onlinelibrary-wiley-com.stanford.idm.oclc.org/doi/full/10.1002/mrm.28378
    """

    _version = 1

    @configurable
    def __init__(self, model: nn.Module, masker: RandomKspaceMask, vis_period: int = None):
        """
        Args:
            model (nn.Module): The base model.
            masker (NoiseModel): The additive noise module.
        """
        super().__init__()
        self.model = model
        self.masker = masker
        # Visualization done by this model
        if hasattr(self.model, "vis_period"):
            if vis_period is not None:
                self.model.vis_period = vis_period
            else:
                vis_period = self.model.vis_period
            self.model.vis_period = -1
        self.vis_period = vis_period

    def augment(self, inputs):
        """Noise augmentation module for the consistency branch.

        Args:
            inputs (Dict[str, Any]): The input dictionary.
                It must contain a key ``'kspace'``, which traditionally
                corresponds to the undersampled kspace when performing
                augmentation for consistency. For supervised examples,
                this can correspond to either the retrospectively
                undersampled k-space or the fully-sampled kspace.

        Returns:
            Dict[str, Any]: The input dictionary with the kspace polluted
                with additive masked complex Gaussian noise.
        """
        masker = self.masker
        kspace = inputs["kspace"].clone()
        mask = cplx.get_mask(kspace)
        edge_mask = inputs["edge_mask"]

        tfm: KspaceMaskTransform = masker.get_transform(kspace)
        train_mask = tfm.generate_mask(kspace, channels_last=True)
        loss_mask = mask - train_mask

        # Pad the train mask so that all unacquired kspace points
        # are included in the train_mask.
        train_mask = (train_mask.type(torch.bool) | edge_mask.type(torch.bool)).type(torch.float32)

        # TODO (arjundd): See if we can remove this check for speed reasons.
        assert torch.all(loss_mask >= 0)

        inputs = {k: v.clone() for k, v in inputs.items() if k != "kspace"}
        inputs["kspace"] = train_mask * kspace
        inputs["mask"] = train_mask
        return inputs, mask[..., 0:1], train_mask, loss_mask[..., 0:1]

    @torch.no_grad()
    def visualize(self, images_dict):
        for name, images in images_dict.items():
            storage = get_event_storage()
            if isinstance(images, (tuple, list)):
                images = torch.stack(images, dim=0)
            if cplx.is_complex_as_real(images) or cplx.is_complex(images):
                images = {
                    f"{name}-phase": cplx.angle(images),
                    f"{name}-mag": cplx.abs(images),
                }
            else:
                images = {name: images}

            for name, data in images.items():
                if data.shape[-1] == 1:
                    data = data.squeeze(-1)
                data = data.unsqueeze(1)
                data = tv_utils.make_grid(
                    data, nrow=len(data), padding=1, normalize=True, scale_each=True
                )
                storage.put_image("ssdu/{}".format(name), data.cpu().numpy(), data_format="CHW")

    def forward(self, inputs):
        if not self.training:
            assert (
                "unsupervised" not in inputs
            ), "unsupervised inputs should not be provided in eval mode"
            inputs = inputs.get("supervised", inputs)
            mask = cplx.get_mask(inputs["kspace"])
            # The mask should be the union of the edge mask and the sampled data mask.
            # https://github.com/byaman14/SSDU
            # If the edge mask is not passed in, we assume that we do not want to get
            # the edge mask.
            if "edge_mask" not in inputs:
                edge_mask = torch.tensor(0, device=mask.device, dtype=mask.dtype)
                warnings.warn("Edge mask not found in `inputs`. Assuming no edge mask.")
            else:
                edge_mask = inputs["edge_mask"]
            dc_mask = (mask + edge_mask).bool().to(mask.dtype)
            inputs["mask"] = dc_mask
            # inputs["postprocessing_mask"] = dc_mask - mask
            return self.model(inputs)

        storage = get_event_storage()
        vis_training = self.training and self.vis_period > 0 and storage.iter % self.vis_period == 0

        # Put supervised and unsupervised scans in a single tensor.
        sup = inputs.get("supervised", {})
        unsup = inputs.get("unsupervised", {})
        if sup or unsup:
            inputs = {
                k: torch.cat([sup.get(k, torch.tensor([])), unsup.get(k, torch.tensor([]))])
                for k in sup.keys() | unsup.keys()
            }
        assert all(k in inputs for k in ["kspace"])

        device = next(self.model.parameters()).device
        inputs = move_to_device(inputs, device=device, non_blocking=True)

        kspace = inputs["kspace"]
        inputs_aug, orig_mask, train_mask, loss_mask = self.augment(inputs)
        outputs = self.model(inputs_aug, vis_training=vis_training and len(sup) > 0)

        # Get the signal model reconstructed images.
        # TODO: Make it possible to use these are the target instead of multi-coil images.
        pred_img = outputs["pred"]
        target_img, zf_image = outputs.get("target", None), outputs.get("zf_image", None)

        # Use signal model (SENSE) to get weighted kspace.
        A = SenseModel(maps=inputs_aug["maps"])  # no weights - we do not want to mask the data.
        loss_pred_kspace = loss_mask * A(outputs["pred"], adjoint=False)
        loss_kspace = loss_mask * kspace

        # A hacky way to prepare the predictions and target for the loss.
        # This may result in inaccurate training metrics outside of the loss.
        # TODO (arjundd): Fix this.
        # Shape: B x H x W x #coils
        outputs["pred"] = oF.ifft2c(loss_pred_kspace, channels_last=True)
        outputs["target"] = oF.ifft2c(loss_kspace, channels_last=True)

        # Visualize.
        if self.training and self.vis_period > 0:
            with torch.no_grad():
                storage = get_event_storage()
                if storage.iter % self.vis_period == 0:
                    A = SenseModel(maps=inputs["maps"][0:1], weights=train_mask[0:1])
                    base_image = A(kspace[0:1], adjoint=True)
                    self.visualize(
                        {
                            "masks": [orig_mask[0], train_mask[0], loss_mask[0]],
                            "kspace": [
                                kspace[0, ..., 0:1],
                                inputs_aug["kspace"][0, ..., 0:1],
                                loss_pred_kspace[0, ..., 0:1],
                                loss_kspace[0, ..., 0:1],
                            ],
                            "images": [
                                x[0]
                                for x in [base_image, zf_image, pred_img, target_img]
                                if x is not None
                            ],
                        }
                    )

        return outputs

    @classmethod
    def from_config(cls, cfg):
        model_cfg = cfg.clone()
        model_cfg.defrost()
        model_cfg.MODEL.META_ARCHITECTURE = cfg.MODEL.SSDU.META_ARCHITECTURE
        model_cfg.freeze()
        model = build_model(model_cfg)

        # TODO: Configure this
        params = cfg.MODEL.SSDU.MASKER.PARAMS
        masker = RandomKspaceMask(**params)
        masker.to(cfg.MODEL.DEVICE)

        return {"model": model, "masker": masker}
