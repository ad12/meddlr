from typing import Sequence, Union

import ss_recon.utils.complex_utils as cplx
import ss_recon.utils.transforms as T
from ss_recon.data.transforms.transform import Normalizer
from ss_recon.transforms.build import build_transforms, seed_tfm_gens
from ss_recon.transforms.mixins import GeometricMixin
from ss_recon.transforms.transform import NoOpTransform, Transform, TransformList
from ss_recon.transforms.transform_gen import RandomTransformChoice, TransformGen
from ss_recon.utils import env


class MRIReconAugmentor:
    """
    The class that manages the organization, generation, and application
    of deterministic and random transforms for MRI reconstruction.
    """

    def __init__(
        self, tfms_or_gens: Sequence[Union[Transform, TransformGen]], seed: int = None
    ) -> None:
        if isinstance(tfms_or_gens, TransformList):
            tfms_or_gens = tfms_or_gens.transforms
        if seed is None:
            seed_tfm_gens(tfms_or_gens, seed=seed)
        self.tfms_or_gens = tfms_or_gens

    def __call__(
        self,
        kspace,
        maps=None,
        target=None,
        normalizer: Normalizer = None,
        mask=None,
        mask_gen=None,
    ):
        # For now, we assume that transforms generated by RandomTransformChoice
        # return random transforms of the same type (equivariant or invariant).
        # We don't have to filter these transforms as a result.
        transform_gens = [
            x.get_transform() if isinstance(x, RandomTransformChoice) else x
            for x in self.tfms_or_gens
        ]

        tfms_equivariant, tfms_invariant = self._classify_transforms(transform_gens)

        # Apply equivariant transforms to the SENSE reconstructed image.
        # Note, RSS reconstruction is not currently supported.
        if mask is True:
            mask = cplx.get_mask(kspace)
        A = T.SenseModel(maps, weights=mask)
        img = A(kspace, adjoint=True)

        img, target, maps = self._permute_data(img, target, maps, spatial_last=True)
        img, target, maps, tfms_equivariant = self._apply_te(tfms_equivariant, img, target, maps)
        img, target, maps = self._permute_data(img, target, maps, spatial_last=False)

        if len(tfms_equivariant) > 0:
            A = T.SenseModel(maps)
            kspace = A(img)

        if mask_gen is not None:
            kspace, mask = mask_gen(kspace)
            img = T.SenseModel(maps, weights=mask)(kspace, adjoint=True)

        if normalizer:
            normalized = normalizer.normalize(
                **{
                    "masked_kspace": kspace,
                    "image": img,
                    "target": target,
                    "mask": mask,
                }
            )
            kspace = normalized["masked_kspace"]
            target = normalized["target"]
            mean = normalized["mean"]
            std = normalized["std"]
        else:
            mean, std = None, None

        # Apply invariant transforms.
        kspace = self._permute_data(kspace, spatial_last=True)
        kspace, tfms_invariant = self._apply_ti(tfms_invariant, kspace)
        kspace = self._permute_data(kspace, spatial_last=False)

        out = {"kspace": kspace, "maps": maps, "target": target, "mean": mean, "std": std}
        return out, tfms_equivariant, tfms_invariant

    def _classify_transforms(self, transform_gens):
        tfms_equivariant = []
        tfms_invariant = []
        for tfm in transform_gens:
            if isinstance(tfm, TransformGen):
                tfm_kind = tfm._base_transform
            else:
                tfm_kind = type(tfm)
            assert issubclass(tfm_kind, Transform)

            if issubclass(tfm_kind, GeometricMixin):
                tfms_equivariant.append(tfm)
            else:
                tfms_invariant.append(tfm)
        return tfms_equivariant, tfms_invariant

    def _permute_data(self, *args, spatial_last: bool = False):
        out = []
        if spatial_last:
            for x in args:
                dims = (0,) + tuple(range(3, x.ndim)) + (1, 2)
                out.append(x.permute(dims))
        else:
            for x in args:
                dims = (0, x.ndim - 2, x.ndim - 1) + tuple(range(1, x.ndim - 2))
                out.append(x.permute(dims))
        return out[0] if len(out) == 1 else tuple(out)

    def _apply_te(self, tfms_equivariant, image, target, maps):
        """Apply equivariant transforms.

        These transforms affect both the input and the target.
        """
        tfms = []
        for g in tfms_equivariant:
            tfm: Transform = g.get_transform(image) if isinstance(g, TransformGen) else g
            if isinstance(tfm, NoOpTransform):
                continue
            image = tfm.apply_image(image)
            if target is not None:
                target = tfm.apply_image(target)
            if maps is not None:
                maps = tfm.apply_maps(maps)
            tfms.append(tfm)
        return image, target, maps, TransformList(tfms, ignore_no_op=True)

    def _apply_ti(self, tfms_invariant, kspace):
        """Apply invariant transforms.

        These transforms affect only the input, not the target.
        """
        tfms = []
        for g in tfms_invariant:
            tfm: Transform = g.get_transform(kspace) if isinstance(g, TransformGen) else g
            if isinstance(tfm, NoOpTransform):
                continue
            kspace = tfm.apply_kspace(kspace)
            tfms.append(tfm)
        return kspace, TransformList(tfms, ignore_no_op=True)

    def reset(self):
        for g in self.tfms_or_gens:
            if isinstance(g, TransformGen):
                g.reset()

    @classmethod
    def from_cfg(cls, cfg, aug_kind, seed=None, **kwargs):
        assert aug_kind in ("aug_train", "consistency")
        if aug_kind == "aug_train":
            vals = cfg.AUG_TRAIN.MRI_RECON
        elif aug_kind == "consistency":
            vals = cfg.MODEL.CONSISTENCY.AUG.MRI_RECON
        if seed is None and env.is_repro():
            seed = cfg.SEED
        tfms_or_gens = build_transforms(cfg, vals.TRANSFORMS, seed=seed, **kwargs)
        return cls(tfms_or_gens)
