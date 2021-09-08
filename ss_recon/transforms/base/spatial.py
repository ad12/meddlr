from typing import Sequence, Tuple

import numpy as np
import torch
import torchvision.transforms.functional as TF

import ss_recon.utils.complex_utils as cplx
from ss_recon.transforms.build import TRANSFORM_REGISTRY
from ss_recon.transforms.mixins import GeometricMixin
from ss_recon.transforms.transform import Transform


@TRANSFORM_REGISTRY.register()
class AffineTransform(GeometricMixin, Transform):
    def __init__(
        self,
        angle: float = None,
        translate: Sequence[int] = None,
        scale=None,
        shear: Sequence[int] = None,
    ) -> None:
        super().__init__()
        if angle is None:
            angle = 0.0
        if scale is None:
            scale = 1.0
        if translate is None:
            translate = [0, 0]
        if shear is None:
            shear = [0, 0]
        self._set_attributes(locals())

    def _apply_affine(self, x):
        img = x

        is_complex = cplx.is_complex(img)
        permute = is_complex or cplx.is_complex_as_real(img)
        if is_complex:
            img = torch.view_as_real(img)
        if permute:
            img = img.permute((img.ndim - 1,) + tuple(range(0, img.ndim - 1)))

        shape = img.shape
        use_view = img.ndim > 4
        if use_view:
            is_contiguous = img.is_contiguous()
            if is_contiguous:
                img = img.view((np.product(shape[:-3]),) + shape[-3:])
            else:
                img = img.reshape((np.product(shape[:-3]),) + shape[-3:])

        img = TF.affine(
            img,
            angle=self.angle,
            translate=self.translate,
            scale=self.scale,
            shear=self.shear,
        )

        if use_view:
            img = img.view(shape)

        if permute:
            img = img.permute(tuple(range(1, img.ndim)) + (0,))
        if is_complex:
            img = torch.view_as_complex(img.contiguous())
        return img

    def apply_image(self, img: torch.Tensor):
        return self._apply_affine(img)

    def apply_maps(self, maps: torch.Tensor):
        maps = self._apply_affine(maps)  # BxCxMxHxW
        norm = cplx.rss(maps, dim=1).unsqueeze(1)
        norm += 1e-8 * (norm == 0)
        maps = maps / norm
        return maps

    def _eq_attrs(self) -> Tuple[str]:
        return ("angle", "translate", "scale", "shear")


@TRANSFORM_REGISTRY.register()
class FlipTransform(GeometricMixin, Transform):
    def __init__(self, dims):
        super().__init__()
        if isinstance(dims, int):
            dims = (dims,)
        self.dims = dims

    def apply_image(self, img: torch.Tensor):
        if cplx.is_complex_as_real(img):
            img = torch.view_as_complex(img)
        return torch.flip(img, self.dims)

    def apply_kspace(self, kspace):
        return self.apply_image(kspace)

    def inverse(self):
        return FlipTransform(self.dims)

    def _eq_attrs(self) -> Tuple[str]:
        return ("dims",)


@TRANSFORM_REGISTRY.register()
class Rot90Transform(GeometricMixin, Transform):
    def __init__(self, k, dims) -> None:
        super().__init__()
        self.k = k
        self.dims = dims

    def apply_image(self, img: torch.Tensor):
        return torch.rot90(img, self.k, self.dims)

    def apply_kspace(self, kspace):
        return self.apply_image(kspace)

    def inverse(self):
        return Rot90Transform(self.k, self.dims[::-1])

    def _eq_attrs(self) -> Tuple[str]:
        return (
            "k",
            "dims",
        )
