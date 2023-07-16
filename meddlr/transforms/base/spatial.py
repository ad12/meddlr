import logging
from typing import Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from PIL import Image

import meddlr.ops.complex as cplx
from meddlr.transforms.build import TRANSFORM_REGISTRY
from meddlr.transforms.mixins import GeometricMixin
from meddlr.transforms.transform import Transform


@TRANSFORM_REGISTRY.register()
class AffineTransform(GeometricMixin, Transform):
    """A deterministic transform that applies affine transformations."""

    def __init__(
        self,
        angle: float = None,
        translate: Sequence[int] = None,
        scale: float = None,
        shear: Sequence[int] = None,
        pad_like: str = None,
        upsample_factor: float = 1,
        upsample_order: int = 1,
    ) -> None:
        """
        Args:
            angle: Rotation angle in degrees.
            translate: Translation vector.
        """
        super().__init__()
        logger = logging.getLogger(f"{__name__}.{type(self).__name__}")

        if angle is None:
            angle = 0.0
        if scale is None:
            scale = 1.0
        if translate is None:
            translate = [0, 0]
        if shear is None:
            shear = [0, 0]

        if pad_like not in (None, "MRAugment"):
            raise ValueError("`pad_like` must be one of (None, 'MRAugment')")
        if pad_like == "MRAugment" and translate not in ([0, 0], None):
            logger.warning("MRAugment padding may not appropriately account for translation")
        self._set_attributes(locals())

    def _apply_affine(self, x):
        img = x
        angle = self.angle
        translate = self.translate[::-1]
        scale = self.scale
        shear = self.shear[::-1]
        upsample_factor = self.upsample_factor
        upsample_order = self.upsample_order

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

        base_shape = img.shape[-2:]
        upsample = upsample_factor != 1
        interpolation = Image.BICUBIC if upsample_order == 3 else Image.BILINEAR
        if upsample:
            upsampled_shape = (
                img.shape[-2] * self.upsample_factor,
                img.shape[-1] * self.upsample_factor,
            )
            img = TF.resize(img, size=upsampled_shape, interpolation=interpolation)

        h, w = img.shape[-2:]
        if self.pad_like == "MRAugment":
            pad = _get_mraugment_affine_pad(shape[-2:], angle, translate, scale, shear)
            img = TF.pad(img, padding=pad, padding_mode="reflect")

        img = TF.affine(
            img,
            angle=angle,
            translate=translate,
            scale=scale,
            shear=shear,
            interpolation=TF.InterpolationMode.BILINEAR,
        )

        if self.pad_like == "MRAugment":
            img = TF.center_crop(img, (h, w))

        if upsample:
            img = TF.resize(img, size=base_shape, interpolation=interpolation)

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
class TranslationTransform(GeometricMixin, Transform):
    def __init__(self, translate: Sequence[int], pad_mode="constant", pad_value=0) -> None:
        super().__init__()
        self.translate = translate
        self.pad_mode = pad_mode
        self.pad_value = pad_value

    def apply_image(self, img: torch.Tensor):
        translation = self.translate
        max_dims = len(translation) + 2

        is_complex = cplx.is_complex(img)
        permute = is_complex or cplx.is_complex_as_real(img)
        if is_complex:
            img = torch.view_as_real(img)
        if permute:
            img = img.permute((img.ndim - 1,) + tuple(range(0, img.ndim - 1)))

        shape = img.shape
        use_view = img.ndim > max_dims
        if use_view:
            is_contiguous = img.is_contiguous()
            if is_contiguous:
                img = img.view((np.product(shape[: -(max_dims - 1)]),) + shape[-(max_dims - 1) :])
            else:
                img = img.reshape(
                    (np.product(shape[: -(max_dims - 1)]),) + shape[-(max_dims - 1) :]
                )

        pad, sl = _get_mraugment_translate_pad(img.shape, translation)
        img = F.pad(img, pad, mode=self.pad_mode, value=self.pad_value)
        img = img[sl]

        if use_view:
            img = img.view(shape)

        if permute:
            img = img.permute(tuple(range(1, img.ndim)) + (0,))
        if is_complex:
            img = torch.view_as_complex(img.contiguous())
        return img

    def apply_maps(self, maps: torch.Tensor):
        maps = self.apply_image(maps)  # BxCxMxHxW
        norm = cplx.rss(maps, dim=1).unsqueeze(1)
        norm += 1e-8 * (norm == 0)
        maps = maps / norm
        return maps


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
        return ("k", "dims")


def _get_mraugment_affine_pad(im_shape, angle, translate, scale, shear):
    """Calculate the padding size based on MRAugment padding method.

    This padding should be applied before the affine transformation.

    Args:
        im_shape (tuple): Shape as ``(height, width)``.
        angle (float): The rotating angle.
        scale (float): The scale factor.
        shear (tuple): Shear factors (H x W) (i.e. YxX).

    Note:
        This method is adapted from MRAugment.
        https://github.com/MathFLDS/MRAugment/blob/master/mraugment/data_augment.py
    """
    h, w = im_shape
    shear = shear[::-1]
    translate = translate[::-1]

    corners = [
        [-h / 2, -w / 2, 1.0],
        [-h / 2, w / 2, 1.0],
        [h / 2, w / 2, 1.0],
        [h / 2, -w / 2, 1.0],
    ]
    mx = torch.tensor(
        TF._get_inverse_affine_matrix([0.0, 0.0], -angle, translate, scale, [-s for s in shear])
    ).reshape(2, 3)
    corners = torch.cat([torch.tensor(c).reshape(3, 1) for c in corners], dim=1)
    tr_corners = torch.matmul(mx, corners)
    all_corners = torch.cat([tr_corners, corners[:2, :]], dim=1)
    bounding_box = all_corners.amax(dim=1) - all_corners.amin(dim=1)
    py = torch.clip(torch.floor((bounding_box[0] - h) / 2), min=0.0, max=h - 1)
    px = torch.clip(torch.floor((bounding_box[1] - w) / 2), min=0.0, max=w - 1)
    return int(px.item()), int(py.item())


def _get_mraugment_translate_pad(im_shape, translation):
    shape = im_shape[-len(translation) :]

    pad = []
    sl = []

    for s, t in zip(shape, translation):
        if t > 0:
            pad.append((t, 0))
            sl.append(slice(0, s))
        else:
            pad.append((0, abs(t)))
            sl.append(slice(abs(t), None))
    pad = [x for y in pad[::-1] for x in y]
    sl.insert(0, Ellipsis)
    return pad, sl
