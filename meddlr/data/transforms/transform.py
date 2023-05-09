"""Basic Transforms.
"""
from functools import partial
from typing import Any, Dict

import numpy as np
import torch
from fvcore.common.registry import Registry

from meddlr.data.transforms.subsample import MaskFunc
from meddlr.forward import SenseModel
from meddlr.ops import complex as cplx
from meddlr.utils import transforms as T

from .motion import MotionModel
from .noise import NoiseModel

NORMALIZER_REGISTRY = Registry("NORMALIZER")
NORMALIZER_REGISTRY.__doc__ = """
Registry for normalizing images
"""


def build_normalizer(cfg):
    cfg = cfg.MODEL.NORMALIZER
    name = cfg.NAME
    obj = NORMALIZER_REGISTRY.get(name)(keywords=cfg.KEYWORDS)
    return obj


def normalize_affine(x, bias, scale):
    return (x - bias) / scale


def unnormalize_affine(x, bias, scale):
    return x * scale + bias


class Normalizer:
    """Template for normalizing and undoing normalization for scans."""

    # Keywords of dictionary keys to process (if they exist)
    # image: The zero-filled or reconstructed image
    # target: The target (fully-sampled) image
    # masked_kspace: The kspace used to calculate the zero-filled image.
    KEYWORDS = ("image", "target", "masked_kspace")

    def __init__(self, keywords=None):
        if not keywords:
            keywords = self.KEYWORDS
        # Copy the sequence to allow modification down the line.
        self._keywords = tuple(keywords)

    def normalize(self, **kwargs):
        pass

    def undo(self, **kwargs):
        pass


@NORMALIZER_REGISTRY.register()
class NoOpNormalizer(Normalizer):
    def normalize(self, **kwargs):
        outputs = {k: v for k, v in kwargs.items()}
        outputs.update(
            {
                "mean": torch.tensor([0.0], dtype=torch.float32),
                "std": torch.tensor([1.0], dtype=torch.float32),
            }
        )
        return outputs

    def undo(self, **kwargs):
        return {k: v for k, v in kwargs.items()}


@NORMALIZER_REGISTRY.register()
class AffineNormalizer(Normalizer):
    def normalize(self, scale, bias, **kwargs):
        outputs = {}
        for k, v in kwargs.items():
            if k in self._keywords:
                outputs[k] = (v - bias) / scale
        outputs.update({"mean": bias, "std": scale})
        return outputs

    def undo(self, mean, std, channels_last=False, **kwargs):
        image = kwargs["image"]
        mean = self._reshape_params(mean, image.ndim, channels_last=channels_last).to(image.device)
        std = self._reshape_params(std, image.ndim, channels_last=channels_last).to(image.device)

        outputs = {}
        for kw in ("image", "target"):
            if kw in self._keywords:
                outputs[kw] = unnormalize_affine(kwargs[kw], mean, std)
        if any("kspace" in k for k in kwargs.keys()):
            raise ValueError("Currently does not support undoing analysis on kspace")

        # Add other keys that were not computed.
        outputs.update({k: v for k, v in kwargs.items() if k not in outputs})
        return outputs

    def _reshape_params(self, param, ndim: int, channels_last: bool = False) -> torch.Tensor:
        if not channels_last or param.ndim != 2:
            return param.view(param.shape + (1,) * (ndim - param.ndim))
        shape = (param.shape[0],) + (1,) * (ndim - param.ndim) + (param.shape[1],)
        return param.reshape(shape)


@NORMALIZER_REGISTRY.register()
class TopMagnitudeNormalizer(AffineNormalizer):
    """Normalizes by percentile of magnitude values."""

    def __init__(self, keywords=None, percentile=0.95, use_mean=False):
        super().__init__(keywords)
        assert 0 < percentile <= 1, "Percentile must be in range (0,1]"
        self._percentile = percentile
        self._use_mean = use_mean

    def normalize(self, masked_kspace, image, **kwargs):
        magnitude_vals = cplx.abs(image).reshape(-1)
        k = int(round((1 - self._percentile) * magnitude_vals.numel()))
        scale = torch.min(torch.topk(magnitude_vals, k).values)

        outputs = {}
        outputs["masked_kspace"] = masked_kspace / scale
        outputs["image"] = image / scale
        if "target" in self._keywords:
            outputs["target"] = kwargs["target"] / scale

        mean = torch.tensor([0.0], dtype=torch.float32)
        std = scale.unsqueeze(-1)
        outputs.update({"mean": mean, "std": std})

        # Add other keys that were not computed.
        outputs.update({k: v for k, v in kwargs.items() if k not in outputs})
        return outputs


class Subsampler(object):
    def __init__(self, mask_func: MaskFunc):
        self.mask_func = mask_func
        self.zip2_padding = None

    def _get_mask_shape(self, data_shape, mode: str):
        """Returns the shape of the mask based on the data shape.

        Args:
            data_shape (tuple[int]): The data shape.
            mode: Either ``"2D"`` or ``"3D"``
        """
        if mode == "2D":
            extra_dims = len(data_shape) - 3
            mask_shape = (1,) + data_shape[1:3] + (1,) * extra_dims
        elif mode == "3D":
            extra_dims = len(data_shape) - 4
            mask_shape = (1,) + data_shape[1:4] + (1,) * extra_dims
        else:
            raise ValueError("Only 2D and 3D undersampling masks are supported.")
        return mask_shape

    def __call__(self, data, mode: str = "2D", seed: int = None, acceleration: int = None):
        assert mode in ["2D", "3D"]
        data_shape = tuple(data.shape)
        if self.zip2_padding:
            data_shape = (
                data_shape[:1]
                + tuple(
                    s - 2 * p if p is not None else s
                    for s, p in zip(data_shape[1:], self.zip2_padding)
                )
                + data_shape[len(self.zip2_padding) + 1 :]
            )

        mask_shape = self._get_mask_shape(data_shape, mode)
        mask = self.mask_func(mask_shape, seed, acceleration)
        if self.zip2_padding:
            padded_mask_shape = self._get_mask_shape(tuple(data.shape), mode)
            padded_mask_shape = padded_mask_shape[1 : len(self.zip2_padding) + 1]
            mask = T.zero_pad(mask, padded_mask_shape)
        return torch.where(mask == 0, torch.tensor([0], dtype=data.dtype), data), mask

    def edge_mask(self, kspace: torch.Tensor, mode: str = "2D"):
        mask_shape = self._get_mask_shape(kspace.shape, mode=mode)
        mask = self.mask_func.get_edge_mask(kspace, out_shape=mask_shape)
        return mask


class DataTransform:
    """
    Data Transformer for training unrolled reconstruction models.

    For scans that
    """

    def __init__(
        self,
        cfg,
        mask_func,
        is_test: bool = False,
        add_noise: bool = False,
        add_motion: bool = False,
    ):
        """
        Args:
            mask_func (utils.subsample.MaskFunc): A function that can create a
                mask of appropriate shape.
            is_test (bool): If `True`, this class behaves with test-time
                functionality. In particular, it computes a pseudo random number
                generator seed from the filename. This ensures that the same
                mask is used for all the slices of a given volume every time.
        """
        from meddlr.transforms.builtin.mri import MRIReconAugmentor

        self._cfg = cfg
        self.mask_func = mask_func
        self._is_test = is_test

        # Build subsampler.
        # mask_func = build_mask_func(cfg)
        self._subsampler = Subsampler(self.mask_func)
        self.add_noise = add_noise
        self.add_motion = add_motion
        seed = cfg.SEED if cfg.SEED > -1 else None
        self.rng = np.random.RandomState(seed)
        if is_test:
            # When we test we dont want to initialize with certain parameters (e.g. scheduler).
            self.noiser = NoiseModel(cfg.MODEL.CONSISTENCY.AUG.NOISE.STD_DEV, seed=seed)
            self.motion_simulator = MotionModel(cfg.MODEL.CONSISTENCY.AUG.MOTION.RANGE, seed=seed)
        else:
            self.noiser = NoiseModel.from_cfg(cfg, seed=seed)
            self.motion_simulator = MotionModel.from_cfg(cfg, seed=seed)
        self.p_noise = cfg.AUG_TRAIN.NOISE_P
        self.p_motion = cfg.AUG_TRAIN.MOTION_P
        self._normalizer = build_normalizer(cfg)
        self._postprocessor = cfg.TEST.POSTPROCESSOR.NAME
        assert self._postprocessor in ["", "hard_dc_edge", "hard_dc_all"], self._postprocessor

        # Build augmentation pipeline.
        self.augmentor = None
        if not is_test and cfg.AUG_TRAIN.MRI_RECON.TRANSFORMS:
            self.augmentor = MRIReconAugmentor.from_cfg(cfg, aug_kind="aug_train", seed=seed)

    def _get_mask(self, masked_kspace):
        # If any of the coils are non-zero at a coordinate, we assume
        assert torch.is_complex(masked_kspace)
        return cplx.get_mask(masked_kspace, coil_dim=-1)

    def _call_augmentor(
        self, kspace, maps, target, fname, slice_id, is_fixed, acceleration: int = None
    ):
        assert not self._is_test, "Augmentor is not supported with testing yet"

        # Convert everything from numpy arrays to tensors
        kspace = cplx.to_tensor(kspace).unsqueeze(0)
        maps = cplx.to_tensor(maps).unsqueeze(0)
        target_init = cplx.to_tensor(target).unsqueeze(0)
        target = (
            torch.complex(target_init, torch.zeros_like(target_init)).unsqueeze(-1)
            if not torch.is_complex(target_init)
            else target_init
        )  # handle rss vs. sensitivity-integrated
        norm = torch.sqrt(torch.mean(cplx.abs(target) ** 2))

        seed = sum(tuple(map(ord, fname))) if self._is_test or is_fixed else None  # noqa
        mask_gen = partial(
            self._subsampler.__call__, mode="2D", seed=seed, acceleration=acceleration
        )

        out, _, _ = self.augmentor(
            kspace,
            maps=maps,
            target=target,
            normalizer=self._normalizer,
            mask_gen=mask_gen,
            skip_tfm=is_fixed,  # Skip augmentations for unsupervised scans.
        )
        masked_kspace = out["kspace"]
        maps = out["maps"]
        target = out["target"]
        mean = out["mean"]
        std = out["std"]

        # Get rid of batch dimension...
        masked_kspace = masked_kspace.squeeze(0)
        maps = maps.squeeze(0)
        target = target.squeeze(0)

        return {
            "kspace": masked_kspace,
            "maps": maps,
            "target": target,
            "mean": mean,
            "std": std,
            "norm": norm,
            "mask": self._get_mask(masked_kspace),
        }

    def __call__(
        self, kspace, maps, target, fname, slice_id, is_fixed, acceleration: int = None
    ) -> Dict[str, Any]:
        """
        Args:
            kspace (numpy.array): Input k-space of shape
                (num_coils, rows, cols, 2) for multi-coil
                data or (rows, cols, 2) for single coil data.
            target (numpy.array): Target image
            attrs (dict): Acquisition related information stored in the HDF5
                object.
            fname (str): File name
            slice (int): Serial number of the slice.
            is_fixed (bool, optional): If `True`, transform the example
                to have a fixed mask and acceleration factor.
            acceleration (int): Acceleration factor. Must be provided if
                `is_undersampled=True`.
        Returns:
            (tuple): tuple containing:
                image (torch.Tensor): Zero-filled input image.
                target (torch.Tensor): Target image converted to a torch Tensor.
                mean (float): Mean value used for normalization.
                std (float): Standard deviation value used for normalization.
                norm (float): L2 norm of the entire volume.
        """
        if is_fixed and not acceleration:
            raise ValueError("Accelerations must be specified for undersampled scans")

        # If augmentor is defined, use it to do computation.
        if self.augmentor is not None:
            return self._call_augmentor(
                kspace, maps, target, fname, slice_id, is_fixed, acceleration
            )

        # Convert everything from numpy arrays to tensors
        kspace = cplx.to_tensor(kspace).unsqueeze(0)
        maps = cplx.to_tensor(maps).unsqueeze(0)
        target_init = cplx.to_tensor(target).unsqueeze(0)
        target = (
            torch.complex(target_init, torch.zeros_like(target_init)).unsqueeze(-1)
            if not torch.is_complex(target_init)
            else target_init
        )  # handle rss vs. sensitivity-integrated
        norm = torch.sqrt(torch.mean(cplx.abs(target) ** 2))

        # TODO: Add other transforms here.

        # Apply mask in k-space
        seed = sum(tuple(map(ord, fname))) if self._is_test or is_fixed else None  # noqa
        masked_kspace, mask = self._subsampler(
            kspace, mode="2D", seed=seed, acceleration=acceleration
        )

        edge_mask = self._subsampler.edge_mask(kspace, mode="2D")
        postprocessing_mask = None
        if self._is_test and self._postprocessor:
            postprocessing_mask = edge_mask
            if self._postprocessor == "hard_dc_all":
                postprocessing_mask = (postprocessing_mask + mask).bool().type(torch.float32)

        # Zero-filled Sense Recon.
        if torch.is_complex(target_init):
            A = SenseModel(maps, weights=mask)
            image = A(masked_kspace, adjoint=True)
        # Zero-filled RSS Recon.
        else:
            image = T.ifft2(masked_kspace)
            image_rss = torch.sqrt(torch.sum(cplx.abs(image) ** 2, axis=-1))
            image = torch.complex(image_rss, torch.zeros_like(image_rss)).unsqueeze(-1)

        # Normalize
        normalized = self._normalizer.normalize(
            **{"masked_kspace": masked_kspace, "image": image, "target": target, "mask": mask}
        )
        masked_kspace = normalized["masked_kspace"]
        target = normalized["target"]
        mean = normalized["mean"]
        std = normalized["std"]
        add_noise = self.add_noise and (self._is_test or (self.rng.uniform() < self.p_noise))
        add_motion = self.add_motion and (
            self._is_test or (not is_fixed and self.rng.uniform() < self.p_motion)
        )
        if add_noise:
            # Seed should be different for each slice of a scan.
            noise_seed = seed + slice_id if seed is not None else None
            masked_kspace = self.noiser(masked_kspace, mask=mask, seed=noise_seed)
        if add_motion:
            # Motion seed should not be different for each slice for now.
            # TODO: Change this for 2D acquisitions.
            masked_kspace = self.motion_simulator(masked_kspace, seed=seed)
        # Get rid of batch dimension...
        masked_kspace = masked_kspace.squeeze(0)
        maps = maps.squeeze(0)
        target = target.squeeze(0)

        out = {
            "kspace": masked_kspace,
            "maps": maps,
            "target": target,
            "mean": mean,
            "std": std,
            "norm": norm,
            "edge_mask": edge_mask.squeeze(0),
            "mask": self._get_mask(masked_kspace),
        }
        if postprocessing_mask is not None:
            out["postprocessing_mask"] = postprocessing_mask.squeeze(0)
        return out
