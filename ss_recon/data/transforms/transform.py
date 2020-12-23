"""Basic Transforms.
"""
import numpy as np
import torch
from fvcore.common.registry import Registry

from ss_recon.utils import complex_utils as cplx
from ss_recon.utils import transforms as T

from .subsample import build_mask_func


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


class Normalizer():
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
        outputs.update({
            "mean": torch.tensor([0.0], dtype=torch.float32),
            "std": torch.tensor([1.0], dtype=torch.float32),
        })
        return outputs

    def undo(self, **kwargs):
        return {k: v for k, v in kwargs.items()}


@NORMALIZER_REGISTRY.register()
class TopMagnitudeNormalizer(Normalizer):
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
        outputs.update({
            "mean": mean,
            "std": std,
        })
        
        # Add other keys that were not computed.
        outputs.update({k: v for k, v in kwargs.items() if k not in outputs})
        return outputs

    def undo(self, mean, std, **kwargs):
        image = kwargs["image"]
        mean = mean.view(mean.shape + (1,)*(image.ndim - mean.ndim)).to(image.device)
        std = std.view(std.shape + (1,)*(image.ndim - std.ndim)).to(image.device)

        outputs = {}
        for kw in ("image", "target"):
            if kw in self._keywords:
                outputs[kw] = unnormalize_affine(kwargs[kw], mean, std)
        if any("kspace" in k for k in kwargs.keys()):
            raise ValueError("Currently does not support undoing analysis on kspace")

        # Add other keys that were not computed.
        outputs.update({k: v for k, v in kwargs.items() if k not in outputs})
        return outputs


class Subsampler(object):
    def __init__(self, mask_func):
        self.mask_func = mask_func

    def __call__(
        self, data, mode: str = "2D", seed: int = None, acceleration: int = None
    ):
        data_shape = tuple(data.shape)
        assert mode in ["2D", "3D"]
        if mode == "2D":
            extra_dims = data.ndim - 3
            mask_shape = (1,) + data_shape[1:3] + (1,) * extra_dims
        elif mode == "3D":
            extra_dims = data.ndim - 4
            mask_shape = (1,) + data_shape[1:4] + (1,) * extra_dims
        else:
            raise ValueError(
                "Only 2D and 3D undersampling masks are supported."
            )
        mask = self.mask_func(mask_shape, seed, acceleration)
        return torch.where(mask == 0, torch.tensor([0], dtype=data.dtype), data), mask


class DataTransform:
    """
    Data Transformer for training unrolled reconstruction models.

    For scans that
    """

    def __init__(self, cfg, mask_func, is_test: bool = False, add_noise: bool = False):
        """
        Args:
            mask_func (utils.subsample.MaskFunc): A function that can create a
                mask of appropriate shape.
            is_test (bool): If `True`, this class behaves with test-time
                functionality. In particular, it computes a pseudo random number
                generator seed from the filename. This ensures that the same
                mask is used for all the slices of a given volume every time.
        """
        self._cfg = cfg
        self.mask_func = mask_func
        self._is_test = is_test

        # Build subsampler.
        # mask_func = build_mask_func(cfg)
        self._subsampler = Subsampler(self.mask_func)
        self.add_noise = add_noise
        seed = cfg.SEED if cfg.SEED > -1 else None
        self.rng = np.random.RandomState(seed)
        self.noiser = T.NoiseModel(cfg.MODEL.CONSISTENCY.AUG.NOISE.STD_DEV, seed=seed)
        self.p_noise = cfg.AUG_TRAIN.NOISE_P
        self._normalizer = build_normalizer(cfg)

    def __call__(
        self,
        kspace,
        maps,
        target,
        fname,
        slice,
        is_fixed,
        acceleration: int = None,
    ):
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
            raise ValueError(
                "Accelerations must be specified for undersampled scans"
            )

        # Convert everything from numpy arrays to tensors
        kspace = cplx.to_tensor(kspace).unsqueeze(0)
        maps = cplx.to_tensor(maps).unsqueeze(0)
        target = cplx.to_tensor(target).unsqueeze(0)
        norm = torch.sqrt(torch.mean(cplx.abs(target) ** 2))

        # print(kspace.shape)
        # print(maps.shape)
        # print(target.shape)

        # TODO: Add other transforms here.

        # Apply mask in k-space
        seed = sum(tuple(map(ord, fname))) if self._is_test or is_fixed else None  # noqa
        masked_kspace, mask = self._subsampler(
            kspace, mode="2D", seed=seed, acceleration=acceleration
        )

        # Zero-filled Sense Recon.
        A = T.SenseModel(maps, weights=mask)
        image = A(masked_kspace, adjoint=True)

        # Normalize
        normalized = self._normalizer.normalize(**{
            "masked_kspace": masked_kspace, 
            "image": image, 
            "target": target,
            "mask": mask,
        })
        masked_kspace = normalized["masked_kspace"]
        target = normalized["target"]
        mean = normalized["mean"]
        std = normalized["std"]

        add_noise = self.add_noise and (self._is_test or (not is_fixed and self.rng.uniform() < self.p_noise))
        if add_noise:
            masked_kspace = self.noiser(masked_kspace, mask=mask, seed=seed)

        # Get rid of batch dimension...
        masked_kspace = masked_kspace.squeeze(0)
        maps = maps.squeeze(0)
        target = target.squeeze(0)

        return masked_kspace, maps, target, mean, std, norm
