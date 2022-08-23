import warnings
from typing import Sequence, Tuple, Union

import torch

from meddlr.ops import complex as cplx
from meddlr.transforms.build import TRANSFORM_REGISTRY
from meddlr.transforms.transform import Transform


@TRANSFORM_REGISTRY.register()
class KspaceMaskTransform(Transform):
    """A deterministic transform that masks kspace.

    This transform masks either fully-sampled or already undersampled kspace.
    Samples from kspace that are non-zero are dropped with probability defined
    by the kind of masking. ``rho`` is the fraction of pixels to drop.

    Uniform masking drops all non-zero pixels with a uniform probability.
    Gaussian masking drops all non-zero pixels with a probability modulated
    by the gaussian distribution centered at the center of the spatial
    kspace dimensions.

    Attributes:
        rho (float): The fraction of non-zero pixels to drop.
        kind (str): The kind of mask to use.
            Either ``'uniform'`` or ``'gaussian'``.
        std_scale (float): The standard deviation of the gaussian mask.
        per_example (bool): Whether to apply the mask per example
            separately or over the full batch.
        calib_size (int, Tuple[int]): The size of the calibration region to use.
            Samples in the calibration region are not dropped.
        seed (int): The seed to use for the random number generator.
        generator (torch.Generator): The random number generator to use.

    Note:
        This function is currently limited to 2D undersampling.

    Note:
        For this transform to be deterministic, either ``seed`` or ``generator``
        have to be set.

    Examples:
        >>> kspace = torch.randn(2, 100, 100, 8)
        >>> masker = KspaceMaskTransform(rho=0.5, kind='uniform', seed=42)
        >>> masked_kspace = masker(kspace)
        >>> mask = masker.generate_mask(kspace)  # to only generate mask
    """

    _SUPPORTED_MASK_KINDS = ("uniform", "gaussian")

    def __init__(
        self,
        rho: float,
        kind: str = "uniform",
        std_scale: float = 4.0,
        per_example: bool = False,
        calib_size: Union[int, Tuple[int]] = None,
        seed: int = None,
        generator: torch.Generator = None,
    ):
        """
        Args:
            rho (float): The fraction of non-zero pixels to drop.
            kind (str, optional): The kind of mask to use. One of ``'uniform'`` or ``'gaussian'``.
            std_scale (float, optional): The standard deviation of the gaussian mask.
            per_example (bool, optional): Whether to apply the mask per example
                separately or over the full batch. Defaults to False.
            calib_size (int or Tuple[int], optional): The size of the calibration region to use.
                If int, the calibration region is a square of size ``(calib_size, calib_size)``.
                If tuple, the calibration region is a rectangle of size
                ``(calib_size[0], calib_size[1])``. Defaults to not using a calibration region.
            seed (int, optional): The seed to use for the random number generator.
            generator (torch.Generator, optional): The random number generator to use.
                Must be specified if ``seed`` is not set.
        """
        if kind not in self._SUPPORTED_MASK_KINDS:
            raise ValueError(
                f"Unknown kspace mask kind={kind}. " f"Expected one of {self._SUPPORTED_MASK_KINDS}"
            )
        self.rho = rho
        self.kind = kind
        self.seed = seed
        self.calib_size = calib_size
        self.per_example = per_example
        self.std_scale = std_scale

        gen_state = None
        if generator is not None:
            gen_state = generator.get_state()
        self._generator_state = gen_state

    def generate_mask(
        self, kspace: torch.Tensor, mask: torch.Tensor = None, channels_last: bool = False
    ) -> torch.Tensor:
        """Generates mask with approximiately ``1-self.rho`` times number of pixels.

        Args:
            kspace (torch.Tensor): The batch of kspace to generate the mask for.
                Shape: ``[batch, #coils, height, width, ...]`` or
                ``[batch, height, width, ...,  #coils]`` (i.e ``channels_last=True``).
            channels_last (bool, optional): If ``True``, the kspace is
                of shape ``[batch, height, width, ...,  #coils]``.

        Returns:
            torch.Tensor: The generated mask.
        """
        g = self._generator(kspace)
        if mask is None:
            mask = True

        if channels_last:
            kspace = cplx.channels_first(kspace)
            if isinstance(mask, torch.Tensor):
                order = (0, mask.ndim - 1) + tuple(range(1, mask.ndim - 1))
                mask = mask.permute(order)

        func_and_kwargs = {
            "uniform": (
                _uniform_mask,
                {"rho": self.rho, "mask": mask, "calib_size": self.calib_size, "generator": g},
            ),
            "gaussian": (
                _gaussian_mask,
                {
                    "rho": self.rho,
                    "mask": mask,
                    "std_scale": self.std_scale,
                    "calib_size": self.calib_size,
                    "generator": g,
                },
            ),
        }
        func, kwargs = func_and_kwargs[self.kind]

        if self.per_example:
            mask = torch.cat([func(kspace[i : i + 1], **kwargs) for i in range(len(kspace))], dim=0)
        else:
            mask = func(kspace, **kwargs)

        if channels_last:
            order = (0,) + tuple(range(2, mask.ndim)) + (1,)
            mask = mask.permute(order)
        return mask

    def apply_kspace(self, kspace: torch.Tensor, channels_last: bool = False):
        """Apply transform to kspace.

        Args:
            kspace (torch.Tensor): The batch of kspace to generate the mask for.
                Shape: ``[B, C, H, W]`` or ``[B, H, W, C]`` (i.e ``channels_last=True``).
            channels_last (bool, optional): If ``True``, the kspace is
                of shape ``[B, H, W, C]``. Defaults to ``False``.

        Returns:
            torch.Tensor: The masked kspace.
        """
        mask = self.generate_mask(kspace, channels_last=channels_last)
        return mask * kspace

    def _generator(self, data: torch.Tensor) -> torch.Generator:
        seed = self.seed

        g = torch.Generator(device=data.device)
        if seed is None:
            g.set_state(self._generator_state)
        else:
            g = g.manual_seed(seed)
        return g

    def _subsample(self, data: torch.Tensor) -> torch.Tensor:  # pragma: no cover
        klass_name = {type(self).__name__}
        warnings.warn(
            f"{klass_name}._subsample is deprecated. Use {klass_name}.apply_kspace instead",
            DeprecationWarning,
        )

        mask = self.generate_mask(data)
        return mask * data

    def _eq_attrs(self) -> Tuple[str]:
        return ("std_dev", "use_mask", "rho", "seed", "_generator_state")


def _uniform_mask(
    kspace: torch.Tensor,
    rho: float,
    mask: Union[bool, torch.Tensor] = True,
    calib_size: Union[int, Tuple[int]] = None,
    generator: torch.Generator = None,
):
    """Subsamples a mask uniformly at random.

    The output will have approximately ``1-rho`` times number of valid pixels in the input mask.
    Samples will be selected uniformly at random.

    Args:
        kspace: A complex-valued tensor of shape: [batch, #coils, height, width, ...].
            where `...` represents additional spatial dimensions.
        rho: Fraction of samples to drop.
        mask: The mask to use.
            If ``mask is True``, the mask is considered to the be the non-zero
            entries of ``kspace``.
        calib_size: The size of the calibration region to use.
            Samples in the calibration region are not dropped.
        generator: The random number generator to use.

    Returns:
        torch.Tensor: A subsampled mask of shape ``[batch, 1, height, width, ...]``.
    """
    kspace = kspace[:, 0:1, ...]
    orig_mask = cplx.get_mask(kspace) if mask is True else mask
    shape = orig_mask.shape
    ndim = orig_mask.ndim

    mask = orig_mask.clone()

    calib_region = None
    if calib_size is not None:
        if not isinstance(calib_size, Sequence):
            calib_size = (calib_size,) * (ndim - 2)
        center = tuple(s // 2 for s in shape[2:])[-len(calib_size) :]
        calib_region = tuple(slice(s - cs // 2, s + cs // 2) for s, cs in zip(center, calib_size))
        calib_region = (Ellipsis,) + calib_region
        mask[calib_region] = 0

    mask = mask.view(-1) if mask.is_contiguous() else mask.reshape(-1)

    # TODO: this doesnt work if the matrix is > 2*24 in size.
    # TODO: make this a bit more optimized
    num_valid = torch.sum(mask)
    weights = mask / num_valid
    num_samples = int(rho * num_valid)
    samples = torch.multinomial(weights, num_samples, replacement=False, generator=generator)
    mask[samples] = 0

    mask = mask.view(shape)
    if calib_region:
        mask[calib_region] = orig_mask[calib_region]
    return mask


def _gaussian_mask(
    kspace: torch.Tensor,
    rho: float,
    std_scale: float,
    mask: bool = True,
    calib_size: Union[int, Tuple[int]] = None,
    generator: torch.Generator = None,
):
    """Subsamples a mask based on Gaussian weighted distribution.

    The output will have approximately ``1-rho`` times number of valid pixels in the input mask.
    Samples will be selected based on a Gaussian weighted distribution, where the center of kspace
    is most likely to be selected.

    Args:
        kspace: A complex-valued tensor of shape: [batch, #coils, height, width, ...].
            where `...` represents additional spatial dimensions.
        rho: Fraction of samples to drop.
        std_scale: The standard deviation of the Gaussian distribution.
        mask: The mask to use.
            If ``mask is True``, the mask is considered to the be the non-zero
            entries of ``kspace``.
        calib_size: The size of the calibration region to use.
            Samples in the calibration region are not dropped.
        generator: The random number generator to use.

    Returns:
        torch.Tensor: A subsampled mask of shape ``[batch, 1, height, width, ...]``.

    Note:
        Currently this creates the same mask for all the examples.
        Make sure to use per_example=True in :cls:`KspaceMaskTransform`.

    Note:
        This method is currently very slow.
    """
    kspace = kspace[:, 0:1, ...]
    shape = kspace.shape

    if mask is True:
        orig_mask = cplx.get_mask(kspace)
    else:
        orig_mask = mask
    mask = orig_mask.clone()

    spatial_shape = kspace.shape[2:]
    center = tuple(s // 2 for s in spatial_shape)

    calib_region = None
    if calib_size is not None:
        if not isinstance(calib_size, Sequence):
            calib_size = (calib_size,) * (kspace.ndim - 2)
        calib_region = tuple(
            slice(s - cs // 2, s + cs // 2) for s, cs in zip(center[-len(calib_size) :], calib_size)
        )
        calib_region = (Ellipsis,) + calib_region
        mask[calib_region] = 0

    num_valid = torch.sum(mask)
    num_samples = int(rho * num_valid)

    temp_mask = torch.zeros_like(mask)
    count = 0
    while count < num_samples:
        idxs = [
            torch.round(torch.normal(float(c), (s - 1) / std_scale, (1,))).type(torch.long)
            for c, s in zip(center, spatial_shape)
        ]
        if any(i < 0 for i in idxs) or any(i >= s for i, s in zip(idxs, spatial_shape)):
            continue
        idxs.insert(0, Ellipsis)
        if torch.all(mask[idxs] == 0) or torch.all(temp_mask[idxs] == 1):
            continue
        temp_mask[idxs] = 1
        count += 1

    mask = mask - temp_mask

    mask = mask.view(shape)
    if calib_region:
        mask[calib_region] = orig_mask[calib_region]
    return mask
