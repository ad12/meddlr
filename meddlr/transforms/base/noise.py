from typing import Optional, Tuple

import torch

from meddlr.ops import complex as cplx
from meddlr.transforms.build import TRANSFORM_REGISTRY
from meddlr.transforms.transform import Transform


@TRANSFORM_REGISTRY.register()
class NoiseTransform(Transform):
    """A deterministic transform that adds zero-mean complex additive Gaussian (white) noise.

    Note:
        This transform currently only supports adding noise to k-space.
        It does not support adding noise to image data.

    TODO (arjundd): Add support for adding noise to arbitrary data types.
    """

    def __init__(
        self,
        std_dev: float,
        use_mask: bool = True,
        rho: float = None,
        seed: int = None,
        generator: torch.Generator = None,
    ):
        """
        Args:
            std_dev: The standard deviation of the noise.
            use_mask: Whether to only add noise to the non-zero entries of the kspace.
            rho: The fraction of entries to add noise to. If ``use_mask=True``,
                only non-zero entries are considered.
            seed (int, optional): The seed to use for the random number generator.
            generator (torch.Generator, optional): The random number generator to use.
                Must be specified if ``seed`` is not set.
        """
        if seed is None and generator is None:
            raise ValueError("One of `seed` or `generator` must be specified.")
        self.std_dev = std_dev
        self.use_mask = use_mask
        self.rho = rho
        self.seed = seed

        gen_state = None
        if generator is not None:
            gen_state = generator.get_state()
        self._generator_state = gen_state

    def apply_kspace(self, kspace: torch.Tensor) -> torch.Tensor:
        """Add noise to the kspace.

        Args:
            kspace: A complex valued tensor.

        Returns:
            torch.Tensor: A complex valued tensor with noise added.
        """
        return self._add_noise(kspace)

    def _generator(self, data: torch.Tensor) -> torch.Generator:
        seed = self.seed

        g = torch.Generator(device=data.device)
        if seed is None:
            g.set_state(self._generator_state)
        else:
            g = g.manual_seed(seed)
        return g

    def _add_noise(self, data: torch.Tensor) -> torch.Tensor:
        noise_std = self.std_dev
        subsample_masks = self.rho is not None and self.rho != 1

        mask = None
        if self.use_mask:
            mask = cplx.get_mask(data)
        if mask is None and subsample_masks:
            mask = torch.ones(data.shape)

        g = self._generator(data)
        if cplx.is_complex(data):
            noise = noise_std * torch.randn(data.shape + (2,), generator=g, device=data.device)
            noise = torch.view_as_complex(noise)
        else:
            noise = noise_std * torch.randn(data.shape, generator=g, device=data.device)

        if self.rho is not None and self.rho != 1:
            mask = self.subsample_mask(mask)
        if mask is not None:
            noise = noise * mask
        aug_kspace = data + noise

        return aug_kspace

    def subsample_mask(
        self, mask: torch.Tensor, generator: Optional[torch.Generator] = None
    ) -> torch.Tensor:
        """Returns mask with ``1-self.rho`` fraction of valid entries dropped.

        Valid entries in ``mask`` are non-zero entries. Only non-zero entries will be masked out.

        TODO (arjundd): Use :cls:`KspaceMaskTransform` for this.

        Args:
            mask: A binary tensor where ``1`` indicates valid entries.

        Returns:
            torch.Tensor: The subsampled mask.
        """
        rho = self.rho
        shape = mask.shape
        mask = mask.view(-1)

        # TODO: this doesnt work if the matrix is > 2*24 in size.
        # TODO: make this a bit more optimized
        num_valid = torch.sum(mask)
        weights = mask / num_valid
        num_samples = int((1 - rho) * num_valid)
        samples = torch.multinomial(weights, num_samples, replacement=False, generator=generator)
        mask[samples] = 0

        mask = mask.view(shape)
        return mask

    def _eq_attrs(self) -> Tuple[str]:
        return ("std_dev", "use_mask", "rho", "seed", "_generator_state")
