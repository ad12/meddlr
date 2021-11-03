from typing import Sequence, Union

import torch

from meddlr.ops import complex as cplx
from meddlr.utils.events import get_event_storage


class NoiseModel:
    """A model that adds additive white noise.

    This model adds zero-mean Gaussian noise to a real-valued or
    complex-valued input. The standard deviation of the noise
    distribution is chosen from a range ``std_devs`` provided by
    the user. The standard deviation is often referred to as the
    *difficulty*, as higher standard deviations correspond to
    wider distributions from which noise can be sampled.

    To simulate noise during MRI acquisition, we only add noise
    to the samples that are acquired. In other words, if a sample
    in kspace was not acquired (i.e. accelerated acquisition), those
    samples are not corrupted with noise.

    Attributes:
        std_devs (Tuple[float]): The range of standard deviations used for
            noise model. If a single value is provided on initialization,
            it is stored as a tuple with 1 element.
        warmup_method (str): The method that is being used for warmup.
        warmup_iters (int): Number of iterations to use for warmup.
        generator (torch.Generator): The generator that should be used for all
            random logic in this class.

    Note:
        This class is functionally deprecated and will not be maintained.
        Use :cls:`RandomNoise` instead.

    Note:
        We do not store this as a module or else it would be saved to the model
        definition, which we dont want.

    Note:
        There is a known bug that the warmup method does not clamp the upper
        bound at the appropriate value. Thus the upper bound of the range keeps
        growing. We have elected not to correct for this to preserve
        reproducibility for older results. To use schedulers with corrected
        functionality, see :cls:`RandomNoise` instead.
    """

    def __init__(
        self,
        std_devs: Union[float, Sequence[float]],
        scheduler=None,
        mask=None,
        seed=None,
        device=None,
    ):
        """
        Args:
            std_devs (float, Tuple[float, float]): The  noise difficulty range.
            scheduler (CfgNode, optional): Config detailing scheduler.
            mask (CfgNode, optional): Config for masking method. Should have
                an attribute ``RHO`` that specifies the extent of masking.
            seed (int, optional): A seed for reproducibility.
        """
        if not isinstance(std_devs, Sequence):
            std_devs = (std_devs,)
        elif len(std_devs) > 2:
            raise ValueError("`std_devs` must have 2 or fewer values")
        self.std_devs = std_devs

        self.warmup_method = None
        self.warmup_iters = 0
        if scheduler is not None:
            self.warmup_method = scheduler.WARMUP_METHOD
            self.warmup_iters = scheduler.WARMUP_ITERS

        # Amount of the kspace to augment with noise.
        self.rho = None
        if mask is not None:
            self.rho = mask.RHO

        # For reproducibility.
        g = torch.Generator(device=device)
        if seed:
            g = g.manual_seed(seed)
        self.generator = g

    def choose_std_dev(self):
        """Chooses a random acceleration rate given a range."""
        if not isinstance(self.std_devs, Sequence):
            return self.std_devs
        elif len(self.std_devs) == 1:
            return self.std_devs[0]

        if self.warmup_method:
            curr_iter = get_event_storage().iter
            warmup_iters = self.warmup_iters
            if self.warmup_method == "linear":
                std_range = min(curr_iter / warmup_iters, 1.0) * (
                    self.std_devs[1] - self.std_devs[0]
                )
            else:
                raise ValueError(f"`warmup_method={self.warmup_method}` not supported")
        else:
            std_range = self.std_devs[1] - self.std_devs[0]

        g = self.generator
        std_dev = self.std_devs[0] + std_range * torch.rand(1, generator=g, device=g.device).item()
        return std_dev

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def forward(self, kspace, mask=None, seed=None, clone=True) -> torch.Tensor:
        """Performs augmentation on undersampled kspace mask."""
        if clone:
            kspace = kspace.clone()
        mask = cplx.get_mask(kspace)

        g = (
            self.generator
            if seed is None
            else torch.Generator(device=kspace.device).manual_seed(seed)
        )
        noise_std = self.choose_std_dev()
        if cplx.is_complex(kspace):
            noise = noise_std * torch.randn(kspace.shape + (2,), generator=g, device=kspace.device)
            noise = torch.view_as_complex(noise)
        else:
            noise = noise_std * torch.randn(kspace.shape, generator=g, device=kspace.device)

        if self.rho is not None and self.rho != 1:
            mask = self.subsample_mask(mask)
        masked_noise = noise * mask
        aug_kspace = kspace + masked_noise

        return aug_kspace

    def subsample_mask(self, mask: torch.Tensor, generator=None):
        """Subsamples mask to add the noise to.

        Currently done uniformly at random.
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

    @classmethod
    def from_cfg(cls, cfg, seed=None, **kwargs):
        cfg = cfg.MODEL.CONSISTENCY.AUG.NOISE
        return cls(cfg.STD_DEV, scheduler=cfg.SCHEDULER, mask=cfg.MASK, seed=seed, **kwargs)
