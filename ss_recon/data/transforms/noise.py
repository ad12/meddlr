from typing import Sequence, Union

import torch

from ss_recon.utils import complex_utils as cplx
from ss_recon.utils import env
from ss_recon.utils.events import get_event_storage

if env.pt_version() >= [1,6]:
    import torch.fft


class NoiseModel():
    """A model that adds additive white noise.

    Note we do not store this as a module or else it would be saved to the model
    definition, which we dont want.
    """
    def __init__(self, std_devs: Union[float, Sequence[float]], scheduler=None, seed=None):
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

        # For reproducibility.
        g = torch.Generator()
        if seed:
            g = g.manual_seed(seed)
        self.generator = g
    
    def choose_std_dev(self):
        """Chooses a random acceleration rate given a range.
        """
        if not isinstance(self.std_devs, Sequence):
            return self.std_devs
        elif len(self.std_devs) == 1:
            return self.std_devs[0]

        if self.warmup_method:
            curr_iter = get_event_storage().iter
            warmup_iters = self.warmup_iters
            if self.warmup_method == "linear":
                std_range = curr_iter / warmup_iters * (self.std_devs[1] - self.std_devs[0])
            else:
                raise ValueError(f"`warmup_method={self.warmup_method}` not supported")
        else:
            std_range = self.std_devs[1] - self.std_devs[0]

        std_dev = self.std_devs[0] + std_range * torch.rand(1, generator=self.generator).item()
        return std_dev

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def forward(self, kspace, mask=None, seed=None, clone=True) -> torch.Tensor:
        """Performs augmentation on undersampled kspace mask."""
        if clone:
            kspace = kspace.clone()
        mask = cplx.get_mask(kspace)

        g = self.generator if seed is None else torch.Generator().manual_seed(seed)
        noise_std = self.choose_std_dev()
        if cplx.is_complex(kspace):
            noise = noise_std * torch.randn(kspace.shape + (2,), generator=g)
            noise = torch.view_as_complex(noise)
        else:
            noise = noise_std * torch.randn(kspace.shape, generator=g)
        masked_noise = noise * mask
        aug_kspace = kspace + masked_noise

        return aug_kspace

    @classmethod
    def from_cfg(cls, cfg, seed=None, **kwargs):
        cfg = cfg.MODEL.CONSISTENCY.AUG.NOISE
        return cls(cfg.STD_DEV, scheduler=cfg.SCHEDULER, seed=seed, **kwargs)
