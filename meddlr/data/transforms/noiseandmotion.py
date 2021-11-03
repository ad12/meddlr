from typing import Sequence, Union

import numpy as np
import torch

from meddlr.ops import complex as cplx
from meddlr.utils import env
from meddlr.utils.events import get_event_storage

if env.pt_version() >= [1, 6]:
    import torch.fft


class NoiseAndMotionModel:
    """A model that adds additive white noise after adding simple motion. N(M(x))"""

    def __init__(
        self,
        std_devs: Union[float, Sequence[float]],
        motion_range: Union[float, Sequence[float]],
        scheduler=None,
        seed=None,
        device=None,
    ):
        if not isinstance(std_devs, Sequence):
            std_devs = (std_devs,)
        if isinstance(motion_range, (float, int)):
            motion_range = (motion_range,)
        elif len(std_devs) > 2:
            raise ValueError("`std_devs` must have 2 or fewer values")
        elif len(motion_range) > 2:
            raise ValueError("`motion_range` must have 2 or fewer values")
        self.std_devs = std_devs
        self.motion_range = motion_range

        self.warmup_method = None
        self.warmup_iters = 0
        if scheduler is not None:
            self.warmup_method = scheduler.WARMUP_METHOD
            self.warmup_iters = scheduler.WARMUP_ITERS

        # For reproducibility.
        g = torch.Generator(device=device)
        if seed:
            g = g.manual_seed(seed)
        self.generator = g

    def choose_std_dev(self):
        """Chooses std range based on warmup."""
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

        g = self.generator
        std_dev = self.std_devs[0] + std_range * torch.rand(1, generator=g, device=g.device).item()
        return std_dev

    def choose_motion_range(self):
        """Chooses motion range based on warmup."""
        if not isinstance(self.motion_range, Sequence):
            return self.motion_range
        elif len(self.motion_range) == 1:
            return self.motion_range[0]

        if self.warmup_method:
            curr_iter = get_event_storage().iter
            warmup_iters = self.warmup_iters
            if self.warmup_method == "linear":
                motion_range = (
                    curr_iter / warmup_iters * (self.motion_range[1] - self.motion_range[0])
                )
            else:
                raise ValueError(f"`warmup_method={self.warmup_method}` not supported")
        else:
            motion_range = self.motion_range[1] - self.motion_range[0]

        g = self.generator
        motion_range = (
            self.motion_range[0] + motion_range * torch.rand(1, generator=g, device=g.device).item()
        )
        return motion_range

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def forward(self, kspace, mask=None, seed=None, clone=True) -> torch.Tensor:
        """Performs noise augmentation followed by motion simulation on undersampled kspace mask."""
        if clone:
            kspace = kspace.clone()
        mask = cplx.get_mask(kspace)

        g = (
            self.generator
            if seed is None
            else torch.Generator(device=kspace.device).manual_seed(seed)
        )

        phase_matrix = torch.zeros(kspace.shape, dtype=torch.complex64)
        width = kspace.shape[2]

        noise_std = self.choose_std_dev()
        scale = self.choose_motion_range()

        odd_err = (2 * np.pi * scale) * torch.rand(1, generator=g).numpy() - np.pi * scale
        even_err = (2 * np.pi * scale) * torch.rand(1, generator=g).numpy() - np.pi * scale

        for line in range(width):
            if line % 2 == 0:
                rand_err = even_err
            else:
                rand_err = odd_err
            phase_error = torch.from_numpy(np.exp(-1j * rand_err))
            phase_matrix[:, :, line] = phase_error
        aug_kspace = kspace * phase_matrix

        if cplx.is_complex(aug_kspace):
            noise = noise_std * torch.randn(
                aug_kspace.shape + (2,), generator=g, device=aug_kspace.device
            )
            noise = torch.view_as_complex(noise)
        else:
            noise = noise_std * torch.randn(aug_kspace.shape, generator=g, device=aug_kspace.device)
        masked_noise = noise * mask
        noised_aug_kspace = aug_kspace + masked_noise

        return noised_aug_kspace

    @classmethod
    def from_cfg(cls, cfg, seed=None, **kwargs):
        cfg = cfg.MODEL.CONSISTENCY.AUG
        return cls(
            cfg.NOISE.STD_DEV, cfg.MOTION.RANGE, scheduler=cfg.NOISE.SCHEDULER, seed=seed, **kwargs
        )
