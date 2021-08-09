from typing import Sequence, Union
import numpy as np
import torch

from ss_recon.utils import complex_utils as cplx
from ss_recon.utils import env
from ss_recon.utils.events import get_event_storage

if env.pt_version() >= [1, 6]:
    import torch.fft


class MotionModel:
    """A model that corrupts kspace inputs with motion.
    Motion is a common artifact experienced during the MR imaging forward problem.
    When a patient moves, the recorded (expected) location of the kspace sample is
    different than the actual location where the kspace sample that was acquired.
    This module is responsible for simulating different motion artifacts.
    Args:
        seed (int, optional): The fixed seed.
    Attributes:
        generator (torch.Generator): The generator that should be used for all
            random logic in this class.
    Things to consider:
        1. What other information is relevant for inducing motion corruption?
            This could include:
            - ``traj``: The scan trajectory
            - ``etl``: The echo train length - how many readouts per shot.
            - ``num_shots``: Number of shots.
        2. What would a simple translational motion model look?
    Note:
        We do not store this as a module or else it would be saved to the model
        definition, which we dont want.
    """

    def __init__(
        self, motion_range: Union[float, Sequence[float]], scheduler=None, seed=None, device=None
    ):
        if isinstance(motion_range, (float, int)):
            motion_range = (motion_range,)
        elif len(motion_range) > 2:
            raise ValueError("`motion_range` must have 2 or fewer values")

        self.warmup_method = None
        self.warmup_iters = 0
        if scheduler is not None:
            self.warmup_method = scheduler.WARMUP_METHOD
            self.warmup_iters = scheduler.WARMUP_ITERS
        self.motion_range = motion_range

        g = torch.Generator()
        if seed:
            g = g.manual_seed(seed)
        self.generator = g

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
                motion_range = curr_iter / warmup_iters * (self.motion_range[1] - self.motion_range[0])
            else:
                raise ValueError(f"`warmup_method={self.warmup_method}` not supported")
        else:
            motion_range = self.motion_range[1] - self.motion_range[0]

        g = self.generator
        motion_range = self.motion_range[0] + motion_range * torch.rand(1, generator=g, device=g.device).item()
        return motion_range

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def forward(self, kspace, seed=None, clone=True) -> torch.Tensor:
        """Performs motion corruption on kspace image.

        TODO: The current arguments were copied from the NoiseModel.
            Feel free to change.
        Args:
            kspace (torch.Tensor): The complex tensor. Shape ``(N, Y, X, #coils, [2])``.
            mask (torch.Tensor): The undersampling mask. Shape ``(N, Y, X, #coils)``.
            seed (int, optional): Fixed seed at runtime (useful for generating testing vals).
            clone (bool, optional): If ``True``, return a cloned tensor.
        Returns:
            torch.Tensor: The motion corrupted kspace.
        Note:
            For backwards compatibility with torch<1.6, complex tensors may also have the shape
            ``(..., 2)``, where the 2 channels in the last dimension are real and
            imaginary, respectively.
            TODO: This code should account for that case as well.
        """
        # is_complex = False
        if clone:
            kspace = kspace.clone()
        phase_matrix = torch.zeros(kspace.shape, dtype=torch.complex64)
        width = kspace.shape[2]
        g = self.generator if seed is None else torch.Generator().manual_seed(seed)
        motion_range = self.choose_motion_range()

        odd_err = (2 * np.pi * motion_range) * torch.rand(1, generator=g).numpy() - np.pi * motion_range
        even_err = (2 * np.pi * motion_range) * torch.rand(1, generator=g).numpy() - np.pi * motion_range
        for line in range(width):
            if line % 2 == 0:
                rand_err = even_err
            else:
                rand_err = odd_err
            phase_error = torch.from_numpy(np.exp(-1j * rand_err))
            phase_matrix[:, :, line] = phase_error
        aug_kspace = kspace * phase_matrix
        return aug_kspace

    @classmethod
    def from_cfg(cls, cfg, seed=None, **kwargs):
        cfg = cfg.MODEL.CONSISTENCY.AUG.MOTION
        return cls(cfg.RANGE, scheduler=cfg.SCHEDULER, seed=seed, **kwargs)
