from numpy.lib.type_check import imag
import torch

import numpy as np

from ss_recon.utils import env
from ss_recon.utils import complex_utils as cplx

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

    def __init__(self, motion_range = (0.2, 0.5), seed: int = None):
        super().__init__()
        g = torch.Generator()
        if seed:
            g = g.manual_seed(seed)
        self.generator = g
        self.motion_range = motion_range

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
        is_complex = False
        if clone:
            kspace = kspace.clone()
        if kspace.shape[-1] == 2:
            is_complex = True
            kspace = torch.view_as_complex(kspace)
            phase_matrix = torch.zeros(kspace.shape[0:-1], dtype=torch.complex64)
        else:
            phase_matrix = torch.zeros(kspace.shape, dtype=torch.complex64)
        width = kspace.shape[2]
        scale = (self.motion_range[1] - self.motion_range[0]) * \
                torch.rand(1) + self.motion_range[0]

        g = self.generator if seed is None else torch.Generator().manual_seed(seed)
        odd_err = (2 * np.pi * scale) * torch.rand(1, generator=g).numpy() - np.pi * scale
        even_err = (2 * np.pi * scale) * torch.rand(1, generator=g).numpy() - np.pi * scale
        for line in range(width):
            if line % 2 == 0:
                rand_err = even_err
            else:
                rand_err = odd_err
            phase_error = np.exp(-1j * rand_err)
            phase_matrix[:, :, line] = phase_error
        aug_kspace = kspace * phase_matrix
        return aug_kspace

    def choose_motion_range(self, range):
        self.motion_range = range

    @classmethod
    def from_cfg(cls, cfg, seed=None, **kwargs):
        motion_range = cfg.MODEL.CONSISTENCY.AUG.MOTION_RANGE
        return cls(motion_range=motion_range, seed=seed, **kwargs)
