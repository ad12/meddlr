from typing import Optional, Tuple, Sequence, Union

import math
import numpy as np
import torch
import meddlr.ops as F
from meddlr.transforms import RandomAffine, RandomTranslation, TransformList
from meddlr.forward.mri import SenseModel

from meddlr.utils.events import get_event_storage


class MotionModel2D:
    """A model that corrupts kspace inputs with motion.

    Motion is a common artifact experienced during the MR imaging forward problem.
    When a patient moves, the recorded (expected) location of the kspace sample is
    different than the actual location where the kspace sample that was acquired.
    This module is responsible for simulating different motion artifacts.
    """

    def __init__(self, nshots, angle, translate, trajectory, seed=None):
        """
        Args:
            nshots (int) : The number of shots in the image. 
                This should be equivalent to ceil(phase_encode_dim / 
                echo_train_length). 
            angle : The (min, max) angle for rotation. Values should be in 
                degrees and should be >=-180, <=180. Use 'None' to 
                ignore rotation.
            translate: The fraction of (height, width) to translate. 
                e.g. 0.1 => 10% of the corresponding dimension.
                So (0.1, 0.2) => 10% of height, 20% of width.
                Use 'None' to ignore translation. 
            trajectory: One of 'interleaved' or 'consecutive'. 
        """
        self.nshots = nshots
        self.angle = angle 
        self.translate = translate 
        self.trajectory = trajectory
        self.seed = seed

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def forward(self, image) -> torch.Tensor:
        """
        Simulate 2D motion for multi-shot Cartesian MRI.

        This function supports two trajectories:
        - 'blocked' : Where each shot corresponds to a consecutive block of 
          kspace. (e.g. 1 1 2 2 3 3)
        - 'interleaved' : Where shots are interleaved (e.g. 1 2 3 1 2 3) 
        
        We assume the phase encode direction is left to right (i.e. along
        width dimesion). 

        TODO: Add support for sensitivity maps. 

        Args:
            image : The complex-valued iamge. Shape [..., height, width].

        Returns:
            The motion corrupted kspace. 
        """
        if self.seed == None:
            random_motion = RandomAffine
        else:
            random_motion = RandomAffine.seed(self.seed)
            

        tfm_gen = random_motion(p = 1.0, translate=translate, angle=angle) 
        kspace = torch.zeros_like(image)
        offset = int(math.ceil(kspace.shape[-1] / nshots)) 

        for shot in range(nshots):
            motion_image = tfm_gen.get_transform(image).apply_image(image)
            motion_kspace = F.fft2c(motion_image) 
            if trajectory == "blocked":
                kspace[..., shot*offset:(shot+1)*offset] = motion_kspace[..., shot*offset:(shot+1)*offset]
            elif trajectory == "interleaved":
                kspace[..., shot::nshots] = motion_kspace[..., shot::nshots]
            else:
                raise ValueError(f"trajectory '{trajectory}' not supported.")

        return kspace

