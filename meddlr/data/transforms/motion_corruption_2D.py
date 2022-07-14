import math
from typing import Optional, Tuple

import torch

import meddlr.ops as F
from meddlr.transforms import RandomAffine


def add_motion_corruption(
    image: torch.Tensor,
    nshots: int,
    angle: Optional[Tuple[float, float]] = (-5.0, 5.0),
    translate: Optional[Tuple[float, float]] = (0.1, 0.1),
    trajectory: str = "blocked",
    seed: Optional[float] = None,
) -> torch.Tensor:
    """
    Simulate 2D motion for multi-shot Cartesian MRI.

    This function supports two trajectories:
    - 'blocked': Where each shot corresponds to a consecutive block of kspace.
       (e.g. 1 1 2 2 3 3)
    - 'interleaved': Where shots are interleaved (e.g. 1 2 3 1 2 3)

    We assume the phase encode direction is left to right
    (i.e. along width dimension).

    TODO: Add support for sensitivity maps.

    Args:
        image: The complex-valued image. Shape [..., height, width].
        nshots: The number of shots in the image.
            This should be equivalent to ceil(phase_encode_dim / echo_train_length).
        angle: The (min, max) angle for rotation. Values should be in degrees
            and should be >=-180, <=180. Use `None` to ignore rotation.
        translate: The fraction of (height, width) to translate.
            e.g. 0.1 => 10% of the corresponding dimension.
            So (0.1, 0.2) => 10% of height, 20% of width.
            Use `None` to ignore translation.
        trajectory: One of 'interleaved' or 'consecutive'.

    Returns:
        A motion corrupted image.
    """
    if seed is None:
        random_motion = RandomAffine(p=1.0, translate=translate, angle=angle)
    else:
        random_motion = RandomAffine(p=1.0, translate=translate, angle=angle)
        random_motion.seed(seed)

    tfm_gen = random_motion
    kspace = torch.zeros_like(image)
    offset = int(math.ceil(kspace.shape[-1] / nshots))

    for shot in range(nshots):
        motion_image = tfm_gen.get_transform(image).apply_image(image)
        motion_kspace = F.fft2c(motion_image)
        if trajectory == "blocked":
            kspace[..., shot * offset : (shot + 1) * offset] = motion_kspace[
                ..., shot * offset : (shot + 1) * offset
            ]
        elif trajectory == "interleaved":
            kspace[..., shot::nshots] = motion_kspace[..., shot::nshots]
        else:
            raise ValueError(f"trajectory '{trajectory}' not supported.")

    return kspace
