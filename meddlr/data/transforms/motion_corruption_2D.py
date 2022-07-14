import math
from typing import Optional

import torch

import meddlr.ops as F
from meddlr.transforms import RandomAffine


def add_motion_corruption(
    image: torch.Tensor,
    nshots: int,
    translation: Optional[RandomAffine] = None,
    trajectory: str = "blocked",
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
        translation: This is the translation to augment images in the image
            domain. This is either 'None' or 'RandomAffine' for now.
        trajectory: One of 'interleaved' or 'consecutive'.

    Returns:
        A motion corrupted image.
    """
    kspace = torch.zeros_like(image)
    offset = int(math.ceil(kspace.shape[-1] / nshots))

    for shot in range(nshots):
        motion_image = translation.get_transform(image).apply_image(image)
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
