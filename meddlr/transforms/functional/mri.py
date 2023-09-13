import math
from typing import Sequence, Tuple

import numpy as np
import torch

import meddlr.ops as oF
import meddlr.ops.complex as cplx
from meddlr.forward import SenseModel


def add_even_odd_motion(
    kspace: torch.Tensor,
    scale: float,
    channel_first: bool = False,
    seed=None,
    generator: torch.Generator = None,
):
    assert cplx.is_complex(kspace) or cplx.is_complex_as_real(kspace)

    phase_matrix = torch.zeros(kspace.shape, dtype=torch.complex64, device=kspace.device)
    width = kspace.shape[3] if channel_first else kspace.shape[2]
    g = generator if seed is None else torch.Generator(device=kspace.device).manual_seed(seed)

    odd_err = (2 * np.pi * scale) * torch.rand(
        1, generator=g, device=g.device
    ).cpu().numpy() - np.pi * scale
    even_err = (2 * np.pi * scale) * torch.rand(
        1, generator=g, device=g.device
    ).cpu().numpy() - np.pi * scale
    for line in range(width):
        if line % 2 == 0:
            rand_err = even_err
        else:
            rand_err = odd_err
        phase_error = torch.from_numpy(np.exp(-1j * rand_err)).item()
        if channel_first:
            phase_matrix[:, :, :, line] = phase_error
        else:
            phase_matrix[:, :, line] = phase_error
    aug_kspace = kspace * phase_matrix
    return aug_kspace


def add_affine_motion(
    x: torch.Tensor,
    *,
    transform_gens,
    trajectory: torch.Tensor,
    maps: torch.Tensor = None,
    is_batch: bool = False,
    channels_first: bool = True,
    xtype: str = "image",
) -> torch.Tensor:
    """Simulate 2D motion for multi-shot Cartesian MRI.

    This function supports two trajectories:
        - 'blocked': Where each shot corresponds to a consecutive block of kspace.
          (e.g. 1 1 2 2 3 3)
        - 'interleaved': Where shots are interleaved (e.g. 1 2 3 1 2 3).

    We assume the phase encode direction is left to right
    (i.e. along width dimension).
    TODO: Add support for sensitivity maps.

    Args:
        image: The complex-valued image. Shape [(batch,), height, width, 1].
        transforms: A sequence of random transform generators. These transforms
            will be used to augment images in the image domain. We recommend using
            [RandomTranslation, RandomAffine] in that order. This matches the MRAugment
            augmentation strategy.
        trajectory: The trajectory tensor. Shape [height, width].
        maps: The sensitivity maps. Shape [(batch,) height, width, ncoils, nmaps].
        is_batch: Whether the image (and maps) are batched.
            Note, the trajectory should not be batched. To run each example with
            different trajectories, call this method multiple times (once per example).

    Returns:
        A motion corrupted image.
    """
    from meddlr.transforms.transform import TransformList
    from meddlr.transforms.transform_gen import TransformGen

    if xtype not in ["image", "kspace"]:
        raise ValueError(f"Invalid xtype: {xtype}. Must be one of ['image', 'kspace'].")

    if not channels_first:
        if maps is not None:
            if is_batch:
                dims = (0, maps.ndim - 2, maps.ndim - 1, *range(1, maps.ndim - 2))
            else:
                dims = (maps.ndim - 2, maps.ndim - 1, *range(0, maps.ndim - 2))
            maps = maps.permute(dims)  # Shape: [(batch), ncoils, nmaps, ...]
        x = cplx.channels_first(x)

    def _maps_channels_first(_maps):
        if is_batch:
            dims = (0, *range(3, maps.ndim), 1, 2)
        else:
            dims = (*range(2, maps.ndim), 0, 1)
        return _maps.permute(dims)

    def _to_image(_x, _maps):
        if _maps is None:
            _maps = maps
        if _maps is None:
            return oF.ifft2c(_x)
        else:
            _maps = _maps_channels_first(_maps)
            out = SenseModel(_maps)(cplx.channels_last(_x), adjoint=True)
            return cplx.channels_first(out)

    def _to_kspace(_x, _maps):
        if _maps is None:
            _maps = maps
        if _maps is None:
            return oF.fft2c(_x)
        else:
            _maps = _maps_channels_first(_maps)
            out = SenseModel(_maps)(cplx.channels_last(_x), adjoint=False)
            return cplx.channels_first(out)

    image = x if xtype == "image" else _to_image(x, maps)

    transform_gens: Sequence[TransformGen] = transform_gens

    if maps is None:
        shape = image.shape
    elif is_batch:
        shape = (image.shape[0], maps.shape[1], *image.shape[-2:])
    else:
        shape = (maps.shape[1], *image.shape[-2:])
    kspace = torch.zeros(shape, device=image.device, dtype=image.dtype)

    shot_ids = torch.unique(trajectory)  # the sorted shot ids.
    assert trajectory.shape == kspace.shape[-2:], f"{trajectory.shape} != {kspace.shape[-2:]}"
    trajectory = trajectory.expand_as(kspace)

    for shot_id in shot_ids:
        motion_image = image
        motion_maps = maps
        # Apply sequence of random transforms to the image.
        tfms = TransformList([])
        for tfm_gen in transform_gens:
            tfm = tfm_gen.get_transform(motion_image)
            tfms += tfm
            motion_image = tfm.apply_image(motion_image)
        if motion_maps is not None:
            motion_maps = tfms.apply_image(motion_maps)

        motion_kspace = _to_kspace(motion_image, motion_maps)

        # Replace locations in the kspace with the motion corrupted kspace.
        kspace[trajectory == shot_id] = motion_kspace[trajectory == shot_id]

    if xtype == "kspace":
        return cplx.channels_last(kspace) if not channels_first else kspace
    image = _to_image(kspace, maps)
    return cplx.channels_last(image) if not channels_first else image


def get_multishot_trajectory(
    kind: str, nshots: int, shape: Tuple[int], device="cpu"
) -> torch.Tensor:
    """Build a multi-shot cartesian trajectory.

    This function supports two trajectories:
        - 'blocked': Where each shot corresponds to a consecutive block of kspace.
          (e.g. 1 1 2 2 3 3)
        - 'interleaved': Where shots are interleaved (e.g. 1 2 3 1 2 3).

    Args:
        kind: One of 'interleaved' or 'blocked'.
        nshots: The number of shots in the image.
            This should be equivalent to ceil(phase_encode_dim / echo_train_length).
        shape: The shape of the 2D kspace tensor (height, width).

    Returns:
        torch.Tensor: A categorical tensor of shape [height, width].
            Values range from [0, nshots-1] which correspond to the readouts
            per shot.
    """
    if kind not in ["blocked", "interleaved"]:
        raise ValueError(
            f"trajectory '{kind}' not supported. " "Must be one of 'blocked' or 'interleaved'."
        )

    trajectory = torch.zeros(shape, dtype=torch.long, device=device)
    offset = int(math.ceil(shape[-1] / nshots))
    for shot in range(nshots):
        if kind == "blocked":
            trajectory[..., shot * offset : (shot + 1) * offset] = shot
        elif kind == "interleaved":
            trajectory[..., shot::nshots] = shot
    return trajectory
