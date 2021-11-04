import numpy as np
import torch

import meddlr.ops.complex as cplx


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
