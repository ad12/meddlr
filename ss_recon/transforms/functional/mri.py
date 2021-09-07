import numpy as np
import torch

import ss_recon.utils.complex_utils as cplx


def add_even_odd_motion(kspace, scale: float, seed=None, generator: torch.Generator = None):
    assert cplx.is_complex(kspace) or cplx.is_complex_as_real(kspace)

    phase_matrix = torch.zeros(kspace.shape, dtype=torch.complex64)
    width = kspace.shape[2]
    g = generator if seed is None else torch.Generator().manual_seed(seed)

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
    return aug_kspace
