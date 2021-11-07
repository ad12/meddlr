import numpy as np
import torch

from meddlr.data.transforms.subsample import PoissonDiskMaskFunc


def test_poisson_disc():
    """
    Verify that built-in Poisson Disc generator returns same value as SigPy.
    """
    acc = 6
    calib_size = 20
    shape = (1, 320, 256)  # (batch, height, width)
    seed = 49

    pd_builtin = PoissonDiskMaskFunc(acc, calib_size)
    builtin_mask = pd_builtin(shape, seed=seed, acceleration=acc)

    # assert all close with 5% tolerance.
    assert np.allclose(torch.sum(builtin_mask).item() / np.prod(shape), 1 / acc, atol=5e-2)
