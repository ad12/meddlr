import numpy as np
import pytest
import torch

from meddlr.data.transforms.subsample import PoissonDiskMaskFunc, RandomMaskFunc1D


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


@pytest.mark.parametrize("acc", [4, 8])
@pytest.mark.parametrize("cf", [0.04, 0.08])
@pytest.mark.parametrize("shape", [(100, 100), (100, 200)])
@pytest.mark.parametrize("seed", [0, 10000])
def test_random1d_reproducibility(acc, cf, shape, seed):
    shape = (1, *shape)
    a = RandomMaskFunc1D(acc, center_fractions=(cf,))
    b = RandomMaskFunc1D(acc, center_fractions=(cf,))

    a_mask = a(shape, seed=seed, acceleration=acc)
    a_mask2 = a(shape, seed=seed, acceleration=acc)
    b_mask = b(shape, seed=seed, acceleration=acc)

    assert torch.all(a_mask == a_mask2)
    assert torch.all(a_mask == b_mask)

    a_mask = a(shape, seed=seed, acceleration=acc)
    np.random.seed(seed * 2 + 1)
    a_mask2 = a(shape, seed=seed, acceleration=acc)
    assert torch.all(a_mask == a_mask2)


@pytest.mark.parametrize("acc", [4, 8])
@pytest.mark.parametrize("cf", [0.04, 0.08])
@pytest.mark.parametrize("shape", [(100, 100), (100, 200)])
@pytest.mark.parametrize("seed", [0])
def test_random1d_randomness(acc, cf, shape, seed):
    np.random.seed(seed)

    shape = (1, *shape)
    a = RandomMaskFunc1D(acc, center_fractions=(cf,))
    seeds = np.random.randint(0, 2**32, size=100)

    a_mask = a(shape, seed=seed, acceleration=acc)
    for s in seeds:
        a_mask2 = a(shape, seed=s, acceleration=acc)
        if torch.any(a_mask != a_mask2):
            return

    assert False
