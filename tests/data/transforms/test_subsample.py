from typing import Tuple

import numpy as np
import pytest
import torch

from meddlr.data.transforms.subsample import PoissonDiskMaskFunc, RandomMaskFunc1D, _get_center


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


@pytest.mark.parametrize("shape", [(100, 100), (320, 256)])
@pytest.mark.parametrize("use_out_shape", [False, True])
def test_poisson_disc_get_edge_mask(shape: Tuple[int], use_out_shape: bool):
    """Verify the edge computation for the poisson disc method.

    This should work with both square and rectangular images.

    """
    acc = 6
    calib_size = 20
    height, width = shape
    shape = (1, height, width)  # (batch, height, width)

    kspace = torch.randn(shape, dtype=torch.complex64)
    masker = PoissonDiskMaskFunc(acc, calib_size)
    out_shape = (*shape, 1, 1) if use_out_shape else None
    edge_mask = masker.get_edge_mask(kspace, out_shape=out_shape)
    if use_out_shape:
        assert edge_mask.shape == out_shape
    else:
        assert edge_mask.shape == kspace.shape
    edge_mask = edge_mask.squeeze()

    # Points outside of ellipse should be 1.
    coordinates = torch.where(edge_mask == 1)
    coordinates = torch.stack(coordinates, axis=1)  # Shape: (N, 2)
    radius = _elliptical_radius(coordinates, (height, width))
    assert torch.all(radius > 1)

    # Points inside/on the ellipse should be 0.
    coordinates = torch.where(edge_mask == 0)
    coordinates = torch.stack(coordinates, axis=1)  # Shape: (N, 2)
    radius = _elliptical_radius(coordinates, (height, width))
    assert torch.all(radius <= 1)


def _elliptical_radius(coordinates: torch.Tensor, shape: Tuple[int]) -> torch.Tensor:
    """Compute the elliptical radius for a given coordinate.

    Args:
        coordinates: The coordinates. Shape: (N, dims)
        shape: The shape of matrix. Length should equal ``dims``.

    Returns:
        torch.Tensor: The radius. Shape: (N,)
    """
    assert len(shape) == coordinates.shape[1]
    center = torch.tensor([_get_center(x) for x in shape])
    # The centered coordinates.
    coords = (coordinates - center) ** 2
    radius = torch.sum(coords / (torch.as_tensor(shape) / 2) ** 2, axis=1)
    return radius


def test_get_center():
    """Verify that the zero-indexed center is returned."""
    assert _get_center(3) == 1
    assert _get_center(4) == 1.5


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
