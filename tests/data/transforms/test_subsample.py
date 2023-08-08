from typing import Tuple

import numpy as np
import pytest
import torch

from meddlr.data.transforms.subsample import (
    EquispacedMaskFunc1D,
    PoissonDiskMaskFunc,
    RandomMaskFunc,
    RandomMaskFunc1D,
    _get_center,
    get_cartesian_edge_mask,
)


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


def test_get_cartesian_edge_mask_1d():
    shape = (1, 10, 8)  # (batch, height, width)

    kspace = torch.randn(shape, dtype=torch.complex64)
    kspace[:, :, :2] = 0
    kspace[:, :, 5:] = 0
    mask = get_cartesian_edge_mask(kspace, dims=2)
    assert mask.shape == shape
    assert torch.all(mask[:, :, :2] == 1)
    assert torch.all(mask[:, :, 5:] == 1)
    assert torch.all(mask[:, :, 2:5] == 0)

    kspace = torch.randn(shape, dtype=torch.complex64)
    kspace[:, :3, :] = 0
    kspace[:, 8:, :] = 0
    mask = get_cartesian_edge_mask(kspace, dims=1)
    assert mask.shape == shape
    assert torch.all(mask[:, :3, :] == 1)
    assert torch.all(mask[:, 8:, :] == 1)
    assert torch.all(mask[:, 3:8, :] == 0)


@pytest.mark.parametrize("dims", [(1, 2), (-1, -2)])
def test_get_cartesian_edge_mask_2d(dims):
    shape = (1, 10, 8)  # (batch, height, width)

    kspace = torch.randn(shape, dtype=torch.complex64)
    kspace[:, :3, :] = 0
    kspace[:, 8:, :] = 0
    kspace[:, :, :2] = 0
    kspace[:, :, 5:] = 0
    mask = get_cartesian_edge_mask(kspace, dims=dims)
    assert mask.shape == shape
    assert torch.all(mask[:, :3, :] == 1)
    assert torch.all(mask[:, 8:, :] == 1)
    assert torch.all(mask[:, :, :2] == 1)
    assert torch.all(mask[:, :, 5:] == 1)
    assert torch.all(mask[:, 3:8, 2:5] == 0)


@pytest.mark.parametrize("accelerations", [4, [4], (4, 5)])
def test_choose_acceleration(accelerations):
    mask_func = RandomMaskFunc(accelerations, calib_size=20)
    acc = mask_func.choose_acceleration()

    if isinstance(accelerations, int):
        assert acc == accelerations
    elif len(accelerations) == 1:
        assert acc == accelerations[0]
    else:
        assert acc >= accelerations[0] and acc <= accelerations[1]


def test_random_mask_func_center_fractions_error():
    """Test RandomMaskFunc raises and error when center fraction is specified."""
    with pytest.raises(ValueError):
        RandomMaskFunc(accelerations=4, calib_size=20, center_fractions=0.08)


def test_equispaced1d():
    accelerations = 4.5
    mask_func = EquispacedMaskFunc1D(accelerations, calib_size=1)

    # Equispaced sampling factor must be an integer.
    assert mask_func.choose_acceleration() == 4

    shape = (1, 50, 50)
    mask = mask_func(shape=shape)
    assert mask.shape == shape

    mask = mask.squeeze()

    # In 1D undersampling, every row in a column must have the same value.
    # i.e. All 0 or 1
    mask_sum = mask.sum(dim=0).bool()
    assert torch.all((~mask_sum) ^ (mask_sum))

    # Test that the samples are equispaced.
    # For now, the offset is 0, so we can start the count from the top.
    # Every 4th line should be 1.
    # All other lines should 0, except for the center fraction line.
    # In this case the center fraction occurs at line 25, which we account for
    # in the second assert.
    assert torch.all(mask_sum[0::4])
    assert torch.sum(mask_sum[1::4]) == 1 and mask_sum[25] == 1
    assert not torch.any(mask_sum[2::4])
    assert not torch.any(mask_sum[3::4])
