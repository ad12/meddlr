import torch

from ss_recon.data.data_utils import structure_patches


def test_structuring_patches():
    # Single dimension
    d1 = 5
    shape = (6, 7, 8)
    patches = [torch.randn(*shape) for _ in range(d1)]
    coords = tuple(range(0, d1))

    expected = torch.stack(patches, dim=0)
    out = structure_patches(patches, coords)
    assert out.shape == (d1,) + shape
    assert torch.all(out == expected)

    expected = torch.stack(patches, dim=1)
    out = structure_patches(patches, coords, dims=1)
    assert (
        out.shape
        == (
            shape[0],
            d1,
        )
        + shape[1:]
    )
    assert torch.all(out == expected)

    # Two dimensions
    d1, d2 = 2, 5
    shape = (6, 7, 8)
    base_patches = [[torch.randn(*shape) for _ in range(d2)] for _ in range(d1)]
    patches = [y for x in base_patches for y in x]
    coords = [(x, y) for x in range(d1) for y in range(d2)]

    expected = torch.stack([torch.stack(y, dim=0) for y in base_patches], dim=0)
    out = structure_patches(patches, coords)
    assert torch.all(out == expected)

    expected = torch.stack([torch.stack(y, dim=1) for y in base_patches], dim=0)
    out = structure_patches(patches, coords, dims=(0, 2))
    assert out.shape == expected.shape
    assert out.shape == (2, 6, 5, 7, 8)
    assert torch.all(out == expected)
