import h5py
import numpy as np
import torch

from meddlr.data.data_utils import HDF5Manager, structure_patches


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


def test_hdf5_manager(tmpdir):
    N = 5

    files = [tmpdir / f"file_{i:03d}.h5" for i in range(N)]
    data = [np.random.randn(10, 10) for _ in range(N)]
    for idx, fpath in enumerate(files):
        with h5py.File(fpath, "w") as f:
            f.create_dataset("data", data=data[idx])

    # Auto file open/close.
    data_manager = HDF5Manager(files)
    assert len(data_manager.files) == N
    assert all(isinstance(f, h5py.File) for f in data_manager.files.values())
    all_files = data_manager.files.copy()
    del data_manager
    assert all(not f.id for f in all_files.values()), "All files should be closed"

    data_manager = HDF5Manager(files)
    for fpath in files:
        file = data_manager.get(fpath)
        assert file is data_manager.files[fpath]

    idx = np.random.choice(N)
    fpath = files[idx]
    out = data_manager.get(fpath, "data")
    assert np.all(out == data[idx])

    idx = np.random.choice(N)
    sl = (slice(None), slice(5))
    fpath = files[idx]
    out = data_manager.get(fpath, "data", sl)
    assert np.all(out == data[idx][sl])
