import contextlib
from typing import Dict, Sequence, Tuple, Union

import h5py
import numpy as np
import torch

from meddlr.utils import profiler

__all__ = ["HDF5Manager", "structure_patches", "collect_mask"]


class HDF5Manager:
    """Manager for opening and caching HDF5 files."""

    def __init__(self, files: Sequence[str] = None, cache=True):
        self.files: Dict[str, h5py.File] = {}
        self.cache = cache
        if files and cache:
            for fp in files:
                self.open(fp)

    def open(self, filepath):
        """Open and cache file in read mode.
        Args:
            filepath (str): Path to HDF5 file
        """
        if filepath in self.files:
            return
        self.files[filepath] = h5py.File(filepath, "r")

    def close(self, filepath, drop=True):
        """Open and cache file in read mode.
        Args:
            filepath (str): Path to HDF5 file
            drop (bool, optional): If `True`, drop filepath from list of files.
                If `False`, `self.files` will keep a closed copy of the file.
        """
        self.files[filepath].close()
        if drop:
            self.files.pop(filepath)

    @profiler.time_profile()
    def get(self, filepath: str, key: str = None, patch: Union[str, Sequence[slice]] = None):
        """Get data from h5 file.
        Args:
            filepath (str): Filepath to fetch data from.
            key (str, optional): HDF5 key. If `None`, returns open file.
            patch ()
        """

        if patch is not None and key is None:
            raise ValueError("`key` must be specified to use `patch`.")

        if filepath in self.files:
            file = self.files[filepath]
            return self._load_data(file, key=key, patch=patch)
        else:
            assert key
            with h5py.File(filepath, "r") as file:
                data = self._load_data(file, key=key, patch=patch)
            return data

    @profiler.time_profile()
    @contextlib.contextmanager
    def yield_file(self, filepath):
        is_cached = filepath in self.files
        file = self.files[filepath] if is_cached else h5py.File(filepath, "r")
        if self.cache and not is_cached:
            self.files[filepath] = file

        try:
            yield file
        finally:
            if not is_cached and not self.cache:
                file.close()

    def _load_data(self, file, key=None, patch=None):
        if not key:
            return file

        val = file[key]
        if patch is None:
            return val

        if isinstance(patch, str):
            assert patch in ["all", "load"]
            patch = ()
        elif not isinstance(patch, tuple):
            patch = tuple(patch)
        return val[patch]

    def __del__(self):
        # Copy keys to allow modifying self.files dict.
        for filepath in list(self.files.keys()):
            self.close(filepath)


def structure_patches(
    patches: Sequence[torch.Tensor],
    coords: Union[Sequence[int], Sequence[Tuple[int]]] = None,
    dims=None,
):
    if coords is None:
        coords = list(patches.keys())
        patches = list(patches.values())

    if len(patches) != len(coords):
        raise ValueError("Got unknown length for coordinates")

    if isinstance(coords[0], int):
        coords = [(x,) for x in coords]

    if len(coords) != len(set(coords)):
        raise ValueError(
            "All coordinates must be unique. " "Overlapping patches are not currently supported."
        )

    if dims is not None:
        if isinstance(dims, int):
            dims = (dims,)
        if len(dims) != len(coords[0]):
            raise ValueError(
                f"Specified {len(dims)} dimensions, but have {len(coords[0])}-d coordinates"
            )
        if len(dims) != len(set(dims)):
            raise ValueError(f"Duplicate dimensions specified - {dims}")

    base_shape = patches[0].shape
    if any(x.shape != base_shape for x in patches):
        raise ValueError("All patches must have the same shape")

    stack_shape = []
    coords_arr = np.asarray(coords)
    assert coords_arr.ndim == 2
    stack_shape = tuple(np.max(coords_arr[:, c]).item() + 1 for c in range(coords_arr.shape[1]))

    struct = np.empty(stack_shape, dtype=np.object)
    for coord, patch in zip(coords, patches):
        struct[coord] = patch
    if any(x is None for x in struct.flatten()):
        raise ValueError("Did not get dense labels.")
    struct = struct.tolist()

    out = _recursive_stack(struct)

    if dims is not None:
        # Reorder dimensions of the structured tensor.
        dims = [out.ndim + d if d < 0 else d for d in dims]
        original_dims_new_order = sorted(set(range(out.ndim)) - set(dims))
        order_dict = {v: k for k, v in zip(range(out.ndim), dims + original_dims_new_order)}
        out = out.permute(tuple(order_dict[k] for k in sorted(order_dict.keys())))

    return out


def collect_mask(
    mask: np.ndarray,
    index: Sequence[Union[int, Sequence[int], int]],
    out_channel_first: bool = True,
):
    """Collect masks by index.

    Collated indices will be summed. For example, `index=(1,(3,4))` will return
    `np.stack(mask[...,1], mask[...,3]+mask[...,4])`.

    TODO: Add support for adding background.

    Args:
        mask (ndarray): A (...)xC array.
        index (Sequence[int]): The index/indices to select in mask.
            If sub-indices are collated, they will be summed.
        out_channel_first (bool, optional): Reorders dimensions of output mask to Cx(...)
    """
    if isinstance(index, int):
        index = (index,)

    if not any(isinstance(idx, Sequence) for idx in index):
        mask = mask[..., index]
    else:
        o_seg = []
        for idx in index:
            c_seg = mask[..., idx]
            if isinstance(idx, Sequence):
                c_seg = np.sum(c_seg, axis=-1)
            o_seg.append(c_seg)
        mask = np.stack(o_seg, axis=-1)

    if out_channel_first:
        last_idx = len(mask.shape) - 1
        mask = np.transpose(mask, (last_idx,) + tuple(range(0, last_idx)))

    return mask


def _recursive_stack(tensors) -> torch.Tensor:
    if all(isinstance(x, torch.Tensor) for x in tensors):
        return torch.stack(tensors, dim=0)
    if all(isinstance(x, list) for x in tensors):
        tensors = [_recursive_stack(x) for x in tensors]
        return torch.stack(tensors, dim=0)
    else:
        raise ValueError("Unknown stack")
