import contextlib
import re
import time
from typing import Dict, Iterable, Sequence, Tuple, Union

import h5py
import numpy as np
import torch

from meddlr.utils import profiler

__all__ = ["HDF5Manager", "structure_patches", "collect_mask"]


class HDF5Manager:
    """Manager for large number of HDF5 files.

    HDF5 files are useful for storing large datasets but have a few limitations
    even in the read-only setting:

        1. File opening and closing can be slow
        2. Reading can be volatile for large datasets on NFS filesystems

    Some functionality includes:

        1. **Caching**: This class can be used to keep files open (defaults to read-only mode).
           This eliminates the need to constantly open and close files.
        2. **Retrying**: Data fetching will be retried in the event of volatile NFS connections.

    Attributes:
        files (Dict[str, h5py.File]): Dictionary of open HDF5 files.
        cache (bool): If `True`, keep files files open.
        max_attempts (int): Maximum number of attempts to read from a file.
            This is useful when the connection to a remote file system
            like a network file storage (NFS) can be volatile.
        wait_time (float): Time to wait between attempts to read from a file.
    """

    def __init__(
        self, files: Sequence[str] = None, cache=True, max_attempts=5, wait_time: float = 1.0
    ):
        """
        Args:
            files (Sequence[str], optional): Filepaths to h5 files the manage.
            cache (bool, optional): If `True`, keep files files open once accessed.
            max_attempts (int, optional): Maximum number of attempts to read from a file.
            wait_time (float, optional): Time to wait between attempts to read from a file.
        """
        self.files: Dict[str, h5py.File] = {}
        self.cache = cache
        self.max_attempts = max(1, max_attempts)
        self.wait_time = wait_time
        if files and cache:
            for fp in files:
                self.open(fp)

    def open(self, filepath, mode="r", **kwargs):
        """Open and cache file.

        Args:
            filepath (str): Path to HDF5 file
        """
        if filepath in self.files:
            return
        self.files[filepath] = h5py.File(filepath, mode, **kwargs)

    def close(self, filepath, drop=True):
        """Close the open file.

        Args:
            filepath (str): Path to HDF5 file
            drop (bool, optional): If `True`, drop filepath from list of files.
                If `False`, `self.files` will keep a reference to the closed file.
        """
        self.files[filepath].close()
        if drop:
            self.files.pop(filepath)

    @profiler.time_profile()
    def get(self, filepath: str, key: str = None, patch: Union[str, Sequence[slice]] = None):
        """Get dataset from h5 file.

        Args:
            filepath (str): Filepath to fetch data from.
            key (str, optional): HDF5 key. If `None`, returns open file.
            patch (str or Sequence[slice], optional): Slice to apply to dataset.
                If `None`, returns dataset without slicing.
        """
        if patch is not None and key is None:
            raise ValueError("`key` must be specified to use `patch`.")

        for idx in range(self.max_attempts):
            try:
                if filepath in self.files:
                    file = self.files[filepath]
                    return self._load_data(file, key=key, sl=patch)
                else:
                    assert key
                    with h5py.File(filepath, "r") as file:
                        data = self._load_data(file, key=key, sl=patch)
                    return data
            except (OSError, KeyError) as e:
                # Handle input/output errors by waiting and retrying.
                # This issue is common for NFS mounted file systems.
                # https://github.com/theislab/scanpy/issues/1351#issuecomment-668009684
                matches_error = any(
                    (re.search(pattern, str(e)) is not None)
                    for pattern in ["\[Errno 5\]", "errno = 5"]
                )
                if (idx < self.max_attempts - 1) and matches_error:
                    is_cached = filepath in self.files
                    if is_cached:
                        self.close(filepath)
                        time.sleep(self.wait_time)
                        self.open(filepath)
                    else:
                        time.sleep(self.wait_time)
                else:
                    raise e

    @profiler.time_profile()
    @contextlib.contextmanager
    def File(self, filepath, mode="r", **kwargs):
        """Yields HDF5 file.

        Similar to ``with h5py.File(...)`` but with supported caching.
        If caching is enabled, the file will be opened (if it is not already
        open) and kept open when the context is exited.

        Example:

        .. code-block:: python

            h5manager = HDF5Manager()
            fpath = "file.h5"
            with h5manager.File(fpath, "r") as h5file:
                arr = h5file["my_array"][()]
        """
        is_cached = filepath in self.files
        file = self.files[filepath] if is_cached else h5py.File(filepath, mode, **kwargs)
        if self.cache and not is_cached:
            self.files[filepath] = file

        try:
            yield file
        finally:
            if not is_cached and not self.cache:
                file.close()

    yield_file = File

    @profiler.time_profile()
    @contextlib.contextmanager
    def temp_open(self, filepath, mode="r", **kwargs) -> "HDF5Manager":
        """Temporarily opens file in this manager.

        If the file is already opened, this open file will be used.

        Yields:
            HDF5Manager: This HDF5Manager (i.e. ``self``).

        Example:

        .. code-block:: python

            h5manager = HDF5Manager()
            fpath = "file.h5"
            with h5manager.temp_open(fpath, "r"):
                arr = h5manager.get(fpath, "my_array", ())
        """
        is_cached = filepath in self.files
        if not is_cached:
            self.open(filepath, mode, **kwargs)

        try:
            yield self
        finally:
            if not is_cached:
                self.close(filepath)

    def _load_data(self, file, key=None, sl=None):
        """Load data from HDF5 file.

        If ``key`` and ``sl`` are not passed, the file is returned.

        If ``key`` is passed, the dataset is returned without loading
        the underlying data into memory. Equivalent to ``file[key]``.

        If ``sl`` is passed, the dataset is returned after slicing.
        Equivalent to ``file[key][sl]``.

        """
        if not key:
            return file

        val = file[key]
        if sl is None:
            return val

        if isinstance(sl, str):
            assert sl in ["all", "load"]
            sl = ()
        elif isinstance(sl, Iterable):
            sl = tuple(sl)
        return val[sl]

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

    struct = np.empty(stack_shape, dtype="object")
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
