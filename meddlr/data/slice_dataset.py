import os
from collections import defaultdict
from typing import Dict, List

import h5py
import numpy as np
from torch.utils.data import Dataset

from meddlr.data.transforms.transform import DataTransform

__all__ = ["SliceData"]


class SliceData(Dataset):
    """A PyTorch Dataset class that iterates over 2D slices for MRI reconstruction.

    This dataset handles per-slice data iteration for MRI reconstruction problems.
    This means that slices of K-space, target images, and sensitivity maps (if applicable)
    are returned in ``__getitem__``.

    Because slices are the atomic unit in this dataset, we refer to them as the *examples*.
    This dataset splits a single scan (i.e. a dataset dictionary) into multiple slices
    (called *examples*) along the 0-th axis. ``self.transform`` is used to
    execute transforms (e.g. undersampling, intensity normalization, etc.) on the data.

    This dataset is also useful for organizing and fetching supervised and unsupervised
    examples. This functionality is used with :cls:`AlternatingSampler` and
    :cls:`GroupAlternatingSampler`.

    While this class is designed to be flexible for easy interation with different datasets
    (e.g. fastMRI, mridata 2D/3D FSE, SKM-TEA), there are some nuances in how the data must
    be formatted. You can find more information at in the Prepare Your Own Dataset section
    of the documentation (documentation in progress).

    This class was adapted from https://github.com/facebookresearch/fastMRI.
    """

    # Default mapping for key types.
    _DEFAULT_MAPPING = {"kspace": "kspace", "maps": "maps", "target": "target"}
    _REQUIRED_METADATA = ("file_name", "is_unsupervised", "fixed_acc")

    def __init__(self, dataset_dicts: List[Dict], transform, keys=None, include_metadata=False):
        """
        Args:
            dataset_dicts (List[Dict]): List of dictionaries. Each dictionary
                contains information about a single scan in Meddlr format.
            transform (callable): A callable object that pre-processes the
                raw data into appropriate form. The transform function should
                take 'kspace', 'target', 'attributes', 'filename', and 'slice'
                as inputs. 'target' may be null for test data.
            include_metadata (bool, optional): If `True`, includes scan metadata:
                - "scan_id"
                - "slice_id"
        """
        self.transform = transform

        # Convert dataset dict into slices.
        # Each slice is tuple of (file name, slice id, is_unsupervised)
        self.examples = self._init_examples(dataset_dicts)

        # All examples should have the following keys:
        #   - fname (str): The file name
        #   - is_unsupervised (bool): If `True`, the example should be treated as unsupervised
        #   - fixed_acc (float): The fixed acceleration for unsupervised examples.
        #   - Any other keys required for data loading.
        for idx, example in enumerate(self.examples):
            assert all(k in example for k in self._REQUIRED_METADATA), f"Example {idx}"

        self.mapping = dict(self._DEFAULT_MAPPING)
        if keys:
            self.mapping.update(keys)
        self._include_metadata = include_metadata

    def groups(self, group_by):
        _groups = defaultdict(list)
        for idx, example in enumerate(self.examples):
            try:
                _groups[example[group_by]].append(idx)
            except KeyError:
                raise KeyError(f"Key {group_by} not found. Use one of {example.keys()}")
        return _groups

    def _init_examples(self, dataset_dicts):
        examples = []
        for dd in dataset_dicts:
            file_path = dd["file_name"]
            is_unsupervised = dd.get("_is_unsupervised", False)
            acc = dd.get("_acceleration", None)

            if "kspace_size" in dd:
                num_slices = dd["kspace_size"][0]
                shape = dd["kspace_size"][1:3]
            elif "num_slices" in dd:
                num_slices = dd["num_slices"]
                with h5py.File(file_path, "r") as f:
                    shape = f["kspace"].shape[1:3]
            else:
                with h5py.File(file_path, "r") as f:
                    num_slices = f["kspace"].shape[0]
                    shape = f["kspace"].shape[0]

            examples.extend(
                [
                    {
                        "file_name": file_path,
                        "slice_id": slice_id,
                        "is_unsupervised": is_unsupervised,
                        "fixed_acc": acc,
                        "_metadata": dd.get("_metadata", {}),
                        "inplane_shape": shape,
                    }
                    for slice_id in range(num_slices)
                ]
            )
        return examples

    def _load_data(self, example, idx):
        file_path = example["file_name"]
        slice_id = example["slice_id"]
        with h5py.File(file_path, "r") as data:
            kspace = data[self.mapping["kspace"]][slice_id]
            target = data[self.mapping["target"]][slice_id]
            maps = (
                np.zeros_like(target)
                if self.mapping["target"] == "reconstruction_rss"
                else data[self.mapping["maps"]][slice_id]
            )

        return {"kspace": kspace, "maps": maps, "target": target}

    def __getitem__(self, i):
        example = self.examples[i]
        file_path, slice_id, is_unsupervised, fixed_acc = tuple(
            example[k] for k in ["file_name", "slice_id", "is_unsupervised", "fixed_acc"]
        )

        # TODO: remove this forced check.
        if not is_unsupervised:
            fixed_acc = None

        data = self._load_data(example, i)
        kspace = data["kspace"]
        maps = data["maps"]
        target = data["target"]

        fname = os.path.splitext(os.path.basename(file_path))[0]
        masked_kspace, maps, target, mean, std, norm = self.transform(
            kspace, maps, target, fname, slice_id, is_unsupervised, fixed_acc
        )

        vals = {
            "kspace": masked_kspace,
            "maps": maps,
            "mean": mean,
            "std": std,
            "norm": norm,
            "is_unsupervised": is_unsupervised,
        }
        if (
            hasattr(self.transform, "augmentor")
            and isinstance(self.transform, DataTransform)
            and self.transform.augmentor is not None
        ):
            scheduler_params = self.transform.augmentor.get_tfm_gen_params()
            if len(scheduler_params):
                vals["metrics"] = {"scheduler": scheduler_params}
        if self._include_metadata:
            vals["metadata"] = {"scan_id": fname, "slice_id": slice_id}
        if not is_unsupervised:
            vals["target"] = target
        return vals

    def __len__(self):
        return len(self.examples)

    def get_supervised_idxs(self) -> List[int]:
        """Returns indices corresponding to supervised examples.

        These indices can be used with ``__getitem__`` to fetch supervised examples.

        Returns:
            List[int]: The indices corresponding to supervised examples.
        """
        idxs = [idx for idx, x in enumerate(self.examples) if not x["is_unsupervised"]]
        return idxs

    def get_unsupervised_idxs(self):
        """Returns indices corresponding to unsupervised examples.

        These indices can be used with ``__getitem__`` to fetch unsupervised examples.

        Returns:
            List[int]: The indices corresponding to unsupervised examples.
        """
        supervised_idxs = self.get_supervised_idxs()
        unsupervised_idxs = set(range(len(self))) - set(supervised_idxs)
        return sorted(unsupervised_idxs)
