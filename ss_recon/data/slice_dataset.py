"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.

This code was adapted from Facebook's fastMRI Challenge codebase:

https://github.com/facebookresearch/fastMRI
"""
import os
from typing import Dict, List

import h5py
from torch.utils.data import Dataset
from torch.utils.data.dataloader import default_collate as _default_collate

__all__ = ["collate_by_supervision", "SliceData"]



def default_collate(batch: list):
    metadata = None
    if any("metadata" in b for b in batch):
        metadata = [b.pop("metadata", None) for b in batch]
    out_dict = _default_collate(batch)
    if metadata is not None:
        out_dict["metadata"] = metadata
    return out_dict


def collate_by_supervision(batch: list):
    """Collate supervised/unsupervised batch examples."""
    supervised = [x for x in batch if not x.get("is_unsupervised", False)]
    unsupervised = [x for x in batch if x.get("is_unsupervised", False)]

    out_dict = {}
    if len(supervised) > 0:
        supervised = default_collate(supervised)
        out_dict["supervised"] = supervised
    if len(unsupervised) > 0:
        unsupervised = default_collate(unsupervised)
        out_dict["unsupervised"] = unsupervised
    assert len(out_dict) > 0
    return out_dict


class SliceData(Dataset):
    """A PyTorch Dataset class that iterates over 2D MR image slices.
    """
    # Default mapping for key types
    _DEFAULT_MAPPING = {
        "kspace": "kspace",
        "maps": "maps",
        "target": "target"
    }

    def __init__(self, dataset_dicts: List[Dict], transform, keys=None, include_metadata=False):
        """
        Args:
            dataset_dicts (List[Dict]): List of dictionaries. Each dictionary
                contains information about a single scan in SSRecon format.
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
            assert all(k in example for k in ["file_name", "is_unsupervised", "fixed_acc"]), (
                f"Example {idx}"
            )

        self.mapping = dict(self._DEFAULT_MAPPING)
        if keys:
            self.mapping.update(keys)
        self._include_metadata = include_metadata

    def _init_examples(self, dataset_dicts):
        examples = []
        for dd in dataset_dicts:
            file_path = dd["file_name"]
            is_unsupervised = dd.get("_is_unsupervised", False)
            acc = dd.get("_acceleration", None)

            if "kspace_size" in dd:
                num_slices = dd["kspace_size"][0]
            elif "num_slices" in dd:
                num_slices = dd["num_slices"]
            else:
                with h5py.File(file_path, "r") as f:
                    num_slices = f["kspace"].shape[0]

            examples.extend([
                {
                    "file_name": file_path,
                    "slice_id": slice_id,
                    "is_unsupervised": is_unsupervised,
                    "fixed_acc": acc,
                }
                for slice_id in range(num_slices)
            ])
        return examples

    def _load_data(self, example, idx):
        file_path = example["file_name"]
        slice_id = example["slice_id"]
        with h5py.File(file_path, "r") as data:
            kspace = data[self.mapping["kspace"]][slice_id]
            maps = data[self.mapping["maps"]][slice_id]
            target = data[self.mapping["target"]][slice_id]

        return {
            "kspace": kspace,
            "maps": maps,
            "target": target,
        }

    def __getitem__(self, i):
        example = self.examples[i]
        file_path, slice_id, is_unsupervised, fixed_acc = tuple(
            example[k]
            for k in ["file_name", "slice_id", "is_unsupervised", "fixed_acc"]
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
            kspace, maps, target, fname, slice_id, is_unsupervised, fixed_acc,
        )

        vals = {
            "kspace": masked_kspace,
            "maps": maps,
            "mean": mean,
            "std": std,
            "norm": norm,
            "is_unsupervised": is_unsupervised,
        }
        if self._include_metadata:
            vals["metadata"] = {"scan_id": fname, "slice_id": slice_id}
        if not is_unsupervised:
            vals["target"] = target
        return vals

    def __len__(self):
        return len(self.examples)

    def get_supervised_idxs(self):
        """Get indices of supervised examples."""
        idxs = [
            idx for idx, x in enumerate(self.examples)
            if not x["is_unsupervised"]
        ]
        return idxs

    def get_unsupervised_idxs(self):
        supervised_idxs = self.get_supervised_idxs()
        unsupervised_idxs = set(range(len(self))) - set(supervised_idxs)
        return sorted(list(unsupervised_idxs))
