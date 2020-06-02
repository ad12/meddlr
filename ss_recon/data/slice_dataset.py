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
from torch.utils.data.dataloader import default_collate


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

    def __init__(self, dataset_dicts: List[Dict], transform):
        """
        Args:
            dataset_dicts (List[Dict]): List of dictionaries. Each dictionary
                contains information about a single scan in SSRecon format.
            transform (callable): A callable object that pre-processes the
                raw data into appropriate form. The transform function should
                take 'kspace', 'target', 'attributes', 'filename', and 'slice'
                as inputs. 'target' may be null for test data.
        """
        self.transform = transform

        # Convert dataset dict into slices.
        # Each slice is tuple of (file name, slice id, is_unsupervised)
        self.examples = []
        for dd in dataset_dicts:
            file_name = dd["file_name"]
            is_unsupervised = dd.get("_is_unsupervised", False)
            acc = dd.get("_acceleration", None)
            self.examples.extend([
                {
                    "fname": file_name,
                    "slice_id": slice_id,
                    "is_unsupervised": is_unsupervised,
                    "fixed_acc": acc,
                }
                for slice_id in range(dd["kspace_size"][0])
            ])

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        example = self.examples[i]
        fname, slice_id, is_unsupervised, fixed_acc = tuple(
            example[k]
            for k in ["fname", "slice_id", "is_unsupervised", "fixed_acc"]
        )

        # TODO: remove this forced check.
        if not is_unsupervised:
            fixed_acc = None

        with h5py.File(fname, "r") as data:
            kspace = data["kspace"][slice_id]
            maps = data["maps"][slice_id]
            target = data["target"][slice_id]
            # attrs = data.attrs

        fname = os.path.splitext(os.path.basename(fname))[0]
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
        if not is_unsupervised:
            vals["target"] = target
        return vals

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
