"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.

This code was adapted from Facebook's fastMRI Challenge codebase:

https://github.com/facebookresearch/fastMRI
"""

from typing import List, Dict
import h5py
from torch.utils.data import Dataset


class SliceData(Dataset):
    """A PyTorch Dataset class that iterates over 2D MR image slices.
    """

    def __init__(self, dataset_dicts: List[Dict], transform):
        """
        Args:
            dataset_dicts (List[Dict]): List of dictionaries. Each dictionary
                contains information about a single scan in SSRecon format.
            transform (callable): A callable object that pre-processes the raw data into
                appropriate form. The transform function should take 'kspace', 'target',
                'attributes', 'filename', and 'slice' as inputs. 'target' may be null
                for test data.
        """

        self.transform = transform

        # Convert dataset dict into slices.
        self.examples = []
        for dd in dataset_dicts:
            self.examples.extend([
                (dd["file_name"], slice_id)
                for slice_id in range(dd["kspace_size"][0])
            ])

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        fname, slice = self.examples[i]
        with h5py.File(fname, "r") as data:
            kspace = data["kspace"][slice]
            maps = data["maps"][slice]
            target = data["target"][slice]

        return self.transform(
            kspace, maps, target, data.attrs, fname.name, slice
        )
