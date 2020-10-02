"""This file registers pre-defined datasets at hard-coded paths.

In the future, will also include metadata.

We hard-code some paths to the dataset that's assumed to
exist at the user's preferred dataset location (see utils/cluster.py).
"""

import os

from .register_mrco import register_mrco_scans

# ==== Predefined datasets and splits for MRCO formatted datasets ==========

_PREDEFINED_SPLITS_MRCO = {}
_PREDEFINED_SPLITS_MRCO["mridata_knee_2019"] = {
    "mridata_knee_2019_train": (
        "mridata_knee_2019/train",
        "mridata_knee_2019/annotations/train.json",
    ),
    "mridata_knee_2019_val": (
        "mridata_knee_2019/val",
        "mridata_knee_2019/annotations/val.json",
    ),
    "mridata_knee_2019_test": (
        "mridata_knee_2019/test",
        "mridata_knee_2019/annotations/test.json",
    ),
}


_PREDEFINED_SPLITS_MRCO["fastMRI_knee_multicoil"] = {
    "fastMRI_knee_multicoil_dev_train": (
        None,
        "fastmri/knee_multicoil/annotations/vtoy-dev/train.json"
    ),
    "fastMRI_knee_multicoil_dev_val": (
        None,
        "fastmri/knee_multicoil/annotations/vtoy-dev/val.json"
    ),
    "fastMRI_knee_multicoil_dev_test": (
        None,
        "fastmri/knee_multicoil/annotations/vtoy-dev/test.json"
    ),

    "fastMRI_knee_multicoil_v0.0.1_train": (
        "fastmri/knee_multicoil/train",
        "ann://fastmri/knee_multicoil/v0.0.1-dev/train.json"
    ),
    "fastMRI_knee_multicoil_v0.0.1_val": (
        "fastmri/knee_multicoil/train",
        "ann://fastmri/knee_multicoil/v0.0.1-dev/val.json"
    ),
    "fastMRI_knee_multicoil_v0.0.1_test": (
        "fastmri/knee_multicoil/val",
        "ann://fastmri/knee_multicoil/v0.0.1-dev/test.json"
    ),

    "fastMRI_knee_multicoil_mini_v0.0.1_train": (
        "fastmri/knee_multicoil/val",
        "ann://fastmri/knee_multicoil/mini-v0.0.1/train.json"
    ),
    "fastMRI_knee_multicoil_mini_v0.0.1_val": (
        "fastmri/knee_multicoil/val",
        "ann://fastmri/knee_multicoil/mini-v0.0.1/val.json"
    ),
    "fastMRI_knee_multicoil_mini_v0.0.1_test": (
        "fastmri/knee_multicoil/val",
        "ann://fastmri/knee_multicoil/mini-v0.0.1/test.json"
    )
}

_PREDEFINED_SPLITS_MRCO["fastMRI_brain_multicoil"] = {
    "fastMRI_brain_multicoil_dev_train": (
        "fastmri/brain_multicoil/train",
        "ann://fastmri/brain_multicoil/vtoy-dev/train.json"
    ),
    "fastMRI_brain_multicoil_dev_val": (
        "fastmri/brain_multicoil/train",
        "ann://fastmri/brain_multicoil/vtoy-dev/val.json"
    ),
    "fastMRI_brain_multicoil_dev_test": (
        "fastmri/brain_multicoil/val",
        "ann://fastmri/brain_multicoil/vtoy-dev/test.json"
    )
}


def register_all_mrco(root="data://"):
    for dataset_name, splits_per_dataset in _PREDEFINED_SPLITS_MRCO.items():
        for key, (image_root, json_file) in splits_per_dataset.items():
            # Assume pre-defined datasets live in `./datasets`.
            register_mrco_scans(
                key,
                {},  # TODO: add metadata
                os.path.join(root, json_file)
                if "://" not in json_file
                else json_file,  # noqa
                os.path.join(root, image_root)
                if image_root is not None
                else None,
            )


# Register them all under "data://"
register_all_mrco()
