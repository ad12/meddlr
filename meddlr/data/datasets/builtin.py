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
    "mridata_knee_2019_train": ("mridata_knee_2019/train", "ann://mridata_knee_2019/train.json"),
    "mridata_knee_2019_val": ("mridata_knee_2019/val", "ann://mridata_knee_2019/val.json"),
    "mridata_knee_2019_test": ("mridata_knee_2019/test", "ann://mridata_knee_2019/test.json"),
}


_PREDEFINED_SPLITS_MRCO["fastMRI_knee_multicoil"] = {
    # fastMRI knee multicoil toy dataset - do not use.
    "fastMRI_knee_multicoil_dev_train": (
        None,
        "fastmri/knee_multicoil/annotations/vtoy-dev/train.json",
    ),
    "fastMRI_knee_multicoil_dev_val": (
        None,
        "fastmri/knee_multicoil/annotations/vtoy-dev/val.json",
    ),
    "fastMRI_knee_multicoil_dev_test": (
        None,
        "fastmri/knee_multicoil/annotations/vtoy-dev/test.json",
    ),
    # fastMRI knee multicoil base v0.0.1 dataset.
    "fastMRI_knee_multicoil_v0.0.1_train": (
        "fastmri/knee_multicoil/train",
        "ann://fastmri/knee_multicoil/v0.0.1-dev/train.json",
    ),
    "fastMRI_knee_multicoil_v0.0.1_val": (
        "fastmri/knee_multicoil/train",
        "ann://fastmri/knee_multicoil/v0.0.1-dev/val.json",
    ),
    "fastMRI_knee_multicoil_v0.0.1_test": (
        "fastmri/knee_multicoil/val",
        "ann://fastmri/knee_multicoil/v0.0.1-dev/test.json",
    ),
    # fastMRI knee multicoil mini v0.0.1 dataset.
    "fastMRI_knee_multicoil_mini_v0.0.1_train": (
        "fastmri/knee_multicoil/val",
        "ann://fastmri/knee_multicoil/mini-v0.0.1/train.json",
    ),
    "fastMRI_knee_multicoil_mini_v0.0.1_val": (
        "fastmri/knee_multicoil/val",
        "ann://fastmri/knee_multicoil/mini-v0.0.1/val.json",
    ),
    "fastMRI_knee_multicoil_mini_v0.0.1_test": (
        "fastmri/knee_multicoil/val",
        "ann://fastmri/knee_multicoil/mini-v0.0.1/test.json",
    ),
    # fastMRI knee multicoil ultra mini v0.0.1 dataset.
    "fastMRI_knee_multicoil_ultra_mini_v0.0.1_train": (
        "fastmri/knee_multicoil/val",
        "ann://fastmri/knee_multicoil/ultra-mini-v0.0.1/train.json",
    ),
    "fastMRI_knee_multicoil_ultra_mini_v0.0.1_val": (
        "fastmri/knee_multicoil/val",
        "ann://fastmri/knee_multicoil/ultra-mini-v0.0.1/val.json",
    ),
    "fastMRI_knee_multicoil_ultra_mini_v0.0.1_test": (
        "fastmri/knee_multicoil/val",
        "ann://fastmri/knee_multicoil/ultra-mini-v0.0.1/test.json",
    ),
    # fastMRI knee multicoil - fat-suppressed
    "fastMRI_knee_multicoil_v0.0.1_fs_train": (
        "fastmri/knee_multicoil/train",
        "ann://fastmri/knee_multicoil/v0.0.1-dev-fs/train.json",
    ),
    "fastMRI_knee_multicoil_v0.0.1_fs_val": (
        "fastmri/knee_multicoil/train",
        "ann://fastmri/knee_multicoil/v0.0.1-dev-fs/val.json",
    ),
    "fastMRI_knee_multicoil_v0.0.1_fs_test": (
        "fastmri/knee_multicoil/val",
        "ann://fastmri/knee_multicoil/v0.0.1-dev-fs/test.json",
    ),
    # fastMRI knee multicoil - 3T fat-suppressed
    "fastMRI_knee_multicoil_v0.0.1_fs_3t_train": (
        "fastmri/knee_multicoil/train",
        "ann://fastmri/knee_multicoil/v0.0.1-dev-fs-3t/train.json",
    ),
    "fastMRI_knee_multicoil_v0.0.1_fs_3t_val": (
        "fastmri/knee_multicoil/train",
        "ann://fastmri/knee_multicoil/v0.0.1-dev-fs-3t/val.json",
    ),
    "fastMRI_knee_multicoil_v0.0.1_fs_3t_test": (
        "fastmri/knee_multicoil/val",
        "ann://fastmri/knee_multicoil/v0.0.1-dev-fs-3t/test.json",
    ),
}

_PREDEFINED_SPLITS_MRCO["fastMRI_brain_multicoil"] = {
    "fastMRI_brain_multicoil_dev_train": (
        "fastmri/brain_multicoil/train",
        "ann://fastmri/brain_multicoil/vtoy-dev/train.json",
    ),
    "fastMRI_brain_multicoil_dev_val": (
        "fastmri/brain_multicoil/train",
        "ann://fastmri/brain_multicoil/vtoy-dev/val.json",
    ),
    "fastMRI_brain_multicoil_dev_test": (
        "fastmri/brain_multicoil/val",
        "ann://fastmri/brain_multicoil/vtoy-dev/test.json",
    ),
    # Mini split
    "fastMRI_brain_multicoil_mini_v0.0.1_train": (
        "fastmri/brain_multicoil/val",
        "ann://fastmri/brain_multicoil/mini-v0.0.1/train.json",
    ),
    "fastMRI_brain_multicoil_mini_v0.0.1_val": (
        "fastmri/brain_multicoil/val",
        "ann://fastmri/brain_multicoil/mini-v0.0.1/val.json",
    ),
    "fastMRI_brain_multicoil_mini_v0.0.1_test": (
        "fastmri/brain_multicoil/val",
        "ann://fastmri/brain_multicoil/mini-v0.0.1/test.json",
    ),
}


_METADATA_FILES = {
    "mridata_knee_2019": "ann://mridata_knee_2019/metadata.csv",
    "fastMRI_knee_multicoil": "ann://fastmri/knee_multicoil/fastmri_knee_multicoil_metadata.csv",
    "fastMRI_brain_multicoil": "ann://fastmri/brain_multicoil/fastmri_brain_multicoil_metadata.csv",
}


def register_all_mrco(root="data://"):
    for dataset_name, splits_per_dataset in _PREDEFINED_SPLITS_MRCO.items():
        metadata_file = _METADATA_FILES.get(dataset_name, None)
        for key, data in splits_per_dataset.items():
            # Assume pre-defined datasets live in `./datasets`.
            image_root, json_file = data

            json_file = os.path.join(root, json_file) if "://" not in json_file else json_file
            image_root = os.path.join(root, image_root) if image_root is not None else None
            register_mrco_scans(
                key,
                {"metadata_file": metadata_file},  # TODO: add metadata
                json_file,  # noqa
                image_root,
            )


# Register them all under "data://"
register_all_mrco()
