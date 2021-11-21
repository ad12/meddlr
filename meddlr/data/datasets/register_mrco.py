"""Functions to register a MRCO-format dataset to the DatasetCatalog.
"""
import json
import logging
import os
import time

import h5py
import pandas as pd

from meddlr.data import DatasetCatalog, MetadataCatalog
from meddlr.utils import env

logger = logging.getLogger(__name__)
_PATH_MANAGER = env.get_path_manager()

__all__ = ["load_mrco_json", "register_mrco_scans"]


def load_mrco_json(json_file: str, image_root: str, dataset_name: str):
    """Load a json file with MRCO's scan annotation format.

    Currently supports reconstruction.

    Args:
        json_file (str): Full path to the json file in MRCO scan annotation
            format.
        image_root (str): The directory where the images in this json file
            exist.
        dataset_name (str): The name of the dataset
            (e.g. mridata_knee_2019_train).
        cipher_kwargs (Dict[str, Any]): Keyword arguments for loading metadata from
            cipher(s). See :func:`merge_ciphers_with_dataset_dicts`.

    Returns:
        List[Dict]: A list of dicts in meddlr standard format.

    Notes:
        1. This function does not read the image files.
           The results do not have the "kspace", "target", or "maps" fields.
    """
    json_file = _PATH_MANAGER.get_local_path(json_file)
    start_time = time.perf_counter()
    with open(json_file, "r") as f:
        data = json.load(f)
    logger.info(
        "Loading {} takes {:.2f} seconds".format(json_file, time.perf_counter() - start_time)
    )

    # TODO: Add any relevant metadata.
    dataset_dicts = []
    for d in data["images"]:
        dd = dict(d)
        if image_root is not None:
            file_name = _PATH_MANAGER.get_local_path(os.path.join(image_root, d["file_name"]))
        else:
            file_name = _PATH_MANAGER.get_local_path(d["file_path"])
        dd["file_name"] = file_name

        # TODO: Clean this up
        if any(dataset_name.startswith(x) for x in ["stanford_qDESS_knee_2020"]):
            with h5py.File(dd["file_name"], "r") as f:
                dd["kspace_shape"] = f["kspace"].shape

        dataset_dicts.append(dd)

    # Load metadata from metadata file.
    metadata_file = MetadataCatalog.get(dataset_name).metadata_file
    if metadata_file:
        dataset_dicts = _load_metadata(metadata_file, dataset_dicts)

    return dataset_dicts


def register_mrco_scans(name, metadata, json_file, image_root: str = None):
    """
    Register a dataset in COCO's json annotation format for
    instance detection, instance segmentation and keypoint detection.
    (i.e., Type 1 and 2 in http://cocodataset.org/#format-data.
    `instances*.json` and `person_keypoints*.json` in the dataset).

    This is an example of how to register a new dataset.
    You can do something similar to this function, to register new datasets.

    Args:
        name (str): the name that identifies a dataset, e.g. "coco_2014_train".
        metadata (dict): extra metadata associated with this dataset.  You can
            leave it as an empty dict.
        json_file (str): path to the json instance annotation file.
        image_root (str): directory which contains all the images.
    """
    # 1. register a function which returns dicts
    DatasetCatalog.register(name, lambda: load_mrco_json(json_file, image_root, name))

    # 2. Optionally, add metadata about this dataset,
    # since they might be useful in evaluation, visualization or logging
    MetadataCatalog.get(name).set(
        json_file=json_file, image_root=image_root, evaluator_type="coco", **metadata
    )


def _load_metadata(metadata_file, dataset_dicts):
    """Loads metadata in csv file into dataset dictionaries.

    Args:
        metadata_file (str): A csv file with metadata.
        dataset_dicts (List[Dict]): List of dataset dictionaries.

    Returns:
        dataset_dicts: The dataset dictionaries with "_metadata" key.
            This key maps to the dictionary of metadata.
    """
    metadata = pd.read_csv(_PATH_MANAGER.get_local_path(metadata_file))
    metadata = metadata.loc[:, ~metadata.columns.str.contains("^Unnamed")]

    for dd in dataset_dicts:
        dd_metadata = metadata[metadata["fname"] == os.path.basename(dd["file_name"])]
        assert len(dd_metadata) == 1
        dd_metadata = dd_metadata.to_dict("index")
        dd_metadata = dd_metadata[list(dd_metadata.keys())[0]]
        dd["_metadata"] = dd_metadata

    return dataset_dicts
