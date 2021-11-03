import unittest

from meddlr.data import DatasetCatalog


def test_load_metadata_from_csv():
    """
    Tests that for main datasets, we can load metadata from corresponding csv file.
    """
    datasets_to_test = [
        "fastMRI_knee_multicoil_v0.0.1",
        "fastMRI_brain_multicoil_dev",
        "mridata_knee_2019",
    ]

    for dataset in datasets_to_test:
        for split in ["train", "val", "test"]:
            dataset_split = f"{dataset}_{split}"
            dataset_dicts = DatasetCatalog.get(dataset_split)
            assert all("_metadata" in dd for dd in dataset_dicts)


if __name__ == "__main__":
    unittest.main()
