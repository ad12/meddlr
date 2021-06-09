import unittest
from ss_recon.data import DatasetCatalog
from ss_recon.data.build import get_recon_dataset_dicts


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


def test_filter_by_metadata():
    dataset = "fastMRI_knee_multicoil_v0.0.1_train"
    dataset_dicts = get_recon_dataset_dicts([dataset], filter_by=(("flip Angle_deg", 140),))
    assert all(dd["_metadata"]["flip Angle_deg"] == 140 for dd in dataset_dicts)



if __name__ == "__main__":
    unittest.main()