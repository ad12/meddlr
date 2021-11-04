import numpy as np

from meddlr.data.build import _limit_data_by_group, get_recon_dataset_dicts
from meddlr.data.catalog import DatasetCatalog


def test_filter_by_metadata():
    dataset = "fastMRI_knee_multicoil_v0.0.1_train"
    dataset_dicts = get_recon_dataset_dicts([dataset], filter_by=(("flipAngle_deg", 140),))

    assert len(dataset_dicts) > 0
    assert all(dd["_metadata"]["flipAngle_deg"] == 140 for dd in dataset_dicts)


def test_num_scans_total():
    dataset = "fastMRI_knee_multicoil_v0.0.1_val"
    orig_dataset_dicts = DatasetCatalog.get(dataset)

    dataset_dicts = get_recon_dataset_dicts([dataset], num_scans_total=10)
    assert len(dataset_dicts) == 10

    values = {140: 10}
    dataset_dicts = get_recon_dataset_dicts([dataset], num_scans_total=(("flipAngle_deg", values),))
    orig_flip_angles_to_count = {
        k: v
        for k, v in zip(
            *np.unique(
                [dd["_metadata"]["flipAngle_deg"] for dd in orig_dataset_dicts], return_counts=True
            )
        )
    }
    flip_angles_to_count = {
        k: v
        for k, v in zip(
            *np.unique(
                [dd["_metadata"]["flipAngle_deg"] for dd in dataset_dicts], return_counts=True
            )
        )
    }
    for k in sorted(orig_flip_angles_to_count.keys()):
        if k in values:
            assert flip_angles_to_count[k] <= values[k]
        else:
            assert flip_angles_to_count[k] == orig_flip_angles_to_count[k]


def test_limit_data_by_group():
    dataset_dicts = [
        {"id": 1, "metadata_A": "A1", "metadata_B": "B1"},
        {"id": 2, "metadata_A": "A1", "metadata_B": "B2"},
        {"id": 3, "metadata_A": "A1", "metadata_B": "B3"},
        {"id": 4, "metadata_A": "A1", "metadata_B": "B1"},
        {"id": 5, "metadata_A": "A2", "metadata_B": "B2"},
        {"id": 6, "metadata_A": "A2", "metadata_B": "B3"},
        {"id": 7, "metadata_A": "A3", "metadata_B": "B1"},
        {"id": 8, "metadata_A": "A3", "metadata_B": "B2"},
        {"id": 9, "metadata_A": "A3", "metadata_B": "B3"},
        {"id": 10, "metadata_A": "A4", "metadata_B": "B1"},
        {"id": 11, "metadata_A": "A4", "metadata_B": "B2"},
        {"id": 12, "metadata_A": "A5", "metadata_B": "B3"},
    ]

    out = _limit_data_by_group(dataset_dicts, num_scans_total=(("metadata_A", {"A1": 2, "A2": 1}),))
    out_ids = [o["id"] for o in out]
    assert out_ids == [1, 2, 5, 7, 8, 9, 10, 11, 12]

    out = _limit_data_by_group(dataset_dicts, num_scans_total=(("metadata_A", ("A1", 2, "A2", 1)),))
    out_ids = [o["id"] for o in out]
    assert out_ids == [1, 2, 5, 7, 8, 9, 10, 11, 12]

    out = _limit_data_by_group(
        dataset_dicts, num_scans_total=(("metadata_A", {("A1", "A2"): 5, "A3": 2}),)
    )
    out_ids = [o["id"] for o in out]
    assert out_ids == [1, 2, 3, 4, 5, 7, 8, 10, 11, 12]
