from ss_recon.data.build import get_recon_dataset_dicts


def test_filter_by_metadata():
    dataset = "fastMRI_knee_multicoil_v0.0.1_train"
    dataset_dicts = get_recon_dataset_dicts([dataset], filter_by=(("flip Angle_deg", 140),))
    assert all(dd["_metadata"]["flip Angle_deg"] == 140 for dd in dataset_dicts)