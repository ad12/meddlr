from collections import defaultdict
import unittest

from ss_recon.data import build_recon_train_loader, build_recon_val_loader
from ss_recon.data.slice_dataset import SliceData
from ss_recon.config import get_cfg


class MockSliceData(SliceData):
    def __getitem__(self, i):
        vals = {}
        fname, slice_id, is_unsupervised = self.examples[i]
        vals["fname"] = fname
        vals["slice_id"] = slice_id
        vals["is_unsupervised"] = is_unsupervised


class TestBuildTrainLoader(unittest.TestCase):
    def test_dataset(self):
        """Test number of examples in the dataset"""
        cfg = get_cfg()
        cfg.DATASETS.TRAIN = "mridata_knee_2019_train"
        data_loader = build_recon_train_loader(cfg)
        dataset: SliceData = data_loader.dataset

        assert len(dataset) == len(dataset.examples)

        scan_names = [x[0] for x in dataset.examples]
        assert len(set(scan_names)) == 16

        scans_to_slices = defaultdict(list)
        for x in dataset.examples:
            scans_to_slices[x[0]].append(x[1])
        for x, slice_nums in scans_to_slices.items():
            assert set(slice_nums) == set(range(0, 320))

    def test_data_loader(self):
        cfg = get_cfg()
        cfg.DATASETS.TRAIN = "mridata_knee_2019_train"
        cfg.SOLVER.TRAIN_BATCH_SIZE = 1
        data_loader = build_recon_train_loader(cfg, MockSliceData)

        scans_to_slices = defaultdict(list)
        for d in data_loader:
            fnames = d["fname"].tolist()
            slice_ids = d["slice_id"].tolist()
            for fname, s_id in zip(fnames, slice_ids):
                scans_to_slices[fname].append(s_id)

        for x, slice_nums in scans_to_slices.items():
            assert set(slice_nums) == set(range(0, 320))
        assert len(scans_to_slices) == 16


if __name__ == "__main__":
    unittest.main()
