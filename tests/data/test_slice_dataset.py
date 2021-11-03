import unittest
from collections import defaultdict

from meddlr.config import get_cfg
from meddlr.data import build_recon_train_loader
from meddlr.data.slice_dataset import SliceData


class MockSliceData(SliceData):
    def __getitem__(self, i):
        example = self.examples[i]
        vals = {}
        vals["file_name"] = example["file_name"]
        vals["slice_id"] = example["slice_id"]
        vals["is_unsupervised"] = example["is_unsupervised"]
        return vals


class TestBuildTrainLoader(unittest.TestCase):
    _TRAIN_DATASET = "mridata_knee_2019_train"
    _NUM_SCANS = 14
    _NUM_SLICES_PER_SCAN = 320

    def test_dataset(self):
        """Test number of examples in the dataset"""
        cfg = get_cfg()
        cfg.DATASETS.TRAIN = (self._TRAIN_DATASET,)
        cfg.DATALOADER.NUM_WORKERS = 0
        data_loader = build_recon_train_loader(cfg)
        dataset: SliceData = data_loader.dataset

        assert len(dataset) == len(dataset.examples)
        scan_names = [x["file_name"] for x in dataset.examples]
        assert len(set(scan_names)) == self._NUM_SCANS

        scans_to_slices = defaultdict(list)
        for x in dataset.examples:
            scans_to_slices[x["file_name"]].append(x["slice_id"])
        for _x, slice_nums in scans_to_slices.items():
            assert set(slice_nums) == set(range(0, 320))

    def test_data_loader(self):
        cfg = get_cfg()
        cfg.DATASETS.TRAIN = (self._TRAIN_DATASET,)
        cfg.DATALOADER.NUM_WORKERS = 0
        cfg.SOLVER.TRAIN_BATCH_SIZE = 1
        data_loader = build_recon_train_loader(cfg, MockSliceData)
        assert len(data_loader) == self._NUM_SCANS * self._NUM_SLICES_PER_SCAN

        scans_to_slices = defaultdict(list)
        for d in data_loader:
            fnames = d["file_name"]
            slice_ids = d["slice_id"].tolist()
            for fname, s_id in zip(fnames, slice_ids):
                scans_to_slices[fname].append(s_id)

        for _x, slice_nums in scans_to_slices.items():
            assert set(slice_nums) == set(range(0, 320))
        assert len(scans_to_slices) == self._NUM_SCANS


if __name__ == "__main__":
    unittest.main()
