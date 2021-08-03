import unittest

from ss_recon.data import DatasetCatalog
from ss_recon.data.samplers.group_sampler import GroupSampler
from ss_recon.data.slice_dataset import SliceData


class _MockDataset:
    def __init__(self):
        self.examples = [
            {"letter": "A", "number": 1},
            {"letter": "A", "number": 2},
            {"letter": "A", "number": 3},
            {"letter": "B", "number": 1},
            {"letter": "B", "number": 2},
            {"letter": "B", "number": 3},
            {"letter": "C", "number": 1},
            {"letter": "C", "number": 4},
        ]

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]


class TestGroupSampler(unittest.TestCase):
    def test_basic_batch_by(self):
        dataset = _MockDataset()

        batch_size = 2
        sampler = GroupSampler(
            dataset, batch_by="letter", shuffle=False, batch_size=batch_size, as_batch_sampler=True
        )
        assert len(sampler) == 5
        for batch in sampler:
            assert len(batch) <= batch_size

        batch_size = 2
        sampler = GroupSampler(
            dataset, batch_by="letter", shuffle=False, batch_size=batch_size, as_batch_sampler=False
        )
        assert len(sampler) == len(dataset.examples)
        for batch in sampler:
            assert isinstance(batch, int)

    def test_fastmri_brain(self):
        """Utility test for fastmri brain split."""
        dataset_dicts = DatasetCatalog.get("fastMRI_brain_multicoil_mini_v0.0.1_val")
        dataset = SliceData(dataset_dicts, transform=None)
        batch_by = "receiverChannels"
        batch_size = 7

        all_channels = {(x["_metadata"][batch_by],) for x in dataset.examples}
        sampler = GroupSampler(
            dataset,
            batch_by=batch_by,
            as_batch_sampler=True,
            batch_size=batch_size,
        )
        assert all_channels == sampler._groups.keys()

        for batch in sampler:
            unique_groups = {dataset.examples[idx]["_metadata"][batch_by] for idx in batch}
            assert len(unique_groups) == 1
