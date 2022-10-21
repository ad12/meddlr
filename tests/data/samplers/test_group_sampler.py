import unittest

import numpy as np

from meddlr.data import DatasetCatalog
from meddlr.data.samplers.group_sampler import (
    AlternatingGroupSampler,
    DistributedGroupSampler,
    GroupSampler,
)
from meddlr.data.slice_dataset import SliceData

from .mock import MockSliceDataset, _MockDataset


class TestGroupSampler(unittest.TestCase):
    def test_group_sampler_errors(self):
        """Test init values that would cause errors."""
        dataset = _MockDataset()
        with self.assertRaises(ValueError):
            GroupSampler(dataset, batch_by="letter", shuffle=False)

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
        try:
            dataset_dicts = DatasetCatalog.get("fastMRI_brain_multicoil_mini_v0.0.1_val")
            dataset = SliceData(dataset_dicts, transform=None)
        except FileNotFoundError:
            self.skipTest("fastMRI files not found")

        batch_by = "receiverChannels"
        batch_size = 7

        all_channels = {(x["_metadata"][batch_by],) for x in dataset.examples}
        sampler = GroupSampler(
            dataset, batch_by=batch_by, as_batch_sampler=True, batch_size=batch_size
        )
        assert all_channels == sampler._groups.keys()

        for batch in sampler:
            unique_groups = {dataset.examples[idx]["_metadata"][batch_by] for idx in batch}
            assert len(unique_groups) == 1


class TestAlternatingGroupSampler(unittest.TestCase):
    def _validate_duty_cycle(self, cycle, T_s, T_us):
        for idx, is_unsup in enumerate(cycle):
            rng = cycle[max(0, idx - 2) : min(len(cycle) - 1, idx + 3)]
            if idx % (T_s + T_us) < T_s:
                assert not is_unsup, f"index {idx}: Expected is_unsup=False, got True - {rng}"
            else:
                assert is_unsup, f"index {idx}: Expected is_unsup=True, got False - {rng}"

    def test_basic(self):
        T_s = T_us = 1
        dataset = _MockDataset({"A": (50, 50), "B": (20, 20)})

        sampler = AlternatingGroupSampler(
            dataset, T_s=T_s, T_us=T_us, batch_by="field0", batch_size=1
        )
        assert sampler.counts_per_group == {("A",): 100, ("B",): 40}
        assert sampler.counts == {
            ("A", False): 50,
            ("A", True): 50,
            ("B", False): 20,
            ("B", True): 20,
        }

        indices = list(iter(sampler))
        sup_unsup_cycle = [dataset[idx]["_is_unsupervised"] for idx in indices]
        self._validate_duty_cycle(sup_unsup_cycle, T_s, T_us)
        _, counts = np.unique(indices, return_counts=True)
        assert np.max(counts) - np.min(counts) <= 1
        assert np.all(counts == 1) or (
            (len(indices) == len(dataset)) and all(x in [1, 2] for x in counts)
        )

        sampler = AlternatingGroupSampler(
            dataset, T_s=T_s, T_us=T_us, batch_by="field0", batch_size=2
        )
        sup_unsup_cycle = [dataset[idx]["_is_unsupervised"] for idx in iter(sampler)]
        self._validate_duty_cycle(sup_unsup_cycle, T_s, T_us)

        sampler = AlternatingGroupSampler(
            dataset, T_s=T_s, T_us=T_us, as_batch_sampler=True, batch_by="field0", batch_size=1
        )
        sup_unsup_cycle = [dataset[idx]["_is_unsupervised"] for x in iter(sampler) for idx in x]
        self._validate_duty_cycle(sup_unsup_cycle, T_s, T_us)

    def test_uneven_sizes(self):
        T_s = T_us = 2
        dataset = _MockDataset({"A": (4, 4), "B": (4, 0)})

        sampler = AlternatingGroupSampler(
            dataset, T_s=T_s, T_us=T_us, batch_by="field0", batch_size=2
        )
        assert sampler.counts_per_group == {("A",): 8, ("B",): 4}
        assert sampler.counts == {("A", False): 4, ("A", True): 4, ("B", False): 4}
        indices = list(iter(sampler))
        sup_unsup_cycle = [dataset[idx]["_is_unsupervised"] for idx in indices]
        self._validate_duty_cycle(sup_unsup_cycle, T_s, T_us)

    def test_uneven_sizes2(self):
        T_s = T_us = 2
        dataset = _MockDataset({"A": (2, 4), "B": (2, 2), "C": (2, 0)})
        sampler = AlternatingGroupSampler(
            dataset,
            T_s=T_s,
            T_us=T_us,
            batch_by="field0",
            batch_size=1,
            seed=10,
            as_batch_sampler=True,
        )
        indices = list(iter(sampler))
        sup_unsup_cycle = [dataset[idx]["_is_unsupervised"] for idx in indices]
        self._validate_duty_cycle(sup_unsup_cycle, T_s, T_us)


class TestDistributedGroupSampler(unittest.TestCase):
    def _build_mock_dataset(self, num_scans=10, num_slices=320) -> MockSliceDataset:
        """Build mock dataset with unsupervised data."""
        dataset_dicts = [
            {
                "file_name": "scan{}".format(idx),
                "kspace_size": (num_slices, 320, 256, 4, 2),
                "_is_unsupervised": False,
            }
            for idx in range(num_scans)
        ]

        return MockSliceDataset(dataset_dicts)

    def test_no_distributed(self):
        dataset = self._build_mock_dataset()
        sampler = DistributedGroupSampler(dataset, group_by="file_name")

        assert len(sampler) == len(dataset), (
            f"Expected same size for sampler ({len(sampler)}) and dataset ({len(dataset)}. "
            f"All groups allocated to same sampler"
        )

    def test_distributed_even_partition(self):
        world_size = 4
        num_scans = 12
        samplers = []

        # Simulate what happens in DDP
        for i in range(world_size):
            dataset = self._build_mock_dataset(num_scans=num_scans)
            sampler = DistributedGroupSampler(
                dataset, group_by="file_name", num_replicas=world_size, rank=i
            )
            samplers.append(sampler)

        expected_dataset = self._build_mock_dataset(num_scans=num_scans)
        length = sum(len(sampler) for sampler in samplers)
        assert length == len(expected_dataset), (
            f"Sum of lengths of samplers ({length}) != "
            f"length of full dataset ({len(expected_dataset)})"
        )

        groups = [{expected_dataset[idx][0] for idx in sampler.indices} for sampler in samplers]
        assert all(len(grps) == num_scans // world_size for grps in groups), (
            f"Groups not equally divided among {world_size} "
            f"processes ({len(grps) for grps in groups}). "
        )
        for i in range(world_size):
            for j in range(i + 1, world_size):
                assert (
                    len(groups[i] & groups[j]) == 0
                ), f"Overlapping groups in processes {i} ({groups[i]}) and {j} ({groups[j]})"

    def test_distributed_uneven_partition(self):
        world_size = 4
        num_scans = 10
        samplers = []

        # Simulate what happens in DDP
        for i in range(world_size):
            dataset = self._build_mock_dataset(num_scans=num_scans)
            sampler = DistributedGroupSampler(
                dataset, group_by="file_name", num_replicas=world_size, rank=i
            )
            samplers.append(sampler)

        expected_dataset = self._build_mock_dataset(num_scans=num_scans)
        length = sum(len(sampler) for sampler in samplers)
        assert length == len(expected_dataset), (
            f"Sum of lengths of samplers ({length}) != "
            f"length of full dataset ({len(expected_dataset)})"
        )

        groups = [{expected_dataset[idx][0] for idx in sampler.indices} for sampler in samplers]
        num_groups = [len(grps) for grps in groups]
        assert sum(x == 2 for x in num_groups) == 2 and sum(x == 3 for x in num_groups) == 2
        for i in range(len(groups)):
            for j in range(i + 1, len(groups)):
                assert (
                    len(groups[i] & groups[j]) == 0
                ), f"Overlapping groups in processes {i} ({groups[i]}) and {j} ({groups[j]})"
