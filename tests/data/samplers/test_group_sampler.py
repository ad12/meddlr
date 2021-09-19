import unittest

import numpy as np

from ss_recon.data import DatasetCatalog
from ss_recon.data.samplers.group_sampler import AlternatingGroupSampler, GroupSampler
from ss_recon.data.slice_dataset import SliceData


class _MockDataset:
    def __init__(self, groups=None):
        if groups is not None:
            self.build_examples(groups)
        else:
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

    def build_examples(self, groups):
        examples = []
        for g, (num_sup, num_unsup) in groups.items():
            if not isinstance(g, (tuple, list)):
                g = (g,)
            base_dict = {f"field{idx}": _g for idx, _g in enumerate(g)}
            exs = [{"_is_unsupervised": False} for _ in range(num_sup)] + [
                {"_is_unsupervised": True} for _ in range(num_unsup)
            ]
            for ex in exs:
                ex.update(base_dict)
            examples.extend(exs)
        self.examples = examples

    def get_supervised_idxs(self):
        return [idx for idx, ex in enumerate(self.examples) if not ex["_is_unsupervised"]]

    def get_unsupervised_idxs(self):
        return [idx for idx, ex in enumerate(self.examples) if ex["_is_unsupervised"]]

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
