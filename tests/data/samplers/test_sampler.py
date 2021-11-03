import unittest

import numpy as np

from meddlr.data import AlternatingSampler

from .mock import MockSliceDataset


class TestAlternatingSampler(unittest.TestCase):
    def _build_mock_dataset(
        self, num_scans=2, num_unsupervised=1, num_slices=320
    ) -> MockSliceDataset:
        """Build mock dataset with unsupervised data."""
        assert num_unsupervised <= num_scans
        dataset_dicts = [
            {
                "file_name": "scan{}".format(idx),
                "kspace_size": (num_slices, 320, 256, 4, 2),
                "_is_unsupervised": idx < num_unsupervised,
            }
            for idx in range(num_scans)
        ]

        return MockSliceDataset(dataset_dicts)

    def test_basic(self):
        """
        Test with same number of supervised/unsupervised samples and same
        period.
        """
        T_s = T_us = 4
        dataset = self._build_mock_dataset(2, 1, 8)

        expected_supervised_idxs = set(dataset.get_supervised_idxs())
        expected_unsupervised_idxs = set(dataset.get_unsupervised_idxs())

        sampler = AlternatingSampler(dataset, T_s, T_us, seed=0)
        assert len(sampler) == 16

        idxs = np.asarray(list(iter(sampler)))
        assert len(idxs) == len(sampler)
        idxs = idxs.reshape((-1, T_s + T_us))
        supervised_idxs = idxs[:, 0:T_s].flatten()
        unsupervised_idxs = idxs[:, T_s:].flatten()

        assert set(supervised_idxs) == expected_supervised_idxs
        assert set(unsupervised_idxs) == expected_unsupervised_idxs

        # Verify that each index occurs only once.
        assert np.all(np.bincount(idxs.flatten()) == 1)

    def test_period(self):
        """Test when periods are not identical."""
        T_s = 4
        T_us = 2
        dataset = self._build_mock_dataset(2, 1, 8)

        expected_supervised_idxs = set(dataset.get_supervised_idxs())
        expected_unsupervised_idxs = set(dataset.get_unsupervised_idxs())

        sampler = AlternatingSampler(dataset, T_s, T_us, seed=0)
        assert len(sampler) == 24

        idxs = np.asarray(list(iter(sampler)))
        assert len(idxs) == len(sampler)
        idxs = idxs.reshape((-1, T_s + T_us))
        supervised_idxs = idxs[:, 0:T_s].flatten()
        unsupervised_idxs = idxs[:, T_s:].flatten()

        assert set(supervised_idxs) == expected_supervised_idxs
        assert set(unsupervised_idxs) == expected_unsupervised_idxs

        # Supervised indices repeat twice. Unsupervised indices repeat once.
        counts = np.bincount(idxs.flatten())
        unsupervised_counts = counts[:8]
        supervised_counts = counts[8:]
        assert np.all(unsupervised_counts == 1)
        assert np.all(supervised_counts == 2)

    def test_data_imbalance(self):
        """Number of supervised/unsupervised examples is imbalanced."""
        T_s = T_us = 4

        # 1 unsupervised, 2 supervised
        dataset = self._build_mock_dataset(3, 1, 8)

        expected_supervised_idxs = set(dataset.get_supervised_idxs())
        expected_unsupervised_idxs = set(dataset.get_unsupervised_idxs())

        sampler = AlternatingSampler(dataset, T_s, T_us, seed=0)
        assert len(sampler) == 32

        idxs = np.asarray(list(iter(sampler)))
        assert len(idxs) == len(sampler)
        idxs = idxs.reshape((-1, T_s + T_us))
        supervised_idxs = set(idxs[:, 0:T_s].flatten())
        unsupervised_idxs = set(idxs[:, T_s:].flatten())

        assert supervised_idxs == expected_supervised_idxs
        assert unsupervised_idxs == expected_unsupervised_idxs

        # Unsupervised indices repeat twice.
        counts = np.bincount(idxs.flatten())
        unsupervised_counts = counts[:8]
        supervised_counts = counts[8:]
        assert np.all(unsupervised_counts == 2)
        assert np.all(supervised_counts == 1)

        # 2 unsupervised, 1 supervised
        dataset = self._build_mock_dataset(3, 2, 8)

        expected_supervised_idxs = set(dataset.get_supervised_idxs())
        expected_unsupervised_idxs = set(dataset.get_unsupervised_idxs())

        sampler = AlternatingSampler(dataset, T_s, T_us, seed=0)
        assert len(sampler) == 32

        idxs = np.asarray(list(iter(sampler)))
        assert len(idxs) == len(sampler)
        idxs = idxs.reshape((-1, T_s + T_us))
        supervised_idxs = set(idxs[:, 0:T_s].flatten())
        unsupervised_idxs = set(idxs[:, T_s:].flatten())

        assert supervised_idxs == expected_supervised_idxs
        assert unsupervised_idxs == expected_unsupervised_idxs

        # Unsupervised indices repeat twice.
        counts = np.bincount(idxs.flatten())
        unsupervised_counts = counts[:16]
        supervised_counts = counts[16:]
        assert np.all(unsupervised_counts == 1)
        assert np.all(supervised_counts == 2)

    def test_advanced(self):
        """Imbalanced data and different periods."""
        T_s = 4
        T_us = 2
        # 2 unsupervised, 1 supervised
        dataset = self._build_mock_dataset(3, 2, 8)

        expected_supervised_idxs = set(dataset.get_supervised_idxs())
        expected_unsupervised_idxs = set(dataset.get_unsupervised_idxs())

        sampler = AlternatingSampler(dataset, T_s, T_us, seed=0)
        assert len(sampler) == 16 + 32

        idxs = np.asarray(list(iter(sampler)))
        assert len(idxs) == len(sampler)
        idxs = idxs.reshape((-1, T_s + T_us))
        supervised_idxs = set(idxs[:, 0:T_s].flatten())
        unsupervised_idxs = set(idxs[:, T_s:].flatten())

        assert supervised_idxs == expected_supervised_idxs
        assert unsupervised_idxs == expected_unsupervised_idxs

        # Unsupervised indices repeat twice.
        counts = np.bincount(idxs.flatten())
        unsupervised_counts = counts[:16]
        supervised_counts = counts[16:]
        assert np.all(unsupervised_counts == 1)
        assert np.all(supervised_counts == 4)
