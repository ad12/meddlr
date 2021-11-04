"""Verify experimental configurations (random seeds, etc.)."""
import unittest

from meddlr.data.build import get_recon_dataset_dicts


class TestMRIDataExperiments(unittest.TestCase):
    TRAIN_DATASET = "mridata_knee_2019_train"
    VAL_DATASET = "mridata_knee_2019_val"
    TEST_DATASET = "mridata_knee_2019_test"

    def test_subject_seeds(self):
        """Verify randomness of seeds for selecting subjects in mridata.org dataset.

        Because the dataset is so small, it is possible that two different
        random seeds can result in the same scans being selected.
        """
        dataset_sizes = [1, 2, 5, 10, 14]
        overlap_thresholds = [0, 1, 3, 8, None]

        # Find 3 seeds that work relatively well
        # 1000 is fixed as it is used by most of our runs.
        scans = {}
        seeds = [1000, 2000, 3000, 9860, 9970]  # last two seeds found by random search
        print(seeds)
        for num_total in dataset_sizes:
            for seed in seeds:
                dataset_dicts = get_recon_dataset_dicts(
                    (self.TRAIN_DATASET,), num_scans_total=num_total, seed=seed
                )
                assert len(dataset_dicts) == num_total
                selected_scans = [dd["file_name"] for dd in dataset_dicts]
                assert len(set(selected_scans)) == num_total
                scans[(num_total, seed)] = selected_scans

        # For a fixed seed, set of scans must be supersets
        for seed in seeds:
            seed_scans = {
                num_total: _scans for (num_total, _seed), _scans in scans.items() if seed == _seed
            }
            ordered_scans = [seed_scans[x] for x in dataset_sizes]
            for idx, (x, y) in enumerate(zip(ordered_scans[:-1], ordered_scans[1:])):
                assert set(y).issuperset(set(x)), (
                    f"Case num_total={dataset_sizes[idx+1]} is not superset of "
                    f"num_total={dataset_sizes[idx]} for seed {seed}"
                )

        # All combinations for 1 & 2 subject cases must be different.
        for num_total, overlap_thresh in zip(dataset_sizes, overlap_thresholds):
            if overlap_thresh is None:
                continue
            scan_names = [
                (_seed, set(_scans))
                for (_num_total, _seed), _scans in scans.items()
                if _num_total == num_total
            ]
            for idx, (seed1, scans1) in enumerate(scan_names):
                for seed2, scans2 in scan_names[idx + 1 :]:
                    overlap_scans = scans1 & scans2
                    num_same = len(overlap_scans)
                    assert (
                        num_same <= overlap_thresh
                    ), "Seeds {} and {} have the same " "{}/{} scan(s):\n{}".format(
                        seed1, seed2, num_same, num_total, "\n".join(sorted(overlap_scans))
                    )
                    print(f"{seed1} - {seed2}: {num_same}/{num_total}")


if __name__ == "__main__":
    unittest.main()
