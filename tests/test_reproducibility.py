import unittest

import torch

from ss_recon.data.build import build_data_loaders_per_scan
from ss_recon.config import get_cfg


def test_eval_reproducibility():
    cfg = get_cfg()
    cfg.DATALOADER.NUM_WORKERS = 4
    cfg.SOLVER.TEST_BATCH_SIZE = 4
    dataset_name = "mridata_knee_2019_test"
    loaders = build_data_loaders_per_scan(cfg, dataset_name, (6,))
    kspace_data = []
    for acc in loaders:
        for scan_name, loader in loaders[acc].items():
            for idx, (kspace, maps, target, mean, std, norm) in enumerate(loader):  # noqa
                kspace_data.append(kspace)
            break
        break
    kspace = torch.cat(kspace_data, dim=0)

    loaders2 = build_data_loaders_per_scan(cfg, dataset_name, (6,))
    kspace_data2 = []
    for acc in loaders:
        for scan_name, loader in loaders2[acc].items():
            for idx, (kspace, maps, target, mean, std, norm) in enumerate(loader):  # noqa
                kspace_data2.append(kspace)
            break
        break
    kspace2 = torch.cat(kspace_data2, dim=0)

    assert torch.allclose(kspace, kspace2)


if __name__ == "__main__":
    unittest.main()
