import os
import unittest

import torch

from meddlr.config import get_cfg
from meddlr.data.build import build_data_loaders_per_scan
from meddlr.ops import complex as cplx


@unittest.skipIf(
    os.environ.get("MEDDLR_TEST_REPRO", "").lower() != "true", "Repro eval is time-consuming"
)
def test_eval_reproducibility():
    cfg = get_cfg()
    cfg.DATALOADER.NUM_WORKERS = 4
    cfg.SOLVER.TEST_BATCH_SIZE = 16
    dataset_name = "mridata_knee_2019_test"
    loaders = build_data_loaders_per_scan(cfg, dataset_name, (6,))
    kspace_data = []
    mask = None
    for acc in loaders:
        for scan_name, loader in loaders[acc].items():
            scan_name1 = scan_name
            for idx, inputs in enumerate(loader):  # noqa
                kspace = inputs["kspace"]
                kspace_data.append(kspace)
                c_mask = cplx.get_mask(kspace)
                if mask is not None:
                    pass
                    # assert torch.all(mask == c_mask)
                else:
                    mask = c_mask
            break
        break
    kspace1 = torch.cat(kspace_data, dim=0)

    loaders2 = build_data_loaders_per_scan(cfg, dataset_name, (6,))
    kspace_data2 = []
    for acc in loaders:
        for scan_name, loader in loaders2[acc].items():
            scan_name2 = scan_name
            for idx, inputs in enumerate(loader):  # noqa
                kspace = inputs["kspace"]
                kspace_data2.append(kspace)
            break
        break
    kspace2 = torch.cat(kspace_data2, dim=0)

    assert scan_name1 == scan_name2
    assert torch.allclose(kspace1, kspace2)


if __name__ == "__main__":
    unittest.main()
