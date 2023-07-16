import os
import tarfile
from pathlib import Path

import torch

from meddlr.config import get_cfg
from meddlr.engine.sharing import prepare_model
from meddlr.evaluation.testing import find_weights
from meddlr.utils import env

from .. import util


def test_prepare_model():
    exp_name = "basic-cpu"
    cache_file = util.TEMP_CACHE_DIR / f"{exp_name}.tar.gz"
    exp_dir = util.TEMP_CACHE_DIR / exp_name

    pm = env.get_path_manager()
    tar_path = pm.get_local_path(
        f"https://huggingface.co/datasets/arjundd/meddlr-data/resolve/main/test-data/test-exps/{exp_name}.tar.gz",  # noqa: E501
        cache=cache_file,
    )
    if not os.path.isdir(exp_dir):
        with tarfile.open(tar_path, "r:gz") as tfile:
            tfile.extractall(util.TEMP_CACHE_DIR)
    exp_dir = Path(exp_dir)

    cfg = get_cfg()
    cfg.merge_from_file(exp_dir / "config.yaml")
    cfg.OUTPUT_DIR = str(exp_dir)

    # Replace the weights file with a real torch pickled file.
    criterion = "ssim (Wang)_scan"
    weights_file, _, _ = find_weights(cfg, criterion=criterion)
    torch.save({"state_dict": {}}, weights_file)

    sharing_dir = prepare_model(
        cfg,
        name="dummy-model",
        output_dir=util.TEMP_CACHE_DIR / "shareable-models",
        criterion=criterion,
    )
    assert os.path.isdir(sharing_dir)
    assert os.path.isfile(sharing_dir / "config.yaml")
    assert os.path.isfile(sharing_dir / "model.ckpt")
