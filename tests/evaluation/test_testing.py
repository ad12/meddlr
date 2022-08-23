import os
import tarfile
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Optional

import pytest
from torch import nn

from meddlr.config import get_cfg
from meddlr.evaluation.testing import (
    check_consistency,
    find_metrics,
    find_weights,
    flatten_results_dict,
)
from meddlr.utils import env

from .. import util


def test_flatten_results_dict():
    x = {"a": 1, "b": {"b1": 5, "b2": {"b2-1": 2}}, "c": {"c1": 8, "c2": 10}}
    expected = {"a": 1, "b/b1": 5, "b/b2/b2-1": 2, "c/c1": 8, "c/c2": 10}
    out = flatten_results_dict(x)

    assert out.keys() == expected.keys()
    for k in expected:
        assert out[k] == expected[k], f"{k}: {out[k]} != {expected[k]}"


def test_check_consistency():
    model = nn.Sequential(
        nn.Linear(10, 10),
        nn.ReLU(),
        nn.Linear(10, 10),
        nn.ReLU(),
        nn.Linear(10, 10),
        nn.ReLU(),
    )

    state_dict = model.state_dict()
    check_consistency(state_dict, model)

    state_dict = deepcopy(state_dict)
    state_dict["0.weight"] += 1
    with pytest.raises(AssertionError):
        check_consistency(state_dict, model)


@pytest.mark.parametrize(
    "func_kwargs,expected_iteration",
    [
        ({"criterion": "psnr_scan"}, 1399),
        ({"criterion": "ssim (Wang)_scan"}, 799),
        ({"criterion": "psnr_scan", "iter_limit": 800}, 399),
        ({"criterion": "ssim_psnr"}, None),
        ({"criterion": "foobar", "operation": "max"}, None),
    ],
)
def test_find_metrics_basic(func_kwargs: Dict[str, Any], expected_iteration: Optional[int]):
    """
    Test that we can find the iteration corresponding to the
    best metrics from a basic experiment.
    """
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

    if expected_iteration is None:
        with pytest.raises(ValueError):
            find_metrics(cfg, **func_kwargs)
    else:
        iteration, _ = find_metrics(cfg, **func_kwargs)
        assert iteration == expected_iteration


@pytest.mark.parametrize(
    "func_kwargs,expected_file",
    [
        ({"criterion": "psnr_scan"}, "model_0001399.pth"),
        ({"criterion": "ssim (Wang)_scan"}, "model_0000799.pth"),
        ({"criterion": "psnr_scan", "iter_limit": 800}, "model_0000399.pth"),
        ({"criterion": "ssim_psnr"}, None),
        ({"criterion": "foobar"}, None),
        ({"criterion": "foobar", "operation": "max"}, None),
        ({"criterion": "psnr_scan", "operation": "foobar"}, None),
    ],
)
def test_find_weights_basic(func_kwargs, expected_file):
    """Test that we can find the best weights from a basic experiment."""
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

    if expected_file is None:
        with pytest.raises(ValueError):
            find_weights(cfg, **func_kwargs)
    else:
        weights, _, _ = find_weights(cfg, **func_kwargs)
        assert os.path.basename(weights) == expected_file
