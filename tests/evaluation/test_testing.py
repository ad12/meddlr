import os
import uuid
from copy import deepcopy
from pathlib import Path

import pytest
from torch import nn

from meddlr.config import get_cfg
from meddlr.evaluation.testing import check_consistency, find_weights, flatten_results_dict
from meddlr.utils import env


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
    "func_kwargs,expected_file",
    [
        ({"criterion": "psnr_scan"}, "model_0001399.pth"),
        ({"criterion": "ssim (Wang)_scan"}, "model_0000799.pth"),
        ({"criterion": "psnr_scan", "iter_limit": 800}, "model_0000399.pth"),
    ],
)
def test_find_weights_basic(tmpdir, func_kwargs, expected_file):
    """Test that we can find the best weights from a basic experiment."""
    pm = env.get_path_manager()
    exp_dir = pm.get_local_path(
        "gdrive://https://drive.google.com/drive/folders/1aKXuSmLgfZVHor6Tq47HXLXLUNIS5E6e?usp=sharing",  # noqa: E501
        cache_file=tmpdir / str(uuid.uuid4()) / "sample-dir",
    )
    exp_dir = Path(exp_dir)
    cfg = get_cfg().merge_from_file(exp_dir / "config.yaml")
    cfg.OUTPUT_DIR = str(exp_dir)

    weights, _, _ = find_weights(cfg, **func_kwargs)
    assert os.path.basename(weights) == expected_file
