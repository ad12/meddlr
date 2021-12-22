import os
import pathlib
import re
import unittest
from typing import Any, Dict

import pytest
import torch
from torch import nn

import meddlr.config.util as config_util
from meddlr.config.config import get_cfg
from meddlr.engine.model_zoo import get_model_from_zoo, load_weights
from meddlr.modeling import build_model
from meddlr.utils import env

from .. import util

REPO_DIR = pathlib.Path(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))


class TestModelZooExceptionsAndWarnings(unittest.TestCase):
    """Write tests in this class specifically for exceptions and warnings."""

    def test_get_model_from_zoo_dependency_warning(self):
        """Test that dependencies for configs get parsed."""
        cfg_gdrive = "download://https://drive.google.com/file/d/1b_HU9p2iFUcQu7_8KadDEfhVer6v68Uj/view?usp=sharing"  # noqa: E501
        with self.assertWarnsRegex(UserWarning, expected_regex=".*dependencies.*"):
            get_model_from_zoo(cfg_gdrive, force_download=True)


def test_get_model_from_zoo():
    cfg_gdrive = "download://https://drive.google.com/file/d/1fRn5t4qGVVR6PyaRWPO6tAmug6-oTBjc/view?usp=sharing"  # noqa: E501
    weights_gdrive = "download://https://drive.google.com/file/d/1BGARoUWLKg_DLfQN4AA2HnzktJzHaAy7/view?usp=sharing"  # noqa: E501
    path_mgr = env.get_path_manager()

    model = get_model_from_zoo(cfg_gdrive, weights_gdrive, force_download=True)
    assert isinstance(model, nn.Module)
    weights_path = path_mgr.get_local_path(weights_gdrive)
    weights = torch.load(weights_path)
    for name, param in model.named_parameters():
        assert name in weights
        assert torch.allclose(param, weights[name])

    model2 = get_model_from_zoo(cfg_gdrive, force_download=True)
    assert isinstance(model2, nn.Module)
    assert type(model2) == type(model)

    cfg = get_cfg().merge_from_file(path_mgr.get_local_path(cfg_gdrive))
    model2 = get_model_from_zoo(cfg, weights_gdrive)
    state_dict = model.state_dict()
    for name, param in model.named_parameters():
        assert name in state_dict
        assert torch.allclose(param, state_dict[name])


def test_load_weights_shape_mismatch():
    cfg_gdrive = "download://https://drive.google.com/file/d/1fRn5t4qGVVR6PyaRWPO6tAmug6-oTBjc/view?usp=sharing"  # noqa: E501
    weights_gdrive = "download://https://drive.google.com/file/d/1BGARoUWLKg_DLfQN4AA2HnzktJzHaAy7/view?usp=sharing"  # noqa: E501
    path_mgr = env.get_path_manager()

    cfg = get_cfg().merge_from_file(path_mgr.get_local_path(cfg_gdrive))
    model = build_model(cfg)
    model.resnets[0] = None
    model = load_weights(model, weights_gdrive, ignore_shape_mismatch=True)

    weights_path = path_mgr.get_local_path(weights_gdrive)
    weights = torch.load(weights_path)
    for name, param in model.named_parameters():
        assert name in weights
        assert torch.allclose(param, weights[name])


def test_load_weights_find_device():
    cfg_gdrive = "download://https://drive.google.com/file/d/1fRn5t4qGVVR6PyaRWPO6tAmug6-oTBjc/view?usp=sharing"  # noqa: E501
    weights_gdrive = "download://https://drive.google.com/file/d/1BGARoUWLKg_DLfQN4AA2HnzktJzHaAy7/view?usp=sharing"  # noqa: E501
    path_mgr = env.get_path_manager()

    cfg = get_cfg().merge_from_file(path_mgr.get_local_path(cfg_gdrive))
    model = build_model(cfg)
    model = load_weights(model, weights_gdrive, find_device=False)

    weights_path = path_mgr.get_local_path(weights_gdrive)
    weights = torch.load(weights_path)
    for name, param in model.named_parameters():
        assert name in weights
        assert torch.allclose(param, weights[name])


@pytest.mark.skipif(not util.TEST_MODEL_ZOOS, reason="Model zoo testing not activated")
@pytest.mark.parametrize("project", ["noise2recon", "vortex"])
def test_model_zoo_configs_for_projects(project):
    """
    Test that all models in the zoo can be built with the appropriate config
    and weight files.

    Note:
        This function does not test if the config can be used for training.
        For now, the user must validate this manually.
    """
    model_zoo_file = REPO_DIR / "projects" / project / "MODEL_ZOO.md"
    models = _parse_model_zoo(model_zoo_file)
    for name, model_info in models.items():
        # Skip models that have failed dependencies (for now).
        # TODO: Auto-configure github actions to run this test with different
        # combinations of dependencies.
        path_manager = env.get_path_manager()
        cfg_file = path_manager.get_local_path(f"download://{model_info['cfg_url']}", force=True)
        failed_deps = config_util.check_dependencies(cfg_file, return_failed_deps=True)
        if len(failed_deps) > 0:
            continue

        try:
            model = get_model_from_zoo(
                f"download://{model_info['cfg_url']}",
                f"download://{model_info['weights_url']}",
                force_download=True,
            )
            assert isinstance(model, nn.Module)
        except Exception as e:
            raise type(e)(f"Failed to build model '{name}':\n{e}")


def _parse_model_zoo(model_zoo_file) -> Dict[str, Dict[str, Any]]:
    """Returns a dictionary representation of the model zoo.

    This function parses the MODEL_ZOO.md file and returns a dictionary
    mapping from the model name to the diction of model information.
    """

    def _has_model_line(line):
        return re.match("^\|.*\[cfg\].*$", line) is not None

    def _parse_model_line(line):
        columns = line.split("|")[1:]
        return {
            "name": columns[0].strip(),
            "cfg_url": re.search("\[cfg\]\((.+?)\)", line).group(1),
            "weights_url": re.search("\[model\]\((.+?)\)", line).group(1),
        }

    with open(model_zoo_file) as f:
        lines = f.readlines()
    lines = [line.strip() for line in lines]
    lines = [line for line in lines if line]
    node, _ = util.parse_markdown(lines)

    models = {}
    for heading, content in node.to_dict(flatten=True).items():
        if not content or not any(_has_model_line(line) for line in content):
            continue
        model_content = [line for line in content if _has_model_line(line)]

        for model_line in model_content:
            model_data = _parse_model_line(model_line)
            models[f"{heading}/{model_data['name']}"] = model_data
    return models
