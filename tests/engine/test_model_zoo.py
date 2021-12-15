import unittest

import torch
from torch import nn

from meddlr.config.config import get_cfg
from meddlr.engine.model_zoo import get_model_from_zoo, load_weights
from meddlr.modeling import build_model
from meddlr.utils import env


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
