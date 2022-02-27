import pytest

from meddlr.config.config import get_cfg
from meddlr.config.util import check_dependencies, configure_params, stringify

from .. import util


def test_check_dependencies():
    cfg_file = util.get_cfg_path("tests/basic-with-deps.yaml")
    assert check_dependencies(cfg_file)


def test_configure_params():
    params = {"DESCRIPTION.BRIEF": ["foo", "bar", "foobar"]}
    fixed = {"DESCRIPTION.PROJECT_NAME": "test"}
    base_cfg = get_cfg()

    cfgs = configure_params(params)
    assert len(cfgs) == 3
    for cfg, val in zip(cfgs, params["DESCRIPTION.BRIEF"]):
        assert cfg["DESCRIPTION.BRIEF"] == val

    cfgs = configure_params(params, base_cfg=base_cfg)
    assert len(cfgs) == 3
    for cfg, val in zip(cfgs, params["DESCRIPTION.BRIEF"]):
        assert cfg.DESCRIPTION.BRIEF == val

    cfgs = configure_params(params, fixed=fixed, base_cfg=base_cfg)
    assert len(cfgs) == 3
    for cfg, val in zip(cfgs, params["DESCRIPTION.BRIEF"]):
        assert cfg.DESCRIPTION.BRIEF == val
        assert cfg.DESCRIPTION.PROJECT_NAME == "test"


@pytest.mark.parametrize(
    "cfg,expected_str",
    [
        (
            {"ALPHA": "foo", "BETA": ("bar", "tie")},
            "ALPHA \"'foo'\" BETA \"\('\"'\"'bar'\"'\"','\"'\"'tie'\"'\"',\)\"",
        ),
        (
            {"ALPHA": "foo", "BETA": ["bar", "tie"]},
            "ALPHA \"'foo'\" BETA \"['\"'\"'bar'\"'\"','\"'\"'tie'\"'\"']\"",
        ),
        (
            {"ALPHA": "foo", "BETA": {"bar": "tie"}},
            "ALPHA \"'foo'\" BETA \"\{'\"'\"'bar'\"'\"':'\"'\"'tie'\"'\"'\}\"",
        ),
    ],
)
def test_stringify(cfg, expected_str):
    assert stringify(cfg) == expected_str
