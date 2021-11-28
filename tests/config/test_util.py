from meddlr.config.util import check_dependencies

from .. import util


def test_check_dependencies():
    cfg_file = util.get_cfg_path("tests/basic-with-deps.yaml")
    assert check_dependencies(cfg_file)
