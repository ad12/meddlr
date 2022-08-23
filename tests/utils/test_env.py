import os
import unittest

import pytest
from packaging import version

from meddlr.utils import env


class TestEnvVariables(unittest.TestCase):
    _env = None

    @classmethod
    def setUpClass(cls):
        cls._env = dict(os.environ)

    @classmethod
    def tearDownClass(cls):
        os.environ.clear()
        os.environ.update(cls._env)

    def _reset_var(self, env_var, value, force=False):
        if force:
            os.environ[env_var] = value
            return

        if value == "":
            os.environ.pop(env_var, None)
        else:
            os.environ[env_var] = value

    def test_supports_cplx_tensors(self):
        env_var = "MEDDLR_ENABLE_CPLX_TENSORS"
        orig_val = os.environ.get(env_var, "")
        is_pt17 = env.pt_version() >= [1, 7]

        # auto
        os.environ[env_var] = "auto"
        if is_pt17:
            assert env.supports_cplx_tensor()
        else:
            assert not env.supports_cplx_tensor()

        # True
        os.environ[env_var] = "True"
        is_pt16 = env.pt_version() >= [1, 6]
        if is_pt17:
            assert env.supports_cplx_tensor()
        elif is_pt16:
            assert env.supports_cplx_tensor()
        else:
            with self.assertRaises(RuntimeError):
                env.supports_cplx_tensor()

        # False
        os.environ[env_var] = "False"
        assert not env.supports_cplx_tensor()

        self._reset_var(env_var, orig_val)

    def test_debug(self):
        env_var = "MEDDLR_DEBUG"
        orig_val = os.environ.get(env_var, "")

        os.environ[env_var] = ""
        assert not env.is_debug()

        os.environ[env_var] = "True"
        assert env.is_debug()

        self._reset_var(env_var, orig_val)

    def test_reproducibility_mode(self):
        env_var = "MEDDLR_REPRO"
        orig_val = os.environ.get(env_var, "")

        os.environ[env_var] = ""
        assert not env.is_repro()

        os.environ[env_var] = "True"
        assert env.is_repro()

        self._reset_var(env_var, orig_val)


def test_is_package_installed():
    assert env.is_package_installed("numpy")
    assert env.is_package_installed("numpy>=0.0.1")
    assert env.is_package_installed("numpy>=0.0.1,<=1000.0.0")
    assert not env.is_package_installed("numpy<=0.0.1")
    assert not env.is_package_installed("numpy>=1000.0.0")

    numpy_version = env.get_package_version("numpy")
    assert not env.is_package_installed("numpy==0.0.1")
    assert env.is_package_installed(f"numpy=={numpy_version}")


@pytest.mark.parametrize("comp_type", ["str", "int_list", "version"])
def test_version(comp_type):
    """Test that we can compare with list and strings."""

    def _format_value(val):
        if comp_type == "str":
            return val
        elif comp_type == "int_list":
            return [int(x) for x in val.split(".")]
        elif comp_type == "version":
            return version.Version(val)
        else:
            raise ValueError(f"Unknown comp_type '{comp_type}'.")

    curr_version = env.Version("1.0.0")
    assert curr_version == _format_value("1.0.0")
    assert curr_version > _format_value("0.9.9")
    assert curr_version >= _format_value("0.9.9")
    assert curr_version < _format_value("1.1.1")
    assert curr_version <= _format_value("1.1.1")
    assert curr_version != _format_value("1.1.1")


def test_custom_pt_version_parsing():
    """Test that custom builds of PyTorch can be parsed and compared."""
    assert env.Version("1.13.0a0+08820cb") >= [1, 10]


if __name__ == "__main__":
    unittest.main()
