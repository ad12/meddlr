import os
import unittest

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


if __name__ == "__main__":
    unittest.main()
