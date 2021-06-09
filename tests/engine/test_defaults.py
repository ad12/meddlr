import os
import unittest

import torch

from ss_recon.config.config import get_cfg
from ss_recon.engine.defaults import _init_reproducible_mode
from ss_recon.utils import env


class TestDefaultSetup(unittest.TestCase):
    """Test that default setup and initialization works as expected."""

    _env = None

    @classmethod
    def setUpClass(cls):
        cls._env = dict(os.environ)

    @classmethod
    def tearDownClass(cls):
        cls._reset_env_vars()

    @classmethod
    def _reset_env_vars(cls):
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

    def test_init_reproducible_mode(self):
        """Test that we properly initialize reproducibility."""
        base_cfg = get_cfg()
        base_cfg.defrost()
        base_cfg.SEED = -1
        base_cfg.DATALOADER.SUBSAMPLE_TRAIN.SEED = -1
        base_cfg.freeze()

        os.environ["SSRECON_REPRO"] = ""
        cfg = base_cfg.clone()
        _init_reproducible_mode(cfg, eval_only=False)
        assert cfg.SEED > 0
        assert cfg.DATALOADER.SUBSAMPLE_TRAIN.SEED > 0
        assert torch.backends.cudnn.deterministic
        assert not torch.backends.cudnn.benchmark
        assert env.is_repro()
        self._reset_env_vars()

        cfg = base_cfg.clone()
        cfg.defrost()
        cfg.SEED = 1000
        cfg.freeze()
        _init_reproducible_mode(cfg, eval_only=False)
        assert cfg.SEED == 1000
        assert cfg.DATALOADER.SUBSAMPLE_TRAIN.SEED > 0
        assert torch.backends.cudnn.deterministic
        assert not torch.backends.cudnn.benchmark
        self._reset_env_vars()

        cfg = base_cfg.clone()
        cfg.defrost()
        cfg.CUDNN_BENCHMARK = True
        cfg.freeze()
        _init_reproducible_mode(cfg, eval_only=True)
        assert torch.backends.cudnn.benchmark
        self._reset_env_vars()

        cfg = base_cfg.clone()
        cfg.defrost()
        cfg.CUDNN_BENCHMARK = True
        cfg.freeze()
        _init_reproducible_mode(cfg, eval_only=False)
        assert not torch.backends.cudnn.benchmark
        self._reset_env_vars()


if __name__ == "__main__":
    unittest.main()
