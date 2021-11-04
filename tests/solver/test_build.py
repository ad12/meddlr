import unittest

from torch.optim import Adam

from meddlr.config import get_cfg
from meddlr.solver import GradAccumOptimizer
from meddlr.solver.build import build_optimizer

from .test_optimizer import build_mock_model


class TestBuildOptimizer(unittest.TestCase):
    def test_build_grad_accumulation(self):
        """Test building optimizer with gradient accumulation."""
        model = build_mock_model()

        cfg = get_cfg()
        cfg.SOLVER.GRAD_ACCUM_ITERS = 4
        optimizer = build_optimizer(cfg, model)
        assert isinstance(optimizer, GradAccumOptimizer)

        cfg = get_cfg()
        cfg.SOLVER.GRAD_ACCUM_ITERS = 1
        optimizer = build_optimizer(cfg, model)
        assert isinstance(optimizer, Adam)
