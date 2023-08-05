import inspect

import pytest
import torch
from torch.optim import Adam

from meddlr.config import get_cfg
from meddlr.solver import GradAccumOptimizer
from meddlr.solver.build import OPTIMIZER_REGISTRY, build_optimizer

from .test_optimizer import build_mock_model


def test_build_grad_accumulation():
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


@pytest.mark.parametrize(
    "optim",
    [
        # torch optimizers
        "Adadelta",
        "Adagrad",
        "Adam",
        "AdamW",
        "SparseAdam",
        "Adamax",
        "ASGD",
        "SGD",
        "RAdam",
        "Rprop",
        "RMSprop",
        "NAdam",
        # custom optimizers
        "SophiaG",
    ],
)
def test_build_optimizers(optim):
    model = build_mock_model()

    cfg = get_cfg()
    cfg.SOLVER.OPTIMIZER = optim
    optimizer = build_optimizer(cfg, model)

    if optim in OPTIMIZER_REGISTRY:
        klass = OPTIMIZER_REGISTRY.get(optim)
    elif hasattr(torch.optim, optim):
        klass = getattr(torch.optim, optim)

    assert isinstance(optimizer, klass)

    sig = inspect.signature(klass)
    # Betas should default to that of the optimizer.
    if "betas" in sig.parameters:
        assert optimizer.defaults["betas"] == sig.parameters["betas"].default
    if "momentum" in sig.parameters:
        assert optimizer.defaults["momentum"] == cfg.SOLVER.MOMENTUM


def test_build_optimizer_with_kwargs():
    model = build_mock_model()

    cfg = get_cfg()
    cfg.SOLVER.OPTIMIZER = "Adam"
    betas = (0, 0.4)
    optimizer = build_optimizer(cfg, model, betas=betas)
    assert optimizer.defaults["betas"] == betas
