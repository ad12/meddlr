# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import inspect
from typing import Any, Dict, List

import torch

from meddlr.config import CfgNode
from meddlr.solver.lr_scheduler import NoOpLR, WarmupCosineLR, WarmupMultiStepLR
from meddlr.utils.registry import Registry

__all__ = ["build_optimizer", "build_lr_scheduler"]

OPTIMIZER_REGISTRY = Registry("OPTIMIZER")  # noqa F401 isort:skip
OPTIMIZER_REGISTRY.__doc__ = """
Registry for custom optimizers.
See meddlr/solver/optimizer for custom optimizers.
"""


def build_optimizer(cfg: CfgNode, model: torch.nn.Module, **kwargs) -> torch.optim.Optimizer:
    """Build an optimizer from config.

    Args:
        cfg: The config to build the model from.
        model: The model who's parameters to manage
        **kwargs: Keyword arguments for optimizer.
            These will override arguments in the config.

    Returns:
        torch.optim.Optimizer: The optimizer.
    """
    params: List[Dict[str, Any]] = []
    for key, value in model.named_parameters():
        if not value.requires_grad:
            continue
        lr = cfg.SOLVER.BASE_LR
        weight_decay = cfg.SOLVER.WEIGHT_DECAY
        if key.endswith("norm.weight") or key.endswith("norm.bias"):
            weight_decay = cfg.SOLVER.WEIGHT_DECAY_NORM
        elif key.endswith(".bias"):
            lr = cfg.SOLVER.BASE_LR * cfg.SOLVER.BIAS_LR_FACTOR
            weight_decay = cfg.SOLVER.WEIGHT_DECAY_BIAS
        params += [{"params": [value], "lr": lr, "weight_decay": weight_decay}]

    optimizer = _build_opt(params, cfg, **kwargs)
    return optimizer


def _build_opt(params, cfg, **kwargs):
    from meddlr.solver.optimizer import GradAccumOptimizer

    opt_kwargs = {
        "lr": cfg.SOLVER.BASE_LR,
        # Weight decay is handled by build_optimizer.
        "weight_decay": 0.0,
        "momentum": cfg.SOLVER.MOMENTUM,
    }
    # TODO: Find a better way to indicate values that we don't want to pass in.
    if cfg.SOLVER.BETAS:
        opt_kwargs["betas"] = cfg.SOLVER.BETAS
    opt_kwargs.update(kwargs)

    # TODO: Add support for "torch/<optim>" to default to torch implementation.
    # Only need to implement when meddlr has the same name as a torch optimizer.
    optim = cfg.SOLVER.OPTIMIZER
    if optim in OPTIMIZER_REGISTRY:
        klass = OPTIMIZER_REGISTRY.get(optim)
    elif hasattr(torch.optim, optim):
        klass = getattr(torch.optim, optim)
    else:
        raise ValueError(f"Unknown {optim} not supported")

    sig = inspect.signature(klass)
    opt_kwargs = {k: v for k, v in opt_kwargs.items() if k in sig.parameters}
    optimizer = klass(params, **opt_kwargs)

    # Gradient accumulation wrapper.
    num_grad_accum = cfg.SOLVER.GRAD_ACCUM_ITERS
    if num_grad_accum < 1:
        raise ValueError(f"cfg.SOLVER.GRAD_ACCUM_ITERS must be >= 1. Got {num_grad_accum}")
    elif num_grad_accum > 1:
        optimizer = GradAccumOptimizer(optimizer, num_grad_accum)
    return optimizer


def build_lr_scheduler(
    cfg: CfgNode, optimizer: torch.optim.Optimizer
) -> torch.optim.lr_scheduler._LRScheduler:
    """
    Build a LR scheduler from config.
    """
    if not isinstance(optimizer, torch.optim.Optimizer):
        if hasattr(optimizer, "optimizer"):
            optimizer = optimizer.optimizer

    name = cfg.SOLVER.LR_SCHEDULER_NAME
    if not name:
        return NoOpLR(optimizer)
    elif name == "WarmupMultiStepLR":
        return WarmupMultiStepLR(
            optimizer,
            cfg.SOLVER.STEPS,
            cfg.SOLVER.GAMMA,
            warmup_factor=cfg.SOLVER.WARMUP_FACTOR,
            warmup_iters=cfg.SOLVER.WARMUP_ITERS,
            warmup_method=cfg.SOLVER.WARMUP_METHOD,
        )
    elif name == "WarmupCosineLR":
        return WarmupCosineLR(
            optimizer,
            cfg.SOLVER.MAX_ITER,
            warmup_factor=cfg.SOLVER.WARMUP_FACTOR,
            warmup_iters=cfg.SOLVER.WARMUP_ITERS,
            warmup_method=cfg.SOLVER.WARMUP_METHOD,
        )
    elif name in ("StepLR", "WarmupStepLR"):
        if len(cfg.SOLVER.STEPS) != 1:
            raise ValueError("SOLVER.STEPS must have single value for StepLR")
        lr_step = cfg.SOLVER.STEPS[0]
        max_iter = cfg.SOLVER.MAX_ITER
        num_steps = max_iter // lr_step
        steps = [lr_step * x for x in range(1, 1 + num_steps)]

        # No warmup if plain StepLR.
        warmup_iters = cfg.SOLVER.WARMUP_ITERS if name == "WarmupStepLR" else 0
        return WarmupMultiStepLR(
            optimizer,
            steps,
            cfg.SOLVER.GAMMA,
            warmup_factor=cfg.SOLVER.WARMUP_FACTOR,
            warmup_iters=warmup_iters,
            warmup_method=cfg.SOLVER.WARMUP_METHOD,
        )
    else:
        raise ValueError("Unknown LR scheduler: {}".format(name))
