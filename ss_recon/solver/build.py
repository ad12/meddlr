# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
from typing import Any, Dict, List

import torch

from ss_recon.config import CfgNode

from .lr_scheduler import NoOpLR, WarmupCosineLR, WarmupMultiStepLR


def build_optimizer(
    cfg: CfgNode, model: torch.nn.Module
) -> torch.optim.Optimizer:
    """
    Build an optimizer from config.
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
            # NOTE: unlike Detectron v1, we now default BIAS_LR_FACTOR to 1.0
            # and WEIGHT_DECAY_BIAS to WEIGHT_DECAY so that bias optimizer
            # hyperparameters are by default exactly the same as for regular
            # weights.
            lr = cfg.SOLVER.BASE_LR * cfg.SOLVER.BIAS_LR_FACTOR
            weight_decay = cfg.SOLVER.WEIGHT_DECAY_BIAS
        params += [{"params": [value], "lr": lr, "weight_decay": weight_decay}]

    optimizer = _build_opt(params, cfg)
    return optimizer


def _build_opt(params, cfg):
    optim = cfg.SOLVER.OPTIMIZER
    if optim == "SGD":
        return torch.optim.SGD(
            params, cfg.SOLVER.BASE_LR, momentum=cfg.SOLVER.MOMENTUM
        )
    elif optim == "Adam":
        return torch.optim.Adam(params, cfg.SOLVER.BASE_LR)
    else:
        raise ValueError("Optimizer {} not supported".format(optim))


def build_lr_scheduler(
    cfg: CfgNode, optimizer: torch.optim.Optimizer
) -> torch.optim.lr_scheduler._LRScheduler:
    """
    Build a LR scheduler from config.
    """
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
