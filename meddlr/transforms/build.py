import inspect
import multiprocessing as mp
from typing import Any, Dict, List, Mapping, Union

import numpy as np
from fvcore.common.registry import Registry

from meddlr.config import CfgNode
from meddlr.transforms.tf_scheduler import TFScheduler, WarmupMultiStepTF, WarmupTF
from meddlr.transforms.transform import Transform
from meddlr.transforms.transform_gen import TransformGen

TRANSFORM_REGISTRY = Registry("TRANSFORM_FUNCS")  # noqa F401 isort:skip
TRANSFORM_REGISTRY.__doc__ = """
Registry for transform functions.

The registered object will be called with `obj(cfg)`
and expected to return a `nn.Module` object.
"""


def build_transforms(
    cfg: CfgNode, tfm_cfgs: List[Mapping[str, Any]], seed=None, **kwargs
) -> List[Union[Transform, TransformGen]]:
    # This should only be used for training right now.
    is_single = isinstance(tfm_cfgs, Mapping)
    if is_single:
        tfm_cfgs = [tfm_cfgs]

    tfms = [_build_transform(cfg, tfm_cfg, **kwargs) for tfm_cfg in tfm_cfgs]

    if seed is not None:
        seed_tfm_gens(tfms, seed=seed)

    if is_single:
        tfms = tfms[0]

    return tfms


def _build_transform(cfg: CfgNode, tfm_cfg: Dict[str, Any], **kwargs):
    tfm_cfg = tfm_cfg.copy()
    name: str = tfm_cfg.pop("name")
    schedulers_cfgs = tfm_cfg.pop("schedulers", tfm_cfg.pop("scheduler", None))
    init_args: Mapping[str, Any] = tfm_cfg

    klass = TRANSFORM_REGISTRY.get(name)
    parameters = inspect.signature(klass.__init__).parameters
    init_args.update(**{k: v for k, v in kwargs.items() if k in parameters})

    if hasattr(klass, "from_dict"):
        tfm = klass.from_dict(cfg, init_args, **kwargs)
    else:
        try:
            tfm = klass(**init_args)
        except TypeError as e:
            raise TypeError(f"Failed to initialize {klass.__name__} - {e}")
        except RecursionError as e:
            raise RecursionError(f"Failed to initialize {klass.__name__} - {e}")
    if isinstance(tfm, Transform) or schedulers_cfgs is None:
        return tfm

    tfm_gen: TransformGen = tfm
    if isinstance(schedulers_cfgs, Mapping):
        schedulers_cfgs = [schedulers_cfgs]
    schedulers = [build_scheduler(cfg, scfg, tfm_gen) for scfg in schedulers_cfgs]
    tfm_gen.register_schedulers(schedulers)
    return tfm_gen


def build_scheduler(
    cfg: CfgNode, scheduler_cfg: Dict[str, Any], tfm_gen: TransformGen
) -> Dict[str, TFScheduler]:
    scheduler_cfg = scheduler_cfg.copy()
    name: str = scheduler_cfg.pop("name")
    params: List[str] = scheduler_cfg.pop("params", None)

    if not name:
        return None
    elif name == "WarmupMultiStepLR":
        return WarmupMultiStepTF(tfm_gen, params=params, **scheduler_cfg)
    elif name == "WarmupTF":
        return WarmupTF(tfm_gen, params=params, **scheduler_cfg)
    elif name == "WarmupStepTF":
        step = scheduler_cfg["warmup_milestones"]
        delay_iters = scheduler_cfg.pop("delay_iters", 0)
        if isinstance(step, int):
            step = (step,)
        if len(step) != 1:
            raise ValueError("step must have single value for WarmupStepTF")
        step = step[0]
        max_iter = scheduler_cfg.pop(
            "max_iter", scheduler_cfg.pop("max_iters", cfg.SOLVER.MAX_ITER)
        )
        num_steps = (max_iter - delay_iters) // step
        steps = tuple(step * x + delay_iters for x in range(1, 1 + num_steps))
        if delay_iters > 0:
            steps = (delay_iters,) + steps
        init_kwargs = dict(tfm=tfm_gen, warmup_milestones=steps, params=params)
        init_kwargs.update({k: v for k, v in scheduler_cfg.items() if k not in init_kwargs})
        return WarmupMultiStepTF(**init_kwargs)

    raise ValueError("Unknown LR scheduler: {}".format(name))


def seed_tfm_gens(tfms, seed):
    # Seed all transform generators with unique, but reproducible seeds.
    # Do not change the scaling constant (1e10).
    rng = np.random.RandomState(seed)
    if seed is not None:
        for t in tfms:
            if isinstance(t, TransformGen):
                t.seed(int(rng.rand() * 1e10))


def build_iter_func(batch_size, num_workers):
    def get_iter(step):
        # Worker id is 1-indexed by default.
        worker_id = int(mp.current_process().name.split("-")[1])
        curr_iter = (step * num_workers + worker_id - 1) // batch_size
        return curr_iter

    return get_iter
