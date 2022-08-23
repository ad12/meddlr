"""Default setup in meddlr.

This file contains components with some default boilerplate logic user may need
in training / testing. They will not work for everyone, but many users may find
them useful.

The behavior of functions/classes in this file is subject to change,
since they are meant to represent the "common default behavior" people need in
their projects.
"""

import argparse
import os
import re
import warnings
from typing import Mapping, Sequence

import torch

from meddlr.config.config import CfgNode
from meddlr.utils import env
from meddlr.utils.collect_env import collect_env_info
from meddlr.utils.env import get_available_gpus, seed_all_rng
from meddlr.utils.logger import setup_logger

__all__ = ["default_argument_parser", "default_setup"]

_PATH_MANAGER = env.get_path_manager()


def default_argument_parser() -> argparse.ArgumentParser:
    """Create a parser with some common arguments.

    Returns:
        argparse.ArgumentParser
    """
    parser = argparse.ArgumentParser(description="SS-Recon Training")
    parser.add_argument("--config-file", default="", metavar="FILE", help="path to config file")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="whether to attempt to resume from the checkpoint directory",
    )
    parser.add_argument(
        "--restart-iter",
        action="store_true",
        help="restart iteration count when loading checkpointed weights",
    )
    parser.add_argument("--eval-only", action="store_true", help="perform evaluation only")
    parser.add_argument(
        "--num-gpus", type=int, default=1, help="number of gpus. overrided by --devices"
    )
    parser.add_argument("--devices", type=int, nargs="*", default=None)
    parser.add_argument("--debug", action="store_true", help="use debug mode")
    parser.add_argument(
        "--reproducible", "--repro", action="store_true", help="activate reproducible mode"
    )
    parser.add_argument(
        "--auto-version", action="store_true", help="auto-configure experiment version"
    )

    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )
    return parser


def default_setup(cfg, args, save_cfg: bool = True):
    """
    Perform some basic common setups at the beginning of a job, including:

    1. Set up the meddlr logger
    2. Log basic information about environment, cmdline arguments, and config
    3. Backup the config to the output directory
    4. Enables debug mode if ``args.debug==True``
    5. Enables reproducible model if ``args.reproducible==True``

    Args:
        cfg (CfgNode): the full config to be used
        args (argparse.NameSpace): the command line arguments to be logged
        save_cfg (bool, optional): If `True`, writes config to `cfg.OUTPUT_DIR`.

    Note:
        Project-specific environment variables are modified by this function.
        ``cfg`` is also modified in-place.
    """
    is_repro_mode = (
        env.is_repro() if env.is_repro() else (hasattr(args, "reproducible") and args.reproducible)
    )
    eval_only = hasattr(args, "eval_only") and args.eval_only
    auto_version = hasattr(args, "auto_version") and args.auto_version

    cfg.defrost()
    cfg.format_fields(unroll=True)
    cfg.freeze()

    if auto_version and not eval_only:
        # TODO: Handle race conditions for generating output directories.
        # NOTE: This is not compatible with DDP.
        out_path = _PATH_MANAGER.get_local_path(cfg.OUTPUT_DIR)
        exp_name = cfg.DESCRIPTION.EXP_NAME
        has_exp_name = bool(exp_name)
        if args.debug:
            out_path = os.path.join(out_path, "debug")
            if "debug" not in exp_name:
                exp_name = f"{exp_name}/debug"
        exp_version = _get_next_experiment_version(out_path)
        cfg.defrost()
        cfg.OUTPUT_DIR = os.path.join(out_path, f"version_{exp_version:03d}")
        if has_exp_name:
            cfg.DESCRIPTION.EXP_NAME = f"{exp_name}/v{exp_version}"
        cfg.freeze()

    # Update config parameters before saving.
    cfg.defrost()
    cfg.OUTPUT_DIR = _PATH_MANAGER.get_local_path(cfg.OUTPUT_DIR)
    if is_repro_mode:
        init_reproducible_mode(cfg, eval_only)
    cfg.freeze()

    output_dir = cfg.OUTPUT_DIR
    if output_dir:
        _PATH_MANAGER.mkdirs(output_dir)

    setup_logger(output_dir, name="fvcore")
    logger = setup_logger(output_dir)

    if args.debug:
        os.environ["MEDDLR_DEBUG"] = "True"
        logger.info("Running in debug mode")
    if is_repro_mode:
        os.environ["MEDDLR_REPRO"] = "True"
        logger.info("Running in reproducible mode")

    logger.info("Environment info:\n" + collect_env_info())

    logger.info("Command line arguments: " + str(args))
    if hasattr(args, "config_file") and args.config_file != "":
        logger.info(
            "Contents of args.config_file={}:\n{}".format(
                args.config_file, _PATH_MANAGER.open(args.config_file, "r").read()
            )
        )

    if "CUDA_VISIBLE_DEVICES" not in os.environ:
        if args.devices:
            gpus = args.devices
            if not isinstance(gpus, Sequence):
                gpus = [gpus]
        else:
            gpus = get_available_gpus(args.num_gpus)
            if len(gpus) != args.num_gpus:
                raise ValueError(f"Could not find '{args.num_gpus}' free gpus")

        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join([str(x) for x in gpus]) if gpus else "-1"

        # TODO: Remove this and find a better way to launch the script
        # with the desired gpus.
        if gpus and gpus[0] >= 0:
            torch.cuda.set_device(gpus[0])

    logger.info("Running with full config:\n{}".format(cfg))
    if output_dir and save_cfg:
        # Note: some of our scripts may expect the existence of
        # config.yaml in output directory
        path = os.path.join(output_dir, "config.yaml")
        with _PATH_MANAGER.open(path, "w") as f:
            f.write(cfg.dump())
        logger.info("Full config saved to {}".format(os.path.abspath(path)))

    # make sure each worker has a different, yet deterministic seed if specified
    seed_all_rng(None if cfg.SEED < 0 else cfg.SEED)

    # cudnn benchmark has large overhead.
    # It shouldn't be used considering the small size of
    # typical validation set.
    if not eval_only:
        torch.backends.cudnn.benchmark = cfg.CUDNN_BENCHMARK


def find_wandb_exp_id(cfg):
    """Helper function to find W&B experiment id.

    Args:
        cfg (CfgNode): The config.

    Returns:
        exp_id (Optional[str]): Returns `None` if experiment id
            cannot be determined.
    """
    filepath = os.path.join(cfg.OUTPUT_DIR, "wandb_id")
    exp_id = None
    # Option 1 (preferred): Check if wandb experiment file exists.
    if os.path.isfile(filepath):
        with open(filepath, "r") as f:
            exp_id = f.read()
        return exp_id

    # Option 2: Navigate folder structure to find wandb relevant folders.
    # Note this may not be robust as `wandb` may change their folder structure.
    base_dir = os.path.join(cfg.OUTPUT_DIR, "wandb", "latest-run")
    if os.path.isdir(base_dir):
        run_files = [x for x in os.listdir(base_dir) if os.path.isfile(os.path.join(base_dir, x))]
        assert len(run_files) == 1, "Found multiple ({}) W&B run files:\n\t{}".format(
            len(run_files), "\n\t".join(run_files)
        )
        exp_id = os.path.splitext(run_files[0])[0].split("-")[1]
    return exp_id


def init_wandb_run(
    cfg,
    exp_id=None,
    resume=False,
    project=None,
    entity=None,
    job_type: str = "training",
    sync_tensorboard: bool = None,
    use_api: bool = False,
):
    """Helper function to initialize Weights & Biases based on project file structure.

    TODO:
        1. Configure `sync_tensorboard` with pytorch lightning once
            new W&B version comes out. (escalated to W&B)
        2. Configure `resume` functionality to log to the same run
            without overwriting. (escalated to W&B)
        3. Configure `is_eval` to use `wandb.init` (related to 2)

    """
    import wandb

    if sync_tensorboard is not None:
        os.environ["WANDB_SYNC_TENSORBOARD"] = "true" if sync_tensorboard else "false"
    sync_tensorboard = os.environ.get("WANDB_SYNC_TENSORBOARD", False)

    # Find last run if `exp_id` not specified.
    is_eval = job_type.lower() in ("eval", "evaluation")
    if (resume or is_eval) and not exp_id:
        exp_id = find_wandb_exp_id(cfg)

    # If evaluation, do not run wandb.init. Just return run.
    if use_api:
        if not exp_id:
            raise ValueError("No experiment id found.")
        api = wandb.Api()
        return api.run(f"{project}/{entity}/{exp_id}")

    if project is None:
        project = cfg.DESCRIPTION.PROJECT_NAME
    else:
        warnings.warn(
            "Setting project name with `project` is deprecated. "
            "Use DESCRIPTION.PROJECT_NAME in config instead.",
            DeprecationWarning,
        )
    if entity is None:
        entity = cfg.DESCRIPTION.ENTITY_NAME
    else:
        warnings.warn(
            "Setting entity name with `entity` is deprecated. "
            "Use DESCRIPTION.ENTITY_NAME in config instead.",
            DeprecationWarning,
        )

    # Keyword args to share between resumed and new runs.
    wandb_kwargs = dict(
        config=cfg,
        project=project,
        entity=entity,
        sync_tensorboard=sync_tensorboard,  # TODO: Change when W&B bug fixed
        job_type=job_type,
        dir=cfg.OUTPUT_DIR,
    )

    # Resume run and return.
    if exp_id:
        assert False, "We should not reach this until TODOs 1-3 are resolved."

    # Create new run.
    exp_name = cfg.DESCRIPTION.EXP_NAME
    if not exp_name:
        warnings.warn("DESCRIPTION.EXP_NAME not specified. Defaulting to basename...")
        exp_name = os.path.basename(cfg.OUTPUT_DIR)

    # Write the wandb id to a file.
    exp_id = wandb.util.generate_id()
    with open(os.path.join(cfg.OUTPUT_DIR, "wandb_id"), "w") as f:
        f.write(exp_id)

    wandb.init(
        id=exp_id,
        name=exp_name,
        tags=cfg.DESCRIPTION.TAGS,
        notes=cfg.DESCRIPTION.BRIEF,
        **wandb_kwargs,
    )
    return wandb.run


def init_reproducible_mode(cfg: CfgNode, eval_only: bool):
    """Activates reproducible mode and sets appropriate config paraemters.

    This method does the following:
        1. Sets environment variable indiciating project is in reproducible mode.
        2. Sets all seeds in the ``cfg`` if they are not initialized.
        3. Enables cuda benchmarking ``eval_only=False`` and ``cfg.CUDNN_BENCHMARK=True``.
        4. Sets ``torch.backends.cudann.deterministic=True``.

    Seed fields in ``cfg`` are indicated by keys that end with ``"SEED"``
    and whose corresponding value is an integer. Below are some examples of
    fields that would match as a seed field:

        * cfg.SEED = -1
        * cfg.XX.YY.SEED = -1
        * cfg.A_SEED = -1
        * cfg.SEED_VAL = -1  # this would not match, does not end with ``'SEED'``.
        * cfg.SEED = "alpha"  # this would not match, value is not an int.

    Args:
        cfg (CfgNode): The full config. This will be modified in place.
        eval_only (bool): If ``True``, initialize reproducible
            mode in an evaluation only setting.

    Note:
        This method should typically be called through :func:`default_setup`.
    """
    os.environ["MEDDLR_REPRO"] = "True"

    orig_cfg = cfg.clone()
    cfg.defrost()

    # Set all seeds in the config if they are not set.
    seed = seed_all_rng(None if cfg.SEED < 0 else cfg.SEED)
    _set_all_seeds(cfg, seed)

    # Turn off cuda benchmarking if this run is not only for evaluation.
    # If eval_only, default to current config value.
    cfg.CUDNN_BENCHMARK = False if not eval_only else orig_cfg.CUDNN_BENCHMARK
    torch.backends.cudnn.benchmark = cfg.CUDNN_BENCHMARK

    # Turn on deterministic mode.
    torch.backends.cudnn.deterministic = True

    cfg.freeze()


def _set_all_seeds(cfg, seed_val):
    for key, value in cfg.items():
        if re.match("^.*SEED$", key) and isinstance(value, int) and value < 0:
            cfg.__setattr__(key, seed_val)
        if isinstance(value, Mapping):
            _set_all_seeds(value, seed_val)


def _get_next_experiment_version(path):
    """Get next version number for the experiment in the path.

    Args:
        path (str): Path to the experiment directory.

    Returns:
        int: Next version number for the experiment.
    """
    path = _PATH_MANAGER.get_local_path(path)
    if not os.path.isdir(path):
        max_version = 0
    else:
        versions = [int(x.split("_")[1]) for x in os.listdir(path) if x.startswith("version")]
        max_version = max(versions) if versions else 0
    return max_version + 1
