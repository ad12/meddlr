# -*- coding: utf-8 -*-
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

"""
This file contains components with some default boilerplate logic user may need
in training / testing. They will not work for everyone, but many users may find
them useful.

The behavior of functions/classes in this file is subject to change,
since they are meant to represent the "common default behavior" people need in
their projects.
"""

import argparse
import logging
import os
import warnings
from typing import Sequence

import torch
from fvcore.common.file_io import PathManager

from ss_recon.utils.collect_env import collect_env_info
from ss_recon.utils.env import get_available_gpus, seed_all_rng
from ss_recon.utils.logger import setup_logger

__all__ = [
    "default_argument_parser",
    "default_setup",
    # "DefaultPredictor",
]


def default_argument_parser():
    """
    Create a parser with some common arguments used by detectron2 users.

    Returns:
        argparse.ArgumentParser:
    """
    parser = argparse.ArgumentParser(description="Detectron2 Training")
    parser.add_argument("--config-file", default="", metavar="FILE", help="path to config file")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="whether to attempt to resume from the checkpoint directory",
    )
    parser.add_argument("--eval-only", action="store_true", help="perform evaluation only")
    parser.add_argument(
        "--num-gpus",
        type=int,
        default=1,
        help="number of gpus. overrided by --devices",
    )
    parser.add_argument("--devices", type=int, nargs="*", default=None)
    parser.add_argument("--debug", action="store_true", help="use debug mode")

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

    1. Set up the detectron2 logger
    2. Log basic information about environment, cmdline arguments, and config
    3. Backup the config to the output directory

    Args:
        cfg (CfgNode): the full config to be used
        args (argparse.NameSpace): the command line arguments to be logged
        save_cfg (bool, optional): If `True`, writes config to `cfg.OUTPUT_DIR`.
    """
    cfg.defrost()
    cfg.OUTPUT_DIR = PathManager.get_local_path(cfg.OUTPUT_DIR)
    cfg.freeze()
    output_dir = cfg.OUTPUT_DIR
    if output_dir:
        PathManager.mkdirs(output_dir)

    setup_logger(output_dir, name="fvcore")
    logger = setup_logger(output_dir)

    if args.debug:
        os.environ["SSRECON_DEBUG"] = "True"
        logger.info("Running in debug mode")

    logger.info("Environment info:\n" + collect_env_info())

    logger.info("Command line arguments: " + str(args))
    if hasattr(args, "config_file") and args.config_file != "":
        logger.info(
            "Contents of args.config_file={}:\n{}".format(
                args.config_file, PathManager.open(args.config_file, "r").read()
            )
        )

    if "CUDA_VISIBLE_DEVICES" not in os.environ:
        if args.devices:
            gpus = args.devices
            if not isinstance(gpus, Sequence):
                gpus = [gpus]
        else:
            gpus = get_available_gpus(args.num_gpus)

        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join([str(x) for x in gpus])

        # TODO: Remove this and find a better way to launch the script
        # with the desired gpus.
        if gpus[0] >= 0:
            torch.cuda.set_device(gpus[0])

    logger.info("Running with full config:\n{}".format(cfg))
    if output_dir and save_cfg:
        # Note: some of our scripts may expect the existence of
        # config.yaml in output directory
        path = os.path.join(output_dir, "config.yaml")
        with PathManager.open(path, "w") as f:
            f.write(cfg.dump())
        logger.info("Full config saved to {}".format(os.path.abspath(path)))

    # make sure each worker has a different, yet deterministic seed if specified
    seed_all_rng(None if cfg.SEED < 0 else cfg.SEED)

    # cudnn benchmark has large overhead.
    # It shouldn't be used considering the small size of
    # typical validation set.
    if not (hasattr(args, "eval_only") and args.eval_only):
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
    cfg, exp_id=None, resume=False, project=None, entity=None, job_type="training", use_api=False
):
    import wandb

    logger = logging.getLogger(__name__)

    is_eval = job_type.lower() in ("eval", "evaluation")

    # Find last run if `exp_id` not specified.
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
        sync_tensorboard=True,
        job_type=job_type,
        dir=cfg.OUTPUT_DIR,
    )

    # Resume run and return.
    if exp_id:
        logger.info(f"Loading W&B run {exp_id}")
        wandb.init(id=exp_id, resume="must", **wandb_kwargs)
        return wandb.run

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
