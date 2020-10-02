"""Run inference on test set scans.

This consists of comparing both zero-filled recon and DL-recon to fully-sampled
scans. All comparisons are done per volume (not per slice).

Supported metrics include:
    - ssim
    - ssim_center (ssim_50)
    - psnr
    - nrmse
"""
import itertools
import json
import logging
import os
import datetime
import time

import h5py
import numpy as np
import pandas as pd
import silx.io.dictdump as silx_dd
from tabulate import tabulate
import torch

from ss_recon.config import get_cfg
from ss_recon.engine import (
    DefaultTrainer,
    default_argument_parser,
    default_setup,
)
from ss_recon.engine.defaults import init_wandb_run
from ss_recon.checkpoint import DetectionCheckpointer
from ss_recon.evaluation import metrics
from ss_recon.evaluation.testing import check_consistency, find_weights, SUPPORTED_VAL_METRICS
from ss_recon.data.build import build_data_loaders_per_scan, build_recon_val_loader
from ss_recon.utils.logger import setup_logger, log_every_n_seconds
from ss_recon.evaluation import ReconEvaluator, DatasetEvaluators, inference_on_dataset

import ss_recon.utils.complex_utils as cplx
from ss_recon.utils.transforms import SenseModel

_FILE_NAME = os.path.splitext(os.path.basename(__file__))[0]
_LOGGER_NAME = "{}.{}".format(_FILE_NAME, __name__)
#logger = logging.getLogger(_LOGGER_NAME)
logger = None  # initialize in setup()


class ZFReconEvaluator(ReconEvaluator):
    """Zero-filled recon evaluator."""
    def process(self, inputs, outputs):
        zf_out = {k: outputs[k] for k in ("target", "metadata")}
        zf_out["pred"] = outputs["zf_pred"]
        return super().process(inputs, zf_out)


def setup(args):
    """
    Create configs and perform basic setups.
    We do not save the config.
    """
    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)
    opts = args.opts
    if opts and opts[0] == "--":
        opts = opts[1:]
    cfg.merge_from_list(opts)
    cfg.freeze()
    default_setup(cfg, args, save_cfg=False)

    # Setup logger for test results
    global logger
    dirname = "test_results"
    logger = setup_logger(os.path.join(cfg.OUTPUT_DIR, dirname), name=_FILE_NAME)
    return cfg


@torch.no_grad()
def eval(cfg, model, zero_filled: bool = False, include_noise=False, use_wandb=False):
    # TODO: Set up W&B configuration.
    # if use_wandb:
    #     run = init_wandb_run(cfg, resume=True, job_type="eval", use_api=True)

    device = cfg.MODEL.DEVICE
    model = model.to(device)
    model = model.eval()

    output_dir = os.path.join(cfg.OUTPUT_DIR, "test_results")

    accelerations = cfg.AUG_TEST.UNDERSAMPLE.ACCELERATIONS
    all_results = []

    # Returns average or each scan
    group_by_scan = True

    noise_vals = (0,) + cfg.MODEL.CONSISTENCY.AUG.NOISE.STD_DEV if include_noise else (0,)
    values = itertools.product(cfg.DATASETS.TEST, cfg.AUG_TEST.UNDERSAMPLE.ACCELERATIONS, noise_vals)
    for dataset_name, acc, noise_level in values:
        # Assign the current acceleration
        s_cfg = cfg.clone()
        s_cfg.defrost()
        s_cfg.AUG_TRAIN.UNDERSAMPLE.ACCELERATIONS = (acc,)
        s_cfg.MODEL.CONSISTENCY.AUG.NOISE.STD_DEV = (noise_level,)
        s_cfg.freeze()

        # Build a recon val loader
        dataloader = build_recon_val_loader(
            s_cfg, dataset_name, as_test=True, add_noise=noise_level > 0
        )

        # Build evaluators
        evaluators = [ReconEvaluator(dataset_name, s_cfg, group_by_scan=group_by_scan)]
        # TODO: add support for multiple evaluators.
        if zero_filled:
            evaluators.append(ZFReconEvaluator(dataset_name, s_cfg, group_by_scan=group_by_scan))
        evaluators = DatasetEvaluators(evaluators, as_list=True)

        results = inference_on_dataset(model, dataloader, evaluators)
        results = [
            pd.DataFrame(x).T.reset_index().rename(columns={"index": "scan_name"})
            for x in results
        ]

        results[0]["Method"] = s_cfg.MODEL.META_ARCHITECTURE
        if zero_filled:
            results[1]["Method"] = "Zero-Filled"
        scan_results = pd.concat(results, ignore_index=True)
        scan_results["Acceleration"] = acc
        scan_results["dataset"] = dataset_name
        scan_results["Noise Level"] = noise_level
        logger.info("\n" + tabulate(scan_results, headers=scan_results.columns))

        all_results.append(scan_results)

        # Currently don't support writing data because it takes too long
        # logger.info("Saving data...")
        # file_path = os.path.join(output_dir, dataset_name, "{}.h5".format(scan_name))
        # os.makedirs(os.path.dirname(file_path), exist_ok=True)

    all_results = pd.concat(all_results, ignore_index=True)
    all_results.to_csv(os.path.join(output_dir, "metrics.csv"), mode="w")
    logger.info("Summary:\n{}".format(tabulate(all_results, headers=all_results.columns)))


def main(args):
    cfg = setup(args)
    model = DefaultTrainer.build_model(cfg)
    weights = cfg.MODEL.WEIGHTS if cfg.MODEL.WEIGHTS else find_weights(cfg, args.metric, args.iter_limit)
    model = model.to(cfg.MODEL.DEVICE)
    DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
        weights, resume=args.resume
    )

    # See https://github.com/pytorch/pytorch/issues/42300
    logger.info("Checking weights were properly loaded...")
    check_consistency(torch.load(weights)["model"], model)

    logger.info("\n\n==============================")
    logger.info("Loading weights from {}".format(weights))

    eval(cfg, model, args.zero_filled, args.noise)


if __name__ == "__main__":
    parser = default_argument_parser()
    parser.add_argument(
        "--metric",
        type=str,
        default="",
        help=(
            "Val metric used to select weights. "
            "Defaults to recon loss. "
            "Ignored if `MODEL.WEIGHTS` specified"
        ),
        choices=list(SUPPORTED_VAL_METRICS.keys()),
    )
    parser.add_argument(
        "--zero-filled",
        action="store_true",
        help="Calculate metrics for zero-filled images"
    )
    parser.add_argument(
        "--noise",
        action="store_true",
        help="Include noise evaluation"
    )
    parser.add_argument(
        "--iter-limit", default=None, type=int, help="Iteration limit. Chooses weights below this time point."
    )
    # parser.add_argument(
    #     "--wandb", action="store_true", help="Log to W&B during evaluation"
    # )

    args = parser.parse_args()
    print("Command Line Args:", args)
    main(args)
