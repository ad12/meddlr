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
from copy import deepcopy
from typing import Any, Dict, Sequence

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
from ss_recon.utils.general import find_experiment_dirs

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


def find_metrics(metrics: pd.DataFrame, params: Dict[str, Any], ignore_missing=False, ignore_case=True):
    """Find subset of metrics dictionary that matches parameter configuration.

    Args:
        metrics (pd.DataFrame): Will be filtered based on column values.
        params (Dict[str, Any]): Parameter values to filter by.
            Keys should correspond to column names in `metrics`.
        ignore_missing (bool, optional): If `True`, ignores filtering by
            columns that are missing. 
        ignore_case (bool, optional): If `True`, ignores the column casing.
            Raises `ValueError` if two columns have the same lower case
            form.

    Returns:
        df (pd.DataFrame): The remaining dataframe after filtering.
    """

    df = deepcopy(metrics)
    if ignore_case:
        column_map = {x: x.lower() for x in df.columns}
        df = df.rename(columns=column_map)
        params = {k.lower(): v for k, v in params.items()}

    for k, v in params.items():
        if k not in df.columns:
            if ignore_missing:
                continue
            else:
                raise KeyError(f"No column {k} in `metrics`")
        df = df[df[k] == v]
    
    # Undo matching by lower case.
    if ignore_case:
        df = df.rename(columns={v: k for k, v in column_map.items()})

    return df


def update_metrics(metrics_new: pd.DataFrame, metrics_old: pd.DataFrame, on: Sequence[str]):
    """Update a previous metrics version with the new one.

    Metrics that were previously computed, may not be recomputed.
    To avoid overwriting them when writing to a csv, we want to
    port over any old metrics that we did not recompute.
    """
    # We currently do not support missing columns.
    missing_cols = [k not in metrics_old.columns for k in on]
    if any(missing_cols):
        raise KeyError(f"Column(s) {missing_cols} not found in `metrics_old`")
    missing_cols = [k not in metrics_new.columns for k in on]
    if any(missing_cols):
        raise KeyError(f"Column(s) {missing_cols} not found in `metrics_new`")

    # Find combination of columns to select on that is not
    # available in the new metrics, but is available in the
    # old metrics.
    old_metrics_combos = list(itertools.product(*[metrics_old[k].unique().tolist() for k in on]))
    new_metrics_combos = list(itertools.product(*[metrics_new[k].unique().tolist() for k in on]))

    to_prepend = []
    for combo in old_metrics_combos:
        if combo not in new_metrics_combos:
            combo_as_dict = {k: v for k, v in zip(on, combo)}
            to_prepend.append(find_metrics(metrics_old, combo_as_dict))
    
    if len(to_prepend) > 0:
        to_prepend = pd.concat(to_prepend, ignore_index=True)
        metrics = pd.concat([to_prepend, metrics_new], ignore_index=True)
    else:
        metrics = metrics_new
    return metrics


@torch.no_grad()
def eval(cfg, args, model, weights_basename, criterion, best_value):
    zero_filled = args.zero_filled
    noise_arg = args.noise.lower()
    include_noise = noise_arg != "false"
    noise_sweep_vals = args.sweep_vals
    overwrite = args.overwrite
    # TODO: Set up W&B configuration.
    # use_wandb = args.use_wandb
    # if use_wandb:
    #     run = init_wandb_run(cfg, resume=True, job_type="eval", use_api=True)

    device = cfg.MODEL.DEVICE
    model = model.to(device)
    model = model.eval()

    # Get and load metrics file
    output_dir = os.path.join(cfg.OUTPUT_DIR, "test_results")
    metrics_file = os.path.join(output_dir, "metrics.csv")
    if not overwrite and os.path.isfile(metrics_file):
        metrics = pd.read_csv(metrics_file, index_col=0)
    else:
        metrics = None

    # Returns average or each scan
    group_by_scan = True

    # Find range of noise values to search
    if include_noise:
        noise_vals = [0] + noise_sweep_vals if noise_arg == "sweep" else [0]
        noise_vals += list(cfg.MODEL.CONSISTENCY.AUG.NOISE.STD_DEV)
        noise_vals = sorted(list(set(noise_vals)))
    else:
        noise_vals = [0]

    accelerations = cfg.AUG_TEST.UNDERSAMPLE.ACCELERATIONS
    values = itertools.product(cfg.DATASETS.TEST, cfg.AUG_TEST.UNDERSAMPLE.ACCELERATIONS, noise_vals)
    all_results = []
    for dataset_name, acc, noise_level in values:
        # Check if the current configuration already has metrics computed
        # If so, dont recompute
        params = {"Acceleration": acc, "dataset": dataset_name, "Noise Level": noise_level, "weights": weights_basename}
        if metrics is not None:
            try:
                existing_metrics = find_metrics(metrics, params)
            except KeyError:
                existing_metrics = None
            if existing_metrics is not None and len(existing_metrics) > 0:
                logger.info("Metrics for ({}) exist:\n{}".format(
                    ", ".join([f"{k}: {v}" for k, v in params.items()]),
                    tabulate(existing_metrics, headers=existing_metrics.columns)
                ))
                all_results.append(existing_metrics)
                continue

        # Add criterion and value after to avoid searching by it.
        params.update({"Criterion Name": criterion, "Criterion Val": best_value})

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
        for k, v in params.items():
            scan_results[k] = v
        logger.info("\n" + tabulate(scan_results, headers=scan_results.columns))

        all_results.append(scan_results)

        # Currently don't support writing data because it takes too long
        # logger.info("Saving data...")
        # file_path = os.path.join(output_dir, dataset_name, "{}.h5".format(scan_name))
        # os.makedirs(os.path.dirname(file_path), exist_ok=True)

    all_results = pd.concat(all_results, ignore_index=True)
    logger.info("Summary:\n{}".format(tabulate(all_results, headers=all_results.columns)))

    # Try to copy over old metrics information.
    # TODO: If fails, it automatically saves the old file in a versioned form and prints logging message.
    if metrics is not None:
        try:
            running_results = update_metrics(all_results, metrics, on=["Acceleration", "dataset", "Noise Level", "weights", "Method"])
        except KeyError as e:
            logger.error(e)
            logger.error("Failed to load old metrics information")
            #raise e
            running_results = all_results
    else:
        running_results = all_results
    running_results.to_csv(metrics_file, mode="w")


def main(args):
    cfg = setup(args)
    model = DefaultTrainer.build_model(cfg)
    weights, criterion, best_value = (cfg.MODEL.WEIGHTS, None, None) if cfg.MODEL.WEIGHTS else find_weights(cfg, args.metric, args.iter_limit)
    model = model.to(cfg.MODEL.DEVICE)
    DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
        weights, resume=args.resume
    )

    # See https://github.com/pytorch/pytorch/issues/42300
    logger.info("Checking weights were properly loaded...")
    check_consistency(torch.load(weights)["model"], model)

    logger.info("\n\n==============================")
    logger.info("Loading weights from {}".format(weights))

    eval(cfg, args, model, os.path.basename(weights), criterion, best_value)


if __name__ == "__main__":
    parser = default_argument_parser()
    # parser.add_argument(
    #     "--dir", type=str, default=None,
    #     help="Process all completed experiment directories under this directory"
    # )
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
        default="standard",
        choices=("false", "standard", "sweep"),
        help="Include noise evaluation",
    )
    parser.add_argument(
        "--sweep-vals",
        default=(0, 0.2, 0.4, 0.6, 0.8, 1.0),
        nargs="*",
        type=float,
        help="args to sweep for noise",
    )
    parser.add_argument(
        "--iter-limit", default=None, type=int, help="Iteration limit. Chooses weights below this time point."
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing metrics file",
    )
    # parser.add_argument(
    #     "--wandb", action="store_true", help="Log to W&B during evaluation"
    # )

    args = parser.parse_args()
    print("Command Line Args:", args)
    main(args)
