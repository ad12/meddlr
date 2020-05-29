"""Run inference on test set scans.

This consists of comparing both zero-filled recon and DL-recon to fully-sampled
scans. All comparisons are done per volume (not per slice).

Supported metrics include:
    - ssim
    - ssim_center (ssim_50)
    - psnr
    - nrmse
"""
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
from ss_recon.checkpoint import DetectionCheckpointer
from ss_recon.evaluation import metrics
from ss_recon.data.build import build_data_loaders_per_scan
from ss_recon.utils.logger import setup_logger, log_every_n_seconds

import ss_recon.utils.complex_utils as cplx
from ss_recon.utils.transforms import SenseModel

_FILE_NAME = os.path.splitext(os.path.basename(__file__))[0]
_LOGGER_NAME = "{}.{}".format(_FILE_NAME, __name__)
logger = logging.getLogger(_LOGGER_NAME)


@torch.no_grad()
def compute_metrics(ref: torch.Tensor, x: torch.Tensor):
    """Metrics are computed per volume.

    Args:
        ref (torch.Tensor): The target. Shape (...)x2
        x (torch.Tensor): The prediction. Same shape as `ref`.
    """
    ref = ref.squeeze()
    x = x.squeeze()
    psnr = metrics.compute_psnr(ref, x).item()
    nrmse = metrics.compute_nrmse(ref, x).item()

    ssim = metrics.compute_ssim(ref, x, multichannel=False)
    ssim_mc = metrics.compute_ssim(ref, x, multichannel=True)

    # Average SSIM score for slices, but only compute SSIM for the
    # center 50% of slices and center 50% of volume.
    # Noise in reference contributes to the SSIM score.
    ref_shape = ref.shape[:-1]
    shape_crop = tuple(
        slice(int(0.25*ref.shape[i]), int(0.75 * ref_shape[i]) + 1)
        for i in range(len(ref_shape))
    )

    ref = ref[shape_crop]
    x = x[shape_crop]

    ssim_50 = metrics.compute_ssim(ref, x, multichannel=False)
    ssim_50_mc = metrics.compute_ssim(ref, x, multichannel=True)
    return {
        'psnr': psnr,
        'nrmse': nrmse,
        'ssim': ssim,
        "ssim_mc": ssim_mc,
        "ssim_50": ssim_50,
        "ssim_50_mc": ssim_50_mc,
    }


def setup(args):
    """
    Create configs and perform basic setups.
    We do not save the config.
    """
    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    default_setup(cfg, args, save_cfg=False)

    # Setup logger for test results
    setup_logger(os.path.join(cfg.OUTPUT_DIR, "test_results"), name=_FILE_NAME)
    return cfg


@torch.no_grad()
def eval(cfg, model, zero_filled: bool = False):
    """Evaluate model on per scan metrics with acceleration factors
    between 6-8x.

    Save scan outputs to an h5 file.

    Args:
        cfg:
        model:
        zero_filled (bool, optional): If `True`, calculate metrics
            for zero-filled reconstruction.
    """
    device = cfg.MODEL.DEVICE
    model = model.to(device)
    model = model.eval()

    output_dir = os.path.join(cfg.OUTPUT_DIR, "test_results")
    os.makedirs(output_dir, exist_ok=True)

    results = []
    for dataset_name in cfg.DATASETS.TEST:
        loaders = build_data_loaders_per_scan(cfg, dataset_name, (6,8))
        for acc in loaders:
            for scan_name, loader in loaders[acc].items():
                scan_name = os.path.splitext(os.path.basename(scan_name))[0]
                header = "{} - {} - {}".format(dataset_name, acc, scan_name)
                zf_images = []
                targets = []
                outputs = []
                num_batches = len(loader)
                start_time = data_start_time = time.perf_counter()
                for idx, inputs in enumerate(loader):  # noqa
                    kspace, maps, target = inputs["kspace"], inputs["maps"], inputs["target"]  # noqa
                    mean, std, norm = inputs["mean"], inputs["std"], inputs["norm"]  # noqa
                    data_load_time = time.perf_counter() - data_start_time

                    output_dict = model(inputs)
                    targets.append(output_dict["target"].cpu())
                    outputs.append(output_dict["pred"].cpu())
                    zf_images.append(output_dict["zf_image"].cpu())

                    eta = datetime.timedelta(
                        seconds=int(
                            (time.perf_counter() - start_time) / (idx + 1) * (num_batches - idx - 1)
                        )
                    )

                    log_every_n_seconds(
                        logging.INFO,
                        "{}: Processed {}/{} - data_time: {:.6f} - ETA: {}".format(
                            header, idx+1, num_batches,  data_load_time, eta
                        ),
                        n=5,
                        name=_LOGGER_NAME,
                    )

                    data_start_time = time.perf_counter()

                recon_time = time.perf_counter() - start_time

                zf_images = torch.cat(zf_images, dim=0)
                targets = torch.cat(targets, dim=0)
                outputs = torch.cat(outputs, dim=0)

                logger.info("Computing metrics...")
                dl_results = compute_metrics(targets, outputs)
                dl_results["recon_time"] = recon_time
                dl_results = pd.DataFrame([dl_results])
                dl_results["Method"] = "DL-Recon"
                if zero_filled:
                    zf_results = pd.DataFrame([compute_metrics(targets, zf_images)])
                    zf_results["Method"] = "zero-filled"
                    scan_results = pd.concat([zf_results, dl_results])
                else:
                    scan_results = dl_results

                logger.info("Results:\n{}".format(tabulate(scan_results, headers=scan_results.columns)))

                scan_results["Acceleration"] = acc
                scan_results["dataset"] = dataset_name
                scan_results["scan_name"] = scan_name
                results.append(scan_results)

                logger.info("Saving data...")
                file_path = os.path.join(output_dir, dataset_name, "{}.h5".format(scan_name))
                os.makedirs(os.path.dirname(file_path), exist_ok=True)
                silx_dd.dicttoh5(
                    {
                        acc: {
                            "zf": zf_images.numpy(),
                            "dl": outputs.numpy(),
                        },
                        "fs": targets.numpy()
                    },
                    file_path,
                    mode="a",
                    overwrite_data = True,
                )

    results = pd.concat(results, ignore_index=True)
    results.to_csv(os.path.join(output_dir, "metrics.csv"))
    logger.info("Summary:\n{}".format(tabulate(results, headers=results.columns)))


# Supported validation metrics and the operation to perform on them.
SUPPORTED_VAL_METRICS = {
    "l1": "min",
    "l2": "min",
    "psnr": "max",
    "ssim": "max",
}


def find_weights(cfg, criterion=""):
    """Find the best weights based on a validation criterion/metric.

    Args:
        criterion (str): The criterion that we can select from 
    """
    if not criterion:
        criterion = cfg.MODEL.RECON_LOSS.NAME.lower()
        operation = "min"  # loss is always minimized
    else:
        operation = SUPPORTED_VAL_METRICS[criterion]

    assert operation in ["min", "max"]
    criterion = "val_{}".format(criterion)

    logger.info("Finding best weights in {} using {}...".format(cfg.OUTPUT_DIR, criterion))

    # Filter metrics to find reporting of real validation metrics.
    # If metric is wrapped (e.g. "mridata_knee_2019_val/val_l1"), that means
    # multiple datasets were validated on.
    # We filter out metrics from datasets that contain the word "test".
    # The criterion from all other datasets are averaged and used as the
    # target criterion.
    metrics_file = os.path.join(cfg.OUTPUT_DIR, "metrics.json")
    metrics = []
    with open(metrics_file) as f:
        metrics = [json.loads(line.strip()) for line in f]
    metrics = [
        m for m in metrics 
        if criterion in m or any(k.endswith(criterion) for k in m.keys())
    ]
    is_metric_wrapped = criterion not in metrics[0]
    if is_metric_wrapped:
        metrics = [
            (
                m["iteration"], 
                np.mean([
                    m[k] for k in m 
                    if k.endswith(criterion) and "test" not in k.split("/")[0]
                ]).item(),
            )
            for m in metrics
        ]
    else:
        metrics = [(m["iteration"], m[criterion]) for m in metrics]

    # Retraining does not overwrite the metrics file.
    # We make sure that the metrics correspond only to the most
    # recent run.
    metrics_recent_order = metrics[::-1]
    iterations = [m[0] for m in metrics_recent_order]
    intervals = np.diff(iterations)
    if any(intervals > 0):
        stop_idx = np.argmax(intervals > 0) + 1
        metrics_recent_order = metrics_recent_order[:stop_idx]
        metrics = metrics_recent_order[::-1]

    # Note that resuming can sometimes report metrics for the same
    # iteration. We handle this by taking the most recent metric for the
    # iteration __after__ filtering out old training runs.
    metrics = {iteration: value for iteration, value in metrics}
    metrics = [(k, v) for k, v in metrics.items()]
    best_iter, best_value = sorted(metrics, key=lambda x: x[1], reverse=operation=="max")[0]


    if best_iter == cfg.SOLVER.MAX_ITER - 1:
        file_name = "model_final.pth"
    else:
        file_name = "model_{:07d}.pth".format(best_iter)
    file_path = os.path.join(cfg.OUTPUT_DIR, file_name)

    if not os.path.isfile(file_path):
        raise ValueError("Model for iteration {} does not exist".format(best_iter))

    logger.info("Weights: {} - {}: {:0.4f}".format(file_name, criterion, best_value))
    return file_path


def main(args):
    cfg = setup(args)
    model = DefaultTrainer.build_model(cfg)
    weights = cfg.MODEL.WEIGHTS if cfg.MODEL.WEIGHTS else find_weights(cfg, args.metric)
    DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
        weights, resume=args.resume
    )
    logger.info("\n\n==============================")
    logger.info("Loading weights from {}".format(weights))

    eval(cfg, model, args.zero_filled)


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

    args = parser.parse_args()
    print("Command Line Args:", args)
    main(args)
