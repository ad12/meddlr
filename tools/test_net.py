"""Run inference on test set scans.

This consists of comparing both zero-filled recon and DL-recon to fully-sampled
scans. All comparisons are done per volume (not per slice).

Supported metrics include:
    - ssim
    - ssim_center (ssim_50)
    - psnr
    - nrmse
"""
import logging
import os
import datetime
import time

import h5py
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
def eval(cfg, model):
    """Evaluate model on per scan metrics with acceleration factors
    between 6-8x.

    Save scan outputs to an h5 file.

    """
    device = cfg.MODEL.DEVICE
    model = model.to(device)
    model = model.eval()

    output_dir = os.path.join(cfg.OUTPUT_DIR, "test_results")
    os.makedirs(output_dir, exist_ok=True)

    results = []
    for dataset_name in cfg.DATASETS.TEST:
        loaders = build_data_loaders_per_scan(cfg, dataset_name, (6, 8))
        for acc in loaders:
            for scan_name, loader in loaders[acc].items():
                scan_name = os.path.splitext(os.path.basename(scan_name))[0]
                header = "{} - {} - {}".format(dataset_name, acc, scan_name)
                zf_images = []
                targets = []
                outputs = []
                num_batches = len(loader)
                start_time = data_start_time = time.perf_counter()
                for idx, (kspace, maps, target, mean, std, norm) in enumerate(loader):  # noqa
                    data_load_time = time.perf_counter() - data_start_time

                    output_dict = model(
                        kspace, maps, target=target, mean=mean, std=std,
                        norm=norm
                    )
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
                zf_results = pd.DataFrame([compute_metrics(targets, zf_images)])
                dl_results = compute_metrics(targets, outputs)
                dl_results["recon_time"] = recon_time
                dl_results = pd.DataFrame([dl_results])
                zf_results["Method"] = "zero-filled"
                dl_results["Method"] = "DL-Recon"
                scan_results = pd.concat([zf_results, dl_results])

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
                # with h5py.File(file_path, "a") as f:
                #     f.create_dataset("zf_{}".format(acc), data=zf_images.numpy())
                #     f.create_dataset("dl_{}".format(acc), data=outputs.numpy())
                #     if "fs" not in f:
                #         f.create_dataset("fs", data=targets.numpy())

    results = pd.concat(results, ignore_index=True)
    results.to_csv(os.path.join(output_dir, "metrics.csv"))


def main(args):
    cfg = setup(args)
    model = DefaultTrainer.build_model(cfg)
    DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
        cfg.MODEL.WEIGHTS, resume=args.resume
    )
    logger.info("\n\n==============================")
    logger.info("Loading weights from {}".format(cfg.MODEL.WEIGHTS))

    eval(cfg, model)


if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    print("Command Line Args:", args)
    main(args)
