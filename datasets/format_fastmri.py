"""Format data from the FastMRI Knee Initiative.

Multicoil Knee Dataset::

    ```bash
    # 1. Format train/validation datasets. Can be run simultaneously.

    python format_fastmri format --challenge knee_multicoil \
        --split train --device <GPU DEVICE ID> --num_workers <NUM WORKERS>
    python format_fastmri format --challenge knee_multicoil \
        --split val --device <GPU DEVICE ID> --num_workers <NUM WORKERS>

    # 2. Create annotation files detailing train/val/test splits.
    # For "dev" split method, the train is split into train/val and val is used for testing.
    # This is because fastMRI test dataset does not have ground truth (i.e. fully sampled scans).

    # Format toy dev datasets (only uses 5 scans -> 4 train, 1 val). Useful for debugging.
    python format_fastmri annotate --challenge knee_multicoil --method toy-dev --version vtoy
    python format_fastmri annotate --challenge knee_multicoil --method dev --version v0.0.1
    ```

TODOs:

    - Make checking more efficient
"""
import argparse
import datetime
import getpass
import itertools
import json
import logging
import os
import random
import time
from collections import defaultdict
from typing import Sequence

import h5py
import numpy as np
import sigpy as sp
import silx.io.dictdump as silx_dd
import torch
from sigpy.mri import app
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from utils import data_partition as dp

from meddlr.forward import SenseModel
from meddlr.ops import complex as cplx
from meddlr.utils import env
from meddlr.utils import transforms as T

try:
    import cupy as cp
except ImportError:
    cp = None

from meddlr.utils.logger import setup_logger

_FILE_DIR = os.path.dirname(__file__)
_FILE_NAME = os.path.splitext(os.path.basename(__file__))[0]

OUTPUT_DIR = "data://fastmri"
ANN_DIR = "ann://fastmri"
RAW_DATA_PATH = "data://fastmri/raw"
_LOGGER_NAME = "{}.{}".format(_FILE_NAME, __name__)
logger = logging.getLogger(_LOGGER_NAME)

SEED = 49
NUM_EMAPS = 1
RECON_METHODS = ("espirit", "jsense", "jsense-8", "jsense-12")

# Supported dataset splits
# TODO: Add support for test and challenge datasets.
SPLITS = ["train", "val"]
CHALLENGES = {
    "knee_multicoil": {
        "train": "multicoil_train",
        "val": "multicoil_val",
        "test": "multicoil_test_v2",
        "challenge": "multicoil_challenge",
    },
    "brain_multicoil": {
        "train": "multicoil_train",
        "val": "multicoil_val",
        "test": "multicoil_test",
        "challege": None,  # add once challenge dataset is out
    },
}
# Defines supported annotation methods
# dev: Train files are split into train/val, val is used as test set.
ANN_METHODS = ["dev", "toy-dev", "mini"]

# Path Manager
_PATH_MANAGER = env.get_path_manager()


def seed_everything(device=None):
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    random.seed(SEED)
    if cp is not None:
        cp.random.seed(SEED)
    if device is not None:
        if isinstance(device, int):
            device = sp.Device(device)
        with device:
            device.xp.random.seed(SEED)


def is_valid_file(fpath):
    """Returns `True` if file corresponds to a valid fastMRI file."""
    fname = os.path.basename(fpath)
    return fname.startswith("file") and fname.endswith(".h5")


def get_files(dir_path: str, ignore_missing_dir=False, filenames=None):
    if not os.path.isdir(dir_path):
        if ignore_missing_dir:
            return []
        else:
            raise NotADirectoryError(f"Directory {dir_path} does not exist")
    files = [
        os.path.join(dir_path, x)
        for x in os.listdir(dir_path)
        if is_valid_file(x)
        and (not filenames or x in filenames or os.path.splitext(x)[0] in filenames)
    ]
    return files


def preprocess_slice(kspace, im_shape):
    """Pre-process k-space data.

    Args:
        kspace: Shape `#coils x H x W`
        im_shape (tuple): Shape to crop to.

    Returns:
        ndarray: Shape `#coils x H x W`
    """
    # Pre-process k-space data (PyTorch)
    #   1. Reconstruct fully sampled image
    #   2. Crop to (yres, xres)
    #   3. Take fourier transform
    # Note this preprocessing technique does not work for
    # test/challenge images as scans in these datasets are undersampled.
    kspace = np.expand_dims(kspace, 0)
    kspace_tensor = cplx.to_tensor(kspace)  # (1, 640, 372, 15, 2)
    if cplx.is_complex(kspace_tensor):
        # If using pytorch>=1.7, complex tensors are now supported.
        # This small fix allows for minimal change in code without
        # a significant computational overhead.
        kspace_tensor = torch.view_as_real(kspace_tensor)
    image_tensor = T.ifft2(kspace_tensor)
    image_tensor = image_tensor.permute(0, 3, 1, 2, 4)  # (1, 15, 640, 372, 2)
    image_tensor = T.complex_center_crop_2d(image_tensor, im_shape)
    image_tensor = image_tensor.permute(0, 2, 3, 1, 4)  # (1, 640, 372, 15, 2)
    kspace_tensor = T.fft2(image_tensor)
    kspace_slice = cplx.to_numpy(kspace_tensor.squeeze(0))  # (640, 372, 15)
    return kspace_slice


class FastMRIDataset(Dataset):
    """Wrapper dataset to take advantage of PyTorch multi-worker loading."""

    def __init__(self, files):
        self.files = files

    def __getitem__(self, idx):
        with h5py.File(self.files[idx], "r") as f:
            kspace_orig = f["kspace"][()]
            im_rss = f["reconstruction_rss"][()]
            xres, yres = im_rss.shape[1:3]  # matrix size
            attrs = {k: v for k, v in f.attrs.items()}
            ismrmrd_header = f["ismrmrd_header"][()]

        num_slices, num_coils, num_kx, num_ky = kspace_orig.shape
        kspace = np.zeros((num_slices, xres, yres, num_coils), dtype=np.complex64)
        im_shape = (xres, yres)

        for sl in range(num_slices):
            kspace_slice = kspace_orig[sl]  # #coils x H x W
            kspace_slice = np.transpose(kspace_slice, (1, 2, 0))  # H x W x #coils
            kspace_slice = preprocess_slice(kspace_slice, im_shape)  # H x W x #coils
            # kspace_slice = np.transpose(kspace_slice, ())  # H x W x #coils
            kspace[sl] = kspace_slice

        return {
            "file_path": self.files[idx],
            "kspace_orig": kspace_orig,
            "kspace": kspace,
            "im_shape": im_shape,
            "attrs": attrs,
            "ismrmrd_header": ismrmrd_header,
            "reconstruction_rss": im_rss,
        }

    def __len__(self):
        return len(self.files)


def collate_simple(batch: list):
    return batch


def process_slice(
    kspace, calib_method="jsense", calib_size: int = 20, device: int = -1, nmaps: int = NUM_EMAPS
):
    """Process 2D multi-coil kspace slice.

    Args:
        kspace (np.ndarray): The k-space. Shape ``(Ky, Kz, #coils)
        calib_method (str): The calibration method.
        calib_size (int): The calibration size.
        device (int): The device to perform computation on. ``-1`` indicates CPU.
        nmaps (int): Number of maps to create.

    Returns:
        image (np.ndarray): The coil-combined image(s). Shape ``(Ky, Kz, #maps)``.
        maps (np.ndarray): The maps. Shape ``(Ky, Kz, #coils, #maps)``
    """
    # get data dimensions
    nky, nkz, ncoils = kspace.shape

    if device is -1:
        device = sp.cpu_device
    else:
        device = sp.Device(device)

    # Seed everything before processing the slice.
    seed_everything(device)

    # compute sensitivity maps (BART)
    # cmd = f'ecalib -d 0 -S -m {nmaps} -c {crop_value} -r {calib_size}'
    # maps = bart.bart(1, cmd, kspace[:,:,0,None,:])
    # maps = np.reshape(maps, (nky, nkz, 1, ncoils, nmaps))

    # compute sensitivity maps (SigPy)
    # TODO: Add support for espirit bart
    ksp = np.transpose(kspace, [2, 1, 0])  # #coils x Kz x Ky
    if calib_method == "espirit":
        maps = app.EspiritCalib(
            ksp, calib_width=calib_size, device=device, show_pbar=False, crop=0.1
        ).run()
        # import pdb; pdb.set_trace()
        if not isinstance(maps, np.ndarray):
            maps = cp.asnumpy(maps)
    elif calib_method == "jsense":
        maps = app.JsenseRecon(
            ksp, mps_ker_width=6, ksp_calib_width=calib_size, device=device, show_pbar=False
        ).run()
    elif calib_method == "jsense-8":
        maps = app.JsenseRecon(
            ksp, mps_ker_width=8, ksp_calib_width=calib_size, device=device, show_pbar=False
        ).run()
    elif calib_method == "jsense-12":
        maps = app.JsenseRecon(
            ksp, mps_ker_width=12, ksp_calib_width=calib_size, device=device, show_pbar=False
        ).run()
    else:
        raise ValueError("%s calibration method not implemented..." % calib_method)
    maps = np.reshape(np.transpose(maps, [2, 1, 0]), (nky, nkz, ncoils, nmaps))

    # Convert everything to tensors
    kspace_tensor = cplx.to_tensor(kspace).unsqueeze(0)  # 1 x Ky x Kz x #coils
    maps_tensor = cplx.to_tensor(maps).unsqueeze(0)  # 1 x Ky x Kz x #coils

    # Do coil combination using sensitivity maps (PyTorch)
    A = SenseModel(maps_tensor)
    im_tensor = A(kspace_tensor, adjoint=True)

    # Convert tensor back to numpy array
    image = cplx.to_numpy(im_tensor.squeeze(0))

    return image, maps


def format_train_file(
    data,
    save_dir: str,
    calib_method="jsense",
    device: int = -1,
    center_fraction=0.04,
    recompute: bool = False,
    overwrite: bool = False,
):
    # Seed everything before processing the slice.
    seed_everything(device)

    file_path = data["file_path"]
    kspace_orig = data["kspace_orig"]
    kspace = data["kspace"]
    im_shape = data["im_shape"]
    xres, yres = im_shape
    attrs = data["attrs"]
    ismrmrd_header = data["ismrmrd_header"]
    reconstruction_rss = data["reconstruction_rss"]

    # Fixed data. Only overwritten by overwrite=True.
    # Not written if only `recompute=True`
    # Data is fixed because it either originates from
    # fastMRI h5 files or is a standard operation (fourier transform).
    fixed_data = {
        "ismrmrd_header": ismrmrd_header,
        "kspace": kspace,
        "reconstruction_rss": reconstruction_rss,
    }

    file_name = os.path.basename(file_path)
    out_file = os.path.join(save_dir, file_name)
    group_name = f"{calib_method}-cf={int(center_fraction*100)}"

    skip_recon = False
    recon_data = None
    if not overwrite and not recompute and _PATH_MANAGER.isfile(out_file):
        with h5py.File(out_file, "r") as f:
            skip_recon = (
                group_name in f.keys()
                and "maps" in f[group_name].keys()
                and "target" in f[group_name].keys()
            )

    if not skip_recon:
        num_slices, num_coils, ky, kx = kspace_orig.shape
        # yres, xres = image_rss.shape[1:3]  # matrix size
        calib_size = int(round(center_fraction * xres))

        im_shape = (xres, yres)
        maps = np.zeros((num_slices, xres, yres, num_coils, NUM_EMAPS), dtype=np.complex64)
        im_truth = np.zeros((num_slices, xres, yres, NUM_EMAPS), dtype=np.complex64)

        with torch.no_grad():
            # Make this parallelized on the gpu.
            for sl in tqdm(range(num_slices)):
                kspace_slice = kspace[sl]

                im_slice, maps_slice = process_slice(kspace_slice, calib_method, calib_size, device)

                maps[sl] = maps_slice
                im_truth[sl] = im_slice

        recon_data = {group_name: {"maps": maps, "target": im_truth}}
    else:
        logger.info(f"Skipped: {calib_method} reconstruction found")

    mode = "w" if overwrite else "a"
    with h5py.File(out_file, mode) as f:
        # Attributes are always written because they are light.
        for k, v in attrs.items():
            f.attrs[k] = v

        silx_dd.dicttoh5(fixed_data, f, overwrite_data=overwrite)
        if recon_data:
            silx_dd.dicttoh5(recon_data, f, overwrite_data=recompute or overwrite)


def filter_files(files: Sequence[str], save_dir: str, calib_method: str):
    """Filter out files that have already been processed.

    If the expected output h5df file has keys
    "{calib_method}/maps" and "{calib_method}/target",
    the file is assumed to be processed.

    Args:
        files: Input files to filter through.
        save_dir: Output directory to search
        calib_method: Method used for calibration.

    Returns:
        Sequence[str]: Files that still need to be processed.
    """
    logger.info("Filtering out processed files...")
    filtered_files = []
    for file_path in tqdm(files):
        file_name = os.path.basename(file_path)
        out_file = os.path.join(save_dir, file_name)

        skip_recon = False
        if _PATH_MANAGER.isfile(out_file):
            with h5py.File(out_file, "r") as f:
                skip_recon = (
                    calib_method in f.keys()
                    and "maps" in f[calib_method].keys()
                    and "target" in f[calib_method].keys()
                )
        if not skip_recon:
            filtered_files.append(file_path)

    logger.info(
        f"{len(files) - len(filtered_files)}/{len(files)} files are processed. "
        f"{len(filtered_files)} files remaining"
    )
    return filtered_files


def split_files(files: Sequence[str]):
    """Split files into regular and oversized files.

    Some files are too large for multiprocessing operations.
    This method partitions files into regular sized files and
    files that are too large for multiprocessing.

    Args:
        files (`str(s)`): Input files (from fastmri data) to partition

    Returns:
        reg_files, oversized_files
    """
    logger.info("Finding oversized files...")
    sizes = []
    for file_path in tqdm(files):
        with h5py.File(file_path, "r") as f:
            sizes.append(np.prod(f["kspace"].shape))

    CONSTANT = 1.7
    ref_size = np.median(sizes)
    reg_files = [file_path for size, file_path in zip(sizes, files) if size <= CONSTANT * ref_size]
    oversized_files = [
        file_path for size, file_path in zip(sizes, files) if size > CONSTANT * ref_size
    ]

    assert len(set(reg_files) & set(oversized_files)) == 0
    assert len(reg_files) + len(oversized_files) == len(files)

    logger.info(f"{len(oversized_files)}/{len(files)} files are oversized.")

    return reg_files, oversized_files


def format_data(args, raw_root, formatted_root):
    input_root = raw_root
    output_root = formatted_root
    if args.recompute and args.overwrite:
        logger.warn("Ignorning `--recompute` flag. Overwriting all data")

    all_setups = list(itertools.product(args.split, args.calib_method, args.center_fraction))

    failed_cases = []
    for idx, (split, calib_method, center_fraction) in enumerate(all_setups):
        # Seed everything before processing the slice.
        seed_everything(args.device)

        input_dir = os.path.join(input_root, CHALLENGES[args.challenge][split])
        output_dir = os.path.join(output_root, split)
        _PATH_MANAGER.mkdirs(output_dir)

        logger.info(
            f"({idx}/{len(all_setups)}) Processing {split} split, "
            f"{calib_method} method, {center_fraction} center fraction ..."
        )
        logger.info("=" * 90)

        files = get_files(input_dir, filenames=args.files)
        if not args.overwrite and not args.recompute:
            files = filter_files(files, output_dir, calib_method)
        if len(files) == 0:
            continue
        reg_files, oversized_files = split_files(files)

        for files, num_workers in zip([reg_files, oversized_files], [args.num_workers, 0]):
            num_files = len(files)
            if num_files == 0:
                continue

            dataset = FastMRIDataset(files)
            data_loader = DataLoader(
                dataset,
                shuffle=False,
                batch_size=1,
                num_workers=num_workers,
                pin_memory=True,
                collate_fn=collate_simple,
            )
            eta = None
            start_time = time.perf_counter()
            for idx, data in enumerate(data_loader):
                data = data[0]
                if idx > 0:
                    eta = datetime.timedelta(
                        seconds=int((time.perf_counter() - start_time) / idx * (num_files - idx))
                    )
                logger.info(
                    "Processing [{}/{}] {} {}".format(
                        idx + 1,
                        num_files,
                        data["file_path"],
                        "- ETA: {}".format(str(eta)) if eta is not None else "",
                    )
                )

                try:
                    format_train_file(
                        data,
                        output_dir,
                        calib_method=calib_method,
                        device=args.device,
                        center_fraction=center_fraction,
                        recompute=args.recompute,
                        overwrite=args.overwrite,
                    )
                except Exception as e:
                    logger.error("Failed to format {}".format(e))
                    failed_cases.append(data["file_path"])
    return failed_cases


def create_dev_split(files, seed, split_percentages=(0.8, 0.2)):
    """Split files into two splits (train/val).

    Data is split by patients to avoid overlap of patients between
    different datasets.

    Returns:
        Tuple[List, List]: Train and validation files.
            Both are sorted by filename
    """
    assert len(split_percentages) == 2
    patient_ids = defaultdict(list)
    for fpath in files:
        with h5py.File(fpath, "r") as f:
            patient_ids[f.attrs["patient_id"]].append(fpath)

    # Weight by the number of files they correspond to.
    patient_ids_list = sorted(patient_ids.keys())
    patient_weights = [len(patient_ids[pid]) for pid in patient_ids_list]

    train_patient_ids, val_patient_ids = dp.approximately_split_weighted(
        patient_ids_list,
        list(split_percentages),
        weights=patient_weights,
        balance="greedy",
        buffer=0.005,
        seed=seed,
    )
    assert len(set(train_patient_ids) & set(val_patient_ids)) == 0

    train_files = sorted([x for pid in train_patient_ids for x in patient_ids[pid]])
    val_files = sorted([x for pid in val_patient_ids for x in patient_ids[pid]])
    assert len(set(train_files) & set(val_files)) == 0

    return train_files, val_files


def create_mini_split(files, seed, split_percentages=(0.55, 0.15, 0.3)):
    """Split files into three splits (train/val/test).

    Data is split by patients to avoid overlap of patients between
    different datasets.

    Returns:
        Tuple[List, List]: Train and validation files.
            Both are sorted by filename
    """
    assert len(split_percentages) == 3
    patient_ids = defaultdict(list)
    for fpath in files:
        with h5py.File(fpath, "r") as f:
            patient_ids[f.attrs["patient_id"]].append(fpath)

    # Weight by the number of files they correspond to.
    patient_ids_list = sorted(patient_ids.keys())
    patient_weights = [len(patient_ids[pid]) for pid in patient_ids_list]

    train_patient_ids, val_patient_ids, test_patient_ids = dp.approximately_split_weighted(
        patient_ids_list,
        list(split_percentages),
        weights=patient_weights,
        balance="greedy",
        buffer=0.005,
        seed=seed,
    )
    assert len(set(train_patient_ids) & set(val_patient_ids)) == 0
    assert len(set(train_patient_ids) & set(test_patient_ids)) == 0
    assert len(set(val_patient_ids) & set(test_patient_ids)) == 0

    train_files = sorted([x for pid in train_patient_ids for x in patient_ids[pid]])
    val_files = sorted([x for pid in val_patient_ids for x in patient_ids[pid]])
    test_files = sorted([x for pid in test_patient_ids for x in patient_ids[pid]])
    assert len(set(train_files) & set(val_files)) == 0
    assert len(set(train_files) & set(test_files)) == 0
    assert len(set(val_files) & set(test_files)) == 0

    return train_files, val_files, test_files


def format_annotations(args, raw_root, formatted_root, base_ann_dir):
    ann_method = args.method

    # Formatted 1-to-1 files with the original fastMRI repositories.
    # Sort files to get deterministic behavior.
    formatted_files = {}
    formatted_to_raw_files = {}  # noqa: F841
    for split, fastmri_subdir in CHALLENGES[args.challenge].items():  # noqa: B007
        if ann_method in ("mini",) and split != "val":
            # only need val split for mini dataset
            continue
        formatted_files[split] = sorted(
            get_files(os.path.join(formatted_root, split), ignore_missing_dir=True)
        )
        # for fpath in formatted_files[split]:
        #     fname = os.path.basename(fpath)
        #     raw_file = os.path.join(raw_root, fastmri_subdir, fname)
        #     assert os.path.isfile(raw_file), f"{raw_file} is not a file"
        #     assert raw_file not in formatted_to_raw_files, f"{raw_file} double occurrence"
        #     formatted_to_raw_files[fpath] = raw_file

    if ann_method in ("dev", "toy-dev"):
        # Toy dev dataset takes first 5 scans and splits into train/val datasets
        if ann_method == "toy-dev":
            formatted_files["train"] = formatted_files["train"][:5]
        train_files, val_files = create_dev_split(formatted_files["train"], args.seed)
        test_files = formatted_files["val"]
        ann_files = {"train": train_files, "val": val_files, "test": test_files}
        version = f"{args.version}-dev"
    elif ann_method in ("mini",):
        # Split fastMRI val set into train/val/test. Called the mini split.
        train_files, val_files, test_files = create_mini_split(formatted_files["val"], args.seed)
        ann_files = {"train": train_files, "val": val_files, "test": test_files}
        version = f"mini-{args.version}"
    else:
        raise ValueError(f"Annotation method {ann_method} is not supported")

    info_template = {
        "contributor": getpass.getuser(),
        "description": "fastMRI {} Dataset - {}",
        "year": time.strftime("%Y"),
        "date_created": time.strftime("%Y-%m-%d %H-%M-%S %Z"),
        "version": version,
    }

    for split, files in ann_files.items():
        logger.info(f"Processing {split} split...")
        logger.info("============================")

        ann_file = os.path.join(base_ann_dir, f"{version}/{split}.json")
        ann_dir = os.path.dirname(_PATH_MANAGER.get_local_path(ann_file))
        _PATH_MANAGER.mkdirs(ann_dir)
        image_data = []
        for formatted in files:
            with h5py.File(formatted, "r") as f:
                patient_id = f.attrs["patient_id"]
                acquisition = f.attrs["acquisition"]
                num_slices = f["kspace"].shape[0]
            scan_id = os.path.basename(formatted).split(".h5")[0].split("file")[-1]
            image_data.append(
                {
                    "file_name": os.path.basename(formatted),
                    # "file_path": formatted,
                    "scan_id": scan_id,
                    "patient_id": patient_id,
                    "acquisition": acquisition,
                    "num_slices": num_slices,
                }
            )

        split_info = dict(info_template)
        split_info["description"] = split_info["description"].format(args.challenge, split)
        data = {"info": split_info, "images": image_data}

        num_scans = len(image_data)
        num_subjects = len({x["patient_id"] for x in image_data})
        num_slices = sum(x["num_slices"] for x in image_data)
        logger.info(f"Summary: {num_scans} scans, {num_subjects} subjects, {num_slices} slices")
        with open(ann_file, "w") as f:
            json.dump(data, f, indent=2)


def add_shared_args(parser: argparse.ArgumentParser):
    parser.add_argument(
        "--challenge",
        required=True,
        choices=tuple(CHALLENGES.keys()),
        help="FastMRI challenge type",
    )
    # TODO: Update help comments.
    parser.add_argument(
        "--raw",
        default=None,
        help="Raw data directory (default: datasets/data/fastmri/raw/{challenge})",
    )
    parser.add_argument(
        "--formatted",
        default=None,
        help="Formatted data directory (default: datasets/data/fastmri/{challenge})",
    )


def main():
    parser = argparse.ArgumentParser(description="Data preparation")
    subparsers = parser.add_subparsers(title="sub-commands", dest="subcommand", required=True)

    format_parser = subparsers.add_parser("format", help="Format fastMRI dataset")
    add_shared_args(format_parser)
    format_parser.add_argument(
        "--calib-method",
        choices=RECON_METHODS,
        nargs="*",
        default=["jsense-8"],
        help="Sensitivity map estimation method (default: espirit)",
    )
    format_parser.add_argument(
        "--center-fraction",
        choices=[0.04, 0.08, 0.3],
        nargs="*",
        default=[0.04],
        type=float,
        help="Center fraction for computing sensitivity maps",
    )
    format_parser.add_argument(
        "--split",
        choices=SPLITS,
        nargs="*",
        default=SPLITS,
        help=f"Dataset split(s) (default: {SPLITS})",
    )
    format_parser.add_argument(
        "--files",
        nargs="*",
        default=None,
        help="Specific files to format. This should be the basename of the file",
    )
    format_parser.add_argument(
        "--device",
        type=int,
        default=-1,
        help="Device on which to run sensitivity calibration step.",
    )
    format_parser.add_argument(
        "--num-workers", type=int, default=0, help="Number of workers to use."
    )
    format_parser.add_argument(
        "--recompute",
        action="store_true",
        help="Recompute sensitivity maps and target image. Only overwrites these two fields",
    )
    format_parser.add_argument("--overwrite", action="store_true", help="Overwrite full file.")

    annotate_parser = subparsers.add_parser("annotate", help="Create annotation json files")
    add_shared_args(annotate_parser)
    annotate_parser.add_argument("--version", type=str, required=True, help="Annotation version")
    annotate_parser.add_argument(
        "--seed", type=int, default=1000, help="Random seed (default: 1000)"
    )
    annotate_parser.add_argument(
        "--method", required=True, choices=ANN_METHODS, help="Annotation split method"
    )

    args = parser.parse_args()
    if not args.subcommand:
        raise ValueError("No subcommand specified")

    if not args.raw:
        args.raw = os.path.join(OUTPUT_DIR, "raw", args.challenge)
    raw_root = _PATH_MANAGER.get_local_path(args.raw)
    if not args.formatted:
        args.formatted = os.path.join(OUTPUT_DIR, args.challenge)
    formatted_root = _PATH_MANAGER.get_local_path(args.formatted)

    setup_logger(formatted_root, name=_FILE_NAME)
    logger.info("Args:\n{}".format(args))

    if args.subcommand == "format":
        failed_cases = format_data(args, raw_root, formatted_root)
        with open(os.path.join(formatted_root, "failed_cases.txt"), "w") as f:
            f.writelines("{}\n".format(line) for line in failed_cases)
    elif args.subcommand == "annotate":
        ann_dir = _PATH_MANAGER.get_local_path(os.path.join(ANN_DIR, args.challenge))
        format_annotations(args, raw_root, formatted_root, ann_dir)
    else:
        raise ValueError(f"Subcommand {args.subcommand} not valid")


if __name__ == "__main__":
    main()
