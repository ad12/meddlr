"""Format the mridata.org Knee dataset.

This file is adapted from the repo at https://github.com/MRSRL/dl-cs/.

Note that this formatting file computes sensitivity maps per slice, not
per volume as was initially done in the repository mentioned above.
We find that this leads to more stable reconstruction.

Usage::
    ```bash
    # Run only from this repository (do not run as a module)

    # Ok
    python format_mridata_org.py ...

    # Not ok
    python -m datasets.format_mridata_org.py
    ```
"""
import argparse
import datetime
import functools
import getpass
import json
import logging
import multiprocessing as mp
import os
import time

import h5py
import ismrmrd
import mridata
import numpy as np
import sigpy as sp
import torch
from sigpy.mri import app
from tqdm import tqdm
from utils import fftc

from meddlr.forward import SenseModel
from meddlr.ops import complex as cplx
from meddlr.utils import env
from meddlr.utils.logger import setup_logger

_FILE_DIR = os.path.dirname(__file__)
_FILE_NAME = os.path.splitext(os.path.basename(__file__))[0]

BIN_BART = "bart"
OUTPUT_DIR = "data://mridata_knee_2019"
_LOGGER_NAME = "{}.{}".format(_FILE_NAME, __name__)
logger = logging.getLogger(_LOGGER_NAME)

_PATH_MANAGER = env.get_path_manager()


def download_mridata_org_dataset(file_name, dir_output, overwrite: bool = False):
    """Download datasets from mridata.org.

    Args:
        file_name (str): Path to mridata file containing uuids for files to
            download.
        dir_output (str): Directory path to save all data.
    """
    if os.path.isdir(dir_output):
        logger.warning(
            "Downloading data mridata.org to existing directory {}...".format(dir_output)
        )
    else:
        os.makedirs(dir_output)
        logger.info("Downloading data from mridata.org to {}...".format(dir_output))

    uuids = open(file_name).read().splitlines()
    for uuid in uuids:
        if not os.path.exists("{}/{}.h5".format(dir_output, uuid)) or overwrite:
            mridata.download(uuid, folder=dir_output)


def ismrmrd_to_np(filename):
    """Read ISMRMRD data file to numpy array."""
    logger.debug("Loading file {}...".format(filename))
    dataset = ismrmrd.Dataset(filename, create_if_needed=False)
    header = ismrmrd.xsd.CreateFromDocument(dataset.read_xml_header())
    num_kx = header.encoding[0].encodedSpace.matrixSize.x
    num_ky = header.encoding[0].encodingLimits.kspace_encoding_step_1.maximum
    num_slices = header.encoding[0].encodingLimits.slice.maximum + 1
    num_channels = header.acquisitionSystemInformation.receiverChannels

    try:
        rec_std = dataset.read_array("rec_std", 0)
        rec_weight = 1.0 / (rec_std**2)
        rec_weight = np.sqrt(rec_weight / np.sum(rec_weight))
        logger.debug("  Using rec std...")
    except Exception:
        rec_weight = np.ones(num_channels)
    opt_mat = np.diag(rec_weight)
    kspace = np.zeros([num_channels, num_slices, num_ky, num_kx], dtype=np.complex64)
    num_acq = dataset.number_of_acquisitions()

    for i in tqdm(range(num_acq)):
        acq = dataset.read_acquisition(i)
        i_ky = acq.idx.kspace_encode_step_1  # pylint: disable=E1101
        # i_kz = acq.idx.kspace_encode_step_2 # pylint: disable=E1101
        i_slice = acq.idx.slice  # pylint: disable=E1101
        data = np.matmul(opt_mat.T, acq.data)
        kspace[:, i_slice, i_ky, :] = data * ((-1) ** i_slice)

    dataset.close()
    kspace = fftc.fftc(kspace, axis=1)

    return kspace.astype(np.complex64)


def ismrmrd_to_npy(dir_input, dir_output, overwrite: bool = False):
    """Convert ISMRMRD files to npy files"""
    if os.path.isdir(dir_output):
        logger.warning("Writing npy data to existing directory {}...".format(dir_output))
    else:
        os.makedirs(dir_output)
        logger.info("Writing npy data to {}...".format(dir_output))

    filelist = sorted(os.listdir(dir_input))

    logger.info("Converting files from ISMRMD to npy...")
    for filename in filelist:
        file_input = os.path.join(dir_input, filename)
        filebase = os.path.splitext(filename)[0]
        file_output = os.path.join(dir_output, filebase + ".npy")
        if not os.path.exists(file_output) or overwrite:
            kspace = ismrmrd_to_np(file_input)
            np.save(file_output, kspace.astype(np.complex64))


def process_slice(kspace, calib_method="jsense", calib_size: int = 20, device: int = -1):
    # get data dimensions
    nky, nkz, ncoils = kspace.shape

    # ESPIRiT parameters
    nmaps = 1

    if device is -1:
        device = sp.cpu_device
    else:
        device = sp.Device(device)

    # compute sensitivity maps (BART)
    # cmd = f'ecalib -d 0 -S -m {nmaps} -c {crop_value} -r {calib_size}'
    # maps = bart.bart(1, cmd, kspace[:,:,0,None,:])
    # maps = np.reshape(maps, (nky, nkz, 1, ncoils, nmaps))

    # compute sensitivity maps (SigPy)
    ksp = np.transpose(kspace, [2, 1, 0])  # #coils x Kz x Ky
    if calib_method == "espirit":
        maps = app.EspiritCalib(ksp, calib_width=calib_size, device=device, show_pbar=False).run()
    elif calib_method == "jsense":
        maps = app.JsenseRecon(
            ksp, ksp_calib_width=calib_size, device=device, show_pbar=False
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


def convert_to_h5(
    file_paths,
    dir_output,
    calib_method: str = "jsense",
    calib_size: int = 20,
    device: int = -1,
    num_workers: int = 1,
    is_input_numpy: bool = False,
    overwrite: bool = False,
):
    """Convert ismrmrd files to h5 format suitable for meddlr library.

    Currently, sensitivity maps are computed over full volume. Images are
    reconstructed per slice

    Args:
        file_paths: Paths to ismrmrd files.
        dir_output: Output dir where to store data.
        calib: Calibration region shape in all spatial dimensions.
        is_input_numpy (bool, optional): If `True`, file paths point to numpy
            files storing kspace. In this case, data will be loaded from
            numpy files rather than recomputed.
        overwrite (bool): Overwrite existing files.
    """
    eta = None
    start_time = time.perf_counter()
    num_files = len(file_paths)
    os.makedirs(dir_output, exist_ok=True)
    for idx, fp in enumerate(file_paths):
        fname = os.path.splitext(os.path.basename(fp))[0]
        h5_file = os.path.join(dir_output, "{}.h5".format(fname))
        if os.path.isfile(h5_file) and not overwrite:
            logger.info(
                "Skipping [{}/{}] {} - hdf5 file found".format(idx + 1, len(file_paths), fp)
            )
            continue

        if idx > 0:
            eta = datetime.timedelta(
                seconds=int((time.perf_counter() - start_time) / idx * (num_files - idx))
            )
        logger.info(
            "Processing [{}/{}] {} {}".format(
                idx + 1,
                len(file_paths),
                fp,
                "- ETA: {}".format(str(eta)) if eta is not None else "",
            )
        )
        if is_input_numpy:
            kspace = np.squeeze(np.load(fp))
        else:
            kspace = np.squeeze(ismrmrd_to_np(fp))  # C x Z x Y x X

        shape_x = kspace.shape[-1]
        shape_y = kspace.shape[-2]
        shape_z = kspace.shape[-3]
        num_coils = kspace.shape[-4]
        num_maps = 1

        kspace = fftc.ifftc(kspace, axis=-1)
        kspace = kspace.astype(np.complex64)

        kspace = np.transpose(kspace, (3, 2, 1, 0))  # X x Y x Z x C

        images = np.zeros((shape_x, shape_y, shape_z, num_maps), dtype=np.complex64)
        maps = np.zeros((shape_x, shape_y, shape_z, num_coils, num_maps), dtype=np.complex64)
        with torch.no_grad():
            if num_workers > 0:
                kspace_sliced = [kspace[x] for x in range(shape_x)]
                func = functools.partial(
                    process_slice, calib_method=calib_method, calib_size=calib_size, device=device
                )
                with mp.Pool(num_workers) as pool:
                    info = pool.imap(kspace_sliced, func)
                for x in range(shape_x):
                    im_slice, maps_slice = info[x]
                    images[x] = im_slice
                    maps[x] = maps_slice
            else:
                for x in tqdm(range(shape_x)):
                    im_slice, maps_slice = process_slice(
                        kspace[x], calib_method, calib_size, device
                    )

                    images[x] = im_slice
                    maps[x] = maps_slice

        with h5py.File(h5_file, "w") as f:
            f.create_dataset("kspace", data=kspace)
            f.create_dataset("maps", data=maps)
            f.create_dataset("target", data=images)


def write_ann_file(ann_file, h5_dir, split, **kwargs):
    files = sorted(x for x in os.listdir(h5_dir) if x.endswith(".h5"))

    image_data = []
    for fname in files:
        with h5py.File(os.path.join(h5_dir, fname), "r") as f:
            kspace_size = f["kspace"].shape
        image_data.append({"file_name": fname, "kspace_size": kspace_size})

    data = {
        "info": {
            "contributor": getpass.getuser(),
            "description": "2019 MRIData.org Knee Dataset - {}".format(split),
            "year": time.strftime("%Y"),
            "date_created": time.strftime("%Y-%m-%d %H-%M-%S %Z"),
            "version": "v1.0.0",
        },
        "images": image_data,
    }
    with open(ann_file, "w") as f:
        json.dump(data, f, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Data preparation")
    parser.add_argument(
        "mridata_txt", action="store", help="Text file with mridata.org UUID datasets"
    )
    parser.add_argument(
        "--output",
        default=OUTPUT_DIR,
        help="Root directory (default: datasets/data/mridata_knee_2019)",
    )
    parser.add_argument("--random_seed", default=1000, help="Random seed")
    parser.add_argument("--redownload", action="store_true", help="Redownload raw dataset")
    parser.add_argument(
        "--recompute", action="store_true", help="Recompute sensitivity maps and target image"
    )
    parser.add_argument(
        "--device",
        type=int,
        default=-1,
        help="Device on which to run sensitivity calibration step.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="Number of workers to use for recontruction (default: 0)",
    )

    args = parser.parse_args()

    if args.random_seed >= 0:
        np.random.seed(args.random_seed)

    root_dir = _PATH_MANAGER.get_local_path(args.output)
    setup_logger(root_dir, name=_FILE_NAME)

    logger.info("Args:\n{}".format(args))

    # Download raw mridata.org knee datasets
    dir_mridata_org = os.path.join(root_dir, "raw/ismrmrd")
    download_mridata_org_dataset(args.mridata_txt, dir_mridata_org, args.redownload)

    # Save files as numpy files.
    # kspace is only recalculated if files are redownloaded.
    dir_npy = os.path.join(root_dir, "raw/npy")
    ismrmrd_to_npy(dir_mridata_org, dir_npy, overwrite=args.redownload)

    # Split data into 75/5/20 (train/val/test) after sorting
    # to preserve splits used in other literature.
    uuids = open(args.mridata_txt).read().splitlines()
    file_paths = sorted(
        os.path.join(dir_npy, x)
        for x in os.listdir(dir_npy)
        if x.endswith(".npy") and os.path.splitext(x)[0] in uuids
    )
    data_divide = (0.72, 0.08, 0.2)  # 14 train - 2 val - 3 test
    num_files = len(file_paths)
    train_idx = np.round(data_divide[0] * num_files).astype(int)
    val_idx = np.round(data_divide[1] * num_files).astype(int) + train_idx

    train_files = file_paths[:train_idx]
    val_files = file_paths[train_idx:val_idx]
    test_files = file_paths[val_idx:]

    # Write h5 files.
    for split, files in zip(["train", "val", "test"], [train_files, val_files, test_files]):
        logger.info("Processing {} split...".format(split))
        dir_h5_data = os.path.join(root_dir, split)
        convert_to_h5(
            files,
            dir_h5_data,
            is_input_numpy=True,
            device=args.device,
            overwrite=args.recompute,
            num_workers=args.num_workers,
        )

        # Save annotation files.
        ann_file = os.path.join(root_dir, "annotations", "{}.json".format(split))
        ann_dir = os.path.dirname(_PATH_MANAGER.get_local_path(ann_file))
        os.makedirs(ann_dir, exist_ok=True)
        write_ann_file(ann_file, dir_h5_data, split)
