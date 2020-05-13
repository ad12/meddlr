"""Format the mridata.org Knee dataset.

This file is adapted from the repo at https://github.com/MRSRL/dl-cs/.

Note that this formatting file computes sensitivity maps per slice, not
per volume as was initially done in the repository mentioned above.
We find that this leads to more stable reconstruction.

Usage::
    ```bash
    # Run only from this repository (do not run as a module)

    # Ok
    python format_mridata.org ...

    # Not ok
    python -m datasets.format_mridata.org
    ```
"""
import argparse
import logging
import os
import shutil
import subprocess
import sys

from fvcore.common.file_io import PathManager
import h5py
import torch
from tqdm import tqdm

import mridata
import ismrmrd
import numpy as np
import sigpy as sp

from utils import fftc

sys.path.append("../")  # noqa
from ss_recon.utils.logger import setup_logger
from ss_recon.utils import cfl
from ss_recon.utils import transforms as T
from ss_recon.utils import complex_utils as cplx


_FILE_DIR = os.path.dirname(__file__)
_FILE_NAME = os.path.splitext(os.path.basename(__file__))[0]

BIN_BART = "bart"
OUTPUT_DIR = "data://mridata_org_knee"
_LOGGER_NAME = "{}.{}".format(_FILE_NAME, __name__)
logger = logging.getLogger(_LOGGER_NAME)


def download_mridata_org_dataset(
    file_name, dir_output, overwrite: bool = False
):
    """Download datasets from mridata.org.

    Args:
        file_name (str): Path to mridata file containing uuids for files to
            download.
        dir_output (str): Directory path to save all data.
    """
    if os.path.isdir(dir_output):
        logger.warning(
            "Downloading data mridata.org to existing directory {}...".format(
                dir_output
            )
        )
    else:
        os.makedirs(dir_output)
        logger.info(
            "Downloading data from mridata.org to {}...".format(dir_output)
        )

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
        rec_weight = 1.0 / (rec_std ** 2)
        rec_weight = np.sqrt(rec_weight / np.sum(rec_weight))
        logger.debug("  Using rec std...")
    except Exception:
        rec_weight = np.ones(num_channels)
    opt_mat = np.diag(rec_weight)
    kspace = np.zeros(
        [num_channels, num_slices, num_ky, num_kx], dtype=np.complex64
    )
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
        logger.warning(
            'Writing npy data to existing directory {}...'.format(dir_output))
    else:
        os.makedirs(dir_output)
        logger.info('Writing npy data to {}...'.format(dir_output))

    filelist = sorted(os.listdir(dir_input))

    logger.info('Converting files from ISMRMD to npy...')
    for filename in filelist:
        file_input = os.path.join(dir_input, filename)
        filebase = os.path.splitext(filename)[0]
        file_output = os.path.join(dir_output, filebase + '.npy')
        if not os.path.exists(file_output) or overwrite:
            kspace = ismrmrd_to_np(file_input)
            np.save(file_output, kspace.astype(np.complex64))


def remove_bart_files(filenames):
    """Remove bart files in list.
    Args:
        filenames: List of bart file names.
    """
    for f in filenames:
        os.remove(f + ".hdr")
        os.remove(f + ".cfl")


def estimate_sense_maps(kspace, calib=20, method="espirit", device: int = -1):
    """Estimate sensitivity maps.

    ESPIRiT is used if bart exists. Otherwise, use JSENSE in sigpy.

    Args:
        kspace: k-Space data input as [coils, spatial dimensions].
        calib: Calibration region shape in all spatial dimensions.
        method: Sensitivity map estimation method.

    Returns:
        ndarray: Estimated sensitivity maps (complex 64-bit)
    """
    if method not in ["espirit", "jsense"]:
        raise ValueError("method must either be 'espirit' or 'jsense'")

    if args.device is -1:
        device = sp.cpu_device
    else:
        device = sp.Device(args.device)

    if method == "espirt":
        if not shutil.which(BIN_BART):
            raise ValueError("bart not installed")
        flags = "-c 1e-9 -m 1 -r %d" % calib
        randnum = np.random.randint(1e8)
        fileinput = "tmp.in.{}".format(randnum)
        fileoutput = "tmp.out.{}".format(randnum)
        cfl.write(fileinput, kspace)
        cmd = "{} ecalib {} {} {}".format(
            BIN_BART, flags, fileinput, fileoutput
        )
        subprocess.check_output(["bash", "-c", cmd])
        sensemap = np.squeeze(cfl.read(fileoutput))
        remove_bart_files([fileoutput, fileinput])
    else:
        JsenseApp = sp.mri.app.JsenseRecon(
            kspace, ksp_calib_width=calib, device=device, show_pbar=True
        )
        sensemap = JsenseApp.run()
        del JsenseApp
        sensemap = sensemap.astype(np.complex64)
    return sensemap


def convert_to_h5(
    file_paths,
    dir_output,
    calib: int = 20,
    device: int = -1,
    is_input_numpy: bool = False,
    overwrite: bool = False,
):
    """Convert ismrmrd files to h5 format suitable for ss_recon library.

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
    os.makedirs(dir_output, exist_ok=True)
    for idx, fp in enumerate(file_paths):
        fname = os.path.splitext(os.path.basename(fp))[0]
        h5_file = os.path.join(dir_output, "{}.h5".format(fname))
        if os.path.isfile(h5_file) and not overwrite:
            logger.info("Skipping [{}/{}] {} - hdf5 file found".format(
                idx + 1, len(file_paths), fp
            ))

        logger.info("Processing [{}/{}] {}...".format(
            idx+1, len(file_paths), fp
        ))
        if is_input_numpy:
            kspace = np.squeeze(np.load(fp))
        else:
            kspace = np.squeeze(ismrmrd_to_np(fp))  # C x Z x Y x X

        shape_x = kspace.shape[-1]
        shape_y = kspace.shape[-2]
        shape_z = kspace.shape[-3]
        shape_c = kspace.shape[-4]
        logger.debug("Slice shape: (%d, %d)" % (shape_z, shape_y))
        logger.debug("Num channels: %d" % shape_c)

        logger.info("Estimating sensitivity maps...")
        sensemap = estimate_sense_maps(kspace, calib=calib, device=device)
        sensemap = np.expand_dims(sensemap, axis=0)  # M x C x Z x Y x X

        kspace = fftc.ifftc(kspace, axis=-1)
        kspace = kspace.astype(np.complex64)

        kspace = np.transpose(kspace, (3, 2, 1, 0))  # X x Y x Z x C
        sensemap = np.transpose(sensemap, (4, 3, 2, 1, 0))  # X x Y x Z x C x M
        shape_m = sensemap.shape[-1]

        images = np.zeros(shape_x, shape_y, shape_z, shape_m)
        with torch.no_grad():
            for x in range(shape_x):
                map_tensor = cplx.to_tensor(sensemap[x]).unsqueeze(0)
                k_tensor = cplx.to_tensor(kspace[x]).unsqueeze(0)

                A = T.SenseModel(map_tensor)
                im_tensor = A(k_tensor, adjoint=True)

                # Convert tensor back to numpy array
                images[x] = cplx.to_numpy(im_tensor.squeeze(0))

        with h5py.File(h5_file, "w") as f:
            f.create_dataset("kspace", data=kspace)
            f.create_dataset("maps", data=sensemap)
            f.create_dataset("target", data=images)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Data preparation")
    parser.add_argument(
        "mridata_txt",
        action="store",
        help="Text file with mridata.org UUID datasets",
    )
    parser.add_argument(
        "--output",
        default=OUTPUT_DIR,
        help="Root directory (default: datasets/data/mridata_org_knee)"
    )
    parser.add_argument("--random_seed", default=1000, help="Random seed")
    parser.add_argument(
        "--redownload", action="store_true", help="Redownload raw dataset"
    )
    parser.add_argument(
        "--recompute",
        action="store_true",
        help="Recompute sensitivity maps and target image"
    )
    # Sensitivity map parameters.
    parser.add_argument(
        "--device", type=int, default=-1,
        help='Device on which to run sensitivity calibration step.')

    args = parser.parse_args()

    if args.random_seed >= 0:
        np.random.seed(args.random_seed)

    root_dir = PathManager.get_local_path(args.output)
    setup_logger(root_dir, name=_FILE_NAME)

    # Download raw mridata.org knee datasets
    dir_mridata_org = os.path.join(root_dir, "raw/ismrmrd")
    download_mridata_org_dataset(
        args.mridata_txt, dir_mridata_org, args.redownload
    )

    # Save files as numpy files.
    # kspace is only recalculated if files are redownloaded.
    dir_npy = os.path.join(root_dir, "raw/npy")
    ismrmrd_to_npy(dir_mridata_org, dir_npy, overwrite=args.redownload)

    # Split data into 75/5/20 (train/val/test) after sorting
    # to preserve splits used in other literature.
    file_paths = sorted([
        os.path.join(dir_npy, x)
        for x in os.listdir(dir_npy) if x.endswith(".npy")
    ])
    data_divide = (0.75, 0.05, 2)
    num_files = len(file_paths)
    train_idx = np.round(data_divide[0] * num_files).astype(int) + 1
    val_idx = np.round(data_divide[1] * num_files).astype(int) + train_idx

    train_files = file_paths[:train_idx]
    val_files = file_paths[train_idx: val_idx]
    test_files = file_paths[val_idx:]

    # Write h5 files.
    for split, files in zip(
        ["train", "val", "test"], [train_files, val_files, test_files]
    ):
        dir_h5_data = os.path.join(root_dir, split)
        convert_to_h5(
            files,
            dir_h5_data,
            is_input_numpy=True,
            device=args.device,
            overwrite=args.recompute
        )
