import importlib
import importlib.util
import logging
import os
import random
import subprocess
import sys
from datetime import datetime

import numpy as np
import torch

try:
    import wandb
except:
    pass

__all__ = []


def seed_all_rng(seed=None):
    """
    Set the random seed for the RNG in torch, numpy and python.

    Args:
        seed (int): if None, will use a strong random seed.
    """
    if seed is None:
        seed = (
            os.getpid()
            + int(datetime.now().strftime("%S%f"))
            + int.from_bytes(os.urandom(2), "big")
        )
        logger = logging.getLogger(__name__)
        logger.info("Using a generated random seed {}".format(seed))
    np.random.seed(seed)
    torch.set_rng_state(torch.manual_seed(seed).get_state())
    random.seed(seed)


# from https://stackoverflow.com/questions/67631/how-to-import-a-module-given-the-full-path  # noqa
def _import_file(module_name, file_path, make_importable=False):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if make_importable:
        sys.modules[module_name] = module
    return module


def _configure_libraries():
    """
    Configurations for some libraries.
    """
    # An environment option to disable `import cv2` globally,
    # in case it leads to negative performance impact
    disable_cv2 = int(os.environ.get("SSRECON_DISABLE_CV2", False))
    if disable_cv2:
        sys.modules["cv2"] = None
    else:
        # Disable opencl in opencv since its interaction with cuda often
        # has negative effects
        # This envvar is supported after OpenCV 3.4.0
        os.environ["OPENCV_OPENCL_RUNTIME"] = "disabled"
        try:
            import cv2

            if int(cv2.__version__.split(".")[0]) >= 3:
                cv2.ocl.setUseOpenCL(False)
        except ImportError:
            pass


_ENV_SETUP_DONE = False


def setup_environment():
    """Perform environment setup work. The default setup is a no-op, but this
    function allows the user to specify a Python source file or a module in
    the $MEDSEGPY_ENV_MODULE environment variable, that performs
    custom setup work that may be necessary to their computing environment.
    """
    global _ENV_SETUP_DONE
    if _ENV_SETUP_DONE:
        return
    _ENV_SETUP_DONE = True

    _configure_libraries()

    custom_module_path = os.environ.get("SSRECON_ENV_MODULE")

    if custom_module_path:
        setup_custom_environment(custom_module_path)
    else:
        # The default setup is a no-op
        pass


def setup_custom_environment(custom_module):
    """
    Load custom environment setup by importing a Python source file or a
    module, and run the setup function.
    """
    if custom_module.endswith(".py"):
        module = _import_file("ss_recon.utils.env.custom_module", custom_module)
    else:
        module = importlib.import_module(custom_module)
    assert hasattr(module, "setup_environment") and callable(
        module.setup_environment
    ), (
        "Custom environment module defined in {} does not have the "
        "required callable attribute 'setup_environment'."
    ).format(
        custom_module
    )
    module.setup_environment()


def get_available_gpus(num_gpus: int = None):
    """Get gpu ids for gpus that are >95% free.

    Tensorflow does not support checking free memory on gpus.
    This is a crude method that relies on `nvidia-smi` to
    determine which gpus are occupied and which are free.

    Args:
        num_gpus: Number of requested gpus. If not specified,
            ids of all available gpu(s) are returned.

    Returns:
        List[int]: List of gpu ids that are free. Length
            will equal `num_gpus`, if specified.
    """
    # Built-in tensorflow gpu id.
    assert isinstance(num_gpus, (type(None), int))
    if num_gpus == 0:
        return [-1]

    num_requested_gpus = num_gpus
    num_gpus = (
        len(
            subprocess.check_output("nvidia-smi --list-gpus", shell=True)
            .decode()
            .split("\n")
        )
        - 1
    )

    out_str = subprocess.check_output(
        "nvidia-smi | grep MiB", shell=True
    ).decode()
    mem_str = [x for x in out_str.split() if "MiB" in x]
    # First 2 * num_gpu elements correspond to memory for gpus
    # Order: (occupied-0, total-0, occupied-1, total-1, ...)
    mems = [float(x[:-3]) for x in mem_str]
    gpu_percent_occupied_mem = [
        mems[2 * gpu_id] / mems[2 * gpu_id + 1] for gpu_id in range(num_gpus)
    ]

    available_gpus = [
        gpu_id
        for gpu_id, mem in enumerate(gpu_percent_occupied_mem)
        if mem < 0.05
    ]
    if num_requested_gpus and num_requested_gpus > len(available_gpus):
        raise ValueError(
            "Requested {} gpus, only {} are free".format(
                num_requested_gpus, len(available_gpus)
            )
        )

    return (
        available_gpus[:num_requested_gpus]
        if num_requested_gpus
        else available_gpus
    )


def get_world_size():
    """Returns number of gpus currently being used by this process"""
    gpu_ids = os.environ["CUDA_VISIBLE_DEVICES"].split(",")
    if len(gpu_ids) == 1 and gpu_ids[0] == "-1":
        return 0
    return len(gpu_ids)


def is_debug() -> bool:
    return os.environ.get("SSRECON_DEBUG", "") == "True"


def supports_wandb():
    return "wandb" in sys.modules and not is_debug()
