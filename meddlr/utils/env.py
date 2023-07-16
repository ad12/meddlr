import importlib
import importlib.util
import logging
import multiprocessing as mp
import os
import random
import re
import subprocess
import sys
import warnings
from datetime import datetime
from importlib import util
from typing import Sequence, Union

import numpy as np
import torch
from iopath.common.file_io import PathManager, PathManagerFactory
from packaging import version

__all__ = []

_PT_VERSION = torch.__version__
_SETTINGS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.settings"))
_GITHUB_URL = "https://github.com/ad12/meddlr"
_SUPPORTED_PACKAGES = {}


class Version(version.Version):
    """
    An extension of packaging.version.Version that supports
    comparisons with strings and lists.
    """

    def _format_version(self, other: Union[str, Sequence[Union[str, int]]]):
        if isinstance(other, str):
            return version.Version(other)
        if isinstance(other, (list, tuple)):
            return version.Version(".".join(map(str, other)))
        return other

    def __eq__(self, other):
        return super().__eq__(self._format_version(other))

    def __lt__(self, other):
        return super().__lt__(self._format_version(other))

    def __le__(self, other):
        return super().__le__(self._format_version(other))

    def __gt__(self, other):
        return super().__gt__(self._format_version(other))

    def __ge__(self, other):
        return super().__ge__(self._format_version(other))

    def __ne__(self, other):
        return super().__ne__(self._format_version(other))


def seed_all_rng(seed=None):
    """
    Set the random seed for the RNG in torch, numpy and python.

    Args:
        seed (int): if None, will use a strong random seed.

    Returns:
        seed (int): The seed used.
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
    return seed


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
    disable_cv2 = int(os.environ.get("MEDDLR_DISABLE_CV2", False))
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

    custom_module_path = os.environ.get("MEDDLR_ENV_MODULE")

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
        module = _import_file("meddlr.utils.env.custom_module", custom_module)
    else:
        module = importlib.import_module(custom_module)
    assert hasattr(module, "setup_environment") and callable(module.setup_environment), (
        "Custom environment module defined in {} does not have the "
        "required callable attribute 'setup_environment'."
    ).format(custom_module)
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
            will equal `num_gpus`, if specified and gpus are available.
            If no gpus are found, returns an empty list.
    """
    # Built-in tensorflow gpu id.
    assert isinstance(num_gpus, (type(None), int))
    if num_gpus == 0:
        return []

    num_requested_gpus = num_gpus
    try:
        num_gpus = (
            len(subprocess.check_output("nvidia-smi --list-gpus", shell=True).decode().split("\n"))
            - 1
        )
    except subprocess.CalledProcessError:
        return []

    out_str = subprocess.check_output("nvidia-smi | grep MiB", shell=True).decode()
    mem_str = [x for x in out_str.split() if "MiB" in x]
    # First 2 * num_gpu elements correspond to memory for gpus
    # Order: (occupied-0, total-0, occupied-1, total-1, ...)
    mems = [float(x[:-3]) for x in mem_str]
    gpu_percent_occupied_mem = [
        mems[2 * gpu_id] / mems[2 * gpu_id + 1] for gpu_id in range(num_gpus)
    ]

    available_gpus = [gpu_id for gpu_id, mem in enumerate(gpu_percent_occupied_mem) if mem < 0.05]
    if num_requested_gpus and num_requested_gpus > len(available_gpus):
        raise ValueError(
            "Requested {} gpus, only {} are free".format(num_requested_gpus, len(available_gpus))
        )

    return available_gpus[:num_requested_gpus] if num_requested_gpus else available_gpus


def get_world_size():
    """Returns number of gpus currently being used by this process"""
    gpu_ids = os.environ["CUDA_VISIBLE_DEVICES"].split(",")
    if len(gpu_ids) == 1 and gpu_ids[0] == "-1":
        return 0
    return len(gpu_ids)


def package_available(name: str):
    """Returns if package is available.

    Args:
        name (str): Name of the package.

    Returns:
        bool: Whether module exists in environment.
    """
    global _SUPPORTED_PACKAGES
    if name not in _SUPPORTED_PACKAGES:
        _SUPPORTED_PACKAGES[name] = importlib.util.find_spec(name) is not None
    return _SUPPORTED_PACKAGES[name]


def get_package_version(package_or_name) -> str:
    """Returns package version.

    Args:
        package_or_name (``module`` or ``str``): Module or name of module.
            This package must have the version accessible through ``<module>.__version__``.

    Returns:
        str: The package version.

    Examples:
        >>> get_version("numpy")
        "1.20.0"
    """
    if isinstance(package_or_name, str):
        if not package_available(package_or_name):
            raise ValueError(f"Package {package_or_name} not available")
        spec = util.find_spec(package_or_name)
        package_or_name = util.module_from_spec(spec)
        spec.loader.exec_module(package_or_name)
    version = package_or_name.__version__
    return version


def is_package_installed(pkg_str) -> bool:
    """Verify that a package dependency is installed and in the expected version range.

    This is useful for optional third-party dependencies where implementation
    changes are not backwards-compatible.

    Args:
        pkg_str (str): The pip formatted dependency string.
            E.g. "numpy", "numpy>=1.0.0", "numpy>=1.0.0,<=1.10.0", "numpy==1.10.0"

    Returns:
        bool: Whether dependency is satisfied.

    Note:
        This cannot resolve packages where the pip name does not match the python
        package name. ``'-'`` characters are automatically changed to ``'_'``.
    """
    ops = {
        "==": lambda x, y: x == y,
        "<=": lambda x, y: x <= y,
        ">=": lambda x, y: x >= y,
        "<": lambda x, y: x < y,
        ">": lambda x, y: x > y,
    }
    comparison_patterns = "(==|<=|>=|>|<)"

    pkg_str = pkg_str.strip()
    pkg_str = pkg_str.replace("-", "_")
    dependency = list(re.finditer(comparison_patterns, pkg_str))

    if len(dependency) == 0:
        return package_available(pkg_str)

    pkg_name = pkg_str[: dependency[0].start()]
    if not package_available(pkg_name):
        return False

    pkg_version = version.Version(get_package_version(pkg_name))
    version_limits = pkg_str[dependency[0].start() :].split(",")

    for vlimit in version_limits:
        comp_loc = list(re.finditer(comparison_patterns, vlimit))
        if len(comp_loc) != 1:
            raise ValueError(f"Invalid version string: {pkg_str}")
        comp_op = vlimit[comp_loc[0].start() : comp_loc[0].end()]
        comp_version = version.Version(vlimit[comp_loc[0].end() :])
        if not ops[comp_op](pkg_version, comp_version):
            return False
    return True


def is_debug() -> bool:
    return os.environ.get("MEDDLR_DEBUG", "") == "True"


def is_pt_lightning() -> bool:
    return os.environ.get("MEDDLR_PT_LIGHTNING", "") == "True"


def is_repro() -> bool:
    return os.environ.get("MEDDLR_REPRO", "") == "True"


def is_profiling_enabled() -> bool:
    return os.environ.get("MEDDLR_PROFILE", "True") == "True"


def profile_memory() -> bool:
    return os.environ.get("MEDDLR_MPROFILE", "") == "True"


def supports_wandb():
    return "wandb" in sys.modules and not is_debug()


def supports_d2() -> bool:
    """Supports detectron2."""
    return "detectron2" in sys.modules


def supports_cupy():
    if "cupy" not in _SUPPORTED_PACKAGES:
        try:
            import cupy  # noqa
        except ImportError:
            _SUPPORTED_PACKAGES["cupy"] = False
    return package_available("cupy")


def pt_version() -> Version:
    """Returns the PyTorch version."""
    return Version(_PT_VERSION)


def supports_cplx_tensor() -> bool:
    """Returns `True` if complex tensors supported.

    This can be controlled by the environment variable
    "MEDDLR_ENABLE_CPLX_TENSORS", which should be set to
    "True", "False", or "auto". Defaults to "auto" functionality,
    which enables complex tensors if PyTorch>=1.7.0.
    While complex tensors were introduced in PyTorch 1.6, there were
    known bugs. Enhanced complex tensor support offered in PyTorch >=1.7.0.

    Returns:
        bool: `True` if complex tensors are supported.
    """
    env_var = os.environ.get("MEDDLR_ENABLE_CPLX_TENSORS", "auto")
    is_min_version = pt_version() >= [1, 6]
    is_auto_version = pt_version() >= [1, 7]

    if env_var == "auto":
        env_var = str(is_auto_version)

    if env_var.lower() == "false":
        return False
    elif env_var.lower() == "true":
        if not is_min_version:
            raise RuntimeError(f"Complex tensors not supported in PyTorch {_PT_VERSION}")
        if not is_auto_version:
            warnings.warn("Complex tensor support has known breaking bugs for PyTorch <1.7")
        return True
    else:
        raise ValueError(f"Unknown environment value: {env_var}")


def is_main_process():
    py_version = tuple(sys.version_info[0:2])
    return (py_version < (3, 8) and mp.current_process().name == "MainProcess") or (
        py_version >= (3, 8) and mp.parent_process() is None
    )


def settings_dir():
    return os.environ.get("MEDDLR_SETTINGS", _SETTINGS_DIR)


def get_path_manager(key="meddlr") -> PathManager:
    return PathManagerFactory.get(key)


def get_github_url() -> str:
    return _GITHUB_URL
