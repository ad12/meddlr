import os
from typing import List, Mapping

import torch

from meddlr.utils import env

_PATH_MANAGER = env.get_path_manager()


def move_to_device(obj, device, non_blocking=False, base_types=None):
    """Given a structure (possibly) containing Tensors on the CPU, move all the Tensors
      to the specified GPU (or do nothing, if they should be on the CPU).
        device = -1 -> "cpu"
        device =  0 -> "cuda:0"

    Args:
      obj(Any): The object to convert.
      device(int): The device id, defaults to -1.

    Returns:
      Any: The converted object.
    """
    if not torch.cuda.is_available() or (isinstance(device, int) and device < 0):
        device = "cpu"
    elif isinstance(device, int):
        device = f"cuda:{device}"

    if isinstance(obj, torch.Tensor):
        return obj.to(device, non_blocking=non_blocking)  # type: ignore
    elif base_types is not None and isinstance(obj, base_types):
        return obj.to(device)
    elif isinstance(obj, dict):
        return {
            key: move_to_device(value, device, non_blocking=non_blocking, base_types=base_types)
            for key, value in obj.items()
        }
    elif isinstance(obj, list):
        return [
            move_to_device(item, device, non_blocking=non_blocking, base_types=base_types)
            for item in obj
        ]
    elif isinstance(obj, tuple):
        return tuple(
            [
                move_to_device(item, device, non_blocking=non_blocking, base_types=base_types)
                for item in obj
            ]
        )
    else:
        return obj


def find_experiment_dirs(dirpath, completed=True) -> List[str]:
    """Recursively search for experiment directories under the ``dirpath``.

    Args:
        dirpath (str): The base directory under which to search.
        completed (bool, optional): If `True`, filter directories where runs
            are completed.

    Returns:
        exp_dirs (List[str]): A list of experiment directories.
    """

    def _find_exp_dirs(_dirpath):
        # Directories with "config.yaml" are considered experiment directories.
        if os.path.isfile(os.path.join(_dirpath, "config.yaml")):
            return [_dirpath]
        # Directories with no more subdirectories do not have a path.
        subfiles = [os.path.join(_dirpath, x) for x in os.listdir(_dirpath)]
        subdirs = [x for x in subfiles if os.path.isdir(x)]
        if len(subdirs) == 0:
            return []
        exp_dirs = []
        for dp in subdirs:
            exp_dirs.extend(_find_exp_dirs(dp))
        return exp_dirs

    dirpath = _PATH_MANAGER.get_local_path(dirpath)
    exp_dirs = _find_exp_dirs(dirpath)
    if completed:
        exp_dirs = [x for x in exp_dirs if os.path.isfile(os.path.join(x, "model_final.pth"))]
    return exp_dirs


def flatten_dict(results, delimiter="/"):
    """
    Expand a hierarchical dict of scalars into a flat dict of scalars.
    If results[k1][k2][k3] = v, the returned dict will have the entry
    {"k1/k2/k3": v}.

    Args:
        results (dict):
    """
    r = {}
    for k, v in results.items():
        if isinstance(v, Mapping):
            v = flatten_dict(v)
            for kk, vv in v.items():
                r[k + delimiter + kk] = vv
        else:
            r[k] = v
    return r
