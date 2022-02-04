import itertools
import os
import pathlib
from typing import Any, Dict, List, Sequence, Union

from meddlr.utils import env

from .config import CfgNode


def configure_params(
    params: Dict[str, Sequence], fixed: Dict[str, Any] = None, base_cfg: CfgNode = None
) -> List[Dict]:
    """Unroll parameters into a list of configurations.

    Args:
        params (Dict[Sequence]): Config keys and potential values they can take on.
            These parameters will be mixed and matched to create configs.
            Order is ``{"param1": [valA, valB, ...], "param2": [val1, val2, ...]}``
        fixed (Dict[str, Any]): Fixed parameters to apply to every configuration.

    Returns:
        List[Dict] | List[CfgNode]: Different configurations. If ``base_cfg`` provided,
            the different configurations are merged into the config.
    """

    def _dict_to_list(d):
        return [x for key_value in d.items() for x in key_value]

    configs = itertools.product(*list(params.values()))
    configs = [{k: v for k, v in zip(params.keys(), cfg)} for cfg in configs]
    if fixed is not None:
        for c in configs:
            c.update(fixed)
    if base_cfg:
        configs = [
            base_cfg.clone().defrost().merge_from_list(_dict_to_list(c)).freeze() for c in configs
        ]
    return configs


def stringify(cfg: Dict[str, Any]):
    """Convert param/value pairs into a command-line compatible string.

    Args:
        cfg (Dict[str, Any]): The configuration to stringify.

    Return:
        str: A command line compatible string.
    """
    cfg = {k: f"'{v}'" if isinstance(v, str) else v for k, v in cfg.items()}
    cfg = {k: _stringify_value(v) for k, v in cfg.items()}
    return " ".join(f'{k} "{v}"' for k, v in cfg.items())


def check_dependencies(
    cfg_file_or_lines, return_failed_deps: bool = False
) -> Union[bool, List[str]]:
    """Check that module dependencies are met for the config file.

    Dependencies are specified as comments in the config file starting
    with ``"# DEPENDENCY:"``.

    Args:
        cfg_file (str): The path to the config file.
        return_failed_deps (bool, optional): Whether to return the list of
            dependencies that are not met. Defaults to ``False``.

    Returns:
        bool | List[str]: If ``return_failed_deps=True``, returns the list of
        dependencies that are not met. Else returns boolean of whether all
        dependencies are met.
    """
    keyword = "# DEPENDENCY:"

    if isinstance(cfg_file_or_lines, (str, os.PathLike, pathlib.Path)):
        with open(cfg_file_or_lines, "r") as f:
            lines = f.readlines()
    else:
        lines = cfg_file_or_lines
    lines = [line.strip() for line in lines if keyword in line]
    dependencies = [
        dep.strip() for line in lines for dep in line.split(keyword)[-1].strip().split(";")
    ]

    missing_deps = []
    for dep in dependencies:
        if not env.is_package_installed(dep):
            missing_deps.append(dep)

    return missing_deps if return_failed_deps else len(missing_deps) == 0


def _stringify_value(value, depth=0) -> str:
    """Convert value into command-line comaptible string.

    The string representations of certain types (e.g. list, tuple)
    are not command-line compatible. We have to add extra ``""``
    around the string for proper formatting. This function
    formats values of primitive types (numbers, strs, collections, dicts).

    Args:
        value (Number | str | Collection | Dict): The value to convert
            to a command-line compatible string.

    Returns:
        str: The stringified value.
    """
    if not isinstance(value, (set, tuple, list, dict)):
        return value
    if isinstance(value, dict):
        keys = list(value.keys())
        values = [value[k] for k in keys]
        keys_str = [_stringify_value(k, depth=depth + 1) for k in keys]
        values_str = [_stringify_value(v, depth=depth + 1) for v in values]
        keys_str = _to_str(keys_str, keys)
        values_str = _to_str(values_str, values)
        val_dict_str = {f"{k}:{v}" for k, v in zip(keys_str, values_str)}
        value_str = "\{"
        value_str += ",".join(val_dict_str)
        value_str += "\}"
    else:
        all_vals = [_stringify_value(v, depth=depth + 1) for v in value]
        value_str = ",".join(
            str(v)
            if not isinstance(v, str) or isinstance(ov, (set, tuple, list, dict))
            else f"'\"'\"'{v}'\"'\"'"
            for v, ov in zip(all_vals, value)
        )
        if isinstance(value, tuple):
            value_str = f"\({value_str},\)" if len(value) > 0 else "\(\)"
        else:
            value_str = f"[{value_str}]"

    return value_str


def _to_str(all_vals, value):
    return [
        str(v)
        if not isinstance(v, str) or isinstance(ov, (set, tuple, list, dict))
        else f"'\"'\"'{v}'\"'\"'"
        for v, ov in zip(all_vals, value)
    ]
