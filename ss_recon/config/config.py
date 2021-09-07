"""CfgNode implementation adapted from Detectron2.

Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""

import logging
import re
from typing import Mapping

from fvcore.common.config import CfgNode as _CfgNode


class CfgNode(_CfgNode):
    """
    The same as `fvcore.common.config.CfgNode`, but different in:

    1. Use unsafe yaml loading by default.
      Note that this may lead to arbitrary code execution: you must not
      load a config file from untrusted sources before manually inspecting
      the content of the file.
    2. Support config versioning.
      When attempting to merge an old config, it will convert the old config
      automatically.
    """

    # Note that the default value of allow_unsafe is changed to True
    def merge_from_file(self, cfg_filename: str, allow_unsafe: bool = True) -> None:
        loaded_cfg = _CfgNode.load_yaml_with_base(cfg_filename, allow_unsafe=allow_unsafe)
        loaded_cfg = type(self)(loaded_cfg)

        # defaults.py needs to import CfgNode
        from .defaults import _C

        latest_ver = _C.VERSION
        assert latest_ver == self.VERSION, (
            "CfgNode.merge_from_file is only allowed on a config " "object of latest version!"
        )

        logger = logging.getLogger(__name__)

        loaded_ver = loaded_cfg.get("VERSION", None)
        if loaded_ver is None:
            from .compat import guess_version

            loaded_ver = guess_version(loaded_cfg, cfg_filename)
        assert loaded_ver <= self.VERSION, "Cannot merge a v{} config into a v{} config.".format(
            loaded_ver, self.VERSION
        )

        if loaded_ver == self.VERSION:
            self.merge_from_other_cfg(loaded_cfg)
        else:
            # compat.py needs to import CfgNode
            from .compat import downgrade_config, upgrade_config

            logger.warning(
                "Loading an old v{} config file '{}' by "
                "automatically upgrading to v{}. "
                "See docs/CHANGELOG.md for instructions to "
                "update your files.".format(loaded_ver, cfg_filename, self.VERSION)
            )
            # To convert, first obtain a full config at an old version
            old_self = downgrade_config(self, to_version=loaded_ver)
            old_self.merge_from_other_cfg(loaded_cfg)
            new_config = upgrade_config(old_self)
            self.clear()
            self.update(new_config)

    def format_fields(self):
        """
        Format string fields in the config by filling them in
        with different parameter values.
        """
        return format_config_fields(self, inplace=True)

    def get_recursive(self, key):
        d = self
        for k in key.split("."):
            d = d[k]
        return d

    def dump(self, *args, **kwargs):
        """
        Returns:
            str: a yaml string representation of the config
        """
        # to make it show up in docs
        return super().dump(*args, **kwargs)

    def freeze(self):
        """Make this CfgNode and all of its children immutable."""
        super().freeze()
        return self

    def defrost(self):
        """Make this CfgNode and all of its children mutable."""
        super().defrost()
        return self


global_cfg = CfgNode()


def get_cfg() -> CfgNode:
    """
    Get a copy of the default config.

    Returns:
        a detectron2 CfgNode instance.
    """
    from .defaults import _C

    return _C.clone()


def set_global_cfg(cfg: CfgNode) -> None:
    """
    Let the global config point to the given cfg.

    Assume that the given "cfg" has the key "KEY", after calling
    `set_global_cfg(cfg)`, the key can be accessed by:

    .. code-block:: python

        from detectron2.config import global_cfg
        print(global_cfg.KEY)

    By using a hacky global config, you can access these configs anywhere,
    without having to pass the config object or the values deep into the code.
    This is a hacky feature introduced for quick prototyping / research
    exploration.
    """
    global global_cfg
    global_cfg.clear()
    global_cfg.update(cfg)


def _find_format_str_keys(cfg: Mapping, prefix="", accum=()):
    accum = set(accum)
    for k, v in cfg.items():
        k_prefix = prefix + "." + k if prefix else k
        if isinstance(v, Mapping):
            accum |= _find_format_str_keys(cfg[k], prefix=k_prefix, accum=accum)
        elif isinstance(v, str) and (
            (v.startswith('f"') and v.endswith('"')) or (v.startswith("f'") and v.endswith("'"))
        ):
            accum |= {(k_prefix, v)}
    return accum


def format_config_fields(cfg: CfgNode, inplace=False):
    keys_and_val_str = _find_format_str_keys(cfg)
    values_list = []
    for k, val_str in keys_and_val_str:
        start = [x.start() for x in re.finditer("\{", val_str)]
        end = [x.start() for x in re.finditer("\}", val_str)]
        assert len(start) == len(end), f"Could not determine formatting string: {val_str}"
        cfg_keys_to_search = [val_str[s + 1 : e] for s, e in zip(start, end)]
        values = [cfg.get_recursive(v) for v in cfg_keys_to_search]

        fmt_str = ""
        idxs = [0] + [y for x in zip(start, end) for y in x] + [len(val_str)]
        for i in range(len(idxs) // 2):
            fmt_str += val_str[idxs[2 * i] : idxs[2 * i + 1] + 1]
        fmt_str = eval(fmt_str.format(*values))
        values_list.extend([k, fmt_str])
    if not inplace:
        cfg.clone()
    cfg.defrost().merge_from_list(values_list)
    return cfg
