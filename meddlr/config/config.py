"""CfgNode implementation.
"""
import functools
import inspect
import logging
import re
from typing import Any, List, Mapping, Tuple

import numpy as np
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
        """
        Adapted from https://github.com/facebookresearch/detectron2
        """
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
        return self

        return self

    def merge_from_list(self, cfg_list: list):
        """Update (keys, values) in a list (e.g., from command line).

        For example, ``cfg_list = ['FOO.BAR', 0.5]`` to set ``self.FOO.BAR = 0.5``.

        Args:
            cfg_list (list): list of configs to merge from.
        """
        super().merge_from_list(cfg_list)
        return self

    def format_fields(self, unroll: bool = False):
        """
        Format string fields in the config by filling them in
        with different parameter values.

        The operation is done in place.

        Args:
            unroll (bool, optional): If ``True``, sequence types (e.g. tuple, list)
                stringified as ``seq[0]-seq[1]-...``. Dict types will be
                stringified as ``k[0]=v[0]-k[1]=v[1]-...``.

        Returns:
            CfgNode: self
        """
        return format_config_fields(self, unroll=unroll, inplace=True)

    def get_recursive(self, key, default: Any = np._NoValue):
        """Get a key recursively from the config.

        Args:
            key (str): The dot-separated key.
            default (Any, optional): The value to return if the key is not found.
                If not provided, a KeyError will be raised.

        Returns:
            object: The value corresponding to the key.

        Raises:
            KeyError: If the key is not found and no default was provided.
        """
        d = self
        try:
            for k in key.split("."):
                k, index = _extract_field_index(k)
                d = d[k] if index is None else d[k][index]
        except KeyError:
            if default != np._NoValue:
                return default
            raise KeyError("Config does not have key '{}'".format(key))
        return d

    def set_recursive(self, name: str, value: Any):
        """Set a key recursively in the config.

        Args:
            name (str): The dot-separated key.
            value (object): The value to set.
        """
        cfg = self
        keys = name.split(".")
        for k in keys[:-1]:
            # Extract groups matching sequence-indexing syntax (e.g. 'field[0]').
            k, index = _extract_field_index(k)
            cfg = cfg[k] if index is None else cfg[k][index]

        k, index = _extract_field_index(keys[-1])
        if index is not None:
            if not isinstance(cfg[k], (list, tuple)):
                raise TypeError(f"Cannot set index {index} for non-sequence field '{name}'")
            current_value = cfg[k]
            if isinstance(current_value, tuple):
                current_value = list(current_value)
            current_value[index] = value
            current_value = type(cfg[k])(current_value)
            value = current_value

        setattr(cfg, k, value)

    def update_recursive(self, mapping: Mapping[str, Any]):
        """
        Update this CfgNode and all of its children recursively.

        Args:
            mapping (dict): a dict to update this CfgNode and all of its children.

        Returns:
            CfgNode: self
        """
        for k, v in mapping.items():
            self.set_recursive(k, v)
        return self

    def dump(self, *args, **kwargs):  # pragma: no cover
        """
        Returns:
            str: a yaml string representation of the config
        """
        # to make it show up in docs
        return super().dump(*args, **kwargs)

    def freeze(self):
        """Make this CfgNode and all of its children immutable.

        Returns:
            CfgNode: self
        """
        super().freeze()
        return self

    def defrost(self):
        """Make this CfgNode and all of its children mutable.

        Returns:
            CfgNode: self
        """
        super().defrost()
        return self


global_cfg = CfgNode()
_base_cfg = None


def get_cfg() -> CfgNode:
    """
    Get a copy of the default config.

    Returns:
        a meddlr CfgNode instance.
    """
    from .defaults import _C

    cfg = _C if _base_cfg is None else _base_cfg
    return cfg.clone()


def set_cfg(cfg: CfgNode) -> None:
    """Set the base config object to use.

    This is useful when customizing meddlr for different projects.

    Args:
        cfg (CfgNode): Set the base config.
    """
    global _base_cfg
    _base_cfg = cfg


def set_global_cfg(cfg: CfgNode) -> None:
    """
    Let the global config point to the given cfg.

    Assume that the given "cfg" has the key "KEY", after calling
    `set_global_cfg(cfg)`, the key can be accessed by:

    .. code-block:: python

        from meddlr.config import global_cfg
        print(global_cfg.KEY)

    By using a hacky global config, you can access these configs anywhere,
    without having to pass the config object or the values deep into the code.
    This is a hacky feature introduced for quick prototyping / research
    exploration.
    """
    global global_cfg
    global_cfg.clear()
    global_cfg.update(cfg)


def configurable(init_func=None, *, from_config=None):  # pragma: no cover
    """
    Decorate a function or a class's __init__ method so that it can be called
    with a :class:`CfgNode` object using a :func:`from_config` function that translates
    :class:`CfgNode` to arguments.

    Adapted from https://github.com/facebookresearch/detectron2

    Examples:
    ::
        # Usage 1: Decorator on __init__:
        class A:
            @configurable
            def __init__(self, a, b=2, c=3):
                pass
            @classmethod
            def from_config(cls, cfg):   # 'cfg' must be the first argument
                # Returns kwargs to be passed to __init__
                return {"a": cfg.A, "b": cfg.B}
        a1 = A(a=1, b=2)  # regular construction
        a2 = A(cfg)       # construct with a cfg
        a3 = A(cfg, b=3, c=4)  # construct with extra overwrite
        # Usage 2: Decorator on any function. Needs an extra from_config argument:
        @configurable(from_config=lambda cfg: {"a: cfg.A, "b": cfg.B})
        def a_func(a, b=2, c=3):
            pass
        a1 = a_func(a=1, b=2)  # regular call
        a2 = a_func(cfg)       # call with a cfg
        a3 = a_func(cfg, b=3, c=4)  # call with extra overwrite
    Args:
        init_func (callable): a class's ``__init__`` method in usage 1. The
            class must have a ``from_config`` classmethod which takes `cfg` as
            the first argument.
        from_config (callable): the from_config function in usage 2. It must take `cfg`
            as its first argument.
    """

    if init_func is not None:
        assert (
            inspect.isfunction(init_func)
            and from_config is None
            and init_func.__name__ == "__init__"
        ), "Incorrect use of @configurable. Check API documentation for examples."

        @functools.wraps(init_func)
        def wrapped(self, *args, **kwargs):
            try:
                from_config_func = type(self).from_config
            except AttributeError as e:
                raise AttributeError(
                    "Class with @configurable must have a 'from_config' classmethod."
                ) from e
            if not inspect.ismethod(from_config_func):
                raise TypeError("Class with @configurable must have a 'from_config' classmethod.")

            if _called_with_cfg(*args, **kwargs):
                explicit_args = _get_args_from_config(from_config_func, *args, **kwargs)
                init_func(self, **explicit_args)
            else:
                init_func(self, *args, **kwargs)

        return wrapped

    else:
        if from_config is None:
            return configurable  # @configurable() is made equivalent to @configurable
        assert inspect.isfunction(
            from_config
        ), "from_config argument of configurable must be a function!"

        def wrapper(orig_func):
            @functools.wraps(orig_func)
            def wrapped(*args, **kwargs):
                if _called_with_cfg(*args, **kwargs):
                    explicit_args = _get_args_from_config(from_config, *args, **kwargs)
                    return orig_func(**explicit_args)
                else:
                    return orig_func(*args, **kwargs)

            return wrapped

        return wrapper


def _get_args_from_config(from_config_func, *args, **kwargs):
    """
    Use `from_config` to obtain explicit arguments.
    Returns:
        dict: arguments to be used for cls.__init__
    """
    signature = inspect.signature(from_config_func)
    if list(signature.parameters.keys())[0] != "cfg":
        if inspect.isfunction(from_config_func):
            name = from_config_func.__name__
        else:
            name = f"{from_config_func.__self__}.from_config"
        raise TypeError(f"{name} must take 'cfg' as the first argument!")
    support_var_arg = any(
        param.kind in [param.VAR_POSITIONAL, param.VAR_KEYWORD]
        for param in signature.parameters.values()
    )
    if support_var_arg:  # forward all arguments to from_config, if from_config accepts them
        ret = from_config_func(*args, **kwargs)
    else:
        # forward supported arguments to from_config
        supported_arg_names = set(signature.parameters.keys())
        extra_kwargs = {}
        for name in list(kwargs.keys()):
            if name not in supported_arg_names:
                extra_kwargs[name] = kwargs.pop(name)
        ret = from_config_func(*args, **kwargs)
        # forward the other arguments to __init__
        ret.update(extra_kwargs)
    return ret


def _called_with_cfg(*args, **kwargs):
    """
    Returns:
        bool: whether the arguments contain CfgNode and should be considered
            forwarded to from_config.
    """
    from omegaconf import DictConfig

    if len(args) and isinstance(args[0], (_CfgNode, DictConfig)):
        return True
    if isinstance(kwargs.pop("cfg", None), (_CfgNode, DictConfig)):
        return True
    # `from_config`'s first argument is forced to be "cfg".
    # So the above check covers all cases.
    return False


def _find_format_str_keys(value: Mapping, prefix=""):
    accum = set()

    if isinstance(value, str) and (
        (value.startswith('f"') and value.endswith('"'))
        or (value.startswith("f'") and value.endswith("'"))
    ):
        return {(prefix, value)}
    elif not isinstance(value, (Mapping, List, Tuple)):
        return accum

    if isinstance(value, Mapping):
        for k, v in value.items():
            k_prefix = f"{prefix}.{k}" if prefix else k
            accum |= _find_format_str_keys(v, prefix=k_prefix)
    elif isinstance(value, (list, tuple)):
        for i in range(len(value)):
            accum |= _find_format_str_keys(value[i], prefix=f"{prefix}[{i}]")
    return accum


def _unroll_value_to_str(value) -> str:
    if isinstance(value, (tuple, list)):
        return "-".join(_unroll_value_to_str(v) for v in value)
    elif isinstance(value, dict):
        return "-".join(f"{k}={_unroll_value_to_str(v)}" for k, v in value.items())
    else:
        return str(value)


def _format_str(val_str: str, *, cfg: CfgNode, unroll: bool):
    start = [x.start() for x in re.finditer("\{", val_str)]
    end = [x.start() for x in re.finditer("\}", val_str)]
    assert len(start) == len(end), f"Could not determine formatting string: {val_str}"

    if len(start) == 0:
        return val_str

    cfg_keys_to_search = [val_str[s + 1 : e] for s, e in zip(start, end)]
    values = [cfg.get_recursive(v) for v in cfg_keys_to_search]

    if unroll:
        values = [_unroll_value_to_str(v) for v in values]

    fmt_str = ""
    idxs = [0] + [y for x in zip(start, end) for y in x] + [len(val_str)]
    for i in range(len(idxs) // 2):
        fmt_str += val_str[idxs[2 * i] : idxs[2 * i + 1] + 1]
    fmt_str = eval(fmt_str.format(*values))
    return fmt_str


def format_config_fields(cfg: CfgNode, unroll=False, inplace=False):
    keys_and_val_str = _find_format_str_keys(cfg)
    values_list = []
    for k, value in keys_and_val_str:
        if isinstance(value, (list, tuple)):
            fmt_str = type(value)(
                _format_str(v, cfg=cfg, unroll=unroll) if isinstance(v, str) else v for v in value
            )
        else:
            assert isinstance(value, str)
            fmt_str = _format_str(value, cfg=cfg, unroll=unroll)
        values_list.append([k, fmt_str])
    if not inplace:
        cfg.clone()

    # TODO: Determine if we want to enforce the checks in merge_from_list.
    # This may be important if we want to handle renamed or deprecated keys.
    # cfg.defrost().merge_from_list(values_list)
    cfg.defrost()
    for k, fmt_str in values_list:
        cfg.set_recursive(k, fmt_str)

    return cfg


def _extract_field_index(key):
    match_val = re.match("^(?P<field>[a-zA-Z0-9_]+)\[(?P<index>[-?0-9]+)\]$", key)
    if match_val:
        return match_val.group("field"), int(match_val.group("index"))
    else:
        return key, None
