import os
import warnings
from typing import Any, Callable, Dict, Union

import torch

import meddlr.config.util as config_util
from meddlr.config import CfgNode, get_cfg
from meddlr.modeling.meta_arch import build_model
from meddlr.utils import env

__all__ = ["get_model_from_zoo"]


def get_model_from_zoo(
    cfg_or_file: Union[str, os.PathLike, CfgNode],
    weights_path: Union[str, os.PathLike] = None,
    strict: bool = True,
    ignore_shape_mismatch: bool = False,
    force_download: bool = False,
    build_model_fn: Callable = None,
    build_model_kwargs: Dict[str, Any] = None,
) -> torch.nn.Module:
    """Get model from zoo and optionally load in weights.

    This function is designed for distributing models for use.
    It builds the model from a configuration and optionally loads in pre-trained weights.

    Pre-trained weights can either be specified by the ``weights_file`` argument
    or by setting the config option ``cfg.MODEL.WEIGHTS``. If neither is specified,
    the model is initialized randomly. If ``weights_file`` is an empty string,
    ``cfg.MODEL.WEIGHTS`` will also be ignored, and the model will be initialized randomly.

    Args:
        cfg_or_file (str | path-like | CfgNode): The config (or file).
            If it is a file, it will be merged with the default config.
        weights_path (str | path-like, optional): The weights file to load.
            This can also be specified in the ``cfg.MODEL.WEIGHTS`` config
            field. If neither are provided, the uninitialized model will be returned.
        strict (bool, optional): Strict option to pass to ``load_state_dict``.
        ignore_shape_mismatch (bool, optional): If ``True``, weights that do not
            match the model layer's shape will be ignored instead of raising
            an error.
        force_download (bool, optional): Force download of model config/weights
            if downloading the model.
        build_model_fn (callable, optional): A function to build the model.
            Defaults to :func:`meddlr.modeling.meta_arch.build_model`.
        build_model_kwargs (dict, optional): Keyword arguments to pass to
            ``build_model_fn``.

    Returns:
        torch.nn.Module: The model loaded with pre-trained weights.

    Examples:
        .. code-block:: python
            from meddlr.config import get_cfg
            from meddlr.engine.model_zoo import get_model_from_zoo

            # Load a pretrained model from a config file.
            cfg_file = "https://huggingface.co/arjundd/vortex-release/raw/main/mridata_knee_3dfse/Supervised/config.yaml"  # noqa: E501
            weights_path = "https://huggingface.co/arjundd/vortex-release/resolve/main/mridata_knee_3dfse/Supervised/model.ckpt"  # noqa: E501
            model = get_model_from_zoo(cfg_file, weights_path)

            # Build a model from config file with randomly initialized weights.
            model = get_model_from_zoo(cfg_file, weights_path="")

            # Build model from a config object.
            cfg = get_cfg()
            cfg.MODEL.META_ARCHITECTURE = "GeneralizedUnrolledCNN"
            model = get_model_from_zoo(cfg)
    """
    path_manager = env.get_path_manager()

    if not isinstance(cfg_or_file, CfgNode):
        cfg_file = path_manager.get_local_path(cfg_or_file, force=force_download)
        failed_deps = config_util.check_dependencies(cfg_file, return_failed_deps=True)
        if len(failed_deps) > 0:  # pragma: no cover
            warning_msg = (
                f"Some dependenices are not met. "
                f"This may result in some issues with model construction or weight loading. "
                f"Unmet dependencies: {failed_deps}"
            )
            warnings.warn(warning_msg)
        cfg = get_cfg()
        cfg.merge_from_file(cfg_file)
    else:
        cfg = cfg_or_file

    if build_model_fn is None:
        build_model_fn = build_model
    if not build_model_kwargs:
        build_model_kwargs = {}
    model = build_model_fn(cfg, **build_model_kwargs)

    if weights_path is None:
        weights_path = cfg.MODEL.WEIGHTS
    if not weights_path:
        return model

    weights_path = path_manager.get_local_path(weights_path, force=force_download)
    model = load_weights(
        model, weights_path, strict=strict, ignore_shape_mismatch=ignore_shape_mismatch
    )
    return model


def load_weights(
    model: torch.nn.Module,
    weights_path: Union[str, os.PathLike],
    strict: bool = True,
    ignore_shape_mismatch: bool = False,
    force_download: bool = False,
    find_device: bool = True,
) -> torch.nn.Module:
    """Load model from checkpoint.

    This function is designed for distributing models for use. It loads in pre-trained weights.
    Pre-trained weights can either be specified by the ``weights_file`` argument
    or by setting the config option ``cfg.MODEL.WEIGHTS``. If neither is specified,
    the model is initialized randomly. If ``weights_file`` is an empty string,
    ``cfg.MODEL.WEIGHTS`` will also be ignored, and the model will be initialized randomly.

    Args:
        model (torch.nn.Module): The model to load weights into.
        checkpoint_path (str | path-like): The checkpoint file to load.
        strict (bool, optional): Strict option to pass to ``load_state_dict``.
        ignore_shape_mismatch (bool, optional): If ``True``, weights that do not
            match the model layer's shape will be ignored instead of raising
            an error.
        force_download (bool, optional): Force download of model config/weights
            if downloading the model.
        find_device (bool, optional): If ``True``, find the device the weights
            were stored on and moves the model prior to loading weights.

    Returns:
        torch.nn.Module: The model loaded with pre-trained weights.
            Note this model will be unpacked from the PyTorch Lightnining modules.
    """
    path_manager = env.get_path_manager()

    weights_path = path_manager.get_local_path(weights_path, force=force_download)
    weights = torch.load(weights_path)
    if "state_dict" in weights:
        weights = weights["state_dict"]
    elif "model" in weights:
        weights = weights["model"]

    if ignore_shape_mismatch:
        params_shape_mismatch = _find_mismatch_sizes(model, weights)
        if len(params_shape_mismatch) > 0:
            mismatched_params_str = "".join("\t- {}\n".format(x) for x in params_shape_mismatch)
            warnings.warn(
                "Shape mismatch found for some parameters. Ignoring these weights:\n{}".format(
                    mismatched_params_str
                )
            )
            for p in params_shape_mismatch:
                weights.pop(p)
            strict = False

    if find_device:
        device = next(iter(weights.values())).device
        model = model.to(device)

    model.load_state_dict(weights, strict=strict)

    return model


def _find_mismatch_sizes(model: torch.nn.Module, state_dict):
    """Finds the keys in the state_dict that are different from the model.
    Args:
        model (torch.nn.Module): The model to load weights into.
        state_dict (dict): The state_dict to load.
    Returns:
        list[str]: The list of keys that are different from the model.
    """
    params_shape_mismatch = set()
    for source in [model.state_dict().items(), model.named_parameters()]:
        for name, param in source:
            if name not in state_dict:
                continue
            if param.shape != state_dict[name].shape:
                params_shape_mismatch |= {name}

    return list(params_shape_mismatch)
