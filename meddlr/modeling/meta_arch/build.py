import ast
import inspect
import logging
import re
from collections import defaultdict
from typing import Dict, List, Sequence, Tuple, Union

from torch import nn

from meddlr.modeling.layers.build import get_layer_kind
from meddlr.utils import env
from meddlr.utils.registry import Registry

_SUPPORTS_MONAI = env.is_package_installed("monai")

META_ARCH_REGISTRY = Registry("META_ARCH")  # noqa F401 isort:skip
META_ARCH_REGISTRY.__doc__ = """
Registry for meta-architectures, i.e. the whole model.

The registered object will be called with `obj(cfg)`
and expected to return a `nn.Module` object.
"""

logger = logging.getLogger(__name__)


def build_model(cfg):
    """
    Build the whole model architecture, defined by ``cfg.MODEL.META_ARCHITECTURE``.
    Note that it does not load any weights from ``cfg``.
    """
    meta_arch = cfg.MODEL.META_ARCHITECTURE
    if "/" in meta_arch:
        module_name, meta_arch = tuple(meta_arch.split("/", 1))
    else:
        module_name = None

    # TODO: Consider adding `monai` to the registry and decorate the
    # build_monai_net function.
    if module_name == "monai":
        return build_monai_net(cfg, meta_arch)
    elif module_name is None:
        return META_ARCH_REGISTRY.get(meta_arch)(cfg)
    else:
        raise ValueError(f"Unknown module name: {module_name}")


def build_monai_net(cfg, name: str):
    """Build a MONAI network.

    This function is a wrapper for MONAI networks that
    handles the conversion of the configuration to the
    appropriate arguments for the network.

    TODO:
        - Log warnings only once.

    Args:
        cfg: The network config.
        name: The name of the network.

    Returns:
        torch.nn.Module: The MONAI network.
    """
    if not _SUPPORTS_MONAI:
        raise ImportError("MONAI is not installed. Install with `pip install monai`.")

    # The import statement needs to be in the function to avoid creating temporary
    # files at import time. This can cause issues on Google Colab with skm-tea.
    from monai.networks import nets as monai_nets

    logger.warn(
        "Signatures for MONAI networks may not be backwards compatible. "
        "Newer versions of MONAI may require careful configuration of network arguments."
    )

    klass = getattr(monai_nets, name)
    sig = inspect.signature(klass)

    build_cfg = cfg.get_recursive(f"MODEL.MONAI.{name}")
    kwargs = {}
    for k, v in build_cfg.items():
        if k in sig.parameters:
            kwargs[k] = v
        else:
            logger.warn(f"No parameter `{k}` for MONAI network `{name}`.")

    return klass(**kwargs)


def initialize_model(model: nn.Module, initializers: Union[Dict, Tuple]):
    """Initialize the model.

    This function initializes the model using the initialization method
    specified in ``initializers``.

    ``initializers`` should be a sequence of dicts, where each dict
    defines the layer type (optional), regex pattern of the parameter
    name (optional), or the dict. The dict has the following keys:

        * 'kind' (str, optional): The layer kind to apply this (e.g. 'conv', 'norm')
        * 'patterns' (Tuple[str] | str, optional): The regex patterns of the
                parameters to use initializer on. If not specified, all parameters
                of
        * 'initializers' (Sequence[Callable | str]): The initializers to use on the
                parameters. These should be called as ``initializer(param)``.
                These values should be 1:1 with the values in ``'patterns'``.

    Args:
        model (nn.Model): The model to initialize. Parameters will be fetched
            with ``model.named_parameters()``.
        initializers (Dict[str, Union[str, Callable]] | Tuple[str]):
            See above.
    """
    _kind_kwd = "kind"
    _pattern_kwd = "patterns"
    _init_kwd = "initializers"

    if isinstance(initializers, Dict):
        initializers = [initializers]

    # Backwards compatibility with pattern-only initialization.
    if isinstance(initializers, Sequence) and not any(isinstance(x, Dict) for x in initializers):
        assert len(initializers) % 2 == 0, "Sequence of regex_to_init must be even"
        initializers = [
            {_pattern_kwd: k, _init_kwd: v} for k, v in zip(initializers[::2], initializers[1::2])
        ]

    # Convert string values to python literals.
    initializers = _to_literal(initializers)

    if not all(
        isinstance(x, Dict) or (isinstance(x, Sequence) and len(x) == 2) for x in initializers
    ):
        raise ValueError(
            "All initializers must either be a dict or sequence of 2 elements "
            "(pattern, initializer). Got:\n\t{}".format(initializers)
        )

    initializers: List[Dict] = [
        {_pattern_kwd: x[0], _init_kwd: x[1]} if isinstance(x, Sequence) else x
        for x in initializers
    ]
    matched_patterns = {}
    for init_cfg in initializers:
        pattern = init_cfg.pop(_pattern_kwd, None)
        if pattern is None:
            pattern = (".*",)
        elif not isinstance(pattern, Sequence) or isinstance(pattern, str):
            pattern = (pattern,)

        init_method = init_cfg.pop(_init_kwd)
        if not isinstance(init_method, (list, tuple)):
            init_method = (init_method,)

        if len(pattern) != len(init_method):
            raise ValueError(
                "Got {} pattern(s) but {} initializer(s):\n\t"
                "Patterns: {}\n\tInitializers: {}".format(
                    len(pattern), len(init_method), pattern, init_method
                )
            )

        pattern = tuple(
            (p,) if not isinstance(p, Sequence) or isinstance(p, str) else p for p in pattern
        )

        init_method = [
            getattr(nn.init, v)
            if isinstance(v, str)
            else (getattr(nn.init, v[0]), v[1])
            if isinstance(v, Sequence)
            else v
            for v in init_method
        ]
        init_cfg[_init_kwd] = {k: v for k, v in zip(pattern, init_method)}
        for pattern_val in pattern:
            key = (init_cfg[_kind_kwd], pattern_val) if _kind_kwd in init_cfg else pattern_val
            matched_patterns[key] = False

    model_layers = {}
    if any(_kind_kwd in x for x in initializers):
        model_layers: Dict[str, nn.Module] = _get_model_layers(model, by_kind=True)
    named_parameters = list(model.named_parameters())

    # Initialize parameters
    for init_cfg in initializers:
        layer_kind = ""
        if _kind_kwd in init_cfg:
            layer_kind = init_cfg[_kind_kwd]
            if layer_kind not in model_layers:
                logger.warning(
                    f"Layer kind '{layer_kind}' not found in model {type(model).__name__}"
                )
                continue
            parameters = [
                named_param
                for layer in model_layers[layer_kind]
                for named_param in layer.named_parameters()
            ]
        else:
            parameters = named_parameters

        patterns_to_initializers = init_cfg[_init_kwd]
        for name, param in parameters:
            for patterns, v in patterns_to_initializers.items():
                if any(re.match(k, name) for k in patterns):
                    matched_patterns[(layer_kind, patterns) if layer_kind else patterns] = True
                    if isinstance(v, tuple):
                        v[0](param, **v[1])
                    else:
                        v(param)

    unmatched_patterns = [k for k, v in matched_patterns.items() if not v]
    if unmatched_patterns:
        logger.warning(
            "No matches found for these patterns when initialing model '{}':"
            "\n\t- {}\nCheck the model's named parameters: {}\n".format(
                type(model).__name__,
                "\n\t- ".join(str(x) for x in unmatched_patterns),
                [x[0] for x in model.named_parameters()],
            )
        )


def _get_model_layers(model, by_kind=False):
    """Get all layers of a model.

    A layer is defined as a ``nn.Module`` that does not have
    any children.
    """
    modules = model.modules()
    layers = []
    for module in modules:
        has_children = False
        for _ in module.children():
            has_children = True
            break
        if not has_children:
            layers.append(module)

    if by_kind:
        layers_by_kind = defaultdict(list)
        for layer in layers:
            layers_by_kind[get_layer_kind(type(layer))].append(layer)
        return layers_by_kind
    return layers


def _to_literal(x):
    if isinstance(x, str):
        v = x
        if any(
            x.startswith(c1) and x.endswith(c2) for c1, c2 in [("(", ")"), ("[", "]"), ("{", "}")]
        ):
            try:
                v = ast.literal_eval(v)
            except ValueError:
                pass
        return v
    elif isinstance(x, Dict):
        return {_to_literal(k): _to_literal(v) for k, v in x.items()}
    elif isinstance(x, Sequence):
        return type(x)(_to_literal(v) for v in x)
    else:
        return x
