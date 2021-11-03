from typing import Union

from fvcore.common.registry import Registry
from torch import nn

CUSTOM_LAYERS_REGISTRY = Registry("CUSTOM_LAYERS")
CUSTOM_LAYERS_REGISTRY.__doc__ = """
Registry for custom layers.

Use this registry to identify if the layer is not provided by default in torch.nn.
"""

_LAYER_SHORTCUTS = {
    "bn": "batchnorm",
    "dropout1d": "dropout",
}
_PT_LAYERS_LOWERCASE = {
    name.lower(): layer
    for name, layer in nn.__dict__.items()
    if isinstance(layer, type) and issubclass(layer, nn.Module)
}

__all__ = ["get_layer_type", "get_layer_kind"]


def get_layer_type(name: str, dimension: int = None):
    """Returns the layer type based on the name and, in some cases, the dimension."""
    in_name = name
    name = name.lower()

    # Handle some layer names earlier because of naming conventions.
    if name == "dropout" and dimension:
        name += f"{dimension}d"

    if name in _LAYER_SHORTCUTS:
        name = _LAYER_SHORTCUTS[name]

    custom_layer_names = {x.lower(): x for x in CUSTOM_LAYERS_REGISTRY._obj_map}
    if name in CUSTOM_LAYERS_REGISTRY:
        return CUSTOM_LAYERS_REGISTRY.get(name)
    if name in custom_layer_names:
        return CUSTOM_LAYERS_REGISTRY.get(custom_layer_names[name])
    if name in _PT_LAYERS_LOWERCASE:
        return _PT_LAYERS_LOWERCASE.get(name)

    if any(x in name for x in ["norm", "conv"]):
        if not dimension:
            raise ValueError(f"{in_name} requires dimension")

    if dimension:
        name += f"{dimension}d"
    if name in CUSTOM_LAYERS_REGISTRY:
        return CUSTOM_LAYERS_REGISTRY.get(name)
    if name in custom_layer_names:
        return CUSTOM_LAYERS_REGISTRY.get(custom_layer_names[name])
    if name in _PT_LAYERS_LOWERCASE:
        return _PT_LAYERS_LOWERCASE.get(name)

    raise ValueError(f"No layer found for '{in_name}'")


def get_layer_kind(layer_type: Union[type, str]):
    if isinstance(layer_type, str):
        layer_type = get_layer_type(layer_type, dimension=2)
    assert issubclass(layer_type, nn.Module)
    kinds_in_name = ["norm", "conv"]
    name = layer_type.__name__
    lower_name = name.lower()
    for kind in kinds_in_name:
        if kind in lower_name:
            return kind

    if hasattr(nn.modules.activation, name):
        return "act"

    if "dropout" in lower_name:
        return "dropout"

    return "unknown"
