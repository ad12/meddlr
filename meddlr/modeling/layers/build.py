import copy
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

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


def get_layer_type(name: str, dimension: int = None) -> type:
    """Returns the layer type based on the name and, in some cases, the dimension.

    This function searches both the default PyTorch layers and the custom layers
    registered in :obj:`CUSTOM_LAYERS_REGISTRY`.

    Args:
        name (str): Name of the layer.
        dimension (int, optional): Dimension of the layer.
            Note, not all layers require this argument (e.g. ReLU), but
            it may be safe to pass it regardless.

    Returns:
        type: Layer type.

    Raises:
        ValueError: If the layer type is not found with the name/dimension pair.
    """
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


def get_layer_kind(layer_type: Union[type, str]) -> str:
    """Returns the layer kind based on the layer type.

    The layer kind is a string that describes the kind of layer:

        - "conv": Convolutional layer.
        - "norm": Normalization layer.
        - "act": Activation layer.
        - "dropout": Dropout layer.
        - "unknown": Unknown layer.

    This delineation is useful for building models that order layers based
    on their kind. For example, a model with layers ``conv->norm->act->dropout``
    would need to know the kind of the different types of the layers to organize
    them appropriately.

    TODO: Add support for pooling layers.

    Args:
        layer_type (Union[type, str]): Layer type.

    Returns:
        str: Layer kind.
    """
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


_LayerInfoInitKwargsDict = Dict[str, Any]
_LayerInfoInitKwargsFlatSequence = Union[List[Any], Tuple[Any]]
_NestedInnerArgs = Tuple[str, Any]
_LayerInfoInitKwargsNestedSequence = Union[List[_NestedInnerArgs], Tuple[_NestedInnerArgs, ...]]
_LayerInfoInitKwargsType = Union[
    _LayerInfoInitKwargsDict, _LayerInfoInitKwargsFlatSequence, _LayerInfoInitKwargsNestedSequence
]

# LayerInfo type schema with raw Python types (str, int, float, tuple, list, etc.).
LayerInfoRawType = Union[
    str, Dict[str, _LayerInfoInitKwargsType], Tuple[str, _LayerInfoInitKwargsType]
]


@dataclass
class LayerInfo:
    """Dataclass for managing layer information."""

    name: str  # name of the layer
    dimension: Optional[int] = None  # The dimension of the layer.
    init_kwargs: Dict[str, Any] = field(default_factory=dict)  # keyword args to initialize layer

    @property
    def kind(self) -> str:
        """The layer kind."""
        return get_layer_kind(self.ltype)

    @property
    def ltype(self) -> type:
        """The layer type."""
        return get_layer_type(self.name, dimension=self.dimension)

    @classmethod
    def format(cls, layer_info: LayerInfoRawType) -> "LayerInfo":
        """Formats layer information from Python raw types to LayerInfo object.

        Args:
            layer_info:

        Returns:
            LayerInfo
        """
        if isinstance(layer_info, str):
            return LayerInfo(name=layer_info, init_kwargs={})

        if isinstance(layer_info, Dict):
            if len(layer_info) != 1:
                raise ValueError(
                    "Dictionary format for LayerInfo can only have one key-value pair - "
                    "e.g. {'dropout': {'p': 0.5}}. "
                    f"Got {layer_info}"
                )
            name, init_kwargs = list(layer_info.items())[0]
        elif isinstance(layer_info, (Tuple, List)):
            if len(layer_info) != 2:
                raise ValueError(
                    "Sequence format for LayerInfo should be formatted as (name, init_kwargs) - "
                    "e.g. ('dropout': {'p': 0.5}), ('dropout', ('p', 0.5). "
                    f"Got {layer_info}"
                )
            name, init_kwargs = layer_info[0], layer_info[1]
        else:
            raise ValueError(f"Unsupported layer info format: {type(layer_info)}")

        # Format init kwargs.
        init_kwargs_err_message = (
            "Unknown init_kwargs format. init_kwargs must follow one of these formats: "
            "\n\t- dict: {key1: value1, key2: value2}"
            "\n\t- flat sequence: [key1, value1, key2, value2, ...]"
            "\n\t- nested sequence: [(key1, value1), (key2, value2), ...]\n"
            f"Got {init_kwargs}"
        )
        if isinstance(init_kwargs, (Tuple, List)):
            if all(isinstance(x, (Tuple, List)) and len(x) == 2 for x in init_kwargs):
                # Nested sequence - i.e. [(key1, value1), (key2, value2), ...]
                init_kwargs = {x[0]: x[1] for x in init_kwargs}
            else:
                if len(init_kwargs) % 2 != 0:
                    raise ValueError(init_kwargs_err_message)
                # Flat sequence - i.e. [key1, value1, key2, value2, ...]
                init_kwargs = dict(zip(init_kwargs[::2], init_kwargs[1::2]))
        if not isinstance(init_kwargs, Dict):
            raise ValueError(init_kwargs_err_message)

        return LayerInfo(name=name, init_kwargs=init_kwargs)

    def build(self, *args, **kwargs) -> nn.Module:
        """Builds the layer.

        Args:
            *args: Positional arguments for building the module.
            **kwargs: Keyword arguments to pass to the layer's ``__init__``.
                Note, these will override the init_kwargs of the layer if there
                are conflicts.
        """
        return self.ltype(*args, **{**self.init_kwargs, **kwargs})


def build_layer_info_from_seq(
    layers_info: Sequence[Union[LayerInfoRawType, LayerInfo]], dimension: Optional[int] = None
) -> List[LayerInfo]:
    """Builds a sequence of layer info objects from a sequence of layer info types.

    Args:
        layers_info: Sequence of layer info types.

    Returns:
        List[LayerInfo]: list of LayerInfo objects corresponding to each element
            in layers_info.
    """
    out = [
        copy.deepcopy(x) if isinstance(x, LayerInfo) else LayerInfo.format(x) for x in layers_info
    ]
    if dimension is not None:
        # TODO (arjundd): We probably want some extra logic here where we don't
        # the dimension if it already exists in the name (e.g. conv1d).
        # Currently, if the name contains the dimension and a dimension is passed
        # (e.g. conv1d, 2), then the dimension value is ignored.
        for x in out:
            x.dimension = dimension

    return out
