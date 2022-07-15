import inspect
from typing import Any, Callable, Iterable, List, Mapping, Optional, Tuple, TypeVar

import numpy as np
import torch

from meddlr.transforms.mixins import DeviceMixin

__all__ = ["Transform", "TransformList"]


class Transform(DeviceMixin):
    """
    Base class for implementations of __deterministic__ transfomations for
    _medical_ image and other data structures.

    Like the `fvcore.transforms` module, there should be a higher-level policy
    that generates (likely with random variations) these transform ops.

    By default, all transforms should handle image data types.
    Other data types like segmentations, coordinates, bounding boxes, and polygons
    may not be supported by default. However, these methods can be overloaded if
    generalized methods are written for these data types.

    Additional domain specific data types may also be supported
    (e.g. ``kspace``, ``maps``).

    Medical images are seldom in the uint8 format and are not always
    normalized between [0, 1] in the floating point format. Transforms should
    not expect that data is normalized in this format.

    Note, each method may choose to modify the input data in-place for
    efficiency.

    This structure is adapted from the fvcore library.
    """

    def _set_attributes(self, params: Optional[Mapping[str, Any]] = None) -> None:
        """Set attributes from the input list of parameters.

        Args:
            params (list): list of parameters.
        """

        if params:
            for k, v in params.items():
                if k != "self" and not k.startswith("_"):
                    setattr(self, k, v)

    @classmethod
    def register_type(cls, data_type: str, func: Optional[Callable] = None):
        """
        Register the given function as a handler that this transform will use
        for a specific data type.

        Args:
            data_type (str): the name of the data type (e.g., box)
            func (callable): takes a transform and a data, returns the
                transformed data.

        Examples:

        .. code-block:: python

            # call it directly
            def func(flip_transform, voxel_data):
                return transformed_voxel_data
            HFlipTransform.register_type("voxel", func)

            # or, use it as a decorator
            @HFlipTransform.register_type("voxel")
            def func(flip_transform, voxel_data):
                return transformed_voxel_data

            # ...
            transform = HFlipTransform(...)
            transform.apply_voxel(voxel_data)  # func will be called
        """
        if func is None:  # the decorator style

            def wrapper(decorated_func):
                assert decorated_func is not None
                cls.register_type(data_type, decorated_func)
                return decorated_func

            return wrapper

        assert callable(
            func
        ), "You can only register a callable to a Transform. Got {} instead.".format(func)
        argspec = inspect.getfullargspec(func)
        assert len(argspec.args) == 2, (
            "You can only register a function that takes two positional "
            "arguments to a Transform! Got a function with spec {}".format(str(argspec))
        )
        setattr(cls, "apply_" + data_type, func)

    def __call__(self, *args, data_type: str, **kwargs):
        """Alias for calling the apply method on the appropriate data type.

        Args:
            data_type (str): the name of the data type (e.g., image, kspace, maps, etc.).

        Returns:
            The output of the corresponding apply method.

        Examples:
            .. code-block:: python

                # These two are equivalent
                out1 = self.apply_image(img)
                out2 = self(img, data_type="image")
        """
        return getattr(self, f"apply_{data_type}")(*args, **kwargs)

    def apply_image(self, img: torch.Tensor):
        """
        Apply the transform on an image.

        Args:
            img (ndarray): of shape NxHxWxC, or HxWxC or HxW. The array can be
                of type uint8 in range [0, 255], or floating point in range
                [0, 1] or [0, 255].
        Returns:
            ndarray: image after apply the transformation.
        """
        raise NotImplementedError

    def apply_maps(self, maps: torch.Tensor):
        return self.apply_image(maps)

    def inverse(self) -> "Transform":
        """
        Create a transform that inverts the geometric changes (i.e. change of
        coordinates) of this transform.

        Note that the inverse is meant for geometric changes only.
        The inverse of photometric transforms that do not change coordinates
        is defined to be a no-op, even if they may be invertible.

        Returns:
            Transform:
        """
        raise NotImplementedError

    def __repr__(self):
        attrs = {
            k: v
            for k, v in self.__dict__.items()
            if not k.startswith("_") and not isinstance(v, Callable)
        }
        return "{}({})".format(type(self).__name__, ", ".join(f"{k}={v}" for k, v in attrs.items()))

    def __str__(self):
        return self.__repr__()

    def _eq_attrs(self) -> Tuple[str]:
        raise NotImplementedError

    def __eq__(self, o: object) -> bool:
        if not isinstance(o, type(self)):
            return False
        attrs = self._eq_attrs()
        vals = [getattr(self, a) == getattr(o, a) for a in attrs]
        vals = [
            torch.all(v)
            if isinstance(v, torch.Tensor)
            else np.all(v)
            if isinstance(v, (np.ndarray, list, tuple))
            else all(v)
            if isinstance(v, Iterable)
            else v
            for v in vals
        ]
        return all(vals)


class NoOpTransform(Transform):
    def apply_image(self, image):
        return image

    def inverse(self):
        return NoOpTransform()


_T = TypeVar("_T")


# pyre-ignore-all-errors
class TransformList:
    """
    Maintain a list of transform operations which will be applied in sequence.
    Attributes:
        transforms (list[Transform])
    """

    def __init__(self, transforms: List[Transform], ignore_no_op: bool = True):
        """
        Args:
            transforms (list[Transform]): list of transforms to perform.
        """
        super().__init__()
        for t in transforms:
            assert isinstance(t, Transform), t

        self.ignore_no_op = ignore_no_op
        if ignore_no_op:
            transforms = [t for t in transforms if t not in (None, NoOpTransform)]
        self.transforms = transforms

    def _apply(self, x: _T, meth: str) -> _T:
        """
        Apply the transforms on the input.
        Args:
            x: input to apply the transform operations.
            meth (str): meth.
        Returns:
            x: after apply the transformation.
        """
        for t in self.transforms:
            x = getattr(t, meth)(x)
        return x

    def __getattribute__(self, name: str):
        # use __getattribute__ to win priority over any registered dtypes
        if name.startswith("apply_"):
            return lambda x: self._apply(x, name)
        return super().__getattribute__(name)

    def __add__(self, other: "TransformList") -> "TransformList":
        """
        Args:
            other (TransformList): transformation to add.
        Returns:
            TransformList: list of transforms.
        """
        ignore_no_op = self.ignore_no_op or other.ignore_no_op
        if not other:
            return TransformList(self.transforms, ignore_no_op=ignore_no_op)
        others = other.transforms if isinstance(other, TransformList) else [other]
        return TransformList(self.transforms + others, ignore_no_op=ignore_no_op)

    def __iadd__(self, other: "TransformList") -> "TransformList":
        """
        Args:
            other (TransformList): transformation to add.
        Returns:
            TransformList: list of transforms.
        """
        others = other.transforms if isinstance(other, TransformList) else [other]
        if self.ignore_no_op:
            others = [t for t in others if t not in (None, NoOpTransform)]
        self.transforms.extend(others)
        return self

    def __radd__(self, other: "TransformList") -> "TransformList":
        """
        Args:
            other (TransformList): transformation to add.
        Returns:
            TransformList: list of transforms.
        """
        ignore_no_op = self.ignore_no_op or other.ignore_no_op
        others = other.transforms if isinstance(other, TransformList) else [other]
        return TransformList(others + self.transforms, ignore_no_op=ignore_no_op)

    def __len__(self) -> int:
        """
        Returns:
            Number of transforms contained in the TransformList.
        """
        return len(self.transforms)

    def __getitem__(self, idx) -> Transform:
        return self.transforms[idx]

    def __contains__(self, x):
        return x in self.transforms

    def inverse(self) -> "TransformList":
        """
        Invert each transform in reversed order.
        """
        return TransformList([x.inverse() for x in self.transforms[::-1]])

    def __repr__(self) -> str:
        return "{}(\n\t{}\n)".format(
            type(self).__name__, "\n\t".join(t.__repr__() for t in self.transforms)
        )

    def __str__(self) -> str:
        return self.__repr__()
