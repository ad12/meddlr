import logging
from enum import Enum
from typing import Callable, Sequence, Union

import numpy as np
import torch

logger = logging.getLogger(__name__)

arr_type = Union[np.ndarray, torch.Tensor]


def to_bool(x):
    """Converts a tensor or ndarray into the bool dtype.

    This function first checks if the tensory is already of
    the bool dtype prior to doing conversion.
    NumPy (and potentially torch) has some issues with doing
    this check first for shared-memory tensors.

    Args:
        x (array-like): An array-like or torch Tensor to convert to bool dtype.

    Returns:
        np.ndarray | torch.Tensor: The array with dtype ``bool``.
    """
    if isinstance(x, torch.Tensor):
        x = x.type(torch.bool) if x.dtype != torch.bool else x
    else:
        if not isinstance(x, np.ndarray):
            x = np.asarray(x, dtype=np.bool)
        x = x.astype(np.bool) if x.dtype != np.bool else x
    return x


def flatten_other_dims(
    xs: Union[arr_type, Sequence[arr_type]], dim: Union[int, Sequence[int]] = None
):
    """Flattens all dimensions other than the specified dimension(s).

    Args:
        xs (torch.Tensors | ndarrays): A single or sequence of tensor(s)/array(s) to process.
        dim (int | tuple[int]): The dimensions to not flatten. All other dimensions will be
            flattened for each array in ``xs``.

    Returns:
        torch.Tensor | ndarray | Sequence[torch.Tensor | ndarray]: Flattened array(s).
    """
    single_input = isinstance(xs, (np.ndarray, torch.Tensor))
    is_tensor = isinstance(xs, torch.Tensor) if single_input else isinstance(xs[0], torch.Tensor)

    if single_input:
        xs = [xs]
    xs_type = type(xs)
    dim = (dim,) if isinstance(dim, int) else dim
    if dim is not None:
        dim = tuple(dim)

    if dim:
        shape = tuple(xs[0].shape[d] for d in dim) + (-1,)
        dims_ordered = dim + tuple(i for i in range(xs[0].ndim) if i not in dim)
        if is_tensor:
            xs = [x.permute(dims_ordered) for x in xs]
        else:
            xs = [x.transpose(dims_ordered) for x in xs]
        xs = (x.reshape(shape) for x in xs)
    else:
        xs = (x.flatten() for x in xs)

    xs = xs_type(xs)
    if single_input:
        xs = xs[0]

    return xs


def flatten_non_category_dims(xs: Union[arr_type, Sequence[arr_type]], category_dim: int = None):
    """Flattens all non-category dimensions into a single dimension.

    Args:
        xs (ndarrays): Sequence of ndarrays with the same category dimension.
        category_dim: The dimension/axis corresponding to different categories.
            i.e. `C`. If `None`, behaves like `np.flatten(x)`.

    Returns:
        ndarray: Shape (C, -1) if `category_dim` specified else shape (-1,)
    """
    return flatten_other_dims(xs, dim=category_dim)


def rms_cv(y_pred: arr_type, y_true: arr_type, dim=None):
    """Compute root-mean-squared coefficient of variation.

    This is typically done to compare intra-method variability.
    For example if multiple measurements are taken using the same method.
    However, in many segmentation manuscripts, this is equation is also
    used.

    This quantity is symmetric.

    Args:
        y_pred (ndarray): Measurements from trial 1.
        y_true (ndarray): Measurements from trial 2.
        dim (int, optional): Dimension/axis over which to compute metric.
            If `None`, all dimensions will be reduced.

    Returns:
        ndarray: If `dim=None`, scalar value.
    """
    is_tensor = isinstance(y_pred, torch.Tensor)
    if is_tensor:
        cat_tensor = torch.stack([y_pred, y_true], dim=0).type(torch.float32)
        stds = torch.std(cat_tensor, dim=0)
        means = torch.mean(cat_tensor, dim=0)
        cv = stds / means
        return torch.sqrt(torch.mean(cv**2, dim=dim))
    else:
        stds = np.std([y_pred, y_true], axis=0)
        means = np.mean([y_pred, y_true], axis=0)
        cv = stds / means
        return np.sqrt(np.mean(cv**2, axis=dim))


def rmse_cv(y_pred: np.ndarray, y_true: np.ndarray, dim=None):
    """Compute root-mean-squared error coefficient of variation.

    This quantity is not symmetric.

    Args:
        y_pred (ndarray): Predicted measurements.
        y_true (ndarray): Ground-truth/baseline measurements.
        dim (int, optional): Dimension/axis over which to compute metric.
            If `None`, all dimensions will be reduced.

    Returns:
        ndarray: If `dim=None`, scalar value.
    """
    is_tensor = isinstance(y_pred, torch.Tensor)
    if is_tensor:
        rmse = torch.sqrt(torch.mean((y_pred - y_true) ** 2, dim=dim))
        means = torch.abs(torch.mean(y_true, dim=dim))
    else:
        rmse = np.sqrt(np.mean((y_pred - y_true) ** 2, axis=dim))
        means = np.absolute(np.mean(y_true, axis=dim))
    return rmse / means


class Reductions(Enum):
    RMS_CV = 1, "RMS-CV", rms_cv
    RMSE_CV = 2, "RMSE-CV", rmse_cv

    def __new__(cls, value: int, display_name: str, func: Callable):
        """
        Args:
            value (int): Unique integer value.
            patterns (`List[str]`): List of regex patterns that would match the
                hostname on the compute cluster. There can be multiple hostnames
                per compute cluster because of the different nodes.
            save_dir (str): Directory to save data to.
        """
        obj = object.__new__(cls)
        obj._value_ = value

        obj.display_name = display_name
        obj.func = func

        return obj
