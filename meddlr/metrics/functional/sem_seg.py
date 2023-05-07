import itertools
import logging

import numpy as np
import torch

try:
    from scipy.ndimage.morphology import binary_erosion, distance_transform_edt
except ImportError:  # pragma: no cover
    binary_erosion = None
    distance_transform_edt = None

try:
    from medpy.metric import assd as _assd
except ImportError:  # pragma: no cover
    _assd = None


from meddlr.metrics.functional import util as mFutil

__all__ = [
    "dice_score",
    "volumetric_overlap_error",
    "coefficient_variation",
    "average_symmetric_surface_distance",
    "dice",
    "cv",
    "voe",
    "assd",
]


logger = logging.getLogger(__name__)


def dice_score(y_pred, y_true):
    """Computes dice score coefficient.

    Args:
        y_pred (torch.Tensor): Predicted categorical segmentation.
            Expected shape: ``(B, C, H, ...)``.
        y_true (torch.Tensor): Ground truth categorical segmentation.
            Expected shape: ``(B, C, H, ...)``.

    Returns:
        torch.Tensor: Dice score. Shape: ``(B, C)``.
    """
    is_tensor = isinstance(y_pred, torch.Tensor)

    y_pred = mFutil.to_bool(y_pred)
    y_true = mFutil.to_bool(y_true)
    y_pred, y_true = mFutil.flatten_other_dims((y_pred, y_true), dim=(0, 1))

    count_nonzero = torch.count_nonzero if is_tensor else np.count_nonzero
    size_i1 = count_nonzero(y_pred, -1)
    size_i2 = count_nonzero(y_true, -1)
    intersection = count_nonzero(y_pred & y_true, -1)

    return 2.0 * intersection / (size_i1 + size_i2)


def volumetric_overlap_error(y_pred, y_true):
    """Computes volumetric overlap error.

    Args:
        y_pred (torch.Tensor): Predicted categorical segmentation.
            Expected shape: ``(B, C, H, ...)``.
        y_true (torch.Tensor): Ground truth categorical segmentation.
            Expected shape: ``(B, C, H, ...)``.

    Returns:
        torch.Tensor: Dice score. Shape: ``(B, C)``.
    """
    is_tensor = isinstance(y_pred, torch.Tensor)

    y_pred = mFutil.to_bool(y_pred)
    y_true = mFutil.to_bool(y_true)
    y_pred, y_true = mFutil.flatten_other_dims((y_pred, y_true), dim=(0, 1))

    count_nonzero = torch.count_nonzero if is_tensor else np.count_nonzero
    intersection = count_nonzero(y_true & y_pred, -1)
    union = count_nonzero(y_true | y_pred, -1)
    union = union.type(torch.float) if is_tensor else union.astype("float")

    return 1 - intersection / union


def coefficient_variation(y_pred, y_true):
    """Computes coefficient of variation.

    Args:
        y_pred (torch.Tensor): Predicted categorical segmentation.
            Expected shape: ``(B, C, H, ...)``.
        y_true (torch.Tensor): Ground truth categorical segmentation.
            Expected shape: ``(B, C, H, ...)``.

    Returns:
        torch.Tensor: Dice score. Shape: ``(B, C)``.
    """
    is_tensor = isinstance(y_pred, torch.Tensor)

    y_pred = mFutil.to_bool(y_pred)
    y_true = mFutil.to_bool(y_true)
    y_pred, y_true = mFutil.flatten_other_dims((y_pred, y_true), dim=(0, 1))

    count_nonzero = torch.count_nonzero if is_tensor else np.count_nonzero
    size_i1 = count_nonzero(y_pred, -1)
    size_i2 = count_nonzero(y_true, -1)

    if is_tensor:
        cat_tensor = torch.stack([size_i1, size_i2], dim=0).type(torch.float32)
        std = torch.std(cat_tensor, dim=0)
        mean = torch.mean(cat_tensor, dim=0)
    else:
        std = np.std([size_i1, size_i2], axis=0)
        mean = np.mean([size_i1, size_i2], axis=0)

    return std / mean


def average_symmetric_surface_distance(y_pred, y_true, spacing=None, connectivity=1, crop=True):
    """Computes average symmetric surface distance.

    Args:
        y_pred (torch.Tensor): Predicted categorical segmentation.
            Expected shape: ``(B, C, H, ...)``.
        y_true (torch.Tensor): Ground truth categorical segmentation.
            Expected shape: ``(B, C, H, ...)``.

    Returns:
        torch.Tensor: Dice score. Shape: ``(B, C)``.
    """
    if _assd is None:
        raise ModuleNotFoundError(
            "assd requires the medpy package. Please install using `pip install medpy`."
        )

    # TODO: check why this statement is needed.
    if not connectivity:
        connectivity = 1

    is_tensor = isinstance(y_pred, torch.Tensor)
    if is_tensor:
        if y_pred.requires_grad or y_true.requires_grad:
            logger.warning("Average symmetric surface distance does not preserve gradients.")
        y_pred = y_pred.detach().cpu().numpy()
        y_true = y_true.detach().cpu().numpy()

    label = 1
    if y_pred.dtype not in (np.bool, bool):
        y_pred = y_pred == label
    if y_true.dtype not in (np.bool, bool):
        y_true = y_true == label

    B, C = y_pred.shape[:2]
    out_matrix = torch.zeros(B, C) if is_tensor else np.zeros((B, C))
    for b in range(B):
        for c in range(C):
            bc_pred, bc_true = y_pred[b, c], y_true[b, c]
            if crop:
                bc_pred, bc_true = _crop_to_joint_roi(bc_pred, bc_true)

            out_matrix[b, c] = _assd(
                bc_pred, bc_true, voxelspacing=spacing, connectivity=connectivity
            )
    return out_matrix


def _crop_to_joint_roi(y_pred: np.ndarray, y_true: np.ndarray):
    joint_roi = np.asarray(y_pred | y_true)
    ndim = joint_roi.ndim
    if not np.any(joint_roi):
        return y_pred, y_true

    bbox = []
    for ax in itertools.combinations(reversed(range(ndim)), ndim - 1):
        arr = np.any(joint_roi, ax)
        arg_max = np.where(arr == arr.max())[0]
        min_d = max(arg_max[0], 0)
        max_d = arg_max[-1] + 1
        bbox.append(slice(min_d, max_d))
    bbox = tuple(bbox)

    return y_pred[bbox], y_true[bbox]


dice = dice_score
cv = coefficient_variation
voe = volumetric_overlap_error
assd = average_symmetric_surface_distance
