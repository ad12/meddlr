from typing import Sequence

import torch
import torch.nn.functional as F

import meddlr.ops.complex as cplx

__all__ = [
    "roll",
    "pad",
    "zero_pad",
    "time_average",
    "sliding_window",
    "center_crop",
    "normalize",
    "normalize_instance",
]


def roll(x, shift, dim):
    """
    Similar to np.roll but applies to PyTorch Tensors
    """
    if isinstance(shift, (tuple, list)):
        assert len(shift) == len(dim)
        for s, d in zip(shift, dim):
            x = roll(x, s, d)
        return x
    shift = shift % x.size(dim)
    if shift == 0:
        return x
    left = x.narrow(dim, 0, x.size(dim) - shift)
    right = x.narrow(dim, x.size(dim) - shift, shift)
    return torch.cat((right, left), dim=dim)


def pad(x: torch.Tensor, shape: Sequence[int], mode="constant", value=0):
    """
    Args:
        x: Input tensor of shape (B, ...)
        shape: Shape to zero pad to. Use `None` to skip padding certain dimensions.
    Returns:
    """
    x_shape = x.shape[1 : 1 + len(shape)]
    assert all(
        x_shape[i] <= shape[i] or shape[i] is None for i in range(len(shape))
    ), f"Tensor spatial dimensions {x_shape} smaller than zero pad dimensions"

    total_padding = tuple(
        desired - current if desired is not None else 0 for current, desired in zip(x_shape, shape)
    )
    # Adding no padding for terminal dimensions.
    # torch.nn.functional.pad pads dimensions in reverse order.
    total_padding += (0,) * (len(x.shape) - 1 - len(x_shape))
    total_padding = total_padding[::-1]

    pad = []
    for padding in total_padding:
        pad1 = padding // 2
        pad2 = padding - pad1
        pad.extend([pad1, pad2])

    return F.pad(x, pad, mode=mode, value=value)


def zero_pad(x: torch.Tensor, shape: Sequence[int]):
    return pad(x, shape, mode="constant", value=0)


def time_average(data, dim, eps=1e-6, keepdim=True):
    """
    Computes time average across a specified axis.
    """
    mask = cplx.get_mask(data)
    return data.sum(dim, keepdim=keepdim) / (mask.sum(dim, keepdim=keepdim) + eps)


def sliding_window(data, dim, window_size):
    """
    Computes sliding window with circular boundary conditions across a specified
    axis.
    """
    assert 0 < window_size <= data.shape[dim]

    windows = [None] * data.shape[dim]
    for i in range(data.shape[dim]):
        data_slide = roll(data, int(window_size / 2) - i, dim)
        window = data_slide.narrow(dim, 0, window_size)
        windows[i] = time_average(window, dim)

    return torch.cat(windows, dim=dim)


def center_crop(data, shape):
    """
    Apply a center crop to a batch of images.
    Args:
        data (torch.Tensor): The input tensor to be center cropped.
        shape (list of ints): The output shape. If shape[dim] = -1, then no crop
            will be applied in that dimension.
    """
    for i in range(len(shape)):
        if (shape[i] == data.shape[i]) or (shape[i] == -1):
            continue
        assert 0 < shape[i] <= data.shape[i]
        idx_start = (data.shape[i] - shape[i]) // 2
        data = data.narrow(i, idx_start, shape[i])

    return data


def normalize(data, mean, stddev, eps=0.0):
    """
    Normalize the given tensor using:
        (data - mean) / (stddev + eps)
    Args:
        data (torch.Tensor): Input data to be normalized
        mean (float): Mean value
        stddev (float): Standard deviation
        eps (float): Added to stddev to prevent dividing by zero
    Returns:
        torch.Tensor: Normalized tensor
    """
    return (data - mean) / (stddev + eps)


def normalize_instance(data, eps=0.0):
    """
    Normalize the given tensor using:
        (data - mean) / (stddev + eps)
    where mean and stddev are computed from the data itself.
    Args:
        data (torch.Tensor): Input data to be normalized
        eps (float): Added to stddev to prevent dividing by zero
    Returns:
        torch.Tensor: Normalized tensor
    """
    mean = data.mean()
    std = data.std()
    return normalize(data, mean, std, eps), mean, std
