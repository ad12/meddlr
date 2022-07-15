"""Image metrics.

The key difference between these implementations and those in torchmetrics
is that these metrics support operations on complex data types.
"""

from typing import Optional, Sequence, Union

import numpy as np
import torch
import torch.nn.functional as F
from packaging.version import Version

from meddlr.ops import complex as cplx
from meddlr.utils import env

if Version(env.get_package_version("torchmetrics")) >= Version("0.8.0"):
    from torchmetrics.functional.image.helper import _gaussian
else:
    from torchmetrics.functional.image.ssim import _gaussian

__all__ = ["mae", "mse", "rmse", "psnr", "nrmse", "l2_norm", "ssim"]


# Mapping from str to complex function name.
_IM_TYPES_TO_FUNCS = {
    "mag": cplx.abs,
    "magnitude": cplx.abs,
    "abs": cplx.abs,
    "phase": cplx.angle,
    "angle": cplx.angle,
    "real": cplx.real,
    "imag": cplx.imag,
}


def _check_consistent_type(*args):
    is_complex = [cplx.is_complex(x) or cplx.is_complex_as_real(x) for x in args]
    all_complex = all(is_complex)
    all_not_complex = all(not x for x in is_complex)

    if not all_complex and not all_not_complex:
        raise ValueError("Type mismatch - all inputs must be complex or real")


def mae(pred: torch.Tensor, target: torch.Tensor, im_type: str = None) -> torch.Tensor:
    """Computes mean absolute error.

    Args:
        pred (torch.Tensor): The prediction. Either a complex or real tensor.
        target (torch.Tensor): The target. Either a complex or real tensor.
        im_type (str, optional): The image type to compute metric on.
            This only applies for complex inputs, otherwise ignored.
            Either ``'magnitude'`` (default) to compute metric on magnitude images
            or ``'phase'`` to compute metric on phase/angle images. If ``None``,
            computed on complex images.

    Returns:
        torch.Tensor: The mean square error.
    """
    return _mean_error(pred, target, im_type=im_type, order=1)


def mse(pred: torch.Tensor, target: torch.Tensor, im_type: str = None) -> torch.Tensor:
    """Computes mean square error.

    Args:
        pred (torch.Tensor): The prediction. Either a complex or real tensor.
        target (torch.Tensor): The target. Either a complex or real tensor.
        im_type (str, optional): The image type to compute metric on.
            This only applies for complex inputs, otherwise ignored.
            Either ``'magnitude'`` (default) to compute metric on magnitude images
            or ``'phase'`` to compute metric on phase/angle images. If ``None``,
            computed on complex images.

    Returns:
        torch.Tensor: The mean square error.
    """
    return _mean_error(pred, target, im_type=im_type, order=2)


def _mean_error(
    pred: torch.Tensor, target: torch.Tensor, im_type: str = None, order: int = 2
) -> torch.Tensor:
    """Computes mean error of order ``order``.

    Args:
        pred (torch.Tensor): The prediction. Either a complex or real tensor.
        target (torch.Tensor): The target. Either a complex or real tensor.
        im_type (str, optional): The image type to compute metric on.
        order (int, optional): The order of the error to compute.
            For example, ``order=1`` is mean absolute error,
            ``order=2`` is mean squared error, etc.

    Returns:
        torch.Tensor: The mean square error.
    """
    if im_type is not None:
        pred = _IM_TYPES_TO_FUNCS[im_type](pred)
        target = _IM_TYPES_TO_FUNCS[im_type](target)

    if cplx.is_complex(pred) or cplx.is_complex_as_real(pred):
        err = cplx.abs(pred - target)
    else:
        err = torch.abs(pred - target)
    if order != 1:
        err = err**order
    shape = (pred.shape[0], pred.shape[1], -1)
    return torch.mean(err.view(shape), dim=-1)


def rmse(pred: torch.Tensor, target: torch.Tensor, im_type: str = None) -> torch.Tensor:
    """Computes root mean square error.

    Args:
        pred (torch.Tensor): The prediction. Either a complex or real tensor.
        target (torch.Tensor): The target. Either a complex or real tensor.
        im_type (str, optional): The image type to compute metric on.
            This only applies for complex inputs, otherwise ignored.
            Either ``'magnitude'`` (default) to compute metric on magnitude images
            or ``'phase'`` to compute metric on phase/angle images. If ``None``,
            computed on complex images.

    Returns:
        torch.Tensor: The root mean square error.
    """
    return torch.sqrt(mse(pred, target, im_type=im_type))


def psnr(pred: torch.Tensor, target: torch.Tensor, im_type: str = None) -> torch.Tensor:
    """Computes peak signal-to-noise ratio.

    Args:
        pred (torch.Tensor): The prediction. Either a complex or real tensor.
        target (torch.Tensor): The target. Either a complex or real tensor.
        im_type (str, optional): The image type to compute metric on.
            This only applies for complex inputs, otherwise ignored.
            Either ``'magnitude'`` (default) to compute metric on magnitude images
            or ``'phase'`` to compute metric on phase/angle images. If ``None``,
            computed on complex images.

    Returns:
        torch.Tensor: The peak signal-to-noise ratio.
    """
    is_complex = cplx.is_complex(pred) or cplx.is_complex_as_real(pred)
    abs_func = cplx.abs if is_complex else torch.abs

    l2_val = rmse(pred, target, im_type=im_type)
    shape = (target.shape[0], target.shape[1], -1)
    max_val = torch.amax(abs_func(target).view(shape), dim=-1)
    return 20 * torch.log10(max_val / l2_val)


def nrmse(pred: torch.Tensor, target: torch.Tensor, im_type: str = None) -> torch.Tensor:
    """Computes normalized root mean squared error.

    Normalization is done with respect to :math:`\sqrt{\\frac{\sum^N target[i]^2}{N}}`.

    Args:
        pred (torch.Tensor): The prediction. Either a complex or real tensor.
        target (torch.Tensor): The target. Either a complex or real tensor.
        im_type (str, optional): The image type to compute metric on.
            This only applies for complex inputs, otherwise ignored.
            Either ``'magnitude'`` (default) to compute metric on magnitude images
            or ``'phase'`` to compute metric on phase/angle images. If ``None``,
            computed on complex images.

    Returns:
        torch.Tensor: The normalized root mean squared error.
    """
    is_complex = cplx.is_complex(pred) or cplx.is_complex_as_real(pred)
    abs_func = cplx.abs if is_complex else torch.abs

    rmse_val = rmse(pred, target, im_type=im_type)
    shape = (pred.shape[0], pred.shape[1], -1)
    norm = torch.sqrt(torch.mean((abs_func(target) ** 2).view(shape), dim=-1))
    return rmse_val / norm


def l2_norm(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Computes l2-norm of the error.

    Normalization is done with respect to :math:`\sqrt{\\frac{\sum^N target[i]^2}{N}}`.

    Args:
        pred (torch.Tensor): The prediction. Either a complex or real tensor.
        target (torch.Tensor): The target. Either a complex or real tensor.
        im_type (str, optional): The image type to compute metric on.
            This only applies for complex inputs, otherwise ignored.
            Either ``'magnitude'`` (default) to compute metric on magnitude images
            or ``'phase'`` to compute metric on phase/angle images. If ``None``,
            computed on complex images.

    Returns:
        torch.Tensor: The normalized root mean squared error.
    """
    err = pred - target
    is_complex = cplx.is_complex(err) or cplx.is_complex_as_real(err)
    abs_func = cplx.abs if is_complex else torch.abs
    shape = (pred.shape[0], pred.shape[1], -1)
    return torch.sum(abs_func(err).view(shape), dim=-1)


def ssim(
    pred: torch.Tensor,
    target: torch.Tensor,
    method: str = None,
    kernel_size=11,
    sigma=1.5,
    data_range=None,
    k1=0.01,
    k2=0.03,
    pad_mode: str = "reflect",
    im_type: str = "magnitude",
) -> torch.Tensor:
    """Computes structural similarity index (SSIM).

    Args:
        pred (torch.Tensor): The prediction. Either a complex or real tensor.
        target (torch.Tensor): The target. Either a complex or real tensor.
        im_type (str, optional): The image type to compute metric on.
            This only applies for complex inputs, otherwise ignored.
            Either ``'magnitude'`` (default) to compute metric on magnitude images
            or ``'phase'`` to compute metric on phase/angle images. If ``None``,
            computed on complex images.

    Returns:
        torch.Tensor: The SSIM for each (batch, channel) pair.
    """
    if method is not None:
        if method.lower() == "wang":
            kernel_size = 11
            sigma = 1.5
            data_range = "ref-maxval"
            k1 = 0.01
            k2 = 0.03
        else:
            raise ValueError(f"Unknown method {method}")

    if im_type is not None:
        if cplx.is_complex(pred) or cplx.is_complex_as_real(pred):
            pred = _IM_TYPES_TO_FUNCS[im_type](pred)
        if cplx.is_complex(target) or cplx.is_complex_as_real(target):
            target = _IM_TYPES_TO_FUNCS[im_type](target)

    ssim_idx = _ssim_compute(
        pred,
        target,
        kernel_size=kernel_size,
        sigma=sigma,
        data_range=data_range,
        k1=k1,
        k2=k2,
        pad_mode=pad_mode,
    )

    reduce_dims = tuple(range(2, pred.ndim))
    return ssim_idx.mean(reduce_dims)


def _ssim_compute(
    pred: torch.Tensor,
    target: torch.Tensor,
    kernel_size: Union[int, Sequence[int]] = 11,
    sigma: Sequence[float] = 1.5,
    data_range: Optional[Union[float, torch.Tensor]] = None,
    k1: float = 0.01,
    k2: float = 0.03,
    pad_mode="reflect",
) -> torch.Tensor:
    """Compute structural similarity.

    Args:
        pred (torch.Tensor): The prediction. Shape: ``BxCxHxW`` or ``BxCxDxHxW``.
        target (torch.Tensor): The target. Shape: ``BxCxHxW`` or ``BxCxDxHxW``.
        kernel_size (int | Sequence[int]): The kernel size. If this is a scalar,
            the same size will be used for all spatial dimensions. If this is
            sequence, it should follow the same spatial ordering as ``pred`` and
            ``target``.
    """
    if isinstance(kernel_size, int):
        kernel_size = (kernel_size,) * (pred.ndim - 2)
    if isinstance(sigma, (int, float)):
        sigma = (sigma,) * (pred.ndim - 2)
    if len(kernel_size) != pred.ndim - 2:
        raise ValueError(
            f"Expected `kernel_size` to be an integer or sequence of length equal to the "
            f"number of spatial dimensions. Got {kernel_size}."
        )
    if len(sigma) != pred.ndim - 2:
        raise ValueError(
            f"Expected `sigma` to be an integer or sequence of length equal to the "
            f"number of spatial dimensions. Got {sigma}."
        )

    if any(x % 2 == 0 or x <= 0 for x in kernel_size):
        raise ValueError(f"Expected `kernel_size` to have odd positive number. Got {kernel_size}.")

    if any(y <= 0 for y in sigma):
        raise ValueError(f"Expected `sigma` to have positive number. Got {sigma}.")

    reduce_dims = tuple(range(2, pred.ndim))
    if data_range in ("range", "ref-range"):
        data_range = torch.amax(target, dim=reduce_dims) - torch.amin(target, dim=reduce_dims)
    elif data_range in ("ref-maxval", "maxval"):
        data_range = torch.amax(target, dim=reduce_dims)
    elif data_range == "x-range":
        data_range = torch.amax(pred, dim=reduce_dims) - torch.amin(pred, dim=reduce_dims)
    elif data_range == "x-maxval":
        data_range = torch.amax(pred, dim=reduce_dims)
    elif data_range is None:
        data_range = torch.amax(
            torch.cat(
                [
                    torch.amax(pred, dim=reduce_dims) - torch.amin(pred, dim=reduce_dims),
                    torch.amax(target, dim=reduce_dims) - torch.amin(target, dim=reduce_dims),
                ],
                dim=0,
            ),
            dim=0,
        )
    if not isinstance(data_range, torch.Tensor):
        data_range = torch.as_tensor(data_range)

    ndim = len(kernel_size)
    c1 = (k1 * data_range) ** 2
    c2 = (k2 * data_range) ** 2
    c1 = c1.view(c1.shape + (1,) * (2 + ndim - c1.ndim))
    c2 = c2.view(c2.shape + (1,) * (2 + ndim - c2.ndim))
    device = pred.device

    channel = pred.size(1)
    dtype = pred.dtype
    kernel = _gaussian_kernel(channel, kernel_size, sigma, dtype, device)
    padding = tuple(
        pad for pad_set in [((k - 1) // 2,) * 2 for k in kernel_size[::-1]] for pad in pad_set
    )  # (pad_w, pad_w, pad_h, pad_h, ...)

    pred = _pad(pred, padding, mode=pad_mode)
    target = _pad(target, padding, mode=pad_mode)

    input_list = torch.cat(
        (pred, target, pred * pred, target * target, pred * target)
    )  # (5 * B, C, H, W)
    if ndim == 2:
        outputs = F.conv2d(input_list, kernel, groups=channel)
    else:
        outputs = F.conv3d(input_list, kernel, groups=channel)
    output_list = [outputs[x * pred.size(0) : (x + 1) * pred.size(0)] for x in range(len(outputs))]

    mu_pred_sq = output_list[0].pow(2)
    mu_target_sq = output_list[1].pow(2)
    mu_pred_target = output_list[0] * output_list[1]

    sigma_pred_sq = output_list[2] - mu_pred_sq
    sigma_target_sq = output_list[3] - mu_target_sq
    sigma_pred_target = output_list[4] - mu_pred_target

    upper = 2 * sigma_pred_target + c2
    lower = sigma_pred_sq + sigma_target_sq + c2

    ssim_idx = ((2 * mu_pred_target + c1) * upper) / ((mu_pred_sq + mu_target_sq + c1) * lower)
    if ndim == 2:
        ssim_idx = ssim_idx[..., padding[2] : -padding[3], padding[0] : -padding[1]]
    else:
        ssim_idx = ssim_idx[
            ..., padding[4] : -padding[5], padding[2] : -padding[3], padding[0] : -padding[1]
        ]

    return ssim_idx


def _gaussian_kernel(
    channel: int,
    kernel_size: Sequence[int],
    sigma: Sequence[float],
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    ndim = len(kernel_size)

    gaussian_kernel_x = _gaussian(kernel_size[0], sigma[0], dtype, device)
    gaussian_kernel_y = _gaussian(kernel_size[1], sigma[1], dtype, device)
    kernel = torch.matmul(
        gaussian_kernel_x.t(), gaussian_kernel_y
    )  # (kernel_size, 1) * (1, kernel_size)
    if ndim == 3:
        gaussian_kernel_z = _gaussian(kernel_size[2], sigma[2], dtype, device)
        kernel = kernel.unsqueeze(0) * gaussian_kernel_z.t().unsqueeze(-1)

    return kernel.expand(channel, 1, *kernel_size)


def _pad(x, padding, mode):
    if x.ndim < 5 or mode != "reflect":
        return F.pad(x, padding, mode)

    assert x.ndim == 5

    # 3D reflection padding
    # TODO: This will likely be supported in future PyTorch versions.
    # Update when the support is in a stable release.
    # https://github.com/pytorch/pytorch/pull/59791
    x = _pad_3d_tensor_with_2d_padding(x, padding[:-2], mode)

    dim = 2
    dpad1, dpad2 = padding[-2:]
    x1 = torch.flip(x[:, :, 1 : dpad1 + 1, ...], dims=(dim,))
    x2 = torch.flip(x[:, :, -dpad2 - 1 : -1, ...], dims=(dim,))
    x = torch.cat([x1, x, x2], dim=dim)
    return x


def _pad_3d_tensor_with_2d_padding(x, padding, mode):
    shape = x.shape
    x = x.reshape(shape[0], np.prod(shape[1:-2]), shape[-2], shape[-1])  # B x C*D x H x W
    x = F.pad(x, padding, mode)
    x = x.reshape(shape[0], *shape[1:-2], x.shape[-2], x.shape[-1])  # B x C x D x Hp x Wp
    return x
