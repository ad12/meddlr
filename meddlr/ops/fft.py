from typing import Sequence

import numpy as np
import torch

from meddlr.ops import complex as cplx
from meddlr.ops.utils import roll
from meddlr.utils import env

if env.pt_version() >= [1, 6]:
    import torch.fft

__all__ = [
    "fftnc",
    "ifftnc",
    "fftc",
    "ifftc",
    "fft2c",
    "ifft2c",
    "fft3c",
    "ifft3c",
    "fftshift",
    "ifftshift",
]


def fftnc(input: torch.Tensor, dim=None, norm="ortho", is_real: bool = None) -> torch.Tensor:
    """Apply nD centered fast fourier transform.

    This function is a backwards-compatible wrapper for centered :meth:`torch.fft.fftn`.
    It supports backwards compatibility with ``torch.fft``as implemented in torch<1.7.

    Args:
        input (torch.Tensor): A tensor (typically complex).
        dim (Tuple[int]): Dimensions to be transformed.
        norm (str | bool, optional): The normalization method.
            Defaults to ``'ortho'``. For torch<1.7, only ``'ortho'``
            is supported.
        is_real (bool, optional): If ``True``, ``input`` is a real-valued
            tensor. If ``None`` or ``False`` and ``input.shape[-1] == 2``,
            ``input`` is a real-view of a complex tensor.

    Returns:
        torch.Tensor

    Note:
        Real-valued tensors are not supported with ``torch<1.7``.
    """
    return _fft_template(input, kind="fft", dim=dim, norm=norm, is_real=is_real)


def ifftnc(input: torch.Tensor, dim=None, norm="ortho", is_real: bool = None) -> torch.Tensor:
    """Apply nD centered inverse fast fourier transform.

    This supports backwards compatibility with ``torch.fft``
    as implemented in torch<1.7.

    Args:
        input (torch.Tensor): A tensor (typically complex).
        dim (Tuple[int]): Dimensions to be transformed.
        norm (str | bool, optional): The normalization method.
            Defaults to ``'ortho'``. For torch<1.7, only ``'ortho'``
            is supported.
        is_real (bool, optional): If ``True``, ``input`` is a real-valued
            tensor. If ``None`` or ``False`` and ``input.shape[-1] == 2``,
            ``input`` is a real-view of a complex tensor.

    Returns:
        torch.Tensor

    Note:
        Real-valued tensors are not supported with ``torch<1.7``.
    """
    return _fft_template(input, kind="ifft", dim=dim, norm=norm, is_real=is_real)


def fftc(input: torch.Tensor, norm: str = "ortho", is_real: bool = None, channels_last=False):
    """Apply 1D centered Fast Fourier Transform (FFT).

    Args:
        input (torch.Tensor): A tensor.
        norm (str | bool, optional): The normalization method.
            Defaults to ``'ortho'``. For torch<1.7, only ``'ortho'``
            is supported.
        is_real (bool, optional): If ``True``, ``input`` is a real-valued
            tensor. If ``None`` or ``False`` and ``input.shape[-1] == 2``,
            ``input`` is a real-view of a complex tensor.
        channels_last (bool, optional): If ``True``, apply to first
            non-batch dimensions. If ``False``, apply to last dimension.

    Returns:
        torch.Tensor: The 1D centered FFT of the input.
    """
    dim = _get_fft_dims(input, 1, is_real=is_real, channels_last=channels_last)
    return fftnc(input, dim=dim, norm=norm, is_real=is_real)


def ifftc(input: torch.Tensor, channels_last=False, norm: str = "ortho", is_real: bool = None):
    """Apply centered 2 dimensional Fast Fourier Transform.

    Args:
        data (torch.Tensor): Complex valued input data with the last dimension
            containing real and imaginary components.
        dims (2-tuple): Containing spatial dimension indices.
    Returns:
        torch.Tensor: The FFT of the input.
    """
    dim = _get_fft_dims(input, 1, is_real=is_real, channels_last=channels_last)
    return ifftnc(input, dim=dim, norm=norm, is_real=is_real)


def fft2c(input: torch.Tensor, channels_last=False, norm: str = "ortho", is_real: bool = None):
    """Apply centered 2 dimensional Fast Fourier Transform.

    Args:
        data (torch.Tensor): Complex valued input data with the last dimension
            containing real and imaginary components.
        dims (2-tuple): Containing spatial dimension indices.
    Returns:
        torch.Tensor: The FFT of the input.
    """
    dim = _get_fft_dims(input, 2, is_real=is_real, channels_last=channels_last)
    return fftnc(input, dim=dim, norm=norm, is_real=is_real)


def ifft2c(input, channels_last=False, norm: str = "ortho", is_real: bool = None):
    """Apply centered 2-dimensional Inverse Fast Fourier Transform.

    Args:
        data (torch.Tensor): Complex valued input data with the last dimension
            containing real and imaginary components.
        dims (2-tuple): Containing spatial dimension indices.
    Returns:
        torch.Tensor: The IFFT of the input.
    """
    dim = _get_fft_dims(input, 2, is_real=is_real, channels_last=channels_last)
    return ifftnc(input, dim=dim, norm=norm, is_real=is_real)


def fft3c(input: torch.Tensor, channels_last=False, norm: str = "ortho", is_real: bool = None):
    """Apply centered 2 dimensional Fast Fourier Transform.

    Args:
        data (torch.Tensor): Complex valued input data with the last dimension
            containing real and imaginary components.
        dims (2-tuple): Containing spatial dimension indices.
    Returns:
        torch.Tensor: The FFT of the input.
    """
    dim = _get_fft_dims(input, 3, is_real=is_real, channels_last=channels_last)
    return fftnc(input, dim=dim, norm=norm, is_real=is_real)


def ifft3c(input, channels_last=False, norm: str = "ortho", is_real: bool = None):
    """Apply centered 2-dimensional Inverse Fast Fourier Transform.

    Args:
        data (torch.Tensor): Complex valued input data with the last dimension
            containing real and imaginary components.
        dims (2-tuple): Containing spatial dimension indices.
    Returns:
        torch.Tensor: The IFFT of the input.
    """
    dim = _get_fft_dims(input, 3, is_real=is_real, channels_last=channels_last)
    return ifftnc(input, dim=dim, norm=norm, is_real=is_real)


def fftshift(x, dim=None):
    """
    Similar to np.fft.fftshift but applies to PyTorch Tensors
    """
    if dim is None:
        dim = tuple(range(x.dim()))
        shift = [dim // 2 for dim in x.shape]
    elif isinstance(dim, int):
        shift = x.shape[dim] // 2
    else:
        shift = [x.shape[i] // 2 for i in dim]
    return roll(x, shift, dim)


def ifftshift(x, dim=None):
    """
    Similar to np.fft.ifftshift but applies to PyTorch Tensors
    """
    if dim is None:
        dim = tuple(range(x.dim()))
        shift = [(dim + 1) // 2 for dim in x.shape]
    elif isinstance(dim, int):
        shift = (x.shape[dim] + 1) // 2
    else:
        shift = [(x.shape[i] + 1) // 2 for i in dim]
    return roll(x, shift, dim)


def _fft_template(
    data: torch.Tensor, kind, dim=None, norm="ortho", is_real: bool = None, centered: bool = True
) -> torch.Tensor:
    """Template for fft operations.

    Args:
        data (torch.Tensor): A tensor.
        kind (str): Either ``'fft'`` or ``'ifft'``.
        dim (int(s), optional): The dimension(s) along which to apply the operation.
            Defaults to all dimensions.
        norm (str, optional): The normalization method. Defaults to ``'ortho'``.
        is_real (bool, optional): If ``True``, ``input`` is treated like a real-valued
            tensor. If not specified, this is ``True`` only if ``data`` is not complex
            and data is not inferred to be a real view of a complex tensor
            (i.e. ``data.shape[-1] != 2``).
        centered (bool, optional): If ``True``, apply centered FFT. Defaults to ``True``.

    Returns:
        torch.Tensor: The FFT (or IFFT) of the input.
    """
    if isinstance(dim, int):
        dim = (dim,)
    if norm is True:
        norm = "ortho"
    assert kind in ("fft", "ifft")

    if is_real is None:
        is_real = not (cplx.is_complex_as_real(data) or cplx.is_complex(data))
    ndim = data.ndim

    if not env.supports_cplx_tensor():
        # Defaults to torch.fft method.
        assert norm in ("ortho", False)  # norm not supported
        norm = norm == "ortho"
        assert not is_real  # real tensors not supported
        assert cplx.is_complex_as_real(data)

        dim = tuple(sorted(_to_positive_index(dim)))
        if ndim - 1 in dim:
            raise ValueError("Cannot take fft along the real/imaginary channel.")
        if len(set(dim)) != len(dim):
            raise ValueError(f"Expected unique dimensions, got {dim}.")

        signal_ndim = len(dim)
        if signal_ndim > 3:
            raise ValueError(f"Number of dimensions must be <=3, got {len(dim)}.")

        # Reorder dims (if necessary).
        last_dims = dim + (ndim - 1,)
        permute = last_dims != tuple(range(ndim - signal_ndim - 1, ndim))
        if permute:
            order = tuple(i for i in range(ndim)) + last_dims
            data = data.permute(order)

        shift_dims = tuple([-2 - i for i in range(len(dim))][::-1])

        if kind == "fft":
            if centered:
                data = ifftshift(data, dim=shift_dims)
            data = torch.Tensor.fft(data, signal_ndim, normalized=norm)
            if centered:
                data = fftshift(data, shift_dims)
        elif kind == "ifft":
            if centered:
                data = ifftshift(data, dim=shift_dims)
            data = torch.ifft(data, signal_ndim, normalized=norm)
            if centered:
                data = fftshift(data, shift_dims)
        else:
            raise ValueError(f"Unknown `kind={kind}`")

        # Reorder dims (if necessary).
        if permute:
            reorder = tuple(np.argsort(order))
            data = data.permute(reorder)
        return data

    is_real_view = not is_real and cplx.is_complex_as_real(data)
    if is_real_view:
        # Make dimensions positive relative to input dimensions.
        dim = _to_positive_index(dim, ndim=data.ndim)
        data = torch.view_as_complex(data)

    if kind == "fft":
        if centered:
            data = ifftshift(data, dim=dim)
        data = torch.fft.fftn(data, dim=dim, norm=norm)
        if centered:
            data = fftshift(data, dim=dim)
    elif kind == "ifft":
        if centered:
            data = ifftshift(data, dim=dim)
        data = torch.fft.ifftn(data, dim=dim, norm=norm)
        if centered:
            data = fftshift(data, dim=dim)
    else:
        raise ValueError(f"Unknown `kind={kind}`")
    if is_real_view:
        data = torch.view_as_real(data)
    return data


def _get_fft_dims(x, signal_ndim, is_real, channels_last):
    if channels_last:
        return tuple(range(1, 1 + signal_ndim))

    if not is_real and not cplx.is_complex(x) and cplx.is_complex_as_real(x):
        return tuple(range(-1 - signal_ndim, -1))
    else:
        return tuple(range(-signal_ndim, 0))


def _to_positive_index(idxs: Sequence[int], ndim: int):
    return tuple(ndim + i if i < 0 else i for i in idxs)
