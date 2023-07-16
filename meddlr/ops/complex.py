"""
Utilities for doing complex-valued operations.
"""
import numpy as np
import torch

from meddlr.utils.deprecated import deprecated
from meddlr.utils.env import supports_cplx_tensor

__all__ = [
    "is_complex",
    "is_complex_as_real",
    "conj",
    "mul",
    "abs",
    "angle",
    "real",
    "imag",
    "from_polar",
    "channels_first",
    "channels_last",
    "get_mask",
    "matmul",
    "power_method",
    "svd",
    "to_numpy",
    "to_tensor",
    "rss",
    "center_crop",
]


def is_complex(x):
    """Returns if ``x`` is a complex-tensor.

    This function is a wrapper around torch.is_complex() for PyTorch<1.7.
    torch < 1.7 does not have the ``torch.is_complex`` directive, so
    we can't call it for older PyTorch versions.

    Args:
        x (torch.Tensor): A tensor.

    Returns:
        bool: ``True`` if complex tensors are supported and the tensor is complex.
    """
    return supports_cplx_tensor() and torch.is_complex(x)


def is_complex_as_real(x):
    """
    Returns ``True`` if the tensor follows the real-view
    convention for complex numbers.

    The real-view of a complex tensor has the shape [..., 2].

    Args:
        x (torch.Tensor): A tensor.

    Returns:
        bool: ``True`` if the tensor follows the real-view convention
            for complex numbers.

    Note:
        We recommend using complex tensors instead of the real-view
        convention. This function cannot interpret if the last dimension
        has a size of ``2`` because it is the real-imaginary channel or
        for some other reason.
    """
    return not is_complex(x) and x.size(-1) == 2


def conj(x):
    """
    Computes the complex conjugate of complex-valued input tensor (x).

    ``conj(a + ib)`` = :math:`\\bar{a + ib} = a - ib`

    Args:
        x (torch.Tensor): A tensor.

    Returns:
        torch.Tensor: The conjugate.
    """
    assert is_complex_as_real(x) or is_complex(x)
    if is_complex(x):
        return x.conj()
    else:
        real = x[..., 0]
        imag = x[..., 1]
        return torch.stack((real, -1.0 * imag), dim=-1)


def mul(x, y):
    """
    Multiplies two complex-valued tensors x and y.

    :math:`z = (a + ib) * (c + id) = (ac - bd) + i(ad + bc)`

    Args:
        x (torch.Tensor): A tensor.
        y (torch.Tensor): A tensor.

    Returns:
        torch.Tensor: The matrix multiplication.
    """
    # assert x.size(-1) == 2
    # assert y.size(-1) == 2
    #
    # a = x[..., 0]
    # b = x[..., 1]
    # c = y[..., 0]
    # d = y[..., 1]
    #
    # real = a * c - b * d
    # imag = a * d + b * c

    # return torch.stack((real, imag), dim=-1)

    assert is_complex_as_real(x) or is_complex(x)
    assert is_complex_as_real(y) or is_complex(y)
    if is_complex(x):
        return x * y
    else:
        # note: using select() makes sure that another copy is not made.
        # real = a*c - b*d
        real = x.select(-1, 0) * y.select(-1, 0)  # a*c
        real -= x.select(-1, 1) * y.select(-1, 1)  # b*d
        # imag = a*d + b*c
        imag = x.select(-1, 0) * y.select(-1, 1)  # a*d
        imag += x.select(-1, 1) * y.select(-1, 0)  # b*c
        return torch.stack((real, imag), dim=-1)


def abs(x):
    """
    Computes the absolute value (magnitude) of a complex-valued input tensor (x).

    Args:
        x (torch.Tensor): A tensor.

    Returns:
        torch.Tensor: The magnitude tensor.
    """
    assert is_complex_as_real(x) or is_complex(x)
    if is_complex(x):
        return x.abs()
    else:
        return (x**2).sum(dim=-1).sqrt()


def angle(x, eps=1e-11):
    """
    Computes the phase of a complex-valued input tensor (x).

    Args:
        x (torch.Tensor): A tensor.

    Returns:
        torch.Tensor: The angle tensor.
    """
    assert is_complex_as_real(x) or is_complex(x)
    if is_complex(x):
        return x.angle()
    else:
        return torch.atan(x[..., 1] / (x[..., 0] + eps))


def real(x):
    """
    Gets real component of complex tensor.
    """
    assert is_complex_as_real(x) or is_complex(x)
    if is_complex(x):
        return x.real
    else:
        return x[..., 0]


def imag(x):
    """
    Gets imaginary component of complex tensor.
    """
    assert is_complex_as_real(x) or is_complex(x)
    if is_complex(x):
        return x.imag
    else:
        return x[..., 1]


def from_polar(magnitude, phase, return_cplx: bool = None):
    """
    Computes real and imaginary values from polar representation.
    """
    if return_cplx and not supports_cplx_tensor():
        raise RuntimeError(f"torch {torch.__version__} does not support complex tensors")

    if supports_cplx_tensor():
        out = torch.polar(magnitude, phase)
        if return_cplx is False:
            out = torch.view_as_real(out)
        return out
    else:
        real = magnitude * torch.cos(phase)
        imag = magnitude * torch.sin(phase)
        return torch.stack((real, imag), dim=-1)


polar = from_polar


def channels_first(x: torch.Tensor):
    """Permute complex-valued ``x`` to channels-first convention.

    For complex values, there are two potential conventions:

    1. ``x`` is complex-valued: ``(B,...,C)`` -> ``(B, C, ...)``.
    2. The real and imaginary components are stored in the last dimension.
       ``(B,...,C,2)`` -> ``(B, C, ..., 2)``.

    Args:
        x (torch.Tensor): A complex-valued tensor of shape ``(B,...,C)``
            or a real-valued tensor of shape ``(B,...,C,2)``.

    Returns:
        torch.Tensor: A channels-first tensor. If ``x`` is complex,
            this will also be complex. If ``x`` is the real-view of
            a complex tensor, this will also be the real view.
    """
    assert is_complex_as_real(x) or is_complex(x)
    if is_complex(x):
        return x.permute((0, x.ndim - 1) + tuple(range(1, x.ndim - 1)))
    else:
        return x.permute((0, x.ndim - 2) + tuple(range(1, x.ndim - 2)) + (x.ndim - 1,))


@deprecated(
    reason="Renamed to channels_first",
    vremove="v0.1.0",
    replacement="meddlr.ops.complex.channels_first",
)
def channel_first(x: torch.Tensor):
    """Deprecated alias for :func:`channels_first`."""
    return channels_first(x)


def channels_last(x: torch.Tensor):
    """Permute complex-valued ``x`` to channels-last convention.

    Args:
        x (torch.Tensor): A tensor of shape [B,C,H,W,...] or [B,C,H,W,...,2].

    Returns:
        torch.Tensor: A tensor of shape [B,H,W,...,C] or [B,H,W,...,C,2].
    """
    assert is_complex_as_real(x) or is_complex(x)
    if is_complex(x):
        return x.permute((0,) + tuple(range(2, x.ndim) + (1,)))
    else:
        order = (0,) + tuple(range(2, x.ndim - 2)) + (1, x.ndim - 1)
        return x.permute(order)


def get_mask(x, eps=1e-11, coil_dim=None):
    """Returns a binary mask for where ``x`` is nonzero with ``eps`` tolerance.

      - 0, if both real and imaginary components are zero.
      - 1, if either real and imaginary components are non-zero.

    Args:
        x (torch.Tensor): A complex-valued tensor.
        eps (float): Tolerance for zer0-value.
        coil_dim (int): The coil dimension.
            When this is provided, if a pixel is non-zero for any coil,
            we assume that pixel was acquired. This is useful when
            a coil ``i`` has zero signal but the location was actually
            acquired.

    Returns:
        torch.Tensor: A binary mask of shape ``x.shape``.
    """
    unsqueeze = True
    if is_complex(x):
        unsqueeze = False
        x = torch.view_as_real(x)
    assert x.size(-1) == 2

    absx = abs(x)
    loc = absx > eps  # squashes last dimension
    if coil_dim is not None:
        loc = loc.any(coil_dim, keepdims=True)
    mask = torch.where(loc, torch.ones_like(absx), torch.zeros_like(absx))

    if unsqueeze:
        mask = mask.unsqueeze(-1)
    return mask


def matmul(X, Y):
    """
    Computes complex-valued matrix product of X and Y.
    """
    assert is_complex_as_real(X) or is_complex(X)
    assert is_complex_as_real(Y) or is_complex(Y)
    if is_complex(X):
        return torch.matmul(X, Y)
    else:
        A = X[..., 0]
        B = X[..., 1]
        C = Y[..., 0]
        D = Y[..., 1]
        real = torch.matmul(A, C) - torch.matmul(B, D)
        imag = torch.matmul(A, D) + torch.matmul(B, C)
        return torch.stack((real, imag), dim=-1)


def power_method(X, num_iter=10, eps=1e-6):
    """
    Iteratively computes first singular value of X using power method.
    """
    if is_complex_as_real(X) or is_complex(X):
        X = torch.view_as_real(X)
    assert X.size(-1) == 2

    # get data dimensions
    batch_size, m, n, _ = X.shape

    XhX = matmul(conj(X).permute(0, 2, 1, 3), X)

    # initialize random eigenvector
    if XhX.is_cuda:
        v = torch.cuda.FloatTensor(batch_size, n, 1, 2).uniform_()
    else:
        v = torch.FloatTensor(batch_size, n, 1, 2).uniform_()
    # v = torch.rand(batch_size, n, 1, 2).to(X.device) # slow way

    for _i in range(num_iter):
        v = matmul(XhX, v)
        eigenvals = (abs(v) ** 2).sum(1).sqrt()
        v = v / (eigenvals.reshape(batch_size, 1, 1, 1) + eps)

    return eigenvals.reshape(batch_size)


def svd(X, compute_uv=True):
    """
    Computes singular value decomposition of batch of complex-valued matrices.

    Args:
        matrix (torch.Tensor): batch of complex-valued 2D matrices
            [batch, m, n, 2]
    Returns:
        U, S, V (tuple)
    """
    if is_complex_as_real(X) or is_complex(X):
        X = torch.view_as_real(X)
    assert X.size(-1) == 2

    # Get data dimensions
    batch_size, m, n, _ = X.shape

    # Allocate block-wise matrix
    # (otherwise, need to allocate new arrays three times)
    if X.is_cuda:
        Xb = torch.cuda.FloatTensor(batch_size, 2 * m, 2 * n).fill_(0)
    else:
        Xb = torch.FloatTensor(batch_size, 2 * m, 2 * n).fill_(0)

    # Construct real-valued block matrix
    # Xb = [X.real, X.imag; -X.imag, X.real]
    Xb[:, :m, :n] = X[..., 0]
    Xb[:, :m, n:] = X[..., 1]
    Xb[:, m:, :n] = -X[..., 1]
    Xb[:, m:, n:] = X[..., 0]

    # Perform real-valued SVD
    U, S, V = torch.svd(Xb, compute_uv=compute_uv)

    # Slice U, S, V appropriately
    S = S[:, ::2]
    U = torch.stack((U[:, :m, ::2], -U[:, m:, ::2]), dim=3)
    V = torch.stack((V[:, :n, ::2], -V[:, n:, ::2]), dim=3)

    return U, S, V


def to_numpy(x: torch.Tensor):
    """
    Convert real-valued PyTorch tensor to complex-valued numpy array.
    """
    assert is_complex_as_real(x) or is_complex(x)
    if is_complex(x):
        return x.clone().numpy()  # previously returned copy
    else:
        x = x.numpy()
        return x[..., 0] + 1j * x[..., 1]


def to_tensor(x: np.ndarray):
    """
    Convert complex-valued numpy array to real-valued PyTorch tensor.
    """
    if not supports_cplx_tensor():
        x = np.stack((x.real, x.imag), axis=-1)
    return torch.from_numpy(x)


def rss(x: torch.Tensor, dim: int = 0) -> torch.Tensor:
    """
    Compute the root-sum-of-squares (RSS) for complex inputs.
    RSS is computed assuming that dim is the coil dimension.

    Args:
        data: The complex-valued input tensor.
        dim: The dimensions along which to apply the RSS transform.

    Returns:
        torch.Tensor: The RSS value.
    """
    return torch.sqrt((abs(x) ** 2).sum(dim))


root_sum_of_squares = rss


def center_crop(x: torch.Tensor, shape, channels_last: bool = False):
    """
    Apply a center crop to the input image or batch of complex images.

    Args:
        data (torch.Tensor): The complex input tensor to be center cropped.
        shape (int, int): The output shape. The shape should be smaller than the
            corresponding dimensions of data.
        channels_last (bool, optional): If ``True``, crop dimensions ``range(1, 1+len(shape))``.
            If ``False``, apply to last non-real/imaginary channel dimensions.

    Returns:
        torch.Tensor: The center cropped image.
    """
    if channels_last:
        dims = range(1, 1 + len(shape))
    elif not is_complex(x) and is_complex_as_real(x):
        dims = range(-1 - len(shape), -1)
    else:
        dims = range(-len(shape), 0)

    x_shape = tuple(x.shape[d] for d in dims)
    assert all(0 < shape[idx] <= x_shape[idx] for idx in range(len(shape)))

    sl = [slice(None) for _ in range(x.ndim)]
    for d, shp, x_shp in zip(dims, shape, x_shape):
        start = (x_shp - shp) // 2
        end = start + shp
        sl[d] = slice(start, end)
    return x[sl]


def bdot(x: torch.Tensor, y: torch.Tensor, keepdim: bool = False) -> torch.Tensor:
    """Batch dot product (inner product) of two complex-valued tensors.

    Args:
        x: The first input tensor.
        y: The second input tensor.

    Returns:
        torch.Tensor: The batch inner product :math:`<x, y>_i = sum(conj(x_i) * y_i)`.

    Note:
        To avoid ambiguity, use torch.complex tensors to represent complex values.
    """
    dim = tuple(range(1, x.ndim))
    return torch.sum((x.conj() * y), dim=dim, keepdim=keepdim)
