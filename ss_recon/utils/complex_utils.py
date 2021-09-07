"""
Utilities for doing complex-valued operations.
"""
import numpy as np
import torch

from ss_recon.utils.env import supports_cplx_tensor


def is_complex(x):
    """Wrapper around torch.is_complex() for PyTorch<1.7"""
    return supports_cplx_tensor() and torch.is_complex(x)


def is_complex_as_real(x):
    """Returns `True` if the tensor is real-view convention for complex numbers.

    The real-view of a complex tensor has the shape [..., 2]
    """
    return x.size(-1) == 2


def conj(x):
    """
    Computes the complex conjugate of complex-valued input tensor (x).
    i.e. conj(a + ib) = a - ib
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
     i.e. z = (a + ib) * (c + id)
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
    Computes the absolute value of a complex-valued input tensor (x).
    """
    assert is_complex_as_real(x) or is_complex(x)
    if is_complex(x):
        return x.abs()
    else:
        return (x ** 2).sum(dim=-1).sqrt()


def angle(x, eps=1e-11):
    """
    Computes the phase of a complex-valued input tensor (x).
    """
    assert is_complex_as_real(x) or is_complex(x)
    if is_complex(x):
        return x.angle()
    else:
        return torch.atan(x[..., 1] / (x[..., 0] + eps))


def from_polar(magnitude, phase):
    """
    Computes real and imaginary values from polar representation.
    """
    real = magnitude * torch.cos(phase)
    imag = magnitude * torch.sin(phase)
    return torch.stack((real, imag), dim=-1)


def get_mask(x, eps=1e-11):
    """
    Returns a binary mask of zeros and ones:
      - 0, if both real and imaginary components are zero.
      - 1, if either real and imaginary components are non-zero.
    """
    unsqueeze = True
    if is_complex(x):
        unsqueeze = False
        x = torch.view_as_real(x)
    assert x.size(-1) == 2
    absx = abs(x)  # squashes last dimension
    mask = torch.where(absx > eps, torch.ones_like(absx), torch.zeros_like(absx))
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
    Computes singular value decomposition of batch of complex-valued matrices

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


def to_numpy(x):
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
    Compute the Root Sum of Squares (RSS) for complex inputs.
    RSS is computed assuming that dim is the coil dimension.
    Args:
        data: The input tensor
        dim: The dimensions along which to apply the RSS transform
    Returns:
        The RSS value.
    """
    return torch.sqrt((abs(x) ** 2).sum(dim))
