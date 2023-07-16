from typing import Callable, Union

import torch
from tqdm.auto import tqdm

import meddlr.ops.complex as cplx


def conjgrad(
    x: torch.Tensor,
    b: torch.Tensor,
    A_op: Callable,
    mu: Union[torch.Tensor, float],
    max_iter: int = 10,
    eps: float = 1e-4,
    pbar: bool = False,
):
    """Conjugate gradient descent for solving min_x ||A(x) - b||_2^2 + mu * ||x||_2^2

    Adapted from https://github.com/bo-10000/MoDL_PyTorch/blob/master/models/modl.py.

    Args:
        x: The initial input. Shape (B, ...).
        b: The residual. Shape (B, ...). Must be same shape as `x`.
        A_op: The function performing the normal equations.
            For complex numbers, this should be adjoint(A) * A.
        mu: The L2 lambda, or regularization parameter (must be positive).
        max_iter: Maximum number of times to run conjugate gradient descent.
        eps: Determines how small the residuals must be before termination.
            Stopping criterion is conj(r).T * r < eps ** 2.
        pbar: Whether to show a progress bar.

    Returns:
        torch.Tensor: x for the minimization equation above.

    Note:
        Currently only supports complex numbers.

    Note:
        Complex-as-real tensors are not supported for complex numbers.
        Use complex tensors instead.
    """

    def _dot(x, y, keepdim=False):
        return cplx.bdot(x, y, keepdim=keepdim).real

    assert cplx.is_complex(x)

    r = b - (A_op(x) + mu * x)
    p = r

    rHr = _dot(r, r, keepdim=True)
    rHr_new = rHr

    for _ in tqdm(range(max_iter), disable=not pbar):
        if rHr.max() < eps**2:
            break

        Ap = A_op(p) + mu * p
        pAp = _dot(p, Ap, keepdim=True)

        alpha = rHr / pAp
        x = x + alpha * p
        r = r - alpha * Ap

        rHr_new = _dot(r, r, keepdim=True)
        beta = rHr_new / rHr
        p = r + beta * p

        rHr = rHr_new
    return x
