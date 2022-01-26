import torch

import meddlr.ops.complex as cplx


def test_complex_realview():
    x = torch.randn(5, 10, 10, dtype=torch.complex64)
    y = torch.randn(5, 10, 10, dtype=torch.complex64)
    x_real, y_real = torch.view_as_real(x), torch.view_as_real(y)

    assert cplx.is_complex(x) and cplx.is_complex(y)
    assert cplx.is_complex_as_real(x_real) and cplx.is_complex_as_real(y_real)

    assert torch.allclose(cplx.conj(x), torch.view_as_complex(cplx.conj(x_real)))

    assert torch.allclose(cplx.mul(x, y), torch.view_as_complex(cplx.mul(x_real, y_real)))

    assert torch.allclose(cplx.abs(x), cplx.abs(x_real))

    # assert torch.allclose(cplx.angle(x), cplx.angle(x_real, eps=0))

    assert torch.allclose(cplx.real(x), cplx.real(x_real))

    assert torch.allclose(cplx.imag(x), cplx.imag(x_real))
