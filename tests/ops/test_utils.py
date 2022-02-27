import pytest
import torch

from meddlr.ops.utils import center_crop, zero_pad


@pytest.mark.parametrize("out_shape", [(40, 40), (40, 50), (40, 50, 60), (40, 41)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.complex64])
def test_zero_pad(out_shape, dtype):
    ndim = len(out_shape)

    in_shape = (10,) * ndim  # (B, ...)
    x = torch.randn(*in_shape, dtype=dtype)
    out = zero_pad(x.unsqueeze(0), out_shape).squeeze(0)

    assert out.shape == out_shape
    assert out.dtype == dtype

    for dim, (i, o) in enumerate(zip(in_shape, out_shape)):
        assert torch.all(torch.index_select(out, dim, torch.arange(0, (o - i) // 2)) == 0)
        assert torch.all(torch.index_select(out, dim, torch.arange((o - i) // 2 + i, o)) == 0)

    sl = []
    for i, o in zip(in_shape, out_shape):
        sl.append(slice((o - i) // 2, (o - i) // 2 + i))
    assert torch.all(out[tuple(sl)] == x)


def test_zero_pad_mismatched_dims():
    in_shape = (10, 10, 3)
    out_shape = (20, 30)

    x = torch.randn(*in_shape)
    out = zero_pad(x.unsqueeze(0), out_shape).squeeze(0)
    assert out.shape == (20, 30, 3)


@pytest.mark.parametrize("out_shape", [(40, 40), (40, 50), (40, 50, 60), (40, 41)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.complex64])
@pytest.mark.parametrize("include_batch", [True, False])
def test_center_crop(out_shape, dtype, include_batch):
    in_shape = tuple(d * 2 for d in out_shape)
    x = torch.randn(*in_shape, dtype=dtype)

    if not include_batch:
        x = x.unsqueeze(0)
    out = center_crop(x, out_shape, include_batch=include_batch)
    if not include_batch:
        assert out.shape == (1,) + out_shape
        x = x.squeeze(0)
        out = out.squeeze(0)

    assert out.shape == out_shape
    sl = tuple([slice((i - o) // 2, (i - o) // 2 + o) for i, o in zip(in_shape, out_shape)])
    assert out.shape == x[sl].shape
    assert torch.all(out == x[sl])
