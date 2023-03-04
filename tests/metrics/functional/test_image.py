import numpy as np
import pytest
import torch
import torch.nn.functional as F
from packaging.version import Version
from skimage import data
from skimage.metrics import structural_similarity

from meddlr.metrics.functional.image import (
    _pad,
    _pad_3d_tensor_with_2d_padding,
    mae,
    mse,
    nrmse,
    psnr,
    ssim,
)
from meddlr.metrics.image import compute_mse, compute_nrmse, compute_psnr, compute_ssim
from meddlr.utils import env

if Version(env.get_package_version("torchmetrics")) < Version("0.7.0"):
    from torchmetrics.functional.regression import ssim as tm_ssim
else:
    from torchmetrics.functional.image import structural_similarity_index_measure as tm_ssim


def test_mse_legacy():
    target = data.cells3d().astype(np.float32)[22:37, :, 75:150, 75:150][:, 0]  # D x H x W
    target = target + 1j * target
    pred = target + np.random.randn(*target.shape).astype(np.complex64) * target.mean()

    target = torch.as_tensor(target)
    pred = torch.as_tensor(pred)
    pred_tensor = pred.unsqueeze(0).unsqueeze(0)  # BxCxDxHxW
    target_tensor = target.unsqueeze(0).unsqueeze(0)  # BxCxDxHxW

    legacy_value = torch.as_tensor(compute_mse(target, pred).item())
    out = mse(pred_tensor, target_tensor)
    assert torch.allclose(out, legacy_value)

    legacy_value = torch.as_tensor(compute_mse(target, pred, magnitude=True).item())
    out = mse(pred_tensor, target_tensor, im_type="magnitude")
    assert torch.allclose(out, legacy_value)


def test_psnr_legacy():
    target = data.cells3d().astype(np.float32)[22:37, :, 75:150, 75:150][:, 0]  # D x H x W
    target = target + 1j * target
    pred = target + np.random.randn(*target.shape).astype(np.complex64) * target.mean()

    target = torch.as_tensor(target)
    pred = torch.as_tensor(pred)
    pred_tensor = pred.unsqueeze(0).unsqueeze(0)  # BxCxDxHxW
    target_tensor = target.unsqueeze(0).unsqueeze(0)  # BxCxDxHxW

    legacy_value = torch.as_tensor(compute_psnr(target, pred).item())
    out = psnr(pred_tensor, target_tensor)
    assert torch.allclose(out, legacy_value)

    legacy_value = torch.as_tensor(compute_psnr(target, pred, magnitude=True).item())
    out = psnr(pred_tensor, target_tensor, im_type="magnitude")
    assert torch.allclose(out, legacy_value)


def test_nrmse_legacy():
    target = data.cells3d().astype(np.float32)[22:37, :, 75:150, 75:150][:, 0]  # D x H x W
    target = target + 1j * target
    pred = target + np.random.randn(*target.shape).astype(np.complex64) * target.mean()

    target = torch.as_tensor(target)
    pred = torch.as_tensor(pred)
    pred_tensor = pred.unsqueeze(0).unsqueeze(0)  # BxCxDxHxW
    target_tensor = target.unsqueeze(0).unsqueeze(0)  # BxCxDxHxW

    legacy_value = torch.as_tensor(compute_nrmse(target, pred).item())
    out = nrmse(pred_tensor, target_tensor)
    assert torch.allclose(out, legacy_value)

    legacy_value = torch.as_tensor(compute_nrmse(target, pred, magnitude=True).item())
    out = nrmse(pred_tensor, target_tensor, im_type="magnitude")
    assert torch.allclose(out, legacy_value)


def test_ssim_torchmetrics_reproducibility():
    """
    Test reproducibility between SSIM implementations in meddlr and torchmetrics.

    Torchmetrics only supports 2D SSIM, which is what we test here.
    """
    img = data.brain()[:, np.newaxis, ...].astype(np.float32)  # B x C x H x W
    img_noise = img + np.random.randn(*img.shape).astype(np.float32) * img.mean()
    img_noise[img_noise < 0.0] = 0.0
    data_range = img.max() - img.min()

    tensor_noise = torch.from_numpy(img_noise)
    tensor_base = torch.from_numpy(img)

    ks = 11
    sigma = 1.5
    k1 = 0.01
    k2 = 0.03

    expected_ssim_val = tm_ssim(
        tensor_noise,
        tensor_base,
        data_range=data_range,
        kernel_size=(ks, ks),
        sigma=(sigma, sigma),
        reduction="none",
        k1=k1,
        k2=k2,
    ).mean()

    ssim_val = ssim(
        tensor_noise,
        tensor_base,
        data_range=data_range,
        kernel_size=ks,
        sigma=sigma,
        k1=k1,
        k2=k2,
    ).mean()
    assert torch.allclose(ssim_val, expected_ssim_val)


def test_ssim_scikit_image_reproducibility():
    """Test reproducibility between SSIM implementations in meddlr and scikit-image.

    Scikit-image supports 3D multichannel SSIM.
    We test 2D, 2D multichannel, 3D, and 3D multichannel.
    """
    img3d = data.cells3d().astype(np.float32)[22:37, :, 75:150, 75:150]  # D x C x H x W
    img3d_noise = img3d + np.random.randn(*img3d.shape).astype(np.float32) * img3d.mean()
    img3d_noise[img3d_noise < 0.0] = 0.0
    data_range = img3d.max() - img3d.min()

    # Reshape tensors to [batch, channel, depth, height, width]
    tensor_base = torch.from_numpy(img3d).permute(1, 0, 2, 3).unsqueeze(0)
    tensor_noise = torch.from_numpy(img3d_noise).permute(1, 0, 2, 3).unsqueeze(0)

    ks = 11
    sigma = 1.5
    k1 = 0.01
    k2 = 0.03

    # 2D
    for sl in range(img3d.shape[0] // 2 - 3, img3d.shape[0] // 2 + 3):
        expected_ssim_val = structural_similarity(
            img3d.transpose((0, 2, 3, 1))[sl],  # H x W x C
            img3d_noise.transpose((0, 2, 3, 1))[sl],  # H x W x C
            multichannel=True,
            win_size=ks,
            data_range=data_range,
            gaussian_weights=True,
            use_sample_covariance=False,
            channel_axis=-1,
        )

        ssim_val = ssim(
            tensor_noise[:, :, sl, ...],
            tensor_base[:, :, sl, ...],
            data_range=data_range,
            kernel_size=ks,
            sigma=sigma,
            k1=k1,
            k2=k2,
        ).mean()
        assert torch.allclose(ssim_val, torch.as_tensor(expected_ssim_val.item()))

    # 3D
    expected_ssim_val = structural_similarity(
        img3d.transpose((0, 2, 3, 1)),  # D x H x W x C
        img3d_noise.transpose((0, 2, 3, 1)),  # D x H x W x C
        multichannel=True,
        win_size=ks,
        data_range=data_range,
        gaussian_weights=True,
        use_sample_covariance=False,
        K1=k1,
        K2=k2,
        channel_axis=-1,
    )

    ssim_val = ssim(
        tensor_noise,
        tensor_base,
        data_range=data_range,
        kernel_size=ks,
        sigma=sigma,
        k1=k1,
        k2=k2,
    ).mean()
    assert torch.allclose(ssim_val, torch.as_tensor(expected_ssim_val.item()))


def test_ssim_legacy():
    """Test SSIM is equivalent to legacy function ``compute_ssim``."""
    target = data.cells3d().astype(np.float32)[22:37, :, 75:150, 75:150][:, 0]  # D x H x W
    target = target + 1j * target
    pred = target + np.random.randn(*target.shape).astype(np.complex64) * target.mean()

    target = torch.as_tensor(target)
    pred = torch.as_tensor(pred)

    legacy_value = compute_ssim(
        target,
        pred,
        data_range="ref-maxval",
        gaussian_weights=True,
        use_sample_covariance=False,
    )
    legacy_value = torch.as_tensor(legacy_value.item())

    pred_tensor = pred.unsqueeze(0).unsqueeze(0)  # BxCxDxHxW
    target_tensor = target.unsqueeze(0).unsqueeze(0)  # BxCxDxHxW
    out = ssim(pred_tensor, target_tensor, data_range="ref-maxval").mean()
    out2 = ssim(pred_tensor, target_tensor, method="Wang").mean()

    assert torch.allclose(out, legacy_value)
    assert torch.allclose(out2, legacy_value)


def test_pad_reflect():
    mode = "reflect"

    # 2D Reflection padding
    # This should match torch.nn.functional padding.
    shape, padding = (10, 3, 20, 20), (1, 1, 2, 2)
    x = torch.randn(*shape)
    expected_shape = (10, 3, 24, 22)
    xpad_expected = F.pad(x, padding, mode=mode)
    xpad = _pad(x, padding, mode=mode)
    assert torch.all(xpad == xpad_expected)
    assert xpad.shape == expected_shape
    check_reflection_per_spatial_dim(xpad, padding)

    # 3D Reflection padding
    # Verify that along each dimension we see reflection like property
    shape, padding = (2, 3, 20, 30, 40), (1, 2, 3, 4, 5, 6)
    x = torch.randn(*shape)
    expected_shape = (2, 3, 31, 37, 43)
    xpad = _pad(x, padding, mode=mode)
    assert xpad.shape == expected_shape
    check_reflection_per_spatial_dim(xpad, padding)


def check_reflection_per_spatial_dim(xpad: torch.Tensor, padding):
    spatial_dims = range(2, xpad.ndim)
    assert len(padding) == len(spatial_dims) * 2
    padding = [(padding[2 * i], padding[2 * i + 1]) for i in range(len(padding) // 2)][::-1]

    for dim, pad in zip(spatial_dims, padding):
        x_flat = xpad.permute((dim,) + tuple(d for d in range(xpad.ndim) if d != dim))
        x_flat = x_flat.reshape(x_flat.shape[0], -1)

        x_nopad = x_flat[pad[0] : -pad[1]]
        xtop, xbottom = x_flat[: pad[0]], x_flat[-pad[1] :]
        assert torch.all(torch.flip(x_nopad[1 : 1 + pad[0]], dims=(0,)) == xtop)
        assert torch.all(torch.flip(x_nopad[-pad[1] - 1 : -1], dims=(0,)) == xbottom)


def test_pad_3d_tensor_with_2d_padding_reflect():
    shape, padding = (10, 3, 5, 20, 30), (2, 4, 5, 6)
    x = torch.randn(*shape)
    mode = "reflect"

    expected_out = torch.stack([F.pad(x[:, i], padding, mode) for i in range(shape[1])], dim=1)
    out = _pad_3d_tensor_with_2d_padding(x, padding, mode=mode)

    assert out.shape == expected_out.shape
    assert torch.all(out == expected_out)


@pytest.mark.parametrize("ndim", [2, 3])
@pytest.mark.parametrize("nchannels", [1, 3, 10])
@pytest.mark.parametrize("dtype", [torch.float32, torch.complex64])
def test_mae_accuracy(ndim, nchannels, dtype):
    """Test MAE accuracy."""
    bsz = 10
    shape = (bsz, nchannels) + (20,) * ndim
    x = torch.randn(*shape, dtype=dtype)
    y = torch.randn(*shape, dtype=dtype)
    out = mae(x, y)
    assert out.shape == (bsz, nchannels)
    assert torch.allclose(out, torch.abs(x - y).reshape(bsz, nchannels, -1).mean(-1))
