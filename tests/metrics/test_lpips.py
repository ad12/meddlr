import unittest

import numpy as np
import torch
from skimage import data

from meddlr.metrics.lpip import LPIPS
from meddlr.utils import env

if [int(ver) for ver in env.get_package_version("torchmetrics").split(".")] >= [0, 8]:
    from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity


@unittest.skipIf(
    [int(ver) for ver in env.get_package_version("torchmetrics").split(".")] < [0, 8]
    or not env.package_available("lpips"),
    "TorchMetrics does not support LPIPS before version 0.8",
)
def test_LPIPS_torchmetrics_reproducibility():
    """Test reproducibility between LPIPS implementations in meddlr and torchmetrics."""

    meddlr_metric = LPIPS(mode="rgb", lpips=True)

    targets = data.astronaut().astype(np.float32)  # H x W x 3
    targets = np.transpose(targets[np.newaxis, :, :, :], (0, 3, 1, 2))  # 1 x 3 x H x W
    preds = targets + np.random.randn(*targets.shape).astype(np.float32) * targets.mean()

    targets = torch.as_tensor(targets)
    preds = torch.as_tensor(preds)

    meddlr_metric(preds, targets)
    meddlr_out = meddlr_metric.compute().sum()

    torchmetrics_metric = LearnedPerceptualImagePatchSimilarity(net_type="alex", reduction="sum")
    # torchmetrics lpips has no built-in preprocessing so we use our own
    preds = LPIPS(mode="rgb").preprocess_lpips(preds)
    targets = LPIPS(mode="rgb").preprocess_lpips(targets)

    torchmetrics_out = torchmetrics_metric(preds, targets)

    assert torch.allclose(meddlr_out, torchmetrics_out)


@unittest.skipIf(
    not env.package_available("lpips"),
    "LPIPS metric requires that lpips is installed.",
)
def test_LPIPS_noise():
    """Test that LPIPS increases as Gaussian noise increases."""

    # Complex valued and greyscale mode
    metric = LPIPS()
    noise_levels = [0, 0.01, 0.05, 0.1, 0.25]

    target = data.camera().astype(np.float32)
    target = target + 1j * target
    targets = np.repeat(target[np.newaxis, np.newaxis, :, :], len(noise_levels), axis=0)  # 5x1xHxW

    preds = np.zeros(targets.shape).astype(np.float32)  # 5x1xHxW
    preds = preds + 1j * preds

    for i, noise_level in enumerate(noise_levels):
        pred = (
            target
            + np.random.randn(*target.shape).astype(np.complex64) * noise_level * target.mean()
        )
        preds[i, 0, :, :] = pred

    targets = torch.as_tensor(targets)
    preds = torch.as_tensor(preds)

    metric(preds, targets)
    out = metric.compute().squeeze(-1)
    sorted_out, _ = torch.sort(out, 0)

    assert torch.allclose(sorted_out, out)

    # Real valued and RGB mode
    metric = LPIPS(mode="rgb")
    noise_levels = [0, 0.01, 0.05, 0.1, 0.25]

    target = data.astronaut().astype(np.float32)  # H x W x 3
    target = np.transpose(target[np.newaxis, :, :, :], (0, 3, 1, 2))  # 1 x 3 x H x W
    targets = np.repeat(target, len(noise_levels), axis=0)  # 5 x 3 x H x W

    preds = np.zeros(targets.shape).astype(np.float32)  # 5 x 3 x H x W
    for i, noise_level in enumerate(noise_levels):
        pred = (
            target + np.random.randn(*target.shape).astype(np.float32) * noise_level * target.mean()
        )
        preds[i, :, :, :] = pred

    targets = torch.as_tensor(targets)
    preds = torch.as_tensor(preds)

    metric(preds, targets)
    out = metric.compute().squeeze(-1)
    sorted_out, _ = torch.sort(out, 0)

    assert torch.allclose(sorted_out, out)


@unittest.skipIf(
    not env.package_available("lpips"),
    "LPIPS metric requires that lpips is installed.",
)
def test_LPIPS_mode():
    """Test that the rgb and grayscale modes are handled correctly"""

    targets = np.random.randn(5, 1, 256, 256).astype(np.float32)
    preds = np.random.randn(5, 1, 256, 256).astype(np.float32)
    targets_torch = torch.as_tensor(targets)
    preds_torch = torch.as_tensor(preds)

    metric = LPIPS(mode="grayscale")
    metric(preds_torch, targets_torch)
    grayscale_out = metric.compute()

    targets = np.repeat(targets, 3, axis=1)
    preds = np.repeat(preds, 3, axis=1)
    targets_torch = torch.as_tensor(targets)
    preds_torch = torch.as_tensor(preds)

    metric = LPIPS(mode="rgb")
    metric(preds_torch, targets_torch)
    rgb_out = metric.compute()

    assert torch.allclose(grayscale_out, rgb_out)


@unittest.skipIf(
    not env.package_available("lpips"),
    "LPIPS metric requires that lpips is installed.",
)
def test_LPIPS_batch_channel_shape():
    """Test that multiple channel dimensions are handled properly in grayscale mode"""

    targets = torch.randn((5, 4, 256, 256))
    targets = targets + 1j * targets
    preds = torch.randn((5, 4, 256, 256))
    preds = preds + 1j * preds

    targets_torch = torch.as_tensor(targets)
    preds_torch = torch.as_tensor(preds)

    metric = LPIPS(mode="grayscale")
    metric(preds_torch, targets_torch)
    out = metric.compute()

    assert out.shape == (5, 4)

    targets_torch = targets_torch.view(20, 1, 256, 256)
    preds_torch = preds_torch.view(20, 1, 256, 256)

    metric = LPIPS(mode="grayscale")
    metric(preds_torch, targets_torch)
    out_reshaped = metric.compute()

    assert out_reshaped.shape == (20, 1)

    out_reshaped = out_reshaped.view(5, 4)

    assert torch.allclose(out, out_reshaped)
