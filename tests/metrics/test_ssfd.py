import numpy as np
import torch
from skimage import data

from meddlr.metrics.ssfd import SSFD


def test_SSFD_noise():
    """Test that SSFD increases as Gaussian noise increases."""

    metric = SSFD()
    noise_levels = [0, 0.01, 0.05, 0.1, 0.25]

    target = data.camera().astype(np.float32)
    target = target + 1j * target
    targets = np.repeat(target[np.newaxis, np.newaxis, :, :], len(noise_levels), axis=0)

    preds = np.zeros(targets.shape).astype(np.float32)
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


def test_SSFD_mode():
    """Test that the rgb and grayscale modes are handled correctly."""

    targets = data.astronaut().astype(np.float32)  # H x W x 3
    targets = np.transpose(targets[np.newaxis, :, :, :], (0, 3, 1, 2))  # 1 x 3 x H x W

    preds = targets + np.random.randn(*targets.shape).astype(np.float32) * targets.mean()
    targets = torch.as_tensor(targets)
    preds = torch.as_tensor(preds)

    metric = SSFD(mode="rgb")
    metric(preds, targets)
    rgb_out = metric.compute()

    targets = torch.mean(targets, axis=1, keepdim=True)  # 1 x 1 x H x W
    preds = torch.mean(preds, axis=1, keepdim=True)  # 1 x 1 x H x W

    metric = SSFD(mode="grayscale")
    metric(preds, targets)
    grayscale_out = metric.compute()

    assert torch.allclose(grayscale_out, rgb_out)


def test_SSFD_batch_channel_shape():
    """Test that multiple channel dimensions are handled properly in grayscale mode."""

    targets = np.random.randn(5, 4, 256, 256).astype(np.complex64)
    preds = np.random.randn(5, 4, 256, 256).astype(np.complex64)
    targets_torch = torch.as_tensor(targets)
    preds_torch = torch.as_tensor(preds)

    metric = SSFD(mode="grayscale")
    metric(preds_torch, targets_torch)
    out = metric.compute()

    assert out.shape == (5, 4)

    targets_torch = targets_torch.view(20, 1, 256, 256)
    preds_torch = preds_torch.view(20, 1, 256, 256)

    metric = SSFD(mode="grayscale")
    metric(preds_torch, targets_torch)
    out_reshaped = metric.compute()

    assert out_reshaped.shape == (20, 1)

    out_reshaped = out_reshaped.view(5, 4)

    assert torch.allclose(out, out_reshaped)


def test_SSFD_layers():
    """Test that SSFD can handle different layer names and sums them correctly."""

    # Complex data
    targets = np.random.randn(3, 2, 320, 320).astype(np.complex64)
    preds = np.random.randn(3, 2, 320, 320).astype(np.complex64)
    targets = torch.as_tensor(targets)
    preds = torch.as_tensor(preds)

    layers = ["block1_relu2", "block2_relu2", "block3_relu2", "block4_relu2", "block5_relu2"]
    total = 0
    for layer in layers:
        metric = SSFD(layer_names=[layer])
        metric(preds, targets)
        out = metric.compute()
        total += out

    metric = SSFD(layer_names=layers)
    metric(preds, targets)
    total_one_pass = metric.compute()

    assert torch.allclose(total, total_one_pass)

    # Real data
    targets = np.random.randn(3, 2, 320, 320).astype(np.float32)
    preds = np.random.randn(3, 2, 320, 320).astype(np.float32)
    targets = torch.as_tensor(targets)
    preds = torch.as_tensor(preds)

    layers = ["block1_relu2", "block2_relu2", "block3_relu2", "block4_relu2", "block5_relu2"]
    total = 0
    for layer in layers:
        metric = SSFD(layer_names=[layer])
        metric(preds, targets)
        out = metric.compute()
        total += out

    metric = SSFD(layer_names=layers)
    metric(preds, targets)
    total_one_pass = metric.compute()

    assert torch.allclose(total, total_one_pass)
