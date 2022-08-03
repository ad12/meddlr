import numpy as np
import torch
from skimage import data

from meddlr.metrics.SSFD import SSFD


def test_SSFD_noise():
    """Test that LPIPS increases as Gaussian noise increases"""

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
    """Test that the rgb and grayscale modes are handled correctly"""

    targets = torch.randn((5, 1, 256, 256))
    targets = targets + 1j * targets
    preds = torch.randn((5, 1, 256, 256))
    preds = preds + 1j * preds

    targets_torch = torch.as_tensor(targets)
    preds_torch = torch.as_tensor(preds)

    metric = SSFD(mode="grayscale")
    metric(preds_torch, targets_torch)
    grayscale_out = metric.compute()

    targets = np.repeat(targets, 3, axis=1)
    preds = np.repeat(preds, 3, axis=1)
    targets_torch = torch.as_tensor(targets)
    preds_torch = torch.as_tensor(preds)

    metric = SSFD(mode="rgb")
    metric(preds_torch, targets_torch)
    rgb_out = metric.compute()

    assert torch.allclose(grayscale_out, rgb_out)


def test_SSFD_batch_channel_shape():
    """Test that multiple channel dimensions are handled properly in grayscale mode"""

    targets = torch.randn((5, 4, 256, 256))
    targets = targets + 1j * targets
    preds = torch.randn((5, 4, 256, 256))
    preds = preds + 1j * preds

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
