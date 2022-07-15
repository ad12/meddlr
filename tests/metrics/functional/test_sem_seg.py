import numpy as np
import torch
import torch.nn.functional as F

from meddlr.metrics.functional.sem_seg import assd


def _generate_3d_mock_masks(num_classes=4):
    k = 5
    ndim = 3
    shape = (20,) * ndim

    y_true = []
    for _ in range(num_classes):
        r = (num_classes + 1) * 6
        grids = torch.meshgrid(*[torch.arange(s) for s in shape])
        x, y, z = tuple(g.float() - s // 2 for g, s in zip(grids, shape))
        mask = torch.sqrt(x**2 + y**2 + z**2) <= r
        y_true.append(mask)
    y_true = torch.stack(y_true, dim=0).unsqueeze(0).float()

    kernel = torch.randn(num_classes, num_classes, k, k, k)
    kernel = kernel / kernel.sum((-1, -2, -3), keepdim=True)
    y_pred = F.conv3d(y_true, weight=kernel, padding=[k // 2 for _ in range(ndim)])
    y_pred = y_pred > 0.5
    return y_pred, y_true


def test_assd_crop():
    """Test that cropping method to speed up computation gives same result."""
    y_pred, y_true = _generate_3d_mock_masks()

    expected = assd(y_pred, y_true, crop=False)
    out = assd(y_pred, y_true, crop=True)
    assert np.allclose(out, expected)

    spacing = (0.4, 1.0, 2.0)
    expected = assd(y_pred, y_true, crop=False, spacing=spacing)
    out = assd(y_pred, y_true, crop=True, spacing=spacing)
    assert np.allclose(out, expected)
