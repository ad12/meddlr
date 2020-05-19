"""Metric utilities.

All metrics take a reference scan (ref) and an output/reconstructed scan (x).
Both should be tensors with the last dimension equal to 2 (real/imaginary
channels).
"""
import torch
from ss_recon.utils import complex_utils as cplx
from skimage.metrics import structural_similarity


def compute_mse(ref: torch.Tensor, x: torch.Tensor):
    assert ref.shape[-1] == 2
    assert x.shape[-1] == 2
    squared_err = cplx.abs(x - ref) ** 2
    return torch.mean(squared_err)


def compute_l2(ref: torch.Tensor, x: torch.Tensor):
    """
    Args:
        ref (torch.Tensor): The target. Shape (...)x2
        x (torch.Tensor): The prediction. Same shape as `ref`.
    """
    assert ref.shape[-1] == 2
    assert x.shape[-1] == 2
    return torch.sqrt(compute_mse(ref, x))


def compute_psnr(ref: torch.Tensor, x: torch.Tensor):
    """Compute peak to signal to noise ratio of magnitude image.

    Args:
        ref (torch.Tensor): The target. Shape (...)x2
        x (torch.Tensor): The prediction. Same shape as `ref`.

    Returns:
        Tensor: Scalar in db
    """
    assert ref.shape[-1] == 2
    assert x.shape[-1] == 2

    l2 = compute_l2(ref, x)
    return 20 * torch.log10(cplx.abs(ref).max() / l2)


def compute_nrmse(ref, x):
    """Compute normalized root mean square error.
    The norm of reference is used to normalize the metric.

    Args:
        ref (torch.Tensor): The target. Shape (...)x2
        x (torch.Tensor): The prediction. Same shape as `ref`.
    """
    assert ref.shape[-1] == 2
    assert x.shape[-1] == 2
    rmse = compute_l2(ref, x)
    norm = torch.sqrt(torch.mean(cplx.abs(ref) ** 2))

    return rmse / norm


def compute_ssim(
    ref: torch.Tensor,
    x: torch.Tensor,
    multichannel: bool = False,
):
    """Compute structural similarity index metric. Does not preserve autograd.

    Based on implementation of Wang et. al. [1]_

    The image is first converted to magnitude image and normalized
    before the metric is computed.

    Args:
        ref (torch.Tensor): The target. Shape (...)x2
        x (torch.Tensor): The prediction. Same shape as `ref`.
        multichannel (torch.Tensor): If `True`, computes ssim for real and
            imaginary channels separately and then averages the two.

    References:
    .. [1] Wang, Z., Bovik, A. C., Sheikh, H. R., & Simoncelli, E. P.
       (2004). Image quality assessment: From error visibility to
       structural similarity. IEEE Transactions on Image Processing,
       13, 600-612.
       https://ece.uwaterloo.ca/~z70wang/publications/ssim.pdf,
       :DOI:`10.1109/TIP.2003.819861`
    """
    assert ref.shape[-1] == 2
    assert x.shape[-1] == 2

    x = x.squeeze().numpy()
    ref = ref.squeeze().numpy()

    # if not data_range:
    #     data_range = ref.max() - ref.min()

    return structural_similarity(
        ref,
        x,
        # data_range=data_range,
        gaussian_weights=True,
        sigma=1.5,
        use_sample_covariance=False,
        multichannel=multichannel
    )
