"""Metric utilities.

All metrics take a reference scan (ref) and an output/reconstructed scan (x).
Both should be tensors with the last dimension equal to 2 (real/imaginary
channels).
"""
import torch
from ss_recon.utils import complex_utils as cplx
from skimage.metrics import structural_similarity


def compute_mse(ref: torch.Tensor, x: torch.Tensor, is_batch=False):
    assert ref.shape[-1] == 2
    assert x.shape[-1] == 2
    squared_err = cplx.abs(x - ref) ** 2
    shape = (x.shape[0], -1) if is_batch else -1
    return torch.mean(squared_err.view(shape), dim=-1)


def compute_l2(ref: torch.Tensor, x: torch.Tensor, is_batch=False):
    """
    Args:
        ref (torch.Tensor): The target. Shape (...)x2
        x (torch.Tensor): The prediction. Same shape as `ref`.
    """
    assert ref.shape[-1] == 2
    assert x.shape[-1] == 2
    return torch.sqrt(compute_mse(ref, x, is_batch=is_batch))


def compute_psnr(ref: torch.Tensor, x: torch.Tensor, is_batch=False):
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
    shape = (x.shape[0], -1) if is_batch else -1
    return 20 * torch.log10(cplx.abs(ref).max() / l2)


def compute_nrmse(ref, x, is_batch=False):
    """Compute normalized root mean square error.
    The norm of reference is used to normalize the metric.

    Args:
        ref (torch.Tensor): The target. Shape (...)x2
        x (torch.Tensor): The prediction. Same shape as `ref`.
    """
    assert ref.shape[-1] == 2
    assert x.shape[-1] == 2
    rmse = compute_l2(ref, x, is_batch=is_batch)
    shape = (x.shape[0], -1) if is_batch else -1
    norm = torch.sqrt(torch.mean((cplx.abs(ref) ** 2).view(shape), dim=-1))

    return rmse / norm


def compute_ssim(
    ref: torch.Tensor,
    x: torch.Tensor,
    multichannel: bool = False,
    data_range = None,
    **kwargs,
):
    """Compute structural similarity index metric. Does not preserve autograd.

    Based on implementation of Wang et. al. [1]_

    The image is first converted to magnitude image and normalized
    before the metric is computed.

    Args:
        ref (torch.Tensor): The target. Shape (...)x2
        x (torch.Tensor): The prediction. Same shape as `ref`.
        multichannel (bool, optional): If `True`, computes ssim for real and
            imaginary channels separately and then averages the two.
        data_range(float, optional): The data range of the input image
        (distance between minimum and maximum possible values). By default,
        this is estimated from the image data-type.

    References:
    .. [1] Wang, Z., Bovik, A. C., Sheikh, H. R., & Simoncelli, E. P.
       (2004). Image quality assessment: From error visibility to
       structural similarity. IEEE Transactions on Image Processing,
       13, 600-612.
       https://ece.uwaterloo.ca/~z70wang/publications/ssim.pdf,
       :DOI:`10.1109/TIP.2003.819861`
    """
    gaussian_weights = kwargs.get("gaussian_weights", True)
    sigma = kwargs.get("sigma", 1.5)
    use_sample_covariance = kwargs.get("use_sample_covariance", False)

    assert ref.shape[-1] == 2
    assert x.shape[-1] == 2

    if not multichannel:
        ref = cplx.abs(ref)
        x = cplx.abs(x)

    x = x.squeeze().numpy()
    ref = ref.squeeze().numpy()

    if data_range in ("range", "ref-range"):
        data_range = ref.max() - ref.min()
    elif data_range in ("ref-maxval", "maxval"):
        data_range = ref.max()
    elif data_range == "x-range":
        data_range = x.max() - x.min()
    elif data_range == "x-maxval":
        data_range = x.max()

    return structural_similarity(
        ref,
        x,
        data_range=data_range,
        gaussian_weights=gaussian_weights,
        sigma=sigma,
        use_sample_covariance=use_sample_covariance,
        multichannel=multichannel
    )


compute_rmse = compute_l2
