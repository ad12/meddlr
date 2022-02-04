"""Metric utilities.

All metrics take a reference scan (ref) and an output/reconstructed scan (x).
Both should be tensors with the last dimension equal to 2 (real/imaginary
channels).
"""
from typing import Sequence

import numpy as np
import scipy as scp
import torch
from skimage.metrics import structural_similarity

import meddlr.metrics.functional as mF
from meddlr.metrics.metric import Metric
from meddlr.ops import complex as cplx
from meddlr.utils.deprecated import deprecated

# Mapping from str to complex function name.
_IM_TYPES_TO_FUNCS = {
    "magnitude": cplx.abs,
    "mag": cplx.abs,
    "abs": cplx.abs,
    "phase": cplx.angle,
    "angle": cplx.angle,
    "real": cplx.real,
    "imag": cplx.imag,
}

__all__ = ["PSNR", "MAE", "MSE", "NRMSE", "RMSE", "SSIM"]


class PSNR(Metric):
    """Peak signal-to-noise ratio with complex-valued support.

    :math:`PSNR = 20 * log_{10}(\\frac{max(|x_{gt}|)}{||x_{pred} - x_{gt}||_2})`

    This implementation supports complex tensors.
    ``im_type`` controls how the complex tensor should be processed:

        - ``'magnitude'``: :math:`x_{pred}` and :math:`x_{gt}` are converted to magnitude images.
        - ``'phase'``: :math:`x_{pred}` and :math:`x_{gt}` are converted to phase images.
        - ``'real'``: Real components of :math:`x_{pred}` and :math:`x_{gt}` are used.
        - ``'imag'``: Imaginary components of :math:`x_{pred}` and :math:`x_{gt}` are used.

    Attributes:
        im_type (str): The type of the complex image to compute the metric on.
            This only applies to complex tensors.
        channel_names (Sequence[str]): The names of the channels in the input.
    """

    is_differentiable = True
    higher_is_better = True

    def __init__(
        self,
        im_type: str = None,
        channel_names: Sequence[str] = None,
        reduction="none",
        compute_on_step: bool = False,
        dist_sync_on_step: bool = False,
        process_group: bool = None,
        dist_sync_fn: bool = None,
    ):
        super().__init__(
            channel_names=channel_names,
            units="dB",
            reduction=reduction,
            compute_on_step=compute_on_step,
            dist_sync_on_step=dist_sync_on_step,
            process_group=process_group,
            dist_sync_fn=dist_sync_fn,
        )

        self.im_type = im_type

    def func(self, preds, targets) -> torch.Tensor:
        return mF.psnr(preds, targets, im_type=self.im_type)


class MAE(Metric):
    """Mean absolute error with complex-valued support.

    :math:`MAE = \\frac{1}{N} \sum_{i=1}^{N} |x_{pred} - x_{gt}|`.

    This implementation supports complex tensors.
    ``im_type`` controls how the complex tensor should be processed:

        - ``'magnitude'``: :math:`x_{pred}` and :math:`x_{gt}` are converted to magnitude images.
        - ``'phase'``: :math:`x_{pred}` and :math:`x_{gt}` are converted to phase images.
        - ``'real'``: Real components of :math:`x_{pred}` and :math:`x_{gt}` are used.
        - ``'imag'``: Imaginary components of :math:`x_{pred}` and :math:`x_{gt}` are used.

    Attributes:
        im_type (str): The type of the complex image to compute the metric on.
            This only applies to complex tensors.
        channel_names (Sequence[str]): The names of the channels in the input.
    """

    is_differentiable = True
    higher_is_better = False

    def __init__(
        self,
        im_type: str = None,
        channel_names: Sequence[str] = None,
        reduction="none",
        compute_on_step: bool = False,
        dist_sync_on_step: bool = False,
        process_group: bool = None,
        dist_sync_fn: bool = None,
    ):
        super().__init__(
            channel_names=channel_names,
            units="",
            reduction=reduction,
            compute_on_step=compute_on_step,
            dist_sync_on_step=dist_sync_on_step,
            process_group=process_group,
            dist_sync_fn=dist_sync_fn,
        )

        self.im_type = im_type

    def func(self, preds, targets) -> torch.Tensor:
        return mF.mae(preds, targets, im_type=self.im_type)


class MSE(Metric):
    """Mean squared error with complex-valued support.

    :math:`MSE = ||x_{pred} - x_{gt}||_2^2`.

    This implementation supports complex tensors.
    ``im_type`` controls how the complex tensor should be processed:

        - ``'magnitude'``: :math:`x_{pred}` and :math:`x_{gt}` are converted to magnitude images.
        - ``'phase'``: :math:`x_{pred}` and :math:`x_{gt}` are converted to phase images.
        - ``'real'``: Real components of :math:`x_{pred}` and :math:`x_{gt}` are used.
        - ``'imag'``: Imaginary components of :math:`x_{pred}` and :math:`x_{gt}` are used.

    Attributes:
        im_type (str): The type of the complex image to compute the metric on.
            This only applies to complex tensors.
        channel_names (Sequence[str]): The names of the channels in the input.
    """

    is_differentiable = True
    higher_is_better = False

    def __init__(
        self,
        im_type: str = None,
        channel_names: Sequence[str] = None,
        reduction="none",
        compute_on_step: bool = False,
        dist_sync_on_step: bool = False,
        process_group: bool = None,
        dist_sync_fn: bool = None,
    ):
        super().__init__(
            channel_names=channel_names,
            units="",
            reduction=reduction,
            compute_on_step=compute_on_step,
            dist_sync_on_step=dist_sync_on_step,
            process_group=process_group,
            dist_sync_fn=dist_sync_fn,
        )

        self.im_type = im_type

    def func(self, preds, targets) -> torch.Tensor:
        return mF.mse(preds, targets, im_type=self.im_type)


class NRMSE(Metric):
    """Normalized root-mean-squared error with complex-valued support.

    :math:`NRMSE = \\frac{||x_{pred} - x_{gt}||_2}{||x_{gt}||_2}`.

    This implementation supports complex tensors.
    ``im_type`` controls how the complex tensor should be processed:

        - ``'magnitude'``: :math:`x_{pred}` and :math:`x_{gt}` are converted to magnitude images.
        - ``'phase'``: :math:`x_{pred}` and :math:`x_{gt}` are converted to phase images.
        - ``'real'``: Real components of :math:`x_{pred}` and :math:`x_{gt}` are used.
        - ``'imag'``: Imaginary components of :math:`x_{pred}` and :math:`x_{gt}` are used.

    Attributes:
        im_type (str): The type of the complex image to compute the metric on.
            This only applies to complex tensors.
        channel_names (Sequence[str]): The names of the channels in the input.
    """

    is_differentiable = True
    higher_is_better = False

    def __init__(
        self,
        im_type: str = None,
        channel_names: Sequence[str] = None,
        reduction="none",
        compute_on_step: bool = False,
        dist_sync_on_step: bool = False,
        process_group: bool = None,
        dist_sync_fn: bool = None,
    ):
        super().__init__(
            channel_names=channel_names,
            units="",
            reduction=reduction,
            compute_on_step=compute_on_step,
            dist_sync_on_step=dist_sync_on_step,
            process_group=process_group,
            dist_sync_fn=dist_sync_fn,
        )

        self.im_type = im_type

    def func(self, preds, targets) -> torch.Tensor:
        return mF.nrmse(preds, targets, im_type=self.im_type)


nRMSE = NRMSE


class RMSE(Metric):
    """Root-mean-squared error with complex-valued support.

    :math:`RMSE = ||x_{pred} - x_{gt}||_2`.

    This implementation supports complex tensors.
    ``im_type`` controls how the complex tensor should be processed:

        - ``'magnitude'``: :math:`x_{pred}` and :math:`x_{gt}` are converted to magnitude images.
        - ``'phase'``: :math:`x_{pred}` and :math:`x_{gt}` are converted to phase images.
        - ``'real'``: Real components of :math:`x_{pred}` and :math:`x_{gt}` are used.
        - ``'imag'``: Imaginary components of :math:`x_{pred}` and :math:`x_{gt}` are used.

    Attributes:
        im_type (str): The type of the complex image to compute the metric on.
            This only applies to complex tensors.
        channel_names (Sequence[str]): The names of the channels in the input.
    """

    is_differentiable = True
    higher_is_better = False

    def __init__(
        self,
        im_type: str = None,
        channel_names: Sequence[str] = None,
        reduction="none",
        compute_on_step: bool = False,
        dist_sync_on_step: bool = False,
        process_group: bool = None,
        dist_sync_fn: bool = None,
    ):
        super().__init__(
            channel_names=channel_names,
            units="",
            reduction=reduction,
            compute_on_step=compute_on_step,
            dist_sync_on_step=dist_sync_on_step,
            process_group=process_group,
            dist_sync_fn=dist_sync_fn,
        )

        self.im_type = im_type

    def func(self, preds, targets) -> torch.Tensor:
        return mF.rmse(preds, targets, im_type=self.im_type)


class SSIM(Metric):
    """Structural similarity index measure with complex-valued support.

    This implementation of pSNR supports complex tensors.
    ``im_type`` controls how the complex tensor should be processed:

        - ``'magnitude'``: :math:`x_{pred}` and :math:`x_{gt}` are converted to magnitude images.
        - ``'phase'``: :math:`x_{pred}` and :math:`x_{gt}` are converted to phase images.
        - ``'real'``: Real components of :math:`x_{pred}` and :math:`x_{gt}` are used.
        - ``'imag'``: Imaginary components of :math:`x_{pred}` and :math:`x_{gt}` are used.

    Attributes:
        method (str): The method to use for computing the SSIM.
            Defaults to ``'wang'``.
        im_type (str): The type of the complex image to compute the metric on.
            This only applies to complex tensors.
        channel_names (Sequence[str]): The names of the channels in the input.
    """

    is_differentiable = True
    higher_is_better = True

    def __init__(
        self,
        method: str = "wang",
        im_type: str = "magnitude",
        channel_names: Sequence[str] = None,
        reduction="none",
        compute_on_step: bool = False,
        dist_sync_on_step: bool = False,
        process_group: bool = None,
        dist_sync_fn: bool = None,
    ):
        super().__init__(
            channel_names=channel_names,
            units="",
            reduction=reduction,
            compute_on_step=compute_on_step,
            dist_sync_on_step=dist_sync_on_step,
            process_group=process_group,
            dist_sync_fn=dist_sync_fn,
        )
        self.method = method
        self.im_type = im_type

    def func(self, preds, targets) -> torch.Tensor:
        return mF.ssim(
            preds,
            targets,
            method=self.method,
            im_type=self.im_type,
        )


@deprecated(vremove="0.1.0", replacement="metrics.functional.mse")
def compute_mse(ref: torch.Tensor, x: torch.Tensor, is_batch=False, magnitude=False):
    if cplx.is_complex(ref):
        ref = torch.view_as_real(ref)
        x = torch.view_as_real(x)

    assert ref.shape[-1] == 2
    assert x.shape[-1] == 2
    if magnitude:
        squared_err = torch.abs(cplx.abs(x) - cplx.abs(ref)) ** 2
    else:
        squared_err = cplx.abs(x - ref) ** 2
    shape = (x.shape[0], -1) if is_batch else -1
    return torch.mean(squared_err.view(shape), dim=-1)


def compute_l2(ref: torch.Tensor, x: torch.Tensor, is_batch=False, magnitude=False):
    """
    Args:
        ref (torch.Tensor): The target. Shape (...)x2
        x (torch.Tensor): The prediction. Same shape as `ref`.
    """
    if cplx.is_complex(ref):
        ref = torch.view_as_real(ref)
        x = torch.view_as_real(x)

    assert ref.shape[-1] == 2
    assert x.shape[-1] == 2
    return torch.sqrt(compute_mse(ref, x, is_batch=is_batch, magnitude=magnitude))


@deprecated(vremove="0.1.0", replacement="metrics.functional.psnr")
def compute_psnr(ref: torch.Tensor, x: torch.Tensor, is_batch=False, magnitude=False):
    """Compute peak to signal to noise ratio of magnitude image.

    Args:
        ref (torch.Tensor): The target. Shape (...)x2
        x (torch.Tensor): The prediction. Same shape as `ref`.

    Returns:
        Tensor: Scalar in db
    """
    if cplx.is_complex(ref):
        ref = torch.view_as_real(ref)
        x = torch.view_as_real(x)

    assert ref.shape[-1] == 2
    assert x.shape[-1] == 2
    assert not is_batch, "is_batch not supported"
    l2 = compute_l2(ref, x, magnitude=magnitude, is_batch=False)
    # shape = (x.shape[0], -1) if is_batch else -1
    return 20 * torch.log10(cplx.abs(ref).max() / l2)


@deprecated(vremove="0.1.0", replacement="metrics.functional.nrmse")
def compute_nrmse(ref, x, is_batch=False, magnitude=False):
    """Compute normalized root mean square error.
    The norm of reference is used to normalize the metric.

    Args:
        ref (torch.Tensor): The target. Shape (...)x2
        x (torch.Tensor): The prediction. Same shape as `ref`.
    """
    if cplx.is_complex(ref):
        ref = torch.view_as_real(ref)
        x = torch.view_as_real(x)

    assert ref.shape[-1] == 2
    assert x.shape[-1] == 2
    rmse = compute_l2(ref, x, is_batch=is_batch, magnitude=magnitude)
    shape = (x.shape[0], -1) if is_batch else -1
    norm = torch.sqrt(torch.mean((cplx.abs(ref) ** 2).view(shape), dim=-1))

    return rmse / norm


@deprecated(vremove="0.1.0", replacement="metrics.functional.ssim")
def compute_ssim(
    ref: torch.Tensor,
    x: torch.Tensor,
    multichannel: bool = False,
    data_range=None,
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
    gaussian_weights = kwargs.pop("gaussian_weights", True)
    sigma = kwargs.pop("sigma", 1.5)
    use_sample_covariance = kwargs.pop("use_sample_covariance", False)

    if cplx.is_complex(ref):
        ref = torch.view_as_real(ref)
        x = torch.view_as_real(x)

    assert ref.shape[-1] == 2
    assert x.shape[-1] == 2

    if not multichannel:
        ref = cplx.abs(ref)
        x = cplx.abs(x)

    if not x.is_contiguous():
        x = x.contiguous()
    if not ref.is_contiguous():
        ref = ref.contiguous()

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
        multichannel=multichannel,
        **kwargs,
    )


def compute_vifp_mscale(
    ref: torch.Tensor,
    x: torch.Tensor,
    sigma_nsq: float = 2.0,
    eps: float = 1e-10,
    im_type: str = None,
):  # pragma: no-cover
    """Compute visual information fidelity (VIF) metric.

    This code is adapted from
    https://github.com/aizvorski/video-quality/blob/master/vifp.py

    Args:
        ref (torch.Tensor): The reference image. This can be complex.
        x (torch.Tensor): The target image. This can be complex.
        sigma_nsq (float, optional): The visual noise parameter.
            This may need to be fine-tuned over the dataset of interest.
        eps (float, optional): The threshold below which data is considered to be 0.
        im_type (str, optional): The image type to compute metric on.
            Either ``'magnitude'`` (default) to compute metric on magnitude images
            or ``'phase'`` to compute metric on phase/angle images.

    Returns:
        float: The metric value.

    Note:
        ``im_type`` is only valid if input is complex.
    """
    if cplx.is_complex(ref) or cplx.is_complex_as_real(ref):
        ref = _IM_TYPES_TO_FUNCS[im_type](ref)
    if cplx.is_complex(x) or cplx.is_complex_as_real(x):
        x = _IM_TYPES_TO_FUNCS[im_type](x)

    ref = np.squeeze(ref.numpy())
    x = np.squeeze(x.numpy())

    scale_val = 255.0 / ref.max()
    ref *= scale_val
    x *= scale_val

    num = 0.0
    den = 0.0
    for scale in range(1, 5):

        N = 2 ** (4 - scale + 1) + 1
        sd = N / 5.0

        if scale > 1:
            sl = tuple(slice(None, None, 2) if dim > 1 else slice(None) for dim in ref.shape)
            ref = scp.ndimage.gaussian_filter(ref, sd)
            x = scp.ndimage.gaussian_filter(x, sd)
            ref = ref[sl]
            x = x[sl]

        mu1 = scp.ndimage.gaussian_filter(ref, sd)
        mu2 = scp.ndimage.gaussian_filter(x, sd)
        mu1_sq = mu1 * mu1
        mu2_sq = mu2 * mu2
        mu1_mu2 = mu1 * mu2
        sigma1_sq = scp.ndimage.gaussian_filter(ref * ref, sd) - mu1_sq
        sigma2_sq = scp.ndimage.gaussian_filter(x * x, sd) - mu2_sq
        sigma12 = scp.ndimage.gaussian_filter(ref * x, sd) - mu1_mu2

        sigma1_sq[sigma1_sq < 0] = 0
        sigma2_sq[sigma2_sq < 0] = 0

        g = sigma12 / (sigma1_sq + eps)
        sv_sq = sigma2_sq - g * sigma12

        g[sigma1_sq < eps] = 0
        sv_sq[sigma1_sq < eps] = sigma2_sq[sigma1_sq < eps]
        sigma1_sq[sigma1_sq < eps] = 0

        g[sigma2_sq < eps] = 0
        sv_sq[sigma2_sq < eps] = 0

        sv_sq[g < 0] = sigma2_sq[g < 0]
        g[g < 0] = 0
        sv_sq[sv_sq <= eps] = eps

        num += np.sum(np.log10(1 + g * g * sigma1_sq / (sv_sq + sigma_nsq)))
        den += np.sum(np.log10(1 + sigma1_sq / sigma_nsq))

    vifp = num / den

    return vifp


compute_rmse = compute_l2
