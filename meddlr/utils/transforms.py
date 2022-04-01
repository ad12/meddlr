from typing import Sequence

import torch
from torch import nn

import meddlr.ops as oF
from meddlr.ops import complex as cplx
from meddlr.utils import env
from meddlr.utils.deprecated import deprecated

if env.pt_version() >= [1, 6]:
    import torch.fft


@deprecated(vremove="0.1.0", replacement="forward.SenseModel")
class SenseModel(nn.Module):
    """
    A module that computes forward and adjoint SENSE operations.

    The forward operation converts a complex image -> multi-coil kspace.
    The adjoint operation converts multi-coil kspace -> a complex image.

    This module also supports multiple sensitivity maps. This is useful if
    you would like to generate images from multiple estimated sensitivity maps.
    This module also works with single coil inputs as long as the #coils dimension
    is set to 1.

    Attributes:
        maps (torch.Tensor): Sensitivity maps. Shape ``(B, H, W, #coils, #maps, [2])``.
        weights (torch.Tensor, optional): Undersampling masks (if applicable).
            Shape ``(B, H, W)`` or ``(B, H, W, #coils, #coils)``.
    """

    def __init__(self, maps: torch.Tensor, weights: torch.Tensor = None):
        """
        Args:
            maps (torch.Tensor): Sensitivity maps.
            weights (torch.Tensor): Undersampling masks.
                If ``None``, it is assumed that inputs are fully-sampled.
        """
        super().__init__()

        self.maps = maps
        if weights is None:
            self.weights = 1.0
        else:
            self.weights = weights

    def _adjoint_op(self, kspace):
        """
        Args:
            kspace: Shape (B,H,W,#coils,[2])
        Returns:
            image: Shape (B,H,W,#maps,[2])
        """
        image = ifft2(self.weights * kspace)
        if cplx.is_complex_as_real(kspace):
            image = cplx.mul(image.unsqueeze(-2), cplx.conj(self.maps))  # [B,...,#coils,#maps,2]
            return image.sum(-3)
        else:
            image = cplx.mul(image.unsqueeze(-1), cplx.conj(self.maps))  # [B,...,#coils,#maps,1]
            return image.sum(-2)

    def _forward_op(self, image):
        """
        Args:
            image: Shape (B,H,W,#maps,[2])
        Returns:
            kspace: Shape (B,H,W,#coils,[2])
        """
        if cplx.is_complex_as_real(image):
            kspace = cplx.mul(image.unsqueeze(-3), self.maps)  # [B,...,1,#maps,2]
            kspace = self.weights * fft2(kspace.sum(-2))  # [B,...,#coils,2]
        else:
            kspace = cplx.mul(image.unsqueeze(-2), self.maps)
            kspace = self.weights * fft2(kspace.sum(-1))
        return kspace

    def forward(self, input: torch.Tensor, adjoint: bool = False):
        """Run forward or adjoint SENSE operation on the input.

        Depending on if ``adjoint=True``, the input should either be the
        k-space or the complex image. The shapes for these are as follows:
            - kspace: ``(B, H, W, #coils, [2])
            - image: ``(B, H, W, #maps, [2])``

        Args:
            input (torch.Tensor): If ``adjoint=True``, this is the multi-coil k-space,
                else it is the image.
            adjoint (bool, optional): If ``True``, use adjoint operation.

        Returns:
            torch.Tensor: If ``adjoint=True``, the image, else multi-coil k-space.
        """
        if adjoint:
            output = self._adjoint_op(input)
        else:
            output = self._forward_op(input)
        return output


@deprecated(vremove="0.1.0", replacement="ops.fft2c")
def fft2(data):
    """
    Apply centered 2 dimensional Fast Fourier Transform.
    Args:
        data (torch.Tensor): Complex valued input data with the last dimension
            containing real and imaginary components.
        dims (2-tuple): Containing spatial dimension indices.
    Returns:
        torch.Tensor: The FFT of the input.
    """

    assert data.size(-1) == 2 or env.supports_cplx_tensor()
    if data.size(-1) != 2:
        # Complex tensors supported
        assert env.supports_cplx_tensor(), torch.__version__  # torch.__version__ >= 1.7
        ndims = len(list(data.size()))
        dims = (1, 2)

        data = ifftshift(data, dim=dims)
        data = torch.fft.fftn(data, dim=dims, norm="ortho")
        data = fftshift(data, dim=dims)
        return data

    ndims = len(list(data.size()))

    if ndims == 5:
        data = data.permute(0, 3, 1, 2, 4)
    elif ndims == 6:
        data = data.permute(0, 3, 4, 1, 2, 5)
    else:
        raise ValueError("fft2: ndims > 6 not supported!")

    data = ifftshift(data, dim=(-3, -2))
    data = torch.Tensor.fft(data, 2, normalized=True)
    data = fftshift(data, dim=(-3, -2))

    if ndims == 5:
        data = data.permute(0, 2, 3, 1, 4)
    elif ndims == 6:
        data = data.permute(0, 3, 4, 1, 2, 5)
    else:
        raise ValueError("fft2: ndims > 6 not supported!")

    return data


@deprecated(vremove="0.1.0", replacement="ops.ifft2c")
def ifft2(data):
    """
    Apply centered 2-dimensional Inverse Fast Fourier Transform.
    Args:
        data (torch.Tensor): Complex valued input data with the last dimension
            containing real and imaginary components.
        dims (2-tuple): Containing spatial dimension indices.
    Returns:
        torch.Tensor: The IFFT of the input.
    """
    assert data.size(-1) == 2 or env.supports_cplx_tensor()
    if data.size(-1) != 2:
        # Complex tensors supported
        assert env.supports_cplx_tensor(), torch.__version__  # torch.__version__ >= 1.7
        ndims = len(list(data.size()))
        dims = (1, 2)

        data = ifftshift(data, dim=dims)
        data = torch.fft.ifftn(data, dim=dims, norm="ortho")
        data = fftshift(data, dim=dims)
        return data

    ndims = len(list(data.size()))

    if ndims == 5:
        data = data.permute(0, 3, 1, 2, 4)
    elif ndims == 6:
        data = data.permute(0, 3, 4, 1, 2, 5)
    else:
        raise ValueError("ifft2: ndims > 6 not supported!")

    data = ifftshift(data, dim=(-3, -2))
    data = torch.ifft(data, 2, normalized=True)
    data = fftshift(data, dim=(-3, -2))

    if ndims == 5:
        data = data.permute(0, 2, 3, 1, 4)
    elif ndims == 6:
        data = data.permute(0, 3, 4, 1, 2, 5)
    else:
        raise ValueError("ifft2: ndims > 6 not supported!")

    return data


@deprecated(vremove="0.1.0", replacement="ops.complex.rss")
def root_sum_of_squares(x, dim=0):
    """
    Compute the root sum-of-squares (RSS) transform along a given dimension of
    a complex-valued tensor.
    """
    assert x.size(-1) == 2
    return torch.sqrt((x**2).sum(dim=-1).sum(dim))


@deprecated(vremove="0.1.0", replacement="ops.time_average")
def time_average(data, dim, eps=1e-6, keepdim=True):
    """
    Computes time average across a specified axis.
    """
    return oF.time_average(data, dim, eps=eps, keepdim=keepdim)


@deprecated(vremove="0.1.0", replacement="ops.sliding_window")
def sliding_window(data, dim, window_size):
    """
    Computes sliding window with circular boundary conditions across a specified
    axis.
    """
    return oF.sliding_window(data, dim, window_size)


@deprecated(vremove="0.1.0", replacement="ops.center_crop")
def center_crop(data, shape):
    """
    Apply a center crop to a batch of images.
    Args:
        data (torch.Tensor): The input tensor to be center cropped.
        shape (list of ints): The output shape. If shape[dim] = -1, then no crop
            will be applied in that dimension.
    """
    return oF.center_crop(data, shape, include_batch=True)


@deprecated(vremove="0.1.0", replacement="ops.complex.complex_center_crop_2d")
def complex_center_crop_2d(data, shape):
    """
    Apply a center crop to the input image or batch of complex images.
    Args:
        data (torch.Tensor): The complex input tensor to be center cropped. It should
            have at least 3 dimensions and the cropping is applied along dimensions
            -3 and -2 and the last dimensions should have a size of 2.
        shape (int, int): The output shape. The shape should be smaller than the
            corresponding dimensions of data.
    Returns:
        torch.Tensor: The center cropped image
    """
    assert 0 < shape[0] <= data.shape[-3]
    assert 0 < shape[1] <= data.shape[-2]
    w_from = (data.shape[-3] - shape[0]) // 2
    h_from = (data.shape[-2] - shape[1]) // 2
    w_to = w_from + shape[0]
    h_to = h_from + shape[1]
    return data[..., w_from:w_to, h_from:h_to, :]


@deprecated(vremove="0.1.0", replacement="ops.normalize")
def normalize(data, mean, stddev, eps=0.0):
    """
    Normalize the given tensor using:
        (data - mean) / (stddev + eps)
    Args:
        data (torch.Tensor): Input data to be normalized
        mean (float): Mean value
        stddev (float): Standard deviation
        eps (float): Added to stddev to prevent dividing by zero
    Returns:
        torch.Tensor: Normalized tensor
    """
    return (data - mean) / (stddev + eps)


@deprecated(vremove="0.1.0", replacement="ops.normalize_instance")
def normalize_instance(data, eps=0.0):
    """
    Normalize the given tensor using:
        (data - mean) / (stddev + eps)
    where mean and stddev are computed from the data itself.
    Args:
        data (torch.Tensor): Input data to be normalized
        eps (float): Added to stddev to prevent dividing by zero
    Returns:
        torch.Tensor: Normalized tensor
    """
    mean = data.mean()
    std = data.std()
    return normalize(data, mean, std, eps), mean, std


# Helper functions


@deprecated(vremove="0.1.0", replacement="ops.complex.center_crop")
def roll(x, shift, dim):
    """
    Similar to np.roll but applies to PyTorch Tensors
    """
    return oF.roll(x, shift, dim)


@deprecated(vremove="0.1.0", replacement="ops.fftshift")
def fftshift(x, dim=None):
    """
    Similar to np.fft.fftshift but applies to PyTorch Tensors
    """
    return oF.fftshift(x, dim=dim)


@deprecated(vremove="0.1.0", replacement="ops.ifftshift")
def ifftshift(x, dim=None):
    """
    Similar to np.fft.ifftshift but applies to PyTorch Tensors
    """
    return oF.ifftshift(x, dim=dim)


@deprecated(vremove="0.1.0", replacement="ops.pad")
def pad(x: torch.Tensor, shape: Sequence[int], mode="constant", value=0):
    """
    Args:
        x: Input tensor of shape (B, ...)
        shape: Shape to zero pad to. Use `None` to skip padding certain dimensions.
    Returns:
    """
    return oF.pad(x, shape, mode=mode, value=value)


@deprecated(vremove="0.1.0", replacement="ops.zero_pad")
def zero_pad(x: torch.Tensor, shape: Sequence[int]):
    return oF.zero_pad(x, shape)
