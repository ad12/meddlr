"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""
from typing import Sequence, Union

import numpy as np
import torch
from torch import nn

from ss_recon.utils import complex_utils as cplx
from ss_recon.utils import env

if env.pt_version() >= [1,6]:
    import torch.fft


class SenseModel(nn.Module):
    """
    A module that computes forward and adjoint SENSE operations.
    """

    def __init__(self, maps, coord=None, weights=None):
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

    def forward(self, input, adjoint=False):
        if adjoint:
            output = self._adjoint_op(input)
        else:
            output = self._forward_op(input)
        return output


class NoiseModel():
    """A model that adds additive white noise.

    Note we do not store this as a module or else it would be saved to the model
    definition, which we dont want.
    """
    def __init__(self, std_devs: Union[float, Sequence[float]], seed=None):
        if not isinstance(std_devs, Sequence):
            std_devs = (std_devs,)
        elif len(std_devs) > 2:
            raise ValueError("`std_devs` must have 2 or fewer values")
        self.std_devs = std_devs

        # For reproducibility.
        g = torch.Generator()
        if seed:
            g = g.manual_seed(seed)
        self.generator = g
    
    def choose_std_dev(self):
        """Chooses a random acceleration rate given a range.
        """
        if not isinstance(self.std_devs, Sequence):
            return self.std_devs
        elif len(self.std_devs) == 1:
            return self.std_devs[0]

        std_range = self.std_devs[1] - self.std_devs[0]
        std_dev = self.std_devs[0] + std_range * torch.rand(1, generator=self.generator).item()
        return std_dev

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def forward(self, kspace, mask=None, seed=None, clone=True) -> torch.Tensor:
        """Performs augmentation on undersampled kspace mask."""
        if clone:
            kspace = kspace.clone()
        mask = cplx.get_mask(kspace)

        g = self.generator if seed is None else torch.Generator().manual_seed(seed)
        noise_std = self.choose_std_dev()
        noise = noise_std * torch.randn(kspace.size(), generator=g)
        masked_noise = noise * mask
        aug_kspace = kspace + masked_noise

        return aug_kspace


class ArrayToBlocks(nn.Module):
    def __init__(self, block_size, image_shape, overlapping=False):
        """
        A module that extracts spatial patches from a 6D array with size
        [1, x, y, t, e, 2].
        Output is also a 6D array with size
        [N, block_size, block_size, t, e, 2].
        """
        super().__init__()

        # Get image / block dimensions
        self.block_size = block_size
        self.image_shape = image_shape
        _, self.nx, self.ny, self.nt, self.ne, _ = image_shape

        # Overlapping vs. non-overlapping block settings
        if overlapping:
            block_stride = self.block_size // 2
            # Use Hanning window to reduce blocking artifacts
            win1d = torch.hann_window(block_size, dtype=torch.float32) ** 0.5
            self.win = (
                win1d[None, :, None, None, None, None]
                * win1d[None, None, :, None, None, None]
            )
        else:
            block_stride = self.block_size
            self.win = torch.tensor([1.0], dtype=torch.float32)

        # Figure out padsize (to avoid black bars)
        num_blocks_x = (self.nx // self.block_size) + 2
        num_blocks_y = (self.ny // self.block_size) + 2
        self.pad_x = (self.block_size * num_blocks_x - self.nx) // 2
        self.pad_y = (self.block_size * num_blocks_y - self.ny) // 2
        nx_pad = self.nx + 2 * self.pad_x
        ny_pad = self.ny + 2 * self.pad_y

        # Compute total number of blocks
        num_blocks_x = (
            self.nx - self.block_size + 2 * self.pad_x
        ) / block_stride + 1
        num_blocks_y = (
            self.ny - self.block_size + 2 * self.pad_y
        ) / block_stride + 1
        self.num_blocks = int(num_blocks_x * num_blocks_y)

        # Set fold params
        self.fold_params = dict(
            kernel_size=2 * (block_size,), stride=block_stride
        )
        self.unfold_op = nn.Unfold(**self.fold_params)
        self.fold_op = nn.Fold(output_size=(ny_pad, nx_pad), **self.fold_params)

    def extract(self, images):
        # Re-shape into a 4D array because nn.Unfold requires it >:(
        images = images.reshape(
            [1, self.nx, self.ny, self.nt * self.ne * 2]
        ).permute(0, 3, 2, 1)

        # Pad array
        images = nn.functional.pad(
            images, 2 * (self.pad_x,) + 2 * (self.pad_y,), mode="constant"
        )

        # Unfold array into vectorized blocks
        blocks = self.unfold_op(images)  # [1, nt*ne*2*bx*by, n]

        # Reshape into 2D blocks
        shape_out = (
            self.nt,
            self.ne,
            2,
            self.block_size,
            self.block_size,
            self.num_blocks,
        )
        blocks = blocks.reshape(shape_out).permute(5, 4, 3, 0, 1, 2)

        # Apply window
        blocks *= self.win.to(images.device)

        return blocks

    def combine(self, blocks):
        # Apply window
        blocks *= self.win.to(blocks.device)

        # Reshape back into nn.Fold format
        blocks = blocks.permute(3, 4, 5, 2, 1, 0)
        blocks = blocks.reshape(
            (1, self.nt * self.ne * 2 * self.block_size ** 2, self.num_blocks)
        )

        # Fold blocks back into array
        images = self.fold_op(blocks)  # [1, nt*ne*2, ny, nx]

        # Crop zero-padded images
        images = center_crop(
            images.permute(0, 3, 2, 1),
            [1, self.nx, self.ny, self.nt * self.ne * 2],
        )
        images = images.reshape(self.image_shape)

        return images

    def forward(self, input, adjoint=False):
        if adjoint:
            output = self.combine(input)
        else:
            output = self.extract(input)
        return output


def decompose_LR(
    images, num_basis, block_size=16, overlapping=False, block_op=None
):
    """
    Decomposes spatio-temporal data into spatial and temporal basis functions
    (L, R)
    """
    # Get image dimensions
    _, nx, ny, nt, ne, _ = images.shape
    nb = num_basis

    # Initialize ArrayToBlocks op if it hasn't been initialized already
    if block_op is None:
        block_op = ArrayToBlocks(
            block_size, images.shape, overlapping=overlapping
        )

    # Extract spatial blocks across images
    blocks = block_op(images)
    nblks = blocks.shape[0]  # number of blocks
    blk_size = blocks.shape[1]  # block shape [blk_size, blk_size]

    # Reshape into batch of 2D matrices
    blocks = blocks.permute(0, 1, 2, 4, 3, 5)
    blocks = blocks.reshape((nblks, blk_size * blk_size * ne, nt, 2))

    # Perform SVD to get left and right singular vectors for each patch
    U, S, V = cplx.svd(blocks, compute_uv=True)

    # Truncate singular values and vectors
    U = U[:, :, :nb, :]  # [N, Px*Py*E, B, 2]
    S = S[:, :nb]  # [N, B]
    V = V[:, :, :nb, :]  # [N, T, B, 2]

    # Combine and reshape matrices
    S_sqrt = S.reshape((nblks, 1, 1, 1, 1, nb, 1)).sqrt()
    L = U.reshape((nblks, blk_size, blk_size, 1, ne, nb, 2)) * S_sqrt
    R = V.reshape((nblks, 1, 1, nt, 1, nb, 2)) * S_sqrt

    return L, R


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


def root_sum_of_squares(x, dim=0):
    """
    Compute the root sum-of-squares (RSS) transform along a given dimension of
    a complex-valued tensor.
    """
    assert x.size(-1) == 2
    return torch.sqrt((x ** 2).sum(dim=-1).sum(dim))


def time_average(data, dim, eps=1e-6, keepdim=True):
    """
    Computes time average across a specified axis.
    """
    mask = cplx.get_mask(data)
    return data.sum(dim, keepdim=keepdim) / (
        mask.sum(dim, keepdim=keepdim) + eps
    )


def sliding_window(data, dim, window_size):
    """
    Computes sliding window with circular boundary conditions across a specified
    axis.
    """
    assert 0 < window_size <= data.shape[dim]

    windows = [None] * data.shape[dim]
    for i in range(data.shape[dim]):
        data_slide = roll(data, int(window_size / 2) - i, dim)
        window = data_slide.narrow(dim, 0, window_size)
        windows[i] = time_average(window, dim)

    return torch.cat(windows, dim=dim)


def center_crop(data, shape):
    """
    Apply a center crop to a batch of images.

    Args:
        data (torch.Tensor): The input tensor to be center cropped.
        shape (list of ints): The output shape. If shape[dim] = -1, then no crop
            will be applied in that dimension.
    """
    for i in range(len(shape)):
        if (shape[i] == data.shape[i]) or (shape[i] == -1):
            continue
        assert 0 < shape[i] <= data.shape[i]
        idx_start = (data.shape[i] - shape[i]) // 2
        data = data.narrow(i, idx_start, shape[i])

    return data


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


def roll(x, shift, dim):
    """
    Similar to np.roll but applies to PyTorch Tensors
    """
    if isinstance(shift, (tuple, list)):
        assert len(shift) == len(dim)
        for s, d in zip(shift, dim):
            x = roll(x, s, d)
        return x
    shift = shift % x.size(dim)
    if shift == 0:
        return x
    left = x.narrow(dim, 0, x.size(dim) - shift)
    right = x.narrow(dim, x.size(dim) - shift, shift)
    return torch.cat((right, left), dim=dim)


def fftshift(x, dim=None):
    """
    Similar to np.fft.fftshift but applies to PyTorch Tensors
    """
    if dim is None:
        dim = tuple(range(x.dim()))
        shift = [dim // 2 for dim in x.shape]
    elif isinstance(dim, int):
        shift = x.shape[dim] // 2
    else:
        shift = [x.shape[i] // 2 for i in dim]
    return roll(x, shift, dim)


def ifftshift(x, dim=None):
    """
    Similar to np.fft.ifftshift but applies to PyTorch Tensors
    """
    if dim is None:
        dim = tuple(range(x.dim()))
        shift = [(dim + 1) // 2 for dim in x.shape]
    elif isinstance(dim, int):
        shift = (x.shape[dim] + 1) // 2
    else:
        shift = [(x.shape[i] + 1) // 2 for i in dim]
    return roll(x, shift, dim)
