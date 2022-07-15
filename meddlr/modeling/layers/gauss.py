from typing import Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from meddlr.modeling.layers.build import CUSTOM_LAYERS_REGISTRY

__all__ = ["gaussian", "get_gaussian_kernel", "GaussianBlur"]


def gaussian(window_size, sigma, normalize=True):
    def gauss_fcn(x):
        center = window_size // 2 if window_size % 2 == 1 else window_size // 2 - 0.5
        return -((x - center) ** 2) / float(2 * sigma**2)

    gauss = torch.stack([torch.exp(torch.tensor(gauss_fcn(x))) for x in range(window_size)])
    if normalize:
        gauss = gauss / gauss.sum()
    return gauss


def get_gaussian_kernel(
    kernel_size: Union[int, Sequence[int]],
    sigma: Union[float, Sequence[float]],
    normalize: bool = True,
) -> torch.Tensor:
    """Function that returns Gaussian filter coefficients.

    Args:
        kernel_size (int(s)): filter size. It should be positive.
        sigma (float(s)): gaussian standard deviation.
        normalize (bool, optional): If `True`, kernel will be normalized. i.e. `kernel.sum() == 1)

    Returns:
        Tensor: nD tensor with gaussian filter coefficients. Shape :math:`(\text{kernel_size})`

    Examples::

        >>> medsegpy.layers.get_gaussian_kernel(3, 2.5)
        tensor([0.3243, 0.3513, 0.3243])

        >>> medsegpy.layers.get_gaussian_kernel((3, 3), (1.5, 1.5))
        tensor([[0.0947, 0.1183, 0.0947],
                [0.1183, 0.1478, 0.1183],
                [0.0947, 0.1183, 0.0947]])
    """
    kernel_size_seq = (kernel_size,) if not isinstance(kernel_size, Sequence) else kernel_size
    sigma_seq = (sigma,) if not isinstance(sigma, Sequence) else sigma
    if not isinstance(kernel_size, (int, Tuple)) or any(k <= 0 for k in kernel_size_seq):
        raise TypeError(
            "kernel_size must be a (sequence of) odd positive integer. "
            "Got {}".format(kernel_size)
        )
    if len(kernel_size_seq) != len(sigma_seq):
        raise ValueError(
            "kernel_size and sigma must have same number of elements. "
            "Got kernel_size={}, sigma={}".format(kernel_size, sigma)
        )
    assert len(kernel_size_seq) <= 26
    kernels_1d = tuple(
        gaussian(ksize, sigma, normalize) for ksize, sigma in zip(kernel_size_seq, sigma_seq)
    )
    elems = tuple(chr(ord("a") + i) for i in range(len(kernels_1d)))
    equation = "{}->{}".format(",".join(elems), "".join(elems))
    return torch.einsum(equation, *kernels_1d)


@CUSTOM_LAYERS_REGISTRY.register()
class GaussianBlur(nn.Module):
    r"""Creates an operator that blurs a tensor using a Gaussian filter.

    The operator smooths the given tensor with a gaussian kernel by convolving
    it to each channel. It suports batched operation.

    Arguments:
        kernel_size (Tuple[int, int]): the size of the kernel. Dimensions are `([D, H], W)`
        sigma (Tuple[float, float]): the standard deviation of the kernel.
            Dimensions are `([D, H], W)`

    Returns:
        Tensor: the blurred tensor. Shape :math:`(B, C, [..., H], W)`

    Shape:
        - Input: :math:`(B, C, H, W)`
        - Output: :math:`(B, C, H, W)`

    Examples:

        >>> input = torch.rand(2, 4, 5, 5)
        >>> gauss = GaussianBlur((3, 3), (1.5, 1.5))
        >>> output = gauss(input)  # 2x4x5x5
    """

    def __init__(
        self, kernel_size: Union[int, Tuple[int, ...]], sigma: Union[float, Tuple[float, ...]]
    ):
        super().__init__()
        kernel_size = (
            (kernel_size,) if not isinstance(kernel_size, Sequence) else tuple(kernel_size)
        )
        if any(k % 2 == 0 for k in kernel_size):
            raise ValueError("kernel_size must be odd and positive. Got {}".format(kernel_size))
        sigma = (sigma,) if not isinstance(sigma, Sequence) else tuple(sigma)
        self.kernel_size: Tuple[int, ...] = kernel_size
        self.sigma: Tuple[float, ...] = sigma
        self._padding: Tuple[int, ...] = self.compute_zero_padding(kernel_size)
        self.kernel: torch.Tensor = nn.Parameter(
            get_gaussian_kernel(kernel_size, sigma, normalize=True),
            requires_grad=False,
        )
        self.spatial_dim = len(kernel_size)
        self.conv = [F.conv1d, F.conv2d, F.conv3d][self.spatial_dim - 1]

    @staticmethod
    def compute_zero_padding(kernel_size: Tuple[int, ...]) -> Tuple[int, ...]:
        """Computes zero padding tuple."""
        computed = [(k - 1) // 2 for k in kernel_size]
        return tuple(computed)

    def forward(self, x: torch.Tensor):
        if not torch.is_tensor(x):
            raise TypeError("Input x type is not a torch.Tensor. Got {}".format(type(x)))

        c = x.shape[1]
        spatial_dim = self.spatial_dim
        tmp_kernel: torch.Tensor = self.kernel
        kernel: torch.Tensor = tmp_kernel.repeat([c, 1] + [1] * spatial_dim)

        # TODO: explore solution when using jit.trace since it raises a warning
        # because the shape is converted to a tensor instead to a int.
        # convolve tensor with gaussian kernel
        return self.conv(x, kernel, padding=self._padding, stride=1, groups=c)
