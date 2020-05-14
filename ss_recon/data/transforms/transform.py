"""Basic Transforms.
"""
import torch

from ss_recon.utils import complex_utils as cplx
from .subsample import build_mask_func
from ss_recon.utils import transforms as T


class Subsampler(object):
    def __init__(self, mask_func):
        self.mask_func = mask_func

    def __call__(self, data, mode: str="2D", seed: int=None):
        data_shape = tuple(data.shape)
        assert mode in ["2D", "3D"]
        if mode is "2D":
            mask_shape = (1,) + data_shape[1:3] + (1, 1)
        elif mode is "3D":
            mask_shape = (1,) + data_shape[1:4] + (1, 1)
        else:
            raise ValueError(
                "Only 2D and 3D undersampling masks are supported.")
        mask = self.mask_func(mask_shape, seed)
        return torch.where(mask == 0, torch.Tensor([0]), data), mask


class DataTransform:
    """
    Data Transformer for training unrolled reconstruction models.
    """

    def __init__(self, cfg, mask_func, is_test: bool = False):
        """
        Args:
            mask_func (utils.subsample.MaskFunc): A function that can create a mask of
                appropriate shape.
            is_test (bool): If `True`, this class behaves with test-time
                functionality. In particular, it computes a pseudo random number
                generator seed from the filename. This ensures that the same
                mask is used for all the slices of a given volume every time.
        """
        self._cfg = cfg
        self.mask_func = mask_func
        self._is_test = is_test

        # Build subsampler.
        mask_func = build_mask_func(cfg)
        self._subsampler = Subsampler(mask_func)

    def __call__(self, kspace, maps, target, attrs, fname, slice):
        """
        Args:
            kspace (numpy.array): Input k-space of shape (num_coils, rows, cols, 2) for multi-coil
                data or (rows, cols, 2) for single coil data.
            target (numpy.array): Target image
            attrs (dict): Acquisition related information stored in the HDF5 object.
            fname (str): File name
            slice (int): Serial number of the slice.
        Returns:
            (tuple): tuple containing:
                image (torch.Tensor): Zero-filled input image.
                target (torch.Tensor): Target image converted to a torch Tensor.
                mean (float): Mean value used for normalization.
                std (float): Standard deviation value used for normalization.
                norm (float): L2 norm of the entire volume.
        """
        # Convert everything from numpy arrays to tensors
        kspace = cplx.to_tensor(kspace).unsqueeze(0)
        maps = cplx.to_tensor(maps).unsqueeze(0)
        target = cplx.to_tensor(target).unsqueeze(0)
        norm = torch.sqrt(torch.mean(cplx.abs(target) ** 2))

        # print(kspace.shape)
        # print(maps.shape)
        # print(target.shape)

        # TODO: Add other transforms here.

        # Apply mask in k-space
        seed = None if not self._is_test else tuple(map(ord, fname))
        masked_kspace, mask = self._subsampler(kspace, mode="2D", seed=seed)

        # Normalize data...
        A = T.SenseModel(maps, weights=mask)
        image = A(masked_kspace, adjoint=True)
        magnitude_vals = cplx.abs(image).reshape(-1)
        k = int(round(0.05 * magnitude_vals.numel()))
        scale = torch.min(torch.topk(magnitude_vals, k).values)

        masked_kspace /= scale
        target /= scale
        mean = torch.tensor([0.0], dtype=torch.float32)
        std = scale

        # Get rid of batch dimension...
        masked_kspace = masked_kspace.squeeze(0)
        maps = maps.squeeze(0)
        target = target.squeeze(0)

        return masked_kspace, maps, target, mean, std, norm
