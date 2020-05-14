"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from math import ceil, floor

import numpy as np
import sigpy.mri
import torch


class MaskFunc:
    """
    Abstract MaskFunc class for creating undersampling masks of a specified shape.
    """

    def __init__(self, accelerations):
        """
        Args:
            accelerations (List[int]): Range of acceleration rates to simulate.
        """
        self.accelerations = accelerations
        self.rng = np.random.RandomState()

    def choose_acceleration(self):
        """
        Chooses a random acceleration rate given a range.
        """
        accel_range = self.accelerations[1] - self.accelerations[0]
        acceleration = self.accelerations[0] + accel_range * self.rng.rand()
        return acceleration


class RandomMaskFunc(MaskFunc):
    """
    RandomMaskFunc creates a 2D uniformly random undersampling mask.
    """

    def __init__(self, accelerations, calib_size):
        super().__init__(accelerations)
        self.calib_size = calib_size

    def __call__(self, out_shape, seed=None):
        # self.rng.seed(seed)

        # Design parameters for mask
        nky = out_shape[1]
        nkz = out_shape[2]
        acceleration = self.choose_acceleration()
        prob = 1.0 / acceleration

        # Generate undersampling mask
        mask = torch.rand([nky, nkz], dtype=torch.float32)
        mask = torch.where(mask < prob, torch.Tensor([1]), torch.Tensor([0]))

        # Add calibration region
        calib = [self.calib_size, self.calib_size]
        mask[
            int(nky / 2 - calib[-2] / 2) : int(nky / 2 + calib[-2] / 2),
            int(nkz / 2 - calib[-1] / 2) : int(nkz / 2 + calib[-1] / 2),
        ] = torch.Tensor([1])

        return mask.reshape(out_shape)


class PoissonDiskMaskFunc(MaskFunc):
    """
    PoissonDiskMaskFunc creates a 2D Poisson disk undersampling mask.
    """

    def __init__(self, accelerations, calib_size):
        super().__init__(accelerations)
        self.calib_size = [calib_size, calib_size]

    def __call__(self, out_shape, seed=None):
        # self.rng.seed(seed)

        # Design parameters for mask
        nky = out_shape[1]
        nkz = out_shape[2]
        acceleration = self.choose_acceleration()

        # Generate undersampling mask
        mask = sigpy.mri.poisson(
            [nky, nkz],
            acceleration,
            calib=self.calib_size,
            dtype=np.float32,
            seed=np.random.seed(seed),
        )

        # Reshape the mask
        mask = torch.from_numpy(mask.reshape(out_shape))

        return mask


def subsample(data, mask_func, seed=None, mode="2D"):
    """
    Subsample given k-space by multiplying with a mask.

    Args:
        data (torch.Tensor): The input k-space data. This should have at least 3 dimensions, where
            dimensions -3 and -2 are the spatial dimensions, and the final dimension has size
            2 (for complex values).
        mask_func (callable): A function that takes a shape (tuple of ints) and a random
            number seed and returns a mask.
        seed (int or 1-d array_like, optional): Seed for the random number generator.

    Returns:
        (tuple): tuple containing:
            masked data (torch.Tensor): Subsampled k-space data
            mask (torch.Tensor): The generated mask
    """
    data_shape = tuple(data.shape)
    if mode is "2D":
        mask_shape = (1,) + data_shape[1:3] + (1, 1)
    elif mode is "3D":
        mask_shape = (1,) + data_shape[1:4] + (1, 1)
    else:
        raise ValueError("Only 2D and 3D undersampling masks are supported.")
    mask = mask_func(mask_shape, seed)
    return torch.where(mask == 0, torch.Tensor([0]), data), mask
