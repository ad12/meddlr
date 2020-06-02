import numpy as np
import sigpy.mri
import torch
from fvcore.common.registry import Registry
from typing import Sequence

MASK_FUNC_REGISTRY = Registry("MASK_FUNC")
MASK_FUNC_REGISTRY.__doc__ = """
Registry for mask functions, which create undersampling masks of a specified
shape.
"""


def build_mask_func(cfg):
    name = cfg.UNDERSAMPLE.NAME
    accelerations = cfg.UNDERSAMPLE.ACCELERATIONS
    calibration_size = cfg.UNDERSAMPLE.CALIBRATION_SIZE

    return MASK_FUNC_REGISTRY.get(name)(accelerations, calibration_size)


class MaskFunc:
    """Abstract MaskFunc class for creating undersampling masks of a specified
    shape.

    Adapted from Facebook fastMRI.
    """

    def __init__(self, accelerations):
        """
        Args:
            accelerations (List[int]): Range of acceleration rates to simulate.
        """
        self.accelerations = accelerations
        self.rng = np.random.RandomState()

    def choose_acceleration(self):
        """Chooses a random acceleration rate given a range.

        If self.accelerations is a constant, it will be returned

        """
        if not isinstance(self.accelerations, Sequence):
            return self.accelerations
        elif len(self.accelerations) == 1:
            return self.accelerations[0]
        accel_range = self.accelerations[1] - self.accelerations[0]
        acceleration = self.accelerations[0] + accel_range * self.rng.rand()
        return acceleration


@MASK_FUNC_REGISTRY.register()
class RandomMaskFunc(MaskFunc):
    """
    RandomMaskFunc creates a 2D uniformly random undersampling mask.
    """

    def __init__(self, accelerations, calib_size):
        super().__init__(accelerations)
        self.calib_size = calib_size

    def __call__(self, out_shape, seed=None, acceleration=None):
        # Design parameters for mask
        nky = out_shape[1]
        nkz = out_shape[2]

        if not acceleration:
            acceleration = self.choose_acceleration()
        prob = 1.0 / acceleration

        # Generate undersampling mask.
        rand_kwargs = {"dtype": torch.float32}
        if seed is not None:
            rand_kwargs["generator"] = torch.Generator().manual_seed(seed)
        
        mask = torch.rand([nky, nkz], **rand_kwargs)
        mask = torch.where(mask < prob, torch.Tensor([1]), torch.Tensor([0]))

        # Add calibration region
        calib = [self.calib_size, self.calib_size]
        mask[
            int(nky / 2 - calib[-2] / 2) : int(nky / 2 + calib[-2] / 2),
            int(nkz / 2 - calib[-1] / 2) : int(nkz / 2 + calib[-1] / 2),
        ] = torch.Tensor([1])

        return mask.reshape(out_shape)


@MASK_FUNC_REGISTRY.register()
class PoissonDiskMaskFunc(MaskFunc):
    """
    PoissonDiskMaskFunc creates a 2D Poisson disk undersampling mask.
    """

    def __init__(self, accelerations, calib_size):
        super().__init__(accelerations)
        self.calib_size = (calib_size, calib_size)

    def __call__(self, out_shape, seed=None, acceleration=None):
        # Design parameters for mask
        nky = out_shape[1]
        nkz = out_shape[2]
        if not acceleration:
            acceleration = self.choose_acceleration()

        # Generate undersampling mask
        # NOTE: Due to a sigpy bug, fixing a seed will cause a change
        # in the fixed numpy seed. Note this can cause downstream reproducibility
        # issues.
        mask = sigpy.mri.poisson(
            (nky, nkz),
            acceleration,
            calib=self.calib_size,
            dtype=np.float32,
            seed=np.random.seed(seed),
        )

        # Reshape the mask
        mask = torch.from_numpy(mask.reshape(out_shape))

        return mask

