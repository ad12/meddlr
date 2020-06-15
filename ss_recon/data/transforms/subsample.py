import numba as nb
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
        # in the fixed numpy seed.
        # To avoid this issue this method has been duplicated from
        # sigpy and added below.
        # https://github.com/mikgroup/sigpy/issues/54
        mask = poisson(
            (nky, nkz),
            acceleration,
            calib=self.calib_size,
            dtype=np.float32,
            seed=seed,
        )

        # Reshape the mask
        mask = torch.from_numpy(mask.reshape(out_shape))

        return mask


# ================================================================ #
# Adapted from sigpy.
# Duplicated because of https://github.com/mikgroup/sigpy/issues/54
# TODO: Remove once https://github.com/mikgroup/sigpy/issues/54 is
# solved and added to release.
# ================================================================ #
def poisson(img_shape, accel, K=30, calib=[0, 0], dtype=np.complex,
            crop_corner=True, return_density=False, seed=0):
    """Generate Poisson-disc sampling pattern

    Args:
        img_shape (tuple of ints): length-2 image shape.
        accel (float): Target acceleration factor. Greater than 1.
        K (float): maximum number of samples to reject.
        calib (tuple of ints): length-2 calibration shape.
        dtype (Dtype): data type.
        crop_corner (bool): Toggle whether to crop sampling corners.
        return_density (bool): Toggle whether to return sampling density.
        seed (int): Random seed.

    Returns:
        array: Poisson-disc sampling mask.

    References:
        Bridson, Robert. "Fast Poisson disk sampling in arbitrary dimensions."
        SIGGRAPH sketches. 2007.

    """
    y, x = np.mgrid[:img_shape[-2], :img_shape[-1]]
    x = np.maximum(abs(x - img_shape[-1] / 2) - calib[-1] / 2, 0)
    x /= x.max()
    y = np.maximum(abs(y - img_shape[-2] / 2) - calib[-2] / 2, 0)
    y /= y.max()
    r = np.sqrt(x ** 2 + y ** 2)

    slope_max = 40
    slope_min = 0
    while slope_min < slope_max:
        slope = (slope_max + slope_min) / 2.0
        R = (1.0 + r * slope)
        mask = _poisson(img_shape[-1], img_shape[-2], K, R, calib, seed)
        if crop_corner:
            mask *= r < 1

        est_accel = img_shape[-1] * img_shape[-2] / np.sum(mask[:])

        if abs(est_accel - accel) < 0.1:
            break
        if est_accel < accel:
            slope_min = slope
        else:
            slope_max = slope

    mask = mask.reshape(img_shape).astype(dtype)
    if return_density:
        return mask, r
    else:
        return mask


@nb.jit(nopython=True, cache=True)  # pragma: no cover
def _poisson(nx, ny, K, R, calib, seed):

    mask = np.zeros((ny, nx))
    f = ny / nx

    if seed is not None:
        rand_state = np.random.RandomState(int(seed))
    else:
        rand_state = np.random

    pxs = np.empty(nx * ny, np.int32)
    pys = np.empty(nx * ny, np.int32)
    pxs[0] = rand_state.randint(0, nx)
    pys[0] = rand_state.randint(0, ny)
    m = 1
    while (m > 0):

        i = rand_state.randint(0, m)
        px = pxs[i]
        py = pys[i]
        rad = R[py, px]

        # Attempt to generate point
        done = False
        k = 0
        while not done and k < K:

            # Generate point randomly from R and 2R
            rd = rad * (rand_state.random() * 3 + 1)**0.5
            t = 2 * np.pi * rand_state.random()
            qx = px + rd * np.cos(t)
            qy = py + rd * f * np.sin(t)

            # Reject if outside grid or close to other points
            if qx >= 0 and qx < nx and qy >= 0 and qy < ny:

                startx = max(int(qx - rad), 0)
                endx = min(int(qx + rad + 1), nx)
                starty = max(int(qy - rad * f), 0)
                endy = min(int(qy + rad * f + 1), ny)

                done = True
                for x in range(startx, endx):
                    for y in range(starty, endy):
                        if (mask[y, x] == 1
                            and (((qx - x) / R[y, x]) ** 2 +
                                 ((qy - y) / (R[y, x] * f)) ** 2 < 1)):
                            done = False
                            break

                    if not done:
                        break

            k += 1

        # Add point if done else remove active
        if done:
            pxs[m] = qx
            pys[m] = qy
            mask[int(qy), int(qx)] = 1
            m += 1
        else:
            pxs[i] = pxs[m - 1]
            pys[i] = pys[m - 1]
            m -= 1

    # Add calibration region
    mask[int(ny / 2 - calib[-2] / 2):int(ny / 2 + calib[-2] / 2),
         int(nx / 2 - calib[-1] / 2):int(nx / 2 + calib[-1] / 2)] = 1

    return mask
