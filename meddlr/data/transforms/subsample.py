import inspect
import os
from typing import Sequence

import numba as nb
import numpy as np
import sigpy.mri
import torch
from fvcore.common.registry import Registry

MASK_FUNC_REGISTRY = Registry("MASK_FUNC")
MASK_FUNC_REGISTRY.__doc__ = """
Registry for mask functions, which create undersampling masks of a specified
shape.
"""


def build_mask_func(cfg):
    name = cfg.UNDERSAMPLE.NAME
    accelerations = cfg.UNDERSAMPLE.ACCELERATIONS
    calibration_size = cfg.UNDERSAMPLE.CALIBRATION_SIZE
    center_fractions = cfg.UNDERSAMPLE.CENTER_FRACTIONS

    klass = MASK_FUNC_REGISTRY.get(name)
    parameters = inspect.signature(klass).parameters

    # Optional args
    mapping = {"max_attempts": cfg.UNDERSAMPLE.MAX_ATTEMPTS}
    kwargs = {}
    for param, value in mapping.items():
        if param in parameters:
            kwargs[param] = value

    return klass(accelerations, calibration_size, center_fractions, **kwargs)


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


class CacheableMaskMixin:
    def get_filename(self):
        raise NotImplementedError


@MASK_FUNC_REGISTRY.register()
class RandomMaskFunc(MaskFunc):
    """
    RandomMaskFunc creates a 2D uniformly random undersampling mask.
    """

    def __init__(self, accelerations, calib_size, center_fractions=None):
        if center_fractions:
            raise ValueError(f"center_fractions not yet supported for class {type(self)}.")
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
class PoissonDiskMaskFunc(CacheableMaskMixin, MaskFunc):
    """
    PoissonDiskMaskFunc creates a 2D Poisson disk undersampling mask.
    """

    def __init__(
        self,
        accelerations,
        calib_size,
        center_fractions=None,
        max_attempts=30,
        crop_corner=True,
        module="internal",
    ):
        if center_fractions:
            raise ValueError(f"center_fractions not yet supported for class {type(self)}.")
        if module not in ("internal", "sigpy"):
            raise ValueError("`module` must be one of ('internal', 'sigpy')")
        super().__init__(accelerations)
        if isinstance(calib_size, int):
            calib_size = (calib_size, calib_size)
        self.calib_size = calib_size
        self.max_attempts = max_attempts
        self.crop_corner = crop_corner
        self.module = module

    def __call__(self, out_shape, seed=None, acceleration=None):
        # Design parameters for mask
        nky = out_shape[1]
        nkz = out_shape[2]
        if not acceleration:
            acceleration = self.choose_acceleration()

        # From empirical results, larger dimension should be first
        # for optimal speed.
        if nky < nkz:
            shape = (nkz, nky)
            transpose = True
        else:
            shape = (nky, nkz)
            transpose = False

        # Issue #2: Due to some optimization reasons, using the internal
        # poisson disc module has been faster than using sigpy's
        # default one. In many cases, the call to sigpy hangs.
        module = self.module
        if module == "internal":
            mask = poisson(
                shape,
                acceleration,
                calib=self.calib_size,
                dtype=np.float32,
                seed=seed,
                K=self.max_attempts,
                crop_corner=self.crop_corner,
            )
        elif module == "sigpy":
            mask = sigpy.mri.poisson(
                shape,
                acceleration,
                calib=self.calib_size,
                dtype=np.float32,
                seed=seed,
                max_attempts=self.max_attempts,
                crop_corner=self.crop_corner,
            )
        if transpose:
            mask = mask.transpose()

        # Reshape the mask
        mask = torch.from_numpy(mask.reshape(out_shape))

        return mask

    def _get_args(self):
        return {
            "accelerations": self.accelerations,
            "calib_size": self.calib_size,
            "max_attempts": self.max_attempts,
            "crop_corner": self.crop_corner,
        }

    def get_str_name(self):
        args = self._get_args()
        return f"{type(self).__name__}-" + "-".join(f"{k}={v}" for k, v in args.items())

    def __str__(self) -> str:
        args = self._get_args()
        args_str = "\n\t" + "\n\t".join(f"{k}={v}" for k, v in args.items()) + "\n\t"
        return f"{type(self)}({args_str})"


@MASK_FUNC_REGISTRY.register()
class RandomMaskFunc1D(MaskFunc):
    """
    RandomMaskFunc creates a sub-sampling mask of a given shape.
    The mask selects a subset of columns from the input k-space data. If the k-space data has N
    columns, the mask picks out:
        1. N_low_freqs = (N * center_fraction) columns in the center corresponding to
           low-frequencies
        2. The other columns are selected uniformly at random with a probability equal to:
           prob = (N / acceleration - N_low_freqs) / (N - N_low_freqs).
    This ensures that the expected number of columns selected is equal to (N / acceleration)
    It is possible to use multiple center_fractions and accelerations, in which case one possible
    (center_fraction, acceleration) is chosen uniformly at random each time the RandomMaskFunc
    object is called.

    For example, if accelerations = [4, 8] and center_fractions = [0.08, 0.04], then there
    is a 50% probability that 4-fold acceleration with 8% center fraction is selected and a 50%
    probability that 8-fold acceleration with 4% center fraction is selected.

    Adapted from https://github.com/facebookresearch/fastMRI/blob/master/common/subsample.py
    """

    def __init__(self, accelerations, calib_size=None, center_fractions=None):
        """
        Args:
            center_fractions (List[float]): Fraction of low-frequency columns to be retained.
                If multiple values are provided, then one of these numbers is chosen uniformly
                each time.
            accelerations (List[int]): Amount of under-sampling. This should have the same length
                as center_fractions. If multiple values are provided, then one of these is chosen
                uniformly each time. An acceleration of 4 retains 25% of the columns, but they may
                not be spaced evenly.
            calib_size (List[int]): Calibration size for scans.
        """
        if not calib_size and not center_fractions:
            raise ValueError("Either calib_size or center_fractions must be specified.")
        if calib_size and center_fractions:
            raise ValueError("Only one of calib_size or center_fractions can be specified")

        self.center_fractions = center_fractions
        self.calib_size = calib_size
        self.accelerations = accelerations

    def __call__(self, shape, seed=None, acceleration=None):
        """
        Args:
            shape (iterable[int]): The shape of the mask to be created. The shape should have
                at least 3 dimensions. Samples are drawn along the second last dimension.
            seed (int, optional): Seed for the random number generator. Setting the seed
                ensures the same mask is generated each time for the same shape.
        Returns:
            torch.Tensor: A mask of the specified shape.
        """
        if len(shape) < 3:
            raise ValueError("Shape should have 3 or more dimensions")

        if seed is not None:
            np_state = np.random.get_state()

        num_rows = shape[1]
        num_cols = shape[2]
        if self.center_fractions:
            if isinstance(self.center_fractions, Sequence):
                choice = np.random.randint(0, len(self.center_fractions))
                center_fraction = self.center_fractions[choice]
            else:
                center_fraction = self.center_fractions
        else:
            center_fraction = self.calib_size / num_cols
        if acceleration is None:
            acceleration = self.choose_acceleration()

        # Create the mask
        num_low_freqs = int(round(num_cols * center_fraction))
        prob = (num_cols / acceleration - num_low_freqs) / (num_cols - num_low_freqs)
        mask = np.random.uniform(size=num_cols) < prob
        pad = (num_cols - num_low_freqs + 1) // 2
        mask[pad : pad + num_low_freqs] = True

        # Reshape the mask
        mask_shape = [1 for _ in shape]
        mask_shape[2] = num_cols
        mask = mask.reshape(*mask_shape).astype(np.float32)
        mask = np.concatenate([mask] * num_rows, axis=1)
        mask = torch.from_numpy(mask)

        if seed is not None:
            np.random.set_state(np_state)

        return mask


class MaskLoader(MaskFunc):
    """Loads masks from predefined file format instead of computing on the fly."""

    def __init__(self, accelerations, masks_path, mask_type: str = "poisson", mode="train"):
        assert isinstance(accelerations, (int, float)) or len(accelerations) == 1
        assert mode in ["train", "eval"]
        if isinstance(accelerations, (int, float)):
            accelerations = (accelerations,)
        super().__init__(accelerations)

        accel = float(self.accelerations[0])
        self.train_masks = None
        self.eval_data = torch.load(os.path.join(masks_path, f"{mask_type}_{accel}x_eval.pt"))
        if mode == "train":
            self.train_masks = np.load(os.path.join(masks_path, f"{mask_type}_{accel}x.npy"))

    def __call__(self, out_shape, seed=None, acceleration=None):
        if acceleration is not None and acceleration not in self.accelerations:
            raise RuntimeError(
                "MaskLoader.__call__ does not currently support ``acceleration`` argument"
            )

        if seed is None:
            # Randomly select from the masks we have
            idx = np.random.choice(len(self.train_masks))
            mask = self.train_masks[idx]
        else:
            data = self.eval_data
            masks = self.eval_data["masks"]
            mask = masks[data["seeds"].index(seed)]

        mask = mask.reshape(out_shape)
        return torch.from_numpy(mask)


# ================================================================ #
# Adapted from sigpy.
# Duplicated because of https://github.com/mikgroup/sigpy/issues/54
# TODO: Remove once https://github.com/mikgroup/sigpy/issues/54 is
# solved and added to release.
# ================================================================ #
def poisson(
    img_shape,
    accel,
    K=30,
    calib=(0, 0),
    dtype=np.complex,
    crop_corner=True,
    return_density=False,
    seed=0,
):
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
    y, x = np.mgrid[: img_shape[-2], : img_shape[-1]]
    x = np.maximum(abs(x - img_shape[-1] / 2) - calib[-1] / 2, 0)
    x /= x.max()
    y = np.maximum(abs(y - img_shape[-2] / 2) - calib[-2] / 2, 0)
    y /= y.max()
    r = np.sqrt(x ** 2 + y ** 2)

    slope_max = 40
    slope_min = 0
    if seed is not None:
        rand_state = np.random.get_state()
    else:
        seed = -1  # numba does not play nicely with None types
    while slope_min < slope_max:
        slope = (slope_max + slope_min) / 2.0
        R = 1.0 + r * slope
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

    if seed is not None and seed > 0:
        np.random.set_state(rand_state)
    mask = mask.reshape(img_shape).astype(dtype)
    if return_density:
        return mask, r
    else:
        return mask


@nb.jit(nopython=True, cache=True)  # pragma: no cover
def _poisson(nx, ny, K, R, calib, seed=None):

    mask = np.zeros((ny, nx))
    f = ny / nx

    if seed is not None and seed > 0:
        np.random.seed(int(seed))

    pxs = np.empty(nx * ny, np.int32)
    pys = np.empty(nx * ny, np.int32)
    pxs[0] = np.random.randint(0, nx)
    pys[0] = np.random.randint(0, ny)
    m = 1
    while m > 0:

        i = np.random.randint(0, m)
        px = pxs[i]
        py = pys[i]
        rad = R[py, px]

        # Attempt to generate point
        done = False
        k = 0
        while not done and k < K:

            # Generate point randomly from R and 2R
            rd = rad * (np.random.random() * 3 + 1) ** 0.5
            t = 2 * np.pi * np.random.random()
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
                        if mask[y, x] == 1 and (
                            ((qx - x) / R[y, x]) ** 2 + ((qy - y) / (R[y, x] * f)) ** 2 < 1
                        ):
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
    mask[
        int(ny / 2 - calib[-2] / 2) : int(ny / 2 + calib[-2] / 2),
        int(nx / 2 - calib[-1] / 2) : int(nx / 2 + calib[-1] / 2),
    ] = 1

    return mask
