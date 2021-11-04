from typing import Sequence, Tuple, Union

import torch

import meddlr.transforms.functional as stf
from meddlr.transforms.build import TRANSFORM_REGISTRY
from meddlr.transforms.transform import Transform


@TRANSFORM_REGISTRY.register()
class MRIMotionTransform(Transform):
    """A model that corrupts kspace inputs with motion.
    Motion is a common artifact experienced during the MR imaging forward problem.
    When a patient moves, the recorded (expected) location of the kspace sample is
    different than the actual location where the kspace sample that was acquired.
    This module is responsible for simulating different motion artifacts.
    Args:
        seed (int, optional): The fixed seed.
    Attributes:
        generator (torch.Generator): The generator that should be used for all
            random logic in this class.
    Things to consider:
        1. What other information is relevant for inducing motion corruption?
            This could include:
            - ``traj``: The scan trajectory
            - ``etl``: The echo train length - how many readouts per shot.
            - ``num_shots``: Number of shots.
        2. What would a simple translational motion model look?
    Note:
        We do not store this as a module or else it would be saved to the model
        definition, which we dont want.
    """

    def __init__(
        self,
        std_dev: Union[float, Sequence[float]],
        seed: int = None,
        generator: torch.Generator = None,
    ):
        self.std_dev = std_dev
        self.seed = seed

        gen_state = None
        if generator is not None:
            gen_state = generator.get_state()
        self._generator_state = gen_state

    def _generator(self, data: torch.Tensor):
        seed = self.seed

        g = torch.Generator(device=data.device)
        if seed is None:
            g.set_state(self._generator_state)
        else:
            g = g.manual_seed(seed)
        return g

    def apply_kspace(self, kspace, channel_first: bool = True) -> torch.Tensor:
        """Performs motion corruption on kspace image.

        Args:
            kspace (torch.Tensor): The complex tensor. Shape ``(N, #coils, Y, X, [2])``.

        Returns:
            torch.Tensor: The motion corrupted kspace.

        Note:
            For backwards compatibility with torch<1.6, complex tensors may also have the shape
            ``(..., 2)``, where the 2 channels in the last dimension are real and
            imaginary, respectively.
            TODO: This code should account for that case as well.
        """
        scale = self.std_dev
        g = self._generator(kspace)
        return stf.add_even_odd_motion(
            kspace, scale=scale, channel_first=channel_first, generator=g
        )

    def _eq_attrs(self) -> Tuple[str]:
        return ("std_dev", "seed", "_generator_state")
