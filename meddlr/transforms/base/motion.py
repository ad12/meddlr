from typing import Optional, Sequence, Tuple, Union

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

    This

    Attributes:
        generator (torch.Generator): The generator that should be used for all
            random logic in this class.
    """

    def __init__(
        self,
        std_dev: Union[float, Sequence[float]],
        seed: Optional[int] = None,
        generator: Optional[torch.Generator] = None,
    ):
        """
        Args:
            std_dev: The standard deviation (i.e. extent) of motion to add to kspace.
                Larger standard deviation enables larger phase shifts (i.e. more motion).
                This is equivalent to the :math:`\\alpha` parameter in the paper.
            seed (int, optional): The seed to use for the random number generator.
            generator (torch.Generator, optional): The random number generator to use.
                Must be specified if ``seed`` is not set.
        """
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
