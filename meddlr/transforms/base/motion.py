from typing import Optional, Sequence, Tuple, Union

import torch

import meddlr.transforms.functional as stf
from meddlr.transforms.build import TRANSFORM_REGISTRY
from meddlr.transforms.transform import Transform
from meddlr.transforms.transform_gen import TransformGen


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


@TRANSFORM_REGISTRY.register()
class MRIMultiShotMotion(Transform):
    """A model that simulates motion artifacts in multi-shot MRI.

    To simulate motion, the coil-combined image is augmented with random
    affine transformations. The number of augmentations corresponds to the
    number of echo trains (i.e. number of shots) used during acquisition.
    The multi-coil kspace for each of these augmented versions is combined based
    on the trajectory.
    """

    def __init__(
        self,
        tfm_gens: Sequence[TransformGen],
        trajectory: torch.Tensor,
        seed: Optional[int] = None,
        generator: Optional[torch.Generator] = None,
    ):
        """
        Args:
            tfm_gens (Sequence[TransformGen]): The transform generators to use.
                These will be seeded prior to use.
            trajectory (torch.Tensor): The trajectory to use for the multi-shot
                motion simulation.
            seed (int, optional): The seed to use for the random number generator.
            generator (torch.Generator, optional): The random number generator to use.
                Must be specified if ``seed`` is not set.
        """
        self.tfm_gens = tfm_gens
        self.trajectory = trajectory
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

    def _apply_x(self, x: torch.Tensor, *, maps: torch.Tensor, channel_first: bool, xtype: str):
        g = self._generator(x)
        seeds = torch.randint(0, 2**32, (len(self.tfm_gens),), device=g.device, generator=g)
        seeds = [seed.cpu().item() for seed in seeds]
        for tfm_gen, seed in zip(self.tfm_gens, seeds):
            tfm_gen.seed(seed)

        return stf.add_affine_motion(
            x,
            transform_gens=self.tfm_gens,
            trajectory=self.trajectory,
            maps=maps,
            is_batch=True,
            channels_first=channel_first,
            xtype=xtype,
        )

    def apply_kspace(
        self,
        kspace: torch.Tensor,
        *,
        maps: torch.Tensor = None,
        channel_first: bool = True,
    ) -> torch.Tensor:
        """Performs motion corruption on kspace image.

        Args:
            kspace (torch.Tensor): The complex tensor. Shape ``(N, #coils, Y, X, [2])``.
            maps (torch.Tensor, optional): The sensitivity maps.
                Shape ``(N, #coils, #maps, Y, X, [2])``.

        Returns:
            torch.Tensor: The motion corrupted kspace.
        """
        return self._apply_x(kspace, maps=maps, channel_first=channel_first, xtype="kspace")

    def apply_image(
        self, image: torch.Tensor, *, maps: torch.Tensor = None, channel_first: bool = True
    ) -> torch.Tensor:
        return self._apply_x(image, maps=maps, channel_first=channel_first, xtype="image")

    def _eq_attrs(self) -> Tuple[str]:
        return ("tfm_gens", "trajectory", "seed", "_generator_state")
