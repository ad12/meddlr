from typing import Sequence, Union

import torch

from meddlr.transforms.base.motion import MRIMotionTransform
from meddlr.transforms.build import TRANSFORM_REGISTRY
from meddlr.transforms.transform import NoOpTransform
from meddlr.transforms.transform_gen import TransformGen

__all__ = ["RandomMRIMotion"]


@TRANSFORM_REGISTRY.register()
class RandomMRIMotion(TransformGen):
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

    _base_transform = MRIMotionTransform

    def __init__(self, std_devs: Union[float, Sequence[float]], p: float = 0.0):
        if isinstance(std_devs, (float, int)):
            std_devs = (std_devs, std_devs)
        elif len(std_devs) > 2:
            raise ValueError("`motion_range` must have 2 or fewer values")
        super().__init__(params={"std_devs": std_devs}, p=p)

    def get_transform(self, input: torch.Tensor):
        params = self._get_param_values(use_schedulers=True)
        if self._rand() >= params["p"]:
            return NoOpTransform()

        std_dev = self._rand_range(*params["std_devs"])
        gen = self._generator
        if gen is None or gen.device != input.device:
            gen = torch.Generator(device=input.device).manual_seed(int(self._rand() * 1e10))
        return MRIMotionTransform(std_dev=std_dev, generator=gen)
