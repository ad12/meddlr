from typing import Any, List, Mapping, Sequence, Tuple, Union

import torch

import meddlr.transforms.functional as tF
from meddlr.config.config import CfgNode
from meddlr.transforms.base.motion import MRIMotionTransform, MRIMultiShotMotion
from meddlr.transforms.build import TRANSFORM_REGISTRY, build_transforms
from meddlr.transforms.mixins import DeviceMixin
from meddlr.transforms.tf_scheduler import SchedulableMixin
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


@TRANSFORM_REGISTRY.register()
class RandomMRIMultiShotMotion(TransformGen):
    """A model that corrupts kspace inputs with affine motion.

    Similar to :class:`RandomMRIMotion`, but supports affine transformations.
    Transformations are performed in image space and filled in based on a trajectory.

    Note:
        We do not store this as a module or else it would be saved to the model
        definition, which we dont want.
    """

    _base_transform = MRIMultiShotMotion

    def __init__(
        self,
        tfms_or_gens: Sequence[TransformGen],
        nshots: Union[int, Tuple[int, int]],
        trajectory: str = "blocked",
        p: float = 0.0,
    ):
        self.tfms_or_gens = tfms_or_gens
        self.trajectory = trajectory
        if isinstance(nshots, float):
            if not nshots.is_integer():
                raise ValueError("`nshots` must be an integer")
            nshots = int(nshots)
        if isinstance(nshots, int):
            nshots = (nshots, nshots)
        super().__init__(params={"nshots": nshots}, p=p)

    def get_transform(self, input: torch.Tensor):
        params = self._get_param_values(use_schedulers=True)
        if self._rand() >= params["p"]:
            return NoOpTransform()

        nshots = self._rand_range(*params["nshots"])
        trajectory = tF.get_multishot_trajectory(
            kind=self.trajectory,
            nshots=int(round(nshots)),
            shape=input.shape[-2:],
            device=input.device,
        )

        gen = self._generator
        if gen is None or gen.device != input.device:
            gen = torch.Generator(device=input.device).manual_seed(int(self._rand() * 1e10))

        return MRIMultiShotMotion(
            tfm_gens=self.tfms_or_gens,
            trajectory=trajectory,
            generator=gen,
        )

    def schedulers(self):
        tfms: List[SchedulableMixin] = self._get_tfm_by_type(SchedulableMixin)
        schedulers = list(self._schedulers)
        schedulers.extend([sch for tfm in tfms for sch in tfm.schedulers()])
        return schedulers

    def seed(self, value: int):
        self._generator = torch.Generator(device=self._device).manual_seed(value)
        tfms: List[TransformGen] = self._get_tfm_by_type(TransformGen)
        for t in tfms:
            t.seed(value)
        return self

    def to(self, device):
        super().to(device)
        tfms: List[DeviceMixin] = self._get_tfm_by_type(DeviceMixin)
        for t in tfms:
            t.to(device)
        return self

    def _get_tfm_by_type(self, klass):
        tfms = []
        for tfm in self.tfms_or_gens:
            if isinstance(tfm, (list, tuple)):
                tfms.extend(t for t in tfm if isinstance(t, klass))
            elif isinstance(tfm, klass):
                tfms.append(tfm)
        return tfms

    def __repr__(self):
        classname = type(self).__name__
        argstr = ",\n  ".join("{}".format(repr(t)) for t in zip(self.tfms_or_gens))
        params = ["nshots", "trajectory", "p"]
        params_str = ",\n  ".join("{}={}".format(k, repr(getattr(self, k))) for k in params)
        return "{}(\n  {},\n  {}\n)".format(classname, argstr, params_str)

    @classmethod
    def from_dict(cls, cfg: CfgNode, init_kwargs: Mapping[str, Any], **kwargs):
        init_kwargs = init_kwargs.copy()
        tfms_or_gens = []
        for tfm_cfg in init_kwargs.pop("tfms_or_gens"):
            tfms_or_gens.append(build_transforms(cfg, tfm_cfg, **kwargs))

        return cls(tfms_or_gens, **init_kwargs)
