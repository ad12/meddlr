from typing import Any, List, Mapping, Sequence, Union

import torch

from meddlr.config import CfgNode
from meddlr.transforms.build import TRANSFORM_REGISTRY, build_transforms
from meddlr.transforms.mixins import DeviceMixin
from meddlr.transforms.tf_scheduler import SchedulableMixin
from meddlr.transforms.transform import NoOpTransform, Transform
from meddlr.transforms.transform_gen import TransformGen

__all__ = ["RandomTransformChoice"]

_TRANSFORM_OR_GEN = Union[Transform, TransformGen]


@TRANSFORM_REGISTRY.register()
class RandomTransformChoice(TransformGen):
    def __init__(
        self, tfms_or_gens: Sequence[Union[Transform, TransformGen]], tfm_ps="uniform", p=0.0
    ) -> None:
        N = len(tfms_or_gens)
        if tfm_ps == "uniform":
            tfm_ps = torch.ones(N) / N
        else:
            tfm_ps = torch.as_tensor(tfm_ps)
        assert torch.allclose(torch.sum(tfm_ps), torch.as_tensor(1.0))
        self.tfm_ps = tfm_ps
        self.tfms_or_gens = tfms_or_gens

        super().__init__(p=p)

    def get_transform(self, input=None) -> Union[_TRANSFORM_OR_GEN, Sequence[_TRANSFORM_OR_GEN]]:
        params = self._get_param_values(use_schedulers=True)
        p = params["p"]
        if self._rand() >= p:
            return NoOpTransform()

        return self.tfms_or_gens[self._rand_choice(probs=self.tfm_ps)]

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
        argstr = ",\n  ".join(
            "(p={:0.2f}) {}".format(p, repr(t)) for t, p in zip(self.tfms_or_gens, self.tfm_ps)
        )
        return "{}(\n  {}\n)".format(classname, argstr)

    @classmethod
    def from_dict(cls, cfg: CfgNode, init_kwargs: Mapping[str, Any], **kwargs):
        init_kwargs = init_kwargs.copy()
        tfms_or_gens = []
        for tfm_cfg in init_kwargs.pop("tfms_or_gens"):
            tfms_or_gens.append(build_transforms(cfg, tfm_cfg, **kwargs))

        return cls(tfms_or_gens, **init_kwargs)
