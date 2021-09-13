from typing import Any, Mapping, Sequence, Union

import torch

from ss_recon.config import CfgNode
from ss_recon.transforms.build import TRANSFORM_REGISTRY, build_transforms
from ss_recon.transforms.transform import Transform
from ss_recon.transforms.transform_gen import TransformGen

__all__ = ["RandomTransformChoice"]


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

    def get_transform(self, input=None):
        return self.tfms_or_gens[self._rand_choice(probs=self.tfm_ps)]

    def seed(self, value: int):
        self._generator = torch.Generator(device=self._device).manual_seed(value)
        for g in self.tfms_or_gens:
            if isinstance(g, TransformGen):
                g.seed(value)
        return self

    def __repr__(self):
        classname = type(self).__name__
        argstr = ",\n\t".join(
            "{} - p={:0.2f}".format(t, p) for t, p in zip(self.tfms_or_gens, self.tfm_ps)
        )
        return "{}(\n\t{}\n\t)".format(classname, ", ".join(argstr))

    @classmethod
    def from_dict(cls, cfg: CfgNode, init_kwargs: Mapping[str, Any], **kwargs):
        init_kwargs = init_kwargs.copy()
        tfms_or_gens = init_kwargs.pop("tfms_or_gens")
        tfms_or_gens = build_transforms(cfg, tfms_or_gens, **kwargs)

        return cls(tfms_or_gens, **init_kwargs)
