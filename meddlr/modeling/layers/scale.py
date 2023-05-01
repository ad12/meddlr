import torch
from torch import nn

from meddlr.modeling.layers.build import CUSTOM_LAYERS_REGISTRY


class ScaleNd(nn.Module):
    def __init__(self, factor: float, trainable: bool = False):
        super().__init__()
        self.factor = (
            nn.Parameter(torch.tensor(factor), requires_grad=trainable)
            if isinstance(factor, torch.Tensor) or trainable
            else factor
        )

    def forward(self, x):
        return x * self.factor


@CUSTOM_LAYERS_REGISTRY.register()
class Scale2d(ScaleNd):
    pass


@CUSTOM_LAYERS_REGISTRY.register()
class Scale3d(ScaleNd):
    pass
