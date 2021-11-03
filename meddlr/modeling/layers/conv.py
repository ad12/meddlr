from typing import Tuple, Union

import torch.nn as nn
import torch.nn.functional as F

from meddlr.modeling.layers.build import CUSTOM_LAYERS_REGISTRY

__all__ = ["ConvWS2d", "ConvWS3d"]


@CUSTOM_LAYERS_REGISTRY.register()
class ConvWS2d(nn.Conv2d):
    """Conv2d with Weight Standardization.

    Adapted from
    https://github.com/joe-siyuan-qiao/pytorch-classification/blob/e6355f829e85ac05a71b8889f4fff77b9ab95d0b/models/layers.py

    References:
        S. Qiao, et al. Micro-Batch Training with Batch-Channel Normalization and Weight
        Standardization, ArXiv 2020. https://arxiv.org/abs/1903.10520
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int, int]],
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=True,
        eps=1e-5,
    ):
        super().__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias
        )
        self.eps = eps

    def forward(self, x):
        weight = self.weight
        weight_mean = weight.view(weight.size(0), -1).mean(dim=1).view(-1, 1, 1, 1)
        weight = weight - weight_mean
        std = weight.view(weight.size(0), -1).std(dim=1).view(-1, 1, 1, 1) + self.eps
        weight = weight / std.expand_as(weight)
        return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)


@CUSTOM_LAYERS_REGISTRY.register()
class ConvWS3d(nn.Conv3d):
    """Conv3d with Weight Standardization.

    Adapted from:
    https://github.com/joe-siyuan-qiao/pytorch-classification/blob/e6355f829e85ac05a71b8889f4fff77b9ab95d0b/models/layers.py

    References:
        S. Qiao, et al. Micro-Batch Training with Batch-Channel Normalization and Weight
        Standardization, ArXiv 2020. https://arxiv.org/abs/1903.10520
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int, int]],
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=True,
        eps=1e-5,
    ):
        super().__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias
        )
        self.eps = eps

    def forward(self, x):
        weight = self.weight
        weight_mean = weight.view(weight.size(0), -1).mean(dim=1).view(-1, 1, 1, 1, 1)
        weight = weight - weight_mean
        std = weight.view(weight.size(0), -1).std(dim=1).view(-1, 1, 1, 1, 1) + self.eps
        weight = weight / std.expand_as(weight)
        return F.conv3d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
