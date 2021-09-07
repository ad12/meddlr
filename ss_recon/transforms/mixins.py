from typing import Union

import torch


class TransformCacheMixin:
    def _reset_transform(self):
        self._transform = None

    def cached_transform(self):
        return self._transform


class DeviceMixin:
    device: Union[str, int, torch.device]

    def to(self, device):
        self._device = device

    def cpu(self):
        self._device = torch.device("cpu")

    def cuda(self):
        self._device = torch.device("cuda")


class GeometricMixin:
    def is_geometric(self):
        return True
