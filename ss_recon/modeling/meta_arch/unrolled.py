"""Unrolled Compressed Sensing (2D).

This file contains an implementation of the Unrolled Compressed Sensing
framework by CM Sandino, JY Cheng, et al. See paper below for more details.

It is also based heavily on the codebase below:

https://github.com/MRSRL/dl-cs

Implementation is based on:
    CM Sandino, JY Cheng, et al. "Compressed Sensing: From Research to
    Clinical Practice with Deep Neural Networks" IEEE Signal Processing
    Magazine, 2020.
"""

import torch
from torch import nn

import ss_recon.utils.complex_utils as cplx
from ss_recon.modeling.loss_computer import BasicLossComputer
from ss_recon.utils.transforms import SenseModel

from ..layers.layers2D import ResNet
from .build import META_ARCH_REGISTRY

__all__ = ["GeneralizedUnrolledCNN"]


@META_ARCH_REGISTRY.register()
class GeneralizedUnrolledCNN(nn.Module):
    """
    PyTorch implementation of Unrolled Compressed Sensing.

    Implementation is based on:
        CM Sandino, JY Cheng, et al. "Compressed Sensing: From Research to
        Clinical Practice with Deep Neural Networks" IEEE Signal Processing
        Magazine, 2020.
    """

    def __init__(self, cfg):
        """
        Args:
            params (dict): Dictionary containing network parameters
        """
        super().__init__()
        self.device = torch.device(cfg.MODEL.DEVICE)

        # Extract network parameters
        num_grad_steps = cfg.MODEL.UNROLLED.NUM_UNROLLED_STEPS
        num_resblocks = cfg.MODEL.UNROLLED.NUM_RESBLOCKS
        num_features = cfg.MODEL.UNROLLED.NUM_FEATURES
        kernel_size = cfg.MODEL.UNROLLED.KERNEL_SIZE
        if len(kernel_size) == 1:
            kernel_size = kernel_size[0]
        drop_prob = cfg.MODEL.UNROLLED.DROPOUT
        circular_pad = cfg.MODEL.UNROLLED.PADDING == "circular"
        fix_step_size = cfg.MODEL.UNROLLED.FIX_STEP_SIZE
        share_weights = cfg.MODEL.UNROLLED.SHARE_WEIGHTS

        # Data dimensions
        self.num_emaps = cfg.MODEL.UNROLLED.NUM_EMAPS

        # ResNet parameters
        resnet_params = dict(
            num_resblocks=num_resblocks,
            in_chans=2 * self.num_emaps,
            chans=num_features,
            kernel_size=kernel_size,
            drop_prob=drop_prob,
            circular_pad=circular_pad,
        )

        # Declare ResNets and RNNs for each unrolled iteration
        if share_weights:
            self.resnets = nn.ModuleList(
                [ResNet(**resnet_params)] * num_grad_steps
            )
        else:
            self.resnets = nn.ModuleList(
                [ResNet(**resnet_params) for _ in range(num_grad_steps)]
            )

        # Declare step sizes for each iteration
        init_step_size = torch.tensor([-2.0], dtype=torch.float32)
        if fix_step_size:
            self.step_sizes = [init_step_size] * num_grad_steps
        else:
            self.step_sizes = [
                torch.nn.Parameter(init_step_size)
                for i in range(num_grad_steps)
            ]

        # self.to(self.device)

    def forward(
        self,
        kspace,
        maps,
        target=None,
        init_image=None,
        mean=None,
        std=None,
        norm=None,
        mask=None,
    ):
        """
        TODO: condense into list of dataset dicts.
        Args:
            kspace (torch.Tensor): Input tensor of shape [batch_size, height, width, num_coils, 2]
            maps (torch.Tensor): Input tensor of shape [batch_size, height, width, num_coils, num_emaps, 2]
            mask (torch.Tensor): Input tensor of shape [batch_size, height, width, 1, 1]

        Returns:
            (torch.Tensor): Output tensor of shape [batch_size, height, width, num_emaps, 2]
        """
        if self.num_emaps != maps.size()[-2]:
            raise ValueError(
                "Incorrect number of ESPIRiT maps! Re-prep data..."
            )

        # Need to fetch device at runtime for proper data transfer.
        device = self.resnets[0].final_layer.weight.device
        kspace = kspace.to(device)
        maps = maps.to(device)
        target = target.to(device)
        mean = mean.to(device)
        std = std.to(device)
        norm = norm.to(device)

        # Move step sizes to the right device.
        step_sizes = [x.to(device) for x in self.step_sizes]

        if mask is None:
            mask = cplx.get_mask(kspace)
        kspace *= mask

        # Get data dimensions
        dims = tuple(kspace.size())

        # Declare signal model
        A = SenseModel(maps, weights=mask)

        # Compute zero-filled image reconstruction
        zf_image = A(kspace, adjoint=True)
        image = zf_image

        # Begin unrolled proximal gradient descent
        for resnet, step_size in zip(self.resnets, step_sizes):
            # dc update
            grad_x = A(A(image), adjoint=True) - zf_image
            image = image + step_size * grad_x

            # prox update
            image = image.reshape(dims[0:3] + (self.num_emaps * 2,)).permute(
                0, 3, 1, 2
            )

            image = resnet(image)
            image = image.permute(0, 2, 3, 1).reshape(
                dims[0:3] + (self.num_emaps, 2)
            )

        output_dict = {
            "pred": image,  # N x Y x Z x 1 x 2
            "target": target,  # N x Y x Z x 1 x 2
            "mean": mean,
            "std": std,
            "norm": norm,
        }

        return output_dict
