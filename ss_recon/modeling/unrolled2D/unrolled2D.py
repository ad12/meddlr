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
from ss_recon.utils.layers2D import ResNet
from ss_recon.utils.transforms import SenseModel


class UnrolledModel(nn.Module):
    """
    PyTorch implementation of Unrolled Compressed Sensing.

    Implementation is based on:
        CM Sandino, JY Cheng, et al. "Compressed Sensing: From Research to 
        Clinical Practice with Deep Neural Networks" IEEE Signal Processing 
        Magazine, 2020.
    """

    def __init__(self, params):
        """
        Args:
            params (dict): Dictionary containing network parameters
        """
        super().__init__()

        # Extract network parameters
        num_grad_steps = params.num_grad_steps
        num_resblocks = params.num_resblocks
        num_features = params.num_features
        kernel_size = params.kernel_size
        drop_prob = params.drop_prob
        circular_pad = params.circular_pad
        fix_step_size = params.fix_step_size
        share_weights = params.share_weights

        # Data dimensions
        self.num_emaps = params.num_emaps

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
                [ResNet(**resnet_params) for i in range(num_grad_steps)]
            )

        # Declare step sizes for each iteration
        init_step_size = torch.tensor([-2.0], dtype=torch.float32).to(
            params.device
        )
        if fix_step_size:
            self.step_sizes = [init_step_size] * num_grad_steps
        else:
            self.step_sizes = [
                torch.nn.Parameter(init_step_size)
                for i in range(num_grad_steps)
            ]

    def forward(self, kspace, maps, init_image=None, mask=None):
        """
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

        if mask is None:
            mask = cplx.get_mask(kspace)
        kspace *= mask

        # Get data dimensions
        dims = tuple(kspace.size())

        # Declare signal model
        A = SenseModel(maps, weights=mask)

        # Compute zero-filled image reconstruction
        zf_image = A(kspace, adjoint=True)
        image = zf_image if init_image is None else init_image

        # Begin unrolled proximal gradient descent
        for resnet, step_size in zip(self.resnets, self.step_sizes):
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

        return image
