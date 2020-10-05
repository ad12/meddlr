"""
Copyright (c) Facebook, Inc. and its affiliates.
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

import torch
from torch import nn
from torch.nn import functional as F
import torchvision.utils as tv_utils

from .build import META_ARCH_REGISTRY
import ss_recon.utils.complex_utils as cplx
from ss_recon.utils.transforms import SenseModel
from ss_recon.utils.events import get_event_storage

__all__ = ["UnetModel"]

class ConvBlock(nn.Module):
    """
    A Convolutional Block that consists of two convolution layers each followed by
    instance normalization, LeakyReLU activation and dropout.
    """

    def __init__(self, in_chans, out_chans, drop_prob):
        """
        Args:
            in_chans (int): Number of channels in the input.
            out_chans (int): Number of channels in the output.
            drop_prob (float): Dropout probability.
        """
        super().__init__()

        self.in_chans = in_chans
        self.out_chans = out_chans
        self.drop_prob = drop_prob

        self.layers = nn.Sequential(
            nn.Conv2d(in_chans, out_chans, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(out_chans),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Dropout2d(drop_prob),
            nn.Conv2d(out_chans, out_chans, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(out_chans),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Dropout2d(drop_prob)
        )

    def forward(self, input):
        """
        Args:
            input (torch.Tensor): Input tensor of shape [batch_size, self.in_chans, height, width]
        Returns:
            (torch.Tensor): Output tensor of shape [batch_size, self.out_chans, height, width]
        """
        return self.layers(input)

    def __repr__(self):
        return f'ConvBlock(in_chans={self.in_chans}, out_chans={self.out_chans}, ' \
            f'drop_prob={self.drop_prob})'


class TransposeConvBlock(nn.Module):
    """
    A Transpose Convolutional Block that consists of one convolution transpose layers followed by
    instance normalization and LeakyReLU activation.
    """

    def __init__(self, in_chans, out_chans):
        """
        Args:
            in_chans (int): Number of channels in the input.
            out_chans (int): Number of channels in the output.
        """
        super().__init__()

        self.in_chans = in_chans
        self.out_chans = out_chans

        self.layers = nn.Sequential(
            nn.ConvTranspose2d(in_chans, out_chans, kernel_size=2, stride=2, bias=False),
            nn.InstanceNorm2d(out_chans),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
        )

    def forward(self, input):
        """
        Args:
            input (torch.Tensor): Input tensor of shape [batch_size, self.in_chans, height, width]
        Returns:
            (torch.Tensor): Output tensor of shape [batch_size, self.out_chans, height, width]
        """
        return self.layers(input)

    def __repr__(self):
        return f'ConvBlock(in_chans={self.in_chans}, out_chans={self.out_chans})'

@META_ARCH_REGISTRY.register()
class UnetModel(nn.Module):
    """
    PyTorch implementation of a U-Net model.
    This is based on:
        Olaf Ronneberger, Philipp Fischer, and Thomas Brox. U-net: Convolutional networks
        for biomedical image segmentation. In International Conference on Medical image
        computing and computer-assisted intervention, pages 234–241. Springer, 2015.
    """

    def __init__(self, cfg):
        """
        Args:
            in_chans (int): Number of channels in the input to the U-Net model.
            out_chans (int): Number of channels in the output to the U-Net model.
            chans (int): Number of output channels of the first convolution layer.
            num_pool_layers (int): Number of down-sampling and up-sampling layers.
            drop_prob (float): Dropout probability.
        """
        super().__init__()

        in_chans = cfg.MODEL.UNET.IN_CHANNELS
        out_chans = cfg.MODEL.UNET.OUT_CHANNELS
        chans = cfg.MODEL.UNET.CHANNELS
        num_pool_layers = cfg.MODEL.UNET.NUM_POOL_LAYERS
        drop_prob = cfg.MODEL.UNET.DROPOUT
        
        self.in_chans = in_chans
        self.out_chans = out_chans
        self.chans = chans
        self.num_pool_layers = num_pool_layers
        self.drop_prob = drop_prob

        self.down_sample_layers = nn.ModuleList([ConvBlock(in_chans, chans, drop_prob)])
        ch = chans
        for i in range(num_pool_layers - 1):
            self.down_sample_layers += [ConvBlock(ch, ch * 2, drop_prob)]
            ch *= 2
        self.conv = ConvBlock(ch, ch * 2, drop_prob)

        self.up_conv = nn.ModuleList()
        self.up_transpose_conv = nn.ModuleList()
        for i in range(num_pool_layers - 1):
            self.up_transpose_conv += [TransposeConvBlock(ch * 2, ch)]
            self.up_conv += [ConvBlock(ch * 2, ch, drop_prob)]
            ch //= 2

        self.up_transpose_conv += [TransposeConvBlock(ch * 2, ch)]
        self.up_conv += [
            nn.Sequential(
                ConvBlock(ch * 2, ch, drop_prob),
                nn.Conv2d(ch, self.out_chans, kernel_size=1, stride=1),
            )]
        
        self.vis_period = cfg.VIS_PERIOD

    def visualize_training(self, kspace, zfs, targets, preds):
            """A function used to visualize reconstructions.

            TODO: Refactor out

            Args:
                targets: NxHxWx2 tensors of target images.
                preds: NxHxWx2 tensors of predictions.
            """
            storage = get_event_storage()
            
            with torch.no_grad():
                kspace = kspace[0, ..., 0, :].unsqueeze(0).cpu() # calc mask for first coil only
                targets = targets[0, ...].unsqueeze(0).cpu()
                preds = preds[0, ...].unsqueeze(0).cpu()
                zfs = zfs[0, ...].unsqueeze(0).cpu()

                N = preds.shape[0]

                all_images = torch.cat([zfs, preds, targets], dim=2)

                imgs_to_write = {
                    "phases": cplx.angle(all_images),
                    "images": cplx.abs(all_images),
                    "errors": cplx.abs(preds - targets),
                    "masks": cplx.get_mask(kspace),
                }

                for name, data in imgs_to_write.items():
                    data = data.squeeze(-1).unsqueeze(1)
                    data = tv_utils.make_grid(
                        data, nrow=1, padding=1, normalize=True, scale_each=True,
                    )
                    storage.put_image(
                        "train/{}".format(name), data.numpy(), data_format="CHW"
                    )

    def forward(self, input, return_pp=False, vis_training=False):
        """
        Args:
            input (torch.Tensor): Input tensor of shape [batch_size, self.in_chans, height, width]
        Returns:
            (torch.Tensor): Output tensor of shape [batch_size, self.out_chans, height, width]
        """
        stack = []
        
        ## start 
        # Need to fetch device at runtime for proper data transfer.
        inputs = input
        device = next(self.parameters()).device
        kspace = inputs["kspace"].to(device)
        target = inputs["target"].to(device) if "target" in inputs else None
        mask = inputs["mask"].to(device) if "mask" in inputs else None
        A = inputs["signal_model"].to(device) if "signal_model" in inputs else None
        maps = inputs["maps"].to(device)
        #if self.num_emaps != maps.size()[-2]:
         #   raise ValueError(
          #      "Incorrect number of ESPIRiT maps! Re-prep data..."
           # )

        # Move step sizes to the right device.
        #step_sizes = [x.to(device) for x in self.step_sizes]
        
        if mask is None:
            mask = cplx.get_mask(kspace)
        kspace *= mask

        # Get data dimensions
        dims = tuple(kspace.size())

        # Declare signal model.
        if A is None:
            A = SenseModel(maps, weights=mask)

        # Compute zero-filled image reconstruction
        zf_image = A(kspace, adjoint=True)
        zf_dims = zf_image.size()
        output = zf_image.permute(0, 4, 1, 2, 3).squeeze(-1)
        # output = zf_image.view(zf_dims[0], zf_dims[4],zf_dims[1],zf_dims[2]) #a stupid way of reshaping lol
        ## end

        # Apply down-sampling layers
        for i, layer in enumerate(self.down_sample_layers):
            output = layer(output)
            stack.append(output)
            output = F.avg_pool2d(output, kernel_size=2, stride=2, padding=0)

        output = self.conv(output)

        # Apply up-sampling layers
        for transpose_conv, conv in zip(self.up_transpose_conv, self.up_conv):
            downsample_layer = stack.pop()
            output = transpose_conv(output)

            # Reflect pad on the right/botton if needed to handle odd input dimensions.
            padding = [0, 0, 0, 0]
            if output.shape[-1] != downsample_layer.shape[-1]:
                padding[1] = 1 # Padding right
            if output.shape[-2] != downsample_layer.shape[-2]:
                padding[3] = 1 # Padding bottom
            if sum(padding) != 0:
                output = F.pad(output, padding, "reflect")

            output = torch.cat([output, downsample_layer], dim=1)
            output = conv(output)

        # pred = output.view(zf_dims)
        pred = output.unsqueeze(-1).permute(0, 2, 3, 4, 1)
        output_dict = {
            "pred": pred,  # N x Y x Z x 1 x 2
            "target": target,  # N x Y x Z x 1 x 2
        }

        if return_pp:
            output_dict.update({
                k: inputs[k] for k in ["mean", "std", "norm"]
            })

        if self.training and (vis_training or self.vis_period > 0):
            storage = get_event_storage()
            if vis_training or storage.iter % self.vis_period == 0:
                self.visualize_training(kspace, zf_image, target, pred)

        if not self.training:
            output_dict["zf_image"] = zf_image

        return output_dict