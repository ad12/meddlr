"""
Copyright (c) Facebook, Inc. and its affiliates.
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

import torch
import torchvision.utils as tv_utils
from torch import nn
from torch.nn import functional as F

import meddlr.ops.complex as cplx
from meddlr.config.config import configurable
from meddlr.forward.mri import SenseModel
from meddlr.utils import transforms as T
from meddlr.utils.events import get_event_storage

from .build import META_ARCH_REGISTRY

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
            nn.Dropout2d(drop_prob),
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
        return (
            f"ConvBlock(in_chans={self.in_chans}, out_chans={self.out_chans}, "
            f"drop_prob={self.drop_prob})"
        )


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
        return f"ConvBlock(in_chans={self.in_chans}, out_chans={self.out_chans})"


@META_ARCH_REGISTRY.register()
class UnetModel(nn.Module):
    """U-Net model as implemented in fastMRI.

    This model is designed for to interface with complex data.
    It may be helpful when benchmarking MRI reconstruction models
    as it is a benchmark model used by fastMRI (https://github.com/facebookresearch/fastMRI).

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        channels (int): Number of output channels for the first convolution layer.
            At all subsequent depths, the number of output channels is doubled.
            For example, if `channels=32`, number of channels will be
            ``[32, 64, 128, ...]``.
        num_pool_layers (int): Number of down-sampling and up-sampling layers.
            By definition, the depth of the network is ``num_pool_layers + 1``.
        dropout (float): Dropout probability.
        use_latent (bool): Whether to use latent features

    Reference:
        Olaf Ronneberger, Philipp Fischer, and Thomas Brox. U-net: Convolutional networks
        for biomedical image segmentation. In International Conference on Medical image
        computing and computer-assisted intervention, pages 234–241. Springer, 2015.
    """

    @configurable
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        channels: int = 32,
        num_pool_layers: int = 4,
        dropout: float = 0.0,
        use_latent: bool = False,
        num_latent_layers: int = 1,
        normalize: bool = False,
        vis_period: int = -1,
    ):
        """
        Args:
            in_chans (int): Number of channels in the input to the U-Net model.
            out_chans (int): Number of channels in the output to the U-Net model.
            chans (int): Number of output channels of the first convolution layer.
            num_pool_layers (int): Number of down-sampling and up-sampling layers.
            drop_prob (float): Dropout probability.
            normalize (bool, optional): If ``True``, normalize the input for stability.
        """
        super().__init__()

        self.in_chans = in_channels
        self.out_chans = out_channels
        self.chans = channels
        self.num_pool_layers = num_pool_layers
        self.drop_prob = dropout
        self.use_latent = use_latent
        self.num_latent_layers = num_latent_layers
        self.normalize = normalize

        self.down_sample_layers = nn.ModuleList([ConvBlock(in_channels, channels, dropout)])
        ch = channels
        for _i in range(num_pool_layers - 1):
            self.down_sample_layers += [ConvBlock(ch, ch * 2, dropout)]
            ch *= 2
        self.conv = ConvBlock(ch, ch * 2, dropout)

        self.up_conv = nn.ModuleList()
        self.up_transpose_conv = nn.ModuleList()
        for _i in range(num_pool_layers - 1):
            self.up_transpose_conv += [TransposeConvBlock(ch * 2, ch)]
            self.up_conv += [ConvBlock(ch * 2, ch, dropout)]
            ch //= 2

        self.up_transpose_conv += [TransposeConvBlock(ch * 2, ch)]
        self.up_conv += [
            nn.Sequential(
                ConvBlock(ch * 2, ch, dropout),
                nn.Conv2d(ch, self.out_chans, kernel_size=1, stride=1),
            )
        ]

        self.vis_period = vis_period

    def register_hooks(self):
        if self.use_latent:
            self.feats = {}
            self.hooks = []
            self.hooks.append(self.conv.register_forward_hook(self.get_latent("E4")))
            for _i in range(self.num_latent_layers - 1):
                k = self.num_pool_layers - 1 - _i
                self.hooks.append(
                    self.up_conv[_i].register_forward_hook(self.get_latent("D" + str(k)))
                )
                self.hooks.append(
                    self.down_sample_layers[k].register_forward_hook(self.get_latent("E" + str(k)))
                )

    def remove_hooks(self):
        for _i in range(self.num_latent_layers - 1):
            self.hooks[_i].remove()

    def get_latent(self, layer_name):
        def hook(module, input, output):
            self.feats[layer_name] = output

        return hook

    def visualize_training(self, kspace, zfs, targets, preds):
        """A function used to visualize reconstructions.

        TODO: Refactor out

        Args:
            targets: NxHxWx2 tensors of target images.
            preds: NxHxWx2 tensors of predictions.
        """
        storage = get_event_storage()

        with torch.no_grad():
            if cplx.is_complex(kspace):
                kspace = torch.view_as_real(kspace)
            if cplx.is_complex(targets) and not cplx.is_complex(zfs):
                # Zero-filled needs to be manually converted.
                zfs = torch.view_as_complex(zfs)
            kspace = kspace[0, ..., 0, :].unsqueeze(0).cpu()  # calc mask for first coil only
            targets = targets[0, ...].unsqueeze(0).cpu()
            preds = preds[0, ...].unsqueeze(0).cpu()
            zfs = zfs[0, ...].unsqueeze(0).cpu()

            all_images = torch.cat([zfs, preds, targets], dim=2)

            imgs_to_write = {
                "phases": cplx.angle(all_images),
                "images": cplx.abs(all_images),
                "errors": cplx.abs(preds - targets),
                "masks": cplx.get_mask(kspace),
            }

            for name, data in imgs_to_write.items():
                data = data.squeeze(-1).unsqueeze(1)
                data = tv_utils.make_grid(data, nrow=1, padding=1, normalize=True, scale_each=True)
                storage.put_image("train/{}".format(name), data.numpy(), data_format="CHW")

    def norm(self, x):
        """
        Args:
            x (torch.Tensor): Input tensor. Shape [B, 2*C, H, W, ...]
        """
        ndim = x.ndim
        x = x.view(x.shape[:2] + (-1,))

        mean = x.mean(dim=-1).reshape(x.shape[:2] + (1,) * (ndim - 2))
        std = x.std(dim=-1).reshape(x.shape[:2] + (1,) * (ndim - 2))

        return (x - mean) / std, mean, std

    def unnorm(self, x, mean, std):
        return x * std + mean

    def base_forward(self, zf_image):
        """The base forward function.

        Args:
            zf_image (torch.Tensor): Zero-filled image. Shape [B, C, ..., H, W].
                The real and imaginary channels should be part of the channels
                dimension if the input is complex and the model does not have
                complex weights.

        Returns:
            torch.Tensor: Reconstructed image. Shape [B, C, ..., H, W].
        """
        x = zf_image

        stack = []
        if self.normalize:
            x, mean, std = self.norm(x)

        # Apply down-sampling layers
        for layer in self.down_sample_layers:
            x = layer(x)
            stack.append(x)
            x = F.avg_pool2d(x, kernel_size=2, stride=2, padding=0)

        x = self.conv(x)

        # Apply up-sampling layers
        for transpose_conv, conv in zip(self.up_transpose_conv, self.up_conv):
            downsample_layer = stack.pop()
            x = transpose_conv(x)

            # Reflect pad on the right/botton if needed to handle odd input dimensions.
            padding = [0, 0, 0, 0]
            if x.shape[-1] != downsample_layer.shape[-1]:
                padding[1] = 1  # Padding right
            if x.shape[-2] != downsample_layer.shape[-2]:
                padding[3] = 1  # Padding bottom
            if sum(padding) != 0:
                x = F.pad(x, padding, "reflect")

            x = torch.cat([x, downsample_layer], dim=1)
            x = conv(x)

        if self.normalize:
            x = self.unnorm(x, mean, std)

        return x

    def forward(self, input, return_pp=False, vis_training=False):
        """
        Args:
            input (torch.Tensor): Input tensor of shape [batch_size, self.in_chans, height, width]
        Returns:
            (torch.Tensor): Output tensor of shape [batch_size, self.out_chans, height, width]
        """
        self.register_hooks()

        # Need to fetch device at runtime for proper data transfer.
        inputs = input
        device = next(self.parameters()).device
        kspace = inputs["kspace"].to(device)
        target = inputs["target"].to(device) if "target" in inputs else None
        mask = inputs["mask"].to(device) if "mask" in inputs else None
        A = inputs["signal_model"].to(device) if "signal_model" in inputs else None
        maps = inputs["maps"].to(device)

        if mask is None:
            mask = cplx.get_mask(kspace)
        kspace *= mask

        # Declare signal model.
        if A is None:
            A = SenseModel(maps, weights=mask)

        # Zero-filled Sense Recon.
        if cplx.is_complex(maps):
            zf_image = A(kspace, adjoint=True)
        # Zero-filled RSS Recon.
        else:
            zf_image_init = T.ifft2(kspace)
            zf_image_rss = torch.sqrt(torch.sum(cplx.abs(zf_image_init) ** 2, axis=-1))
            zf_image = torch.complex(zf_image_rss, torch.zeros_like(zf_image_rss)).unsqueeze(-1)

        use_cplx = cplx.is_complex(zf_image)
        if use_cplx:
            zf_image = torch.view_as_real(zf_image)

        num_maps = zf_image.shape[-2]
        if num_maps > 1:
            output = zf_image.reshape(zf_image.shape[:-2] + (-1,)).unsqueeze(-2)
        else:
            output = zf_image
        output = output.permute(0, 4, 1, 2, 3).squeeze(-1)

        # Run U-Net.
        output = self.base_forward(output)

        if num_maps > 1:
            pred = output.permute(0, 2, 3, 1)
            pred = pred.reshape(pred.shape[:-1] + (num_maps, 2))
        else:
            pred = output.unsqueeze(-1).permute(0, 2, 3, 4, 1)

        if use_cplx:
            pred = torch.view_as_complex(pred.contiguous())

        # pred: shape [batch, height, width, #maps, 2]
        # target: shape [batch, height, width, #maps, 2]
        output_dict = {
            "pred": pred,
            "target": target,
            "signal_model": A,
        }

        if return_pp:
            output_dict.update({k: inputs[k] for k in ["mean", "std", "norm"]})

        # TODO (arjundd): Figure out why this is needed during SSDU training.
        images_available = all(x is not None for x in [kspace, zf_image, target, pred])
        if images_available and self.training and (vis_training or self.vis_period > 0):
            storage = get_event_storage()
            if vis_training or storage.iter % self.vis_period == 0:
                self.visualize_training(kspace, zf_image, target, pred)

        if use_cplx:
            zf_image = torch.view_as_complex(zf_image)
        output_dict["zf_image"] = zf_image

        if self.use_latent:
            output_dict["latent"] = self.feats

        self.remove_hooks()

        return output_dict

    @classmethod
    def from_config(cls, cfg):
        return {
            "in_channels": cfg.MODEL.UNET.IN_CHANNELS,
            "out_channels": cfg.MODEL.UNET.OUT_CHANNELS,
            "channels": cfg.MODEL.UNET.CHANNELS,
            "num_pool_layers": cfg.MODEL.UNET.NUM_POOL_LAYERS,
            "dropout": cfg.MODEL.UNET.DROPOUT,
            "use_latent": cfg.get_recursive("MODEL.CONSISTENCY.USE_LATENT", False),
            "num_latent_layers": cfg.get_recursive("MODEL.CONSISTENCY.NUM_LATENT_LAYERS", 1),
            "normalize": cfg.MODEL.UNET.NORMALIZE,
            "vis_period": cfg.VIS_PERIOD,
        }
