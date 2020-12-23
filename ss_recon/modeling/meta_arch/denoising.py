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
import torchvision.utils as tv_utils

from ss_recon.utils.general import move_to_device
import ss_recon.utils.complex_utils as cplx
from ss_recon.utils.transforms import SenseModel

from ..layers.layers2D import ResNet
from .build import META_ARCH_REGISTRY
from ss_recon.utils.events import get_event_storage

__all__ = ["DenoisingModel"]


@META_ARCH_REGISTRY.register()
class DenoisingModel(nn.Module):

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
        noise_std_dev = cfg.MODEL.DENOISING.NOISE.STD_DEV
        assert len(noise_std_dev) == 1, (
            "Noise std dev currently only supports one value."
        )
        self.noise_std_dev = noise_std_dev[0]

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
            act_type=cfg.MODEL.UNROLLED.CONV_BLOCK.ACTIVATION,
            norm_type=cfg.MODEL.UNROLLED.CONV_BLOCK.NORM,
            norm_affine=cfg.MODEL.UNROLLED.CONV_BLOCK.NORM_AFFINE,
            order=cfg.MODEL.UNROLLED.CONV_BLOCK.ORDER,
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
            self.step_sizes = nn.ParameterList([
                torch.nn.Parameter(init_step_size)
                for _ in range(num_grad_steps)
            ])

        self.vis_period = cfg.VIS_PERIOD
    ''''
    def augment(self, inputs):
        """Noise augmentation module.
        TODO: Perform the augmentation here.
        """
        kspace = inputs["kspace"].clone()
        mask = cplx.get_mask(kspace)

        noise_std = self.noise_std_dev
        noise = noise_std * torch.randn(inputs['kspace'].size())
        noise = noise.to(self.device) #wasn't a problem for n2r?
        masked_noise = noise * mask
        aug_kspace = kspace + masked_noise
        import pdb; pdb.set_trace()
        inputs = {k: v.clone() for k, v in inputs.items() if k != "kspace"}
        inputs["kspace"] = aug_kspace
        return inputs
    '''
    
    def augment(self, kspace):
        """Noise augmentation module.
        TODO: Perform the augmentation here.
        """
        #kspace = inputs["kspace"].clone()
        mask = cplx.get_mask(kspace)

        noise_std = self.noise_std_dev
        noise = noise_std * torch.randn(kspace.size())
        noise = noise.to(kspace.device) #wasn't a problem for n2r?
        masked_noise = noise * mask
        aug_kspace = kspace + masked_noise
        #inputs = {k: v.clone() for k, v in inputs.items() if k != "kspace"}
        #inputs["kspace"] = aug_kspace
        return aug_kspace.contiguous()

    def visualize_training(self, kspace, zfs, targets, preds):
        """A function used to visualize reconstructions.

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

    def forward(self, inputs, return_pp=False, vis_training=False):
        """
        TODO: condense into list of dataset dicts.
        Args:
            inputs: Standard ss_recon module input dictionary
                * "kspace": Kspace. If fully sampled, and want to simulate
                    undersampled kspace, provide "mask" argument.
                * "maps": Sensitivity maps
                * "target" (optional): Target image (typically fully sampled).
                * "mask" (optional): Undersampling mask to apply.
                * "signal_model" (optional): The signal model. If provided,
                    "maps" will not be used to estimate the signal model.
                    Use with caution.
            return_pp (bool, optional): If `True`, return post-processing
                parameters "mean", "std", and "norm" if included in the input.
            vis_training (bool, optional): If `True`, force visualize training
                on this pass. Can only be `True` if model is in training mode.

        Returns:
            Dict: A standard ss_recon output dict
                * "pred": The reconstructed image
                * "target" (optional): The target image.
                    Added if provided in the input.
                * "mean"/"std"/"norm" (optional): Pre-processing parameters.
                    Added if provided in the input.
                * "zf_image": The zero-filled image.
                    Added when model is in eval mode.
        """
        if vis_training and not self.training:
            raise ValueError(
                "vis_training is only applicable in training mode."
            )
        # Need to fetch device at runtime for proper data transfer.
        device = self.resnets[0].final_layer.weight.device
        inputs = move_to_device(inputs, device)
        clean_kspace = inputs["kspace"]
        target = inputs.get("target", None)
        mask = inputs.get("mask", None)
        A = inputs.get("signal_model", None)
        maps = inputs["maps"]
        if self.num_emaps != maps.size()[-2]:
            raise ValueError(
                "Incorrect number of ESPIRiT maps! Re-prep data..."
            )

        # Move step sizes to the right device.
        step_sizes = [x.to(device) for x in self.step_sizes]

        if mask is None:
            mask = cplx.get_mask(clean_kspace)
        clean_kspace *= mask

        # Get data dimensions
        dims = tuple(clean_kspace.size())

        # Declare signal model.
        if A is None:
            A = SenseModel(maps, weights=mask)

        # Compute zero-filled image reconstruction
        #inputs_aug = self.augment(inputs)
        #kspace = inputs_aug["kspace"]
        kspace = self.augment(clean_kspace)
        zf_image = A(kspace, adjoint=True)
        target = A(clean_kspace, adjoint=True)

        # Begin unrolled proximal gradient descent
        image = zf_image
        for resnet, step_size in zip(self.resnets, step_sizes):
            # dc update
            grad_x = A(A(image), adjoint=True) - zf_image
            image = image + step_size * grad_x
            use_cplx = cplx.is_complex(image)
            if use_cplx:
                image = torch.view_as_real(image)

            # prox update
            image = image.reshape(dims[0:3] + (self.num_emaps * 2,)).permute(
                0, 3, 1, 2
            )

            image = resnet(image)
            image = image.permute(0, 2, 3, 1).reshape(
                dims[0:3] + (self.num_emaps, 2)
            )
            if use_cplx:
                image = torch.view_as_complex(image)

        output_dict = {
            "pred": image,  # N x Y x Z x 1 x 2
            "target": target,  # N x Y x Z x 1 x 2
        }
        if return_pp:
            output_dict.update({
                k: inputs[k] for k in ["mean", "std", "norm"]
            })

        if self.training and (vis_training or self.vis_period > 0):
            storage = get_event_storage()
            if vis_training or storage.iter % self.vis_period == 0:
                self.visualize_training(kspace, zf_image, target, image)

        if not self.training:
            output_dict["zf_image"] = zf_image

        return output_dict
