from numbers import Number
from typing import Any, Dict, Sequence, Union

import torch
import torchvision.utils as tv_utils
from torch import nn

import meddlr.ops.complex as cplx
from meddlr.config import CfgNode
from meddlr.config.config import configurable
from meddlr.forward.mri import SenseModel
from meddlr.utils.events import get_event_storage
from meddlr.utils.general import move_to_device

from ..layers.layers2D import ResNet
from .build import META_ARCH_REGISTRY, build_model

__all__ = ["GeneralizedUnrolledCNN"]


@META_ARCH_REGISTRY.register()
class GeneralizedUnrolledCNN(nn.Module):
    """Unrolled compressed sensing model.

    This implementation is adapted from:
    https://github.com/MRSRL/dl-cs

    Reference:
        CM Sandino, JY Cheng, et al. "Compressed Sensing: From Research to
        Clinical Practice with Deep Neural Networks" IEEE Signal Processing
        Magazine, 2020.
    """

    @configurable
    def __init__(
        self,
        blocks: Union[nn.Module, Sequence[nn.Module]],
        step_sizes: Union[float, Sequence[float]] = -2.0,
        fix_step_size: bool = False,
        num_emaps: int = 1,
        vis_period: int = -1,
        num_grad_steps: int = None,
    ):
        """
        Args:
            blocks: A sequence of blocks
            step_sizes: Step size for data consistency prior to each block.
                If a single float is given, the same step size is used for all blocks.
            fix_step_size: Whether to fix the step size to a given value --
                i.e. set to ``True`` to make the step size non-trainable.
            num_emaps: Number of sensitivity maps used to estimate the image.
            vis_period: Number of steps between logging visualizations.
            num_grad_steps: Number of unrolled steps in the network.
                This is deprecated - the number of steps will be determined
                from the length of ``blocks``.
        """
        super().__init__()

        self.resnets = blocks
        if num_grad_steps is None:
            if isinstance(blocks, Sequence) and not isinstance(blocks, nn.ModuleList):
                blocks = nn.ModuleList(blocks)
            if not isinstance(blocks, nn.ModuleList):
                raise TypeError("`blocks` must be a sequence of nn.Modules or a nn.ModuleList")
            num_grad_steps = len(blocks)
            num_repeat_steps = 0
        else:
            if not isinstance(num_grad_steps, int) or num_grad_steps <= 0:
                raise ValueError("`num_grad_steps` must be positive integer")
            num_repeat_steps = num_grad_steps

        if isinstance(step_sizes, Number):
            step_sizes = [
                torch.tensor([step_sizes], dtype=torch.float32) for _ in range(num_grad_steps)
            ]
        else:
            if len(step_sizes) != num_grad_steps:
                raise ValueError(
                    "`step_sizes` must be a single value or a list of the "
                    "same length as `blocks` or `num_grad_steps`"
                )
            step_sizes = [torch.tensor(s) for s in step_sizes]
        if not fix_step_size:
            step_sizes = nn.ParameterList([nn.Parameter(s) for s in step_sizes])
        self.step_sizes: Sequence[Union[torch.Tensor, nn.Parameter]] = step_sizes

        self.num_repeat_steps = num_repeat_steps
        self.num_emaps = num_emaps
        self.vis_period = vis_period

    def visualize_training(
        self, kspace: torch.Tensor, zfs: torch.Tensor, targets: torch.Tensor, preds: torch.Tensor
    ):
        """Visualize kspace data and reconstructions.

        Dimension ``(,2)`` indicates optional dimension for real-valued view of complex tensors.
        For example, a real-valued tensor of shape BxHxWx2 will be interpreted as
        a complex-valued tensor of shape BxHxW.

        Args:
            kspace: The complex-valued kspace. Shape: [batch, height, width, #coils, (,2)].
            zfs: The complex-valued zero-filled images.
                Shape: [batch, height, width, (,2)].
            targets: The complex-valued target (reference) images.
                Shape: [batch, height, width, (,2)].
            preds: The complex-valued predicted images.
                Shape: [batch, height, width, (,2)].
        """
        storage = get_event_storage()

        with torch.no_grad():
            if cplx.is_complex(kspace):
                kspace = torch.view_as_real(kspace)
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

    def forward(self, inputs: Dict[str, Any], return_pp: bool = False, vis_training: bool = False):
        """Reconstructs the image from the kspace.

        Dimension ``(,2)`` indicates optional dimension for real-valued view of complex tensors.
        For example, a real-valued tensor of shape BxHxWx2 will be interpreted as
        a complex-valued tensor of shape BxHxW.

        ``#maps`` refers to the number of sensitivity maps used to estimate the image
        (i.e. ``self.num_emaps``).

        Args:
            inputs: Standard meddlr module input dictionary
                * "kspace": The kspace (typically undersampled).
                  Shape: [batch, height, width, #coils, (,2)].
                * "maps": The sensitivity maps used for SENSE coil combination.
                  Shape: [batch, height, width, #coils, #maps, (,2)].
                * "target" (optional): Target (reference) image.
                  Shape: [batch, height, width, #maps, (,2)].
                * "signal_model" (optional): The signal model. If provided,
                    "maps" will not be used to estimate the signal model.
                    Use with caution.
            return_pp (bool, optional): If `True`, return post-processing
                parameters "mean", "std", and "norm" if included in the input.
            vis_training (bool, optional): If `True`, force visualize training
                on this pass. Can only be `True` if model is in training mode.

        Returns:
            Dict: A standard meddlr output dict
                * "pred": The reconstructed image
                * "target" (optional): The target image.
                    Added if provided in the input.
                * "mean"/"std"/"norm" (optional): Pre-processing parameters.
                    Added if provided in the input.
                * "zf_image": The zero-filled image.
                    Added when model is in eval mode.
        """
        if self.num_repeat_steps > 0:
            conv_blocks = [self.resnets] * self.num_repeat_steps
        else:
            conv_blocks = self.resnets

        if vis_training and not self.training:
            raise ValueError("vis_training is only applicable in training mode.")
        # Need to fetch device at runtime for proper data transfer.
        device = next(conv_blocks[0].parameters()).device
        inputs = move_to_device(inputs, device)
        kspace = inputs["kspace"]
        target = inputs.get("target", None)
        mask = inputs.get("mask", None)
        A = inputs.get("signal_model", None)
        maps = inputs["maps"]
        num_maps_dim = -2 if cplx.is_complex_as_real(maps) else -1
        if self.num_emaps != maps.size()[num_maps_dim] and maps.size()[num_maps_dim] != 1:
            raise ValueError("Incorrect number of ESPIRiT maps! Re-prep data...")

        # Move step sizes to the right device.
        step_sizes = [x.to(device) for x in self.step_sizes]

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

        # Begin unrolled proximal gradient descent
        image = zf_image
        for resnet, step_size in zip(conv_blocks, step_sizes):
            # dc update
            grad_x = A(A(image), adjoint=True) - zf_image
            image = image + step_size * grad_x

            # If the image is a complex tensor, we view it as a real image
            # where last dimension has 2 channels (real, imaginary).
            # This may take more time, but is done for backwards compatibility
            # reasons.
            # TODO (arjundd): Fix to auto-detect which version of the model is
            # being used.
            use_cplx = cplx.is_complex(image)
            if use_cplx:
                image = torch.view_as_real(image)

            # prox update
            image = image.reshape(dims[0:3] + (self.num_emaps * 2,)).permute(0, 3, 1, 2)
            if hasattr(resnet, "base_forward") and callable(resnet.base_forward):
                image = resnet.base_forward(image)
            else:
                image = resnet(image)

            image = image.permute(0, 2, 3, 1).reshape(dims[0:3] + (self.num_emaps, 2))
            if not image.is_contiguous():
                image = image.contiguous()
            if use_cplx:
                image = torch.view_as_complex(image)

        output_dict = {"pred": image, "target": target}  # N x Y x Z x 1 x 2  # N x Y x Z x 1 x 2

        if return_pp:
            output_dict.update({k: inputs[k] for k in ["mean", "std", "norm"]})

        if self.training and (vis_training or self.vis_period > 0):
            storage = get_event_storage()
            if vis_training or storage.iter % self.vis_period == 0:
                self.visualize_training(kspace, zf_image, target, image)

        output_dict["zf_image"] = zf_image

        return output_dict

    @classmethod
    def from_config(cls, cfg: CfgNode, **kwargs) -> "GeneralizedUnrolledCNN":
        """Build :cls:`GeneralizedUnrolledCNN` from a config.

        Args:
            cfg: The config.
            kwargs: Keyword arguments to override config-specified parameters.

        Returns:
            Dict[str, Any]: The parameters to pass to the constructor.
        """
        # Extract network parameters
        num_grad_steps = cfg.MODEL.UNROLLED.NUM_UNROLLED_STEPS
        share_weights = cfg.MODEL.UNROLLED.SHARE_WEIGHTS

        # Data dimensions
        num_emaps = cfg.MODEL.UNROLLED.NUM_EMAPS

        # Determine block to use for each unrolled step.
        if cfg.MODEL.UNROLLED.BLOCK_ARCHITECTURE == "ResNet":
            builder = lambda: _build_resblock(cfg)  # noqa: E731
        else:
            # TODO: Fix any inconsistencies between config's IN_CHANNELS
            # and the number of channels that the unrolled net expects.
            mcfg = cfg.clone().defrost()
            mcfg.MODEL.META_ARCHITECTURE = cfg.MODEL.UNROLLED.BLOCK_ARCHITECTURE
            mcfg = mcfg.freeze()
            builder = lambda: build_model(mcfg)  # noqa: E731

        # Declare ResNets and RNNs for each unrolled iteration
        if share_weights:
            blocks = builder()
        else:
            blocks = nn.ModuleList([builder() for _ in range(num_grad_steps)])

        # Step sizes
        step_sizes = cfg.MODEL.UNROLLED.STEP_SIZES
        if len(step_sizes) == 1:
            step_sizes = step_sizes[0]

        out = {
            "blocks": blocks,
            "step_sizes": step_sizes,
            "fix_step_size": cfg.MODEL.UNROLLED.FIX_STEP_SIZE,
            "num_emaps": num_emaps,
            "vis_period": cfg.VIS_PERIOD,
            "num_grad_steps": num_grad_steps if share_weights else None,
        }
        out.update(kwargs)
        return out


def _build_resblock(cfg: CfgNode) -> ResNet:
    """Build the resblock for unrolled network.

    Args:
        cfg (CfgNode): The network configuration.

    Note:
        This is a temporary method used as a base case for building
        unrolled networks with the default resblocks. In the future,
        this will be handled by :func:`meddlr.modeling.meta_arch.build_model`.
    """
    # Data dimensions
    num_emaps = cfg.MODEL.UNROLLED.NUM_EMAPS

    # ResNet parameters
    kernel_size = cfg.MODEL.UNROLLED.KERNEL_SIZE
    if len(kernel_size) == 1:
        kernel_size = kernel_size[0]
    resnet_params = dict(
        num_resblocks=cfg.MODEL.UNROLLED.NUM_RESBLOCKS,
        in_chans=2 * num_emaps,  # complex -> real/imag
        chans=cfg.MODEL.UNROLLED.NUM_FEATURES,
        kernel_size=kernel_size,
        drop_prob=cfg.MODEL.UNROLLED.DROPOUT,
        circular_pad=cfg.MODEL.UNROLLED.PADDING == "circular",
        act_type=cfg.MODEL.UNROLLED.CONV_BLOCK.ACTIVATION,
        norm_type=cfg.MODEL.UNROLLED.CONV_BLOCK.NORM,
        norm_affine=cfg.MODEL.UNROLLED.CONV_BLOCK.NORM_AFFINE,
        order=cfg.MODEL.UNROLLED.CONV_BLOCK.ORDER,
    )

    return ResNet(**resnet_params)
