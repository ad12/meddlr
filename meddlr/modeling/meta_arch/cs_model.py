"""Compressed Sensing (2D).

This file contains an implementation of the Compressed Sensing
framework by Lustig, et al. using the Python Package sigpy.
See the tutorial for details.

Tutorial:
    https://sigpy.readthedocs.io/en/latest/generated/sigpy.mri.app.L1WaveletRecon.html#sigpy.mri.app.L1WaveletRecon

Reference:
    Lustig, M., Donoho, D., & Pauly, J. M. (2007). Sparse MRI: The application of compressed
    sensing for rapid MR imaging. Magnetic Resonance in Medicine, 58(6), 1082-1195.
"""
import numpy as np
import sigpy as sp
import sigpy.mri as mr
import torch
from torch import nn

import meddlr.ops.complex as cplx
from meddlr.config.config import configurable
from meddlr.forward.mri import SenseModel
from meddlr.utils.general import move_to_device

from .build import META_ARCH_REGISTRY

try:
    import cupy as cp

    _CUPY_AVAILABLE = True
except ImportError:
    cp = None
    _CUPY_AVAILABLE = False

__all__ = ["CSModel"]


@META_ARCH_REGISTRY.register()
class CSModel(nn.Module):
    """Compressed sensing reconstruction with l1 wavelet regularization.

    This class is a PyTorch wrapper around the SigPy's L1WaveletRecon class.
    On each forward pass, each example is reconstructed using :math:`\ell_1`
    wavelet-regularized compressed sensing.

    If the model should run on a GPU, `cupy` must be installed.

    Note:
        Gradients are not supported.

    Attributes:
        device (torch.Device | str): Device to use for execution.
        l1_reg (float): :math:`\ell_1` regularization parameter.
        max_iter (int): Maximum number of iterations.
        num_emaps (int): Number of sensitivity maps.
    """

    @configurable
    def __init__(self, reg: float, max_iter: int, device="cpu", num_emaps: int = 1):
        """
        Args:
            reg (float): The regularization strength.
            max_iter (int): Maximum number of iterations.
            device (str | torch.device, optional): The device to execute on.
            num_emaps (int, optional): Number of estimated sensitivity maps.
                Currently only ``1`` is supported.
        """
        super().__init__()
        if device != torch.device("cpu") and not _CUPY_AVAILABLE:
            raise ModuleNotFoundError(
                f"Requested device {device}, but cupy not installed. "
                f"Install cupy>=9.0 following instructions at "
                f"https://docs.cupy.dev/en/stable/install.html"
            )
        self.device = device

        # Extract network parameters
        self.l1_reg = reg
        self.max_iter = max_iter

        # Data dimensions
        self.num_emaps = num_emaps
        if self.num_emaps != 1:
            raise ValueError("CSModel currently only supports one sensitivity map.")

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
        if inputs["kspace"].shape[0] != 1:
            raise ValueError("Only batch size == 1 is supported in compressed sensing")

        # Need to fetch device at runtime for proper data transfer.
        # device = self.resnets[0].final_layer.weight.device
        device = self.device
        inputs = move_to_device(inputs, device)
        kspace = inputs["kspace"]
        target = inputs.get("target", None)
        mask = inputs.get("mask", None)
        A = inputs.get("signal_model", None)
        maps = inputs["maps"]
        num_maps_dim = -2 if cplx.is_complex_as_real(maps) else -1
        if self.num_emaps != maps.size()[num_maps_dim]:
            raise ValueError("Incorrect number of ESPIRiT maps! Re-prep data...")

        if mask is None:
            mask = cplx.get_mask(kspace)
        kspace *= mask

        # Declare signal model.
        if A is None:
            A = SenseModel(maps, weights=mask)
        zf_image = A(kspace, adjoint=True)

        # Channel-first - (#coils, ky, kz)
        # TODO: Generalize to 3D
        kspace = kspace[0].permute((2, 0, 1))
        maps = maps.squeeze(num_maps_dim)[0].permute((2, 0, 1))
        mask = mask[0].permute((2, 0, 1))

        xp = np if device == torch.device("cpu") else cp
        kspace = xp.asarray(kspace)
        maps = xp.asarray(maps)
        mask = xp.asarray(mask)

        image = mr.app.L1WaveletRecon(
            kspace,
            maps,
            self.l1_reg,
            weights=mask,
            max_iter=self.max_iter,
            device=sp.get_device(kspace),
        ).run()
        image = torch.as_tensor(image, device=device)
        image = image.unsqueeze(0).unsqueeze(-1)

        output_dict = {"pred": image, "target": target}  # N x Y x Z x 1 x 2  # N x Y x Z x 1 x 2
        if return_pp:
            output_dict.update({k: inputs[k] for k in ["mean", "std", "norm"]})

        if not self.training:
            output_dict["zf_image"] = zf_image

        return output_dict

    @classmethod
    def from_config(cls, cfg):
        return {
            "reg": cfg.MODEL.CS.REGULARIZATION,
            "max_iter": cfg.MODEL.CS.MAX_ITER,
            "device": cfg.MODEL.DEVICE,
            "num_emaps": cfg.MODEL.UNROLLED.NUM_EMAPS,
        }
