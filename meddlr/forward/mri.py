import torch
from torch import nn

import meddlr.ops as oF
import meddlr.ops.complex as cplx

__all__ = ["SenseModel"]


class SenseModel(nn.Module):
    """
    A module that computes forward and adjoint SENSE operations.

    The forward operation converts a complex image -> multi-coil kspace.
    The adjoint operation converts multi-coil kspace -> a complex image.

    This module also supports multiple sensitivity maps. This is useful if
    you would like to generate images from multiple estimated sensitivity maps.
    This module also works with single coil inputs as long as the #coils dimension
    is set to 1.

    Attributes:
        maps (torch.Tensor): Sensitivity maps. Shape ``(B, H, W, #coils, #maps, [2])``.
        weights (torch.Tensor, optional): Undersampling masks (if applicable).
            Shape ``(B, H, W)`` or ``(B, H, W, #coils, #coils)``.
    """

    def __init__(self, maps: torch.Tensor, weights: torch.Tensor = None):
        """
        Args:
            maps (torch.Tensor): Sensitivity maps.
            weights (torch.Tensor): Undersampling masks.
                If ``None``, it is assumed that inputs are fully-sampled.
        """
        super().__init__()

        self.maps = maps
        if weights is None:
            self.weights = 1.0
        else:
            self.weights = weights

    def _adjoint_op(self, kspace):
        """
        Args:
            kspace: Shape (B,H,W,#coils,[2])
        Returns:
            image: Shape (B,H,W,#maps,[2])
        """
        image = oF.ifft2c(self.weights * kspace, channels_last=True)
        if cplx.is_complex_as_real(kspace):
            image = cplx.mul(image.unsqueeze(-2), cplx.conj(self.maps))  # [B,...,#coils,#maps,2]
            return image.sum(-3)
        else:
            # This is a hacky solution managing multi-channel inputs.
            # Note multi-channel inputs are only supported in complex tensors.
            # TODO (arjundd, issue #18): Fix with tensor ordering.
            if image.ndim != self.maps.ndim:
                image = image.unsqueeze(-1)

            image = cplx.mul(image, cplx.conj(self.maps))  # [B,...,#coils,#maps,1]
            return image.sum(-2)

    def _forward_op(self, image):
        """
        Args:
            image: Shape (B,H,W,#maps,[2])
        Returns:
            kspace: Shape (B,H,W,#coils,[2])
        """
        if cplx.is_complex_as_real(image):
            kspace = cplx.mul(image.unsqueeze(-3), self.maps)  # [B,...,1,#maps,2]
            kspace = self.weights * oF.fft2c(kspace.sum(-2), channels_last=True)  # [B,...,#coils,2]
        else:
            kspace = cplx.mul(image.unsqueeze(-2), self.maps)
            # This is a hacky solution managing multi-channel inputs.
            # Note this change means that multiple maps with multi-channel inputs
            # is not supported for forward operations. This will change in future updates.
            # TODO (arjundd, issue #18): Fix with tensor ordering.
            if image.shape[-1] == self.maps.shape[-1]:
                kspace = kspace.sum(-1)
            kspace = self.weights * oF.fft2c(kspace, channels_last=True)
        return kspace

    def forward(self, input: torch.Tensor, adjoint: bool = False):
        """Run forward or adjoint SENSE operation on the input.

        Depending on if ``adjoint=True``, the input should either be the
        k-space or the complex image. The shapes for these are as follows:
            - kspace: ``(B, H, W, #coils, [2])
            - image: ``(B, H, W, #maps, [2])``

        Args:
            input (torch.Tensor): If ``adjoint=True``, this is the multi-coil k-space,
                else it is the image.
            adjoint (bool, optional): If ``True``, use adjoint operation.

        Returns:
            torch.Tensor: If ``adjoint=True``, the image, else multi-coil k-space.
        """
        if adjoint:
            output = self._adjoint_op(input)
        else:
            output = self._forward_op(input)
        return output


def hard_data_consistency(
    image: torch.Tensor, acq_kspace: torch.Tensor, mask: torch.Tensor, maps: torch.Tensor
):
    """Hard project acquired k-space into reconstructed k-space.

    Args:
        image: The reconstructed image. Shape ``(B, H, W, #maps, [2])``.
        acq_kspace: The acquired k-space. Shape ``(B, H, W, #coils, [2])``.
        mask: The consistency mask. Shape ``(B, H, W)``.
        maps: The sensitivity maps. Shape ``(B, H, W, #coils, #maps, [2])``.

    Returns:
        torch.Tensor: The projected image. Shape ``(B, H, W, #maps, [2])``.
    """
    # Do not pass the mask to the SenseModel. We do not want to mask out any k-space values.
    device = image.device
    A = SenseModel(maps=maps.to(device))
    kspace = A(image, adjoint=False)
    # TODO (arjundd): Profile this operation. It may be faster to do torch.where.
    # Performance may also depend on the device.
    if mask.dtype != torch.bool:
        mask = mask.bool()
    mask = mask.to(device)
    acq_kspace = acq_kspace.to(device)
    kspace = mask * acq_kspace + (~mask) * kspace
    recon = A(kspace, adjoint=True)
    return recon
