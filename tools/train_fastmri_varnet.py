"""Train fastMRI with a varnet model."""
from typing import Any, Dict

import torch
import torchvision.utils as tv_utils
from fastmri.models.varnet import VarNet
from torch import nn
from train_net import setup

import meddlr.ops.complex as cplx
from meddlr.engine.defaults import default_argument_parser
from meddlr.engine.trainer import DefaultTrainer
from meddlr.utils.events import get_event_storage
from meddlr.utils.general import move_to_device


class VarNetWrapper(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = VarNet()
        self._device = None
        self.vis_period = 100

    def visualize_training(self, mask: torch.Tensor, targets: torch.Tensor, preds: torch.Tensor):
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
            mask = mask[0, ..., 0, :].unsqueeze(0).cpu()  # calc mask for first coil only
            targets = targets[0, ...].unsqueeze(0).cpu()
            preds = preds[0, ...].unsqueeze(0).cpu()

            all_images = torch.cat([preds, targets], dim=2)

            imgs_to_write = {
                "images": cplx.abs(all_images),
                "errors": cplx.abs(preds - targets),
                "masks": mask,
            }

            for name, data in imgs_to_write.items():
                data = data.squeeze(-1).unsqueeze(1)
                data = tv_utils.make_grid(data, nrow=1, padding=1, normalize=True, scale_each=True)
                storage.put_image("train/{}".format(name), data.numpy(), data_format="CHW")

    def forward(self, inputs: Dict[str, Any], return_pp: bool = False, vis_training: bool = False):
        if self._device is None:
            self._device = next(self.model.parameters()).device

        inputs = dict(
            masked_kspace=inputs["kspace"],
            mask=inputs["mask"],
            target=cplx.abs(inputs["target"]),
            num_low_frequencies=0,
        )
        inputs = move_to_device(inputs, self._device)
        pred = self.model(**inputs)
        output_dict = dict(pred=pred, target=inputs["target"])

        if return_pp:
            output_dict.update({k: inputs[k] for k in ["mean", "std", "norm"]})

        if self.training and (vis_training or self.vis_period > 0):
            storage = get_event_storage()
            if vis_training or storage.iter % self.vis_period == 0:
                self.visualize_training(inputs["mask"], inputs["target"], pred)

        return output_dict


def main(args):
    cfg = setup(args)

    if args.eval_only:
        raise NotImplementedError("Evaluation is not yet implemented")

    model = VarNetWrapper()
    trainer = DefaultTrainer(cfg, model=model)
    trainer.resume_or_load(resume=args.resume, restart_iter=args.restart_iter)
    return trainer.train()


if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    print("Command Line Args:", args)
    main(args)
