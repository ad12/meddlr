import logging
import os
import pathlib
import shutil

import torch

from meddlr.config import CfgNode
from meddlr.evaluation.testing import find_weights
from meddlr.utils import env
from meddlr.utils.general import move_to_device

logger = logging.getLogger(__name__)


def prepare_model(
    cfg: CfgNode,
    name: str,
    output_dir: str,
    weights_file: str = None,
    criterion: str = None,
    dry_run: bool = False,
    device: str = "cpu",
) -> pathlib.Path:
    """Processes an experiment folder for model deployment.

    This function finds the model checkpoint (based on the val_loss criterion)
    and config files in the experiment folder. If the model checkpoint is on the
    GPU, it is moved to the CPU. The model is then saved to the output directory.

    If your model training was performed with some accelerator (e.g. GPU),
    this function must be run on a node with that accelerator.

    Important fields to set in the config:
        - DESCRIPTION.BRIEF: A brief description of the model.
        - DESCRIPTION.TAGS: A list of tags to describe the model.
        - OUTPUT_DIR: The output directory for users who want to use
          the config for training. This should start with ``results://``.

    Note:
        This function is in pre-alpha stage. There may be several breaking changes
        to this function.

    Examples:
        >>> prepare_model(cfg, "my_model", "shareable-models", criterion="val_ssim")

    Returns:
        os.PathLike: The path to the directory containing the formatted model and config.
    """
    path = cfg.OUTPUT_DIR
    # project = cfg.DESCRIPTION.PROJECT_NAME
    cfg = cfg.clone().defrost()

    path_manager = env.get_path_manager()

    logger.info(f"================== {name} ==================")
    cfg.OUTPUT_DIR = path_manager.get_local_path(path)
    path_manager.get_local_path(os.path.join(path, "metrics.csv"))

    if weights_file is None:
        weights_file, _, _ = find_weights(cfg, criterion=criterion)

    fmt_name = name.replace(" ", "_").replace("(", "").replace(")", "").replace(",", "_")
    exp_out_dir = os.path.join(output_dir, fmt_name)

    if dry_run:
        logger.info(f"{name}: {weights_file} -> {exp_out_dir}")
        return

    if os.path.exists(exp_out_dir):
        shutil.rmtree(exp_out_dir)
    os.makedirs(exp_out_dir, exist_ok=True)

    weights = torch.load(weights_file)
    weights = move_to_device(weights, device=device)
    torch.save(weights, os.path.join(exp_out_dir, "model.ckpt"))

    # Config
    out_cfg_file = f"{exp_out_dir}/config.yaml"
    cfg.DESCRIPTION.EXP_NAME = name
    cfg.DESCRIPTION.ENTITY_NAME = ""
    cfg.DESCRIPTION.PROJECT_NAME = ""
    cfg.MODEL.DEVICE = device
    # exp_type = project if project in exp_name.lower() else "baseline"
    # cfg.DESCRIPTION.TAGS = (
    #     exp_type,
    #     exp_name.split(" ")[0].lower(),
    #     str(cfg.AUG_TRAIN.UNDERSAMPLE.ACCELERATIONS[0]) + "x",
    #     dataset.split("/")[0],
    # )
    cfg.TEST.VAL_METRICS.RECON = [
        metric for metric in cfg.TEST.VAL_METRICS.RECON if "ssim_old" not in metric
    ]
    with open(out_cfg_file, "w") as f:
        f.write(cfg.dump())
    return pathlib.Path(exp_out_dir)
