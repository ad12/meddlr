# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Detection Training Script.

This scripts reads a given config file and runs the training or evaluation.
It is an entry point that is made to train standard models in detectron2.

In order to let one script support training of many models,
this script contains logic that are specific to these built-in models and
therefore may not be suitable for your own project.
For example, your research project perhaps only needs a single "evaluator".

Therefore, we recommend you to use detectron2 as an library and take
this file as an example of how to use the library.
You may want to write your own script with your datasets and other
customizations.
"""
import os
import sys
import warnings

from ss_recon.config import get_cfg
from ss_recon.engine import (
    DefaultTrainer,
    default_argument_parser,
    default_setup,
)
from ss_recon.utils.env import supports_wandb

try:
    import wandb
except:
    pass

def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)
    opts = args.opts
    if opts and opts[0] == "--":
        opts = opts[1:]
    cfg.merge_from_list(opts)
    cfg.freeze()
    
    if not cfg.OUTPUT_DIR:
        raise ValueError("OUTPUT_DIR not specified")

    default_setup(cfg, args)

    # TODO: Add suppport for resume.
    if supports_wandb():
        exp_name = cfg.DESCRIPTION.EXP_NAME
        if not exp_name:
            warnings.warn("DESCRIPTION.EXP_NAME not specified. Defaulting to basename...")
            exp_name = os.path.basename(cfg.OUTPUT_DIR)
        wandb.init(
            project="ss_recon",
            name=exp_name,
            config=cfg,
            sync_tensorboard=True,
            job_type="training",
            dir=cfg.OUTPUT_DIR,
            entity="ss_recon",
            tags=cfg.DESCRIPTION.TAGS,
            notes=cfg.DESCRIPTION.BRIEF,
        )
    return cfg


def main(args):
    cfg = setup(args)

    if args.eval_only:
        raise NotImplementedError("Evaluation is not yet implemented")
        # model = DefaultTrainer.build_model(cfg)
        # DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
        #     cfg.MODEL.WEIGHTS, resume=args.resume
        # )
        # res = Trainer.test(cfg, model)
        # if comm.is_main_process():
        #     verify_results(cfg, res)
        # if cfg.TEST.AUG.ENABLED:
        #     res.update(Trainer.test_with_TTA(cfg, model))
        # return res

    trainer = DefaultTrainer(cfg)
    trainer.resume_or_load(resume=args.resume)
    return trainer.train()


if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    print("Command Line Args:", args)
    main(args)
