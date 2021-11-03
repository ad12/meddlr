"""
Recon Training Script.

This scripts reads a given config file and runs the training or evaluation.
It is an entry point that is made to train standard models in ss_recon.

In order to let one script support training of many models,
this script contains logic that are specific to these built-in models and
therefore may not be suitable for your own project.
For example, your research project perhaps only needs a single "evaluator".

Therefore, we recommend you to use detectron2 as an library and take
this file as an example of how to use the library.
You may want to write your own script with your datasets and other
customizations.
"""

from meddlr.config import get_cfg
from meddlr.engine import DefaultTrainer, default_argument_parser, default_setup
from meddlr.engine.defaults import init_wandb_run
from meddlr.utils.env import supports_wandb


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

    # TODO: Change resume=args.resume once functionality is specified.
    # Currently resuming with the same experiment id overwrites previous data.
    # So for now, even if you are resuming your experiment, it will be logged
    # as a separate run in W&B.
    if supports_wandb():
        init_wandb_run(cfg, resume=False, job_type="training")

    return cfg


def main(args):
    cfg = setup(args)

    if args.eval_only:
        raise NotImplementedError("Evaluation is not yet implemented")

    trainer = DefaultTrainer(cfg)
    trainer.resume_or_load(resume=args.resume, restart_iter=args.restart_iter)
    return trainer.train()


if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    print("Command Line Args:", args)
    main(args)
