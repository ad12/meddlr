import logging
import math
import os
import re
from collections import OrderedDict
from typing import Mapping, Sequence, Union

from torch.nn import DataParallel

from meddlr.checkpoint import Checkpointer
from meddlr.config import CfgNode
from meddlr.data import build_recon_train_loader, build_recon_val_loader
from meddlr.engine import SimpleTrainer, hooks
from meddlr.evaluation import (
    DatasetEvaluator,
    ReconEvaluator,
    inference_on_dataset,
    print_csv_format,
    verify_results,
)
from meddlr.modeling import build_loss_computer, build_model
from meddlr.solver import GradAccumOptimizer, build_lr_scheduler, build_optimizer
from meddlr.utils import env
from meddlr.utils.events import CommonMetricPrinter, JSONWriter, TensorboardXWriter
from meddlr.utils.logger import setup_logger

__all__ = ["DefaultTrainer"]


def format_as_iter(vals: Union[int, Sequence[int]], iters_per_epoch: int, time_scale: str):
    """Format data to be at iteration time scale.

    If values are negative, they correspond to the opposite time scale.
    For example if `time_scale="epoch"` and `val=-1`, `val` corresponds to
    1 iteration.

    Args:
        vals (`int(s)`): Values to format.
        iters_per_epoch (int): Number of iterations per epoch.
        time_scale (str): Time scale of current values.

    Returns:
        int: Time (positive int) formatted in iteration format
    """
    assert time_scale in ["epoch", "iter"]

    single_value = not isinstance(vals, Sequence)
    if single_value:
        vals = [vals]

    epoch_convention = (time_scale == "epoch" and all(x >= 0 for x in vals)) or (
        time_scale == "iter" and all(x <= 0 for x in vals)
    )

    if epoch_convention:
        vals = type(vals)([iters_per_epoch * abs(x) for x in vals])
    else:
        vals = type(vals)([abs(x) for x in vals])

    if single_value:
        return vals[0]
    else:
        return vals


def _convert_time_recursive(
    entity: Union[Mapping, Sequence],
    iters_per_epoch,
    time_scale,
    patterns=("^.*_iters$", "^.*_iter$", "^.*_milestones$"),
):
    otype = type(entity)
    if isinstance(entity, (list, tuple)):
        out = []
        for e in entity:
            out.append(_convert_time_recursive(e, iters_per_epoch, time_scale, patterns))
        return otype(out)
    elif isinstance(entity, Mapping):
        entity = {k: v for k, v in entity.items()}
        for k in entity.keys():
            if not isinstance(entity[k], Mapping):
                if any(re.match(pattern, k) for pattern in patterns):
                    entity[k] = format_as_iter(entity[k], iters_per_epoch, time_scale)
                elif not isinstance(entity[k], (list, tuple)):
                    continue

            entity[k] = _convert_time_recursive(entity[k], iters_per_epoch, time_scale, patterns)
        return otype(entity)
    else:
        return entity


def convert_cfg_time_to_iter(cfg: CfgNode, iters_per_epoch: int, ignore_missing: bool = False):
    """Convert all config time-related parameters to iteration scale.

    Args:
        cfg (CfgNode): The config to
        iters_per_epoch (int): Number of iterations per epoch.
        ignore_missing (bool, optional): If ``True``, silently skip over missing keys.
    """
    cfg = cfg.clone()
    cfg.defrost()

    time_scale = cfg.TIME_SCALE

    time_fmt = (
        "SOLVER.MAX_ITER",
        "SOLVER.STEPS",
        "SOLVER.CHECKPOINT_PERIOD",
        "TEST.EVAL_PERIOD",
        "VIS_PERIOD",
        "MODEL.CONSISTENCY.AUG.NOISE.SCHEDULER.WARMUP_ITERS",
    )
    recursively_fmt = (
        "AUG_TRAIN.MRI_RECON.TRANSFORMS",
        "AUG_TRAIN.MRI_RECON.SCHEDULER_P",
        "MODEL.CONSISTENCY.AUG.MRI_RECON.TRANSFORMS",
        "MODEL.CONSISTENCY.AUG.MRI_RECON.SCHEDULER_P",
    )

    for keys_to_fmt, func in [
        (time_fmt, format_as_iter),
        (recursively_fmt, _convert_time_recursive),
    ]:
        for key in keys_to_fmt:
            try:
                curr_val = cfg.get_recursive(key)
            except KeyError:
                if ignore_missing:
                    continue
                raise KeyError(f"cfg does not contain key '{key}'")
            value = func(curr_val, iters_per_epoch, time_scale)
            cfg.set_recursive(key, value)

    cfg.TIME_SCALE = "iter"
    cfg.freeze()
    return cfg


class DefaultTrainer(SimpleTrainer):
    """
    A trainer with default training logic. Compared to `SimpleTrainer`, it
    contains the following logic in addition:

    1. Create model, optimizer, scheduler, dataloader from the given config.
    2. Load a checkpoint or `cfg.MODEL.WEIGHTS`, if exists.
    3. Register a few common hooks.

    It is created to simplify the **standard model training workflow**
    and reduce code boilerplate for users who only need the standard training
    workflow, with standard features.
    It means this class makes *many assumptions* about your training logic that
    may easily become invalid in a new research. In fact, any assumptions beyond
    those made in the :class:`SimpleTrainer` are too much for research.

    The code of this class has been annotated about restrictive assumptions it
    makes. When they do not work for you, you're encouraged to:

    1. Overwrite methods of this class, OR:
    2. Use :class:`SimpleTrainer`, which only does minimal SGD training and
       nothing else. You can then add your own hooks if needed. OR:
    3. Write your own training loop similar to `tools/plain_train_net.py`.

    Also note that the behavior of this class, like other functions/classes in
    this file, is not stable, since it is meant to represent the
    "common default behavior".
    It is only guaranteed to work well with the standard models and training
    workflow in meddlr.
    To obtain more stable behavior, write your own training logic with other
    public APIs.

    Attributes:
        scheduler:
        checkpointer (DetectionCheckpointer):
        cfg (CfgNode):

    Examples:

    .. code-block:: python

        trainer = DefaultTrainer(cfg)
        trainer.resume_or_load()  # load last checkpoint or MODEL.WEIGHTS
        trainer.train()
    """

    def __init__(self, cfg):
        """
        Args:
            cfg (CfgNode):
        """
        logger = logging.getLogger("meddlr")
        if not logger.isEnabledFor(logging.INFO):  # setup_logger is not called for d2
            setup_logger()

        # Assume these objects must be constructed in this order.
        data_loader = self.build_train_loader(cfg)
        num_iter_per_epoch = len(data_loader.dataset) / cfg.SOLVER.TRAIN_BATCH_SIZE
        if cfg.DATALOADER.DROP_LAST:
            num_iter_per_epoch = int(num_iter_per_epoch)
        else:
            num_iter_per_epoch = math.ceil(num_iter_per_epoch)
        cfg = convert_cfg_time_to_iter(cfg, num_iter_per_epoch)
        logger.info(f"Calculated {num_iter_per_epoch} iterations per epoch")

        # Recreate data loader in case data loader required epoch -> iter conversion.
        del data_loader
        data_loader = self.build_train_loader(cfg)

        model = self.build_model(cfg)
        optimizer = self.build_optimizer(cfg, model)

        # For training, wrap with DP. But don't need this for inference.
        num_gpus = len(os.environ.get("CUDA_VISIBLE_DEVICES", "").split(","))
        if num_gpus > 1:
            logger.info("Using data parallel...")
            model = DataParallel(model)
        model.to(cfg.MODEL.DEVICE)

        loss_computer = self.build_loss_computer(cfg)
        super().__init__(model, data_loader, optimizer, loss_computer)

        self.scheduler = self.build_lr_scheduler(cfg, optimizer)
        # Assume no other objects need to be checkpointed.
        # We can later make it checkpoint the stateful hooks
        self.checkpointer = Checkpointer(
            # Assume you want to save checkpoints together with logs/statistics
            model,
            cfg.OUTPUT_DIR,
            optimizer=optimizer,
            scheduler=self.scheduler,
        )
        self.start_iter = 0
        self.max_iter = cfg.SOLVER.MAX_ITER
        self.cfg = cfg

        self.register_hooks(self.build_hooks())

    def init_fine_tune(self):
        """
        Initialize model with fine-tune weights.

        If fine tune weights not specified, results in no-op.
        """
        fine_tune_weights = self.cfg.MODEL.FINE_TUNE.WEIGHTS
        if not fine_tune_weights:
            return

        logger = logging.getLogger(__name__)
        logger.info("Loading fine-tune weights")
        if not os.path.isfile(fine_tune_weights):
            raise FileNotFoundError("weights not found: {}".format(fine_tune_weights))

        temp_checkpointer = Checkpointer(self.model)
        temp_checkpointer.load(fine_tune_weights)

    def resume_or_load(self, resume=True, restart_iter=False):
        """
        If `resume==True`, and last checkpoint exists, resume from it.

        Otherwise, load a model specified by the config.

        Args:
            resume (bool): whether to do resume or not
        """
        # The checkpoint stores the training iteration that just finished,
        # thus we start at the next iteration
        # (or iter zero if there's no checkpoint).
        iteration = (
            self.checkpointer.resume_or_load(self.cfg.MODEL.WEIGHTS, resume=resume).get(
                "iteration", -1
            )
            + 1
        )
        self.start_iter = iteration if not restart_iter else 0

    def build_hooks(self):
        """
        Build a list of default hooks, including timing, evaluation,
        checkpointing, lr scheduling, precise BN, writing events.

        Returns:
            list[HookBase]:
        """
        cfg = self.cfg.clone()
        cfg.defrost()
        cfg.DATALOADER.NUM_WORKERS = 0  # save some memory and time for PreciseBN

        ret = [
            (
                hooks.FlushOptimizer(self.optimizer, cfg.TEST.EVAL_PERIOD)
                if isinstance(self.optimizer, GradAccumOptimizer)
                else None
            ),
            hooks.IterationTimer(),
            hooks.LRScheduler(self.optimizer, self.scheduler),
            hooks.PeriodicCheckpointer(self.checkpointer, cfg.SOLVER.CHECKPOINT_PERIOD),
        ]

        def test_and_save_results():
            self._last_eval_results = self.test(self.cfg, self.model, use_val=True)
            return self._last_eval_results

        # Do evaluation after checkpointer, because then if it fails,
        # we can use the saved checkpoint to debug.
        ret.append(hooks.EvalHook(cfg.TEST.EVAL_PERIOD, test_and_save_results))
        if env.profile_memory():
            ret.append(hooks.MemoryProfiler(20))

        # Writing should be last
        ret.append(hooks.PeriodicWriter(self.build_writers(), periods=(20, cfg.TEST.EVAL_PERIOD)))
        ret = [x for x in ret if x is not None]
        return ret

    def build_writers(self):
        """
        Build a list of writers to be used. By default it contains
        writers that write metrics to the screen,
        a json file, and a tensorboard event file respectively.
        If you'd like a different list of writers, you can overwrite it in
        your trainer.

        Returns:
            list[EventWriter]: a list of :class:`EventWriter` objects.

        It is now implemented by:

        .. code-block:: python

            return [
                CommonMetricPrinter(self.max_iter),
                JSONWriter(os.path.join(self.cfg.OUTPUT_DIR, "metrics.json")),
                TensorboardXWriter(self.cfg.OUTPUT_DIR),
            ]

        """
        assert (
            self.cfg.TIME_SCALE == "iter" and self.cfg.TEST.EVAL_PERIOD >= 0
        ), "cfg should be formatted in 'iter' time scale"
        eval_period = self.cfg.TEST.EVAL_PERIOD

        # Assume the default print/log frequency.
        return [
            # It may not always print what you want to see,
            # since it prints "common" metrics only.
            CommonMetricPrinter(self.max_iter, eval_period=eval_period),
            JSONWriter(os.path.join(self.cfg.OUTPUT_DIR, "metrics.json")),
            TensorboardXWriter(self.cfg.OUTPUT_DIR),
        ]

    def train(self):
        """
        Run training.

        Returns:
            OrderedDict of results, if evaluation is enabled. Otherwise None.
        """
        super().train(self.start_iter, self.max_iter)
        if hasattr(self, "_last_eval_results"):
            verify_results(self.cfg, self._last_eval_results)
            return self._last_eval_results

    @classmethod
    def build_model(cls, cfg):
        """
        Returns:
            torch.nn.Module:

        It now calls :func:`meddlr.modeling.build_model`.
        Overwrite it if you'd like a different model.
        """
        model = build_model(cfg)
        # logger = logging.getLogger(__name__)
        # printing model doesnt really help
        # logger.info("Model:\n{}".format(model))
        return model

    @classmethod
    def build_loss_computer(cls, cfg):
        loss_computer = (
            "N2RLossComputer"
            if cfg.MODEL.META_ARCHITECTURE == "N2RModel"
            or cfg.MODEL.META_ARCHITECTURE == "M2RModel"
            or cfg.MODEL.META_ARCHITECTURE == "NM2RModel"
            or cfg.MODEL.META_ARCHITECTURE == "A2RModel"
            else "BasicLossComputer"
        )
        return build_loss_computer(cfg, loss_computer)

    @classmethod
    def build_optimizer(cls, cfg, model):
        """
        Returns:
            torch.optim.Optimizer:

        It now calls :func:`meddlr.solver.build_optimizer`.
        Overwrite it if you'd like a different optimizer.
        """
        return build_optimizer(cfg, model)

    @classmethod
    def build_lr_scheduler(cls, cfg, optimizer):
        """
        It now calls :func:`meddlr.solver.build_lr_scheduler`.
        Overwrite it if you'd like a different scheduler.
        """
        return build_lr_scheduler(cfg, optimizer)

    @classmethod
    def build_train_loader(cls, cfg):
        """
        Returns:
            iterable

        It now calls :func:`meddlr.data.build_recon_train_loader`.
        Overwrite it if you'd like a different data loader.
        """
        return build_recon_train_loader(cfg)

    @classmethod
    def build_test_loader(cls, cfg, dataset_name, is_val=False):
        """
        Returns:
            iterable

        It now calls :func:`meddlr.data.build_recon_val_loader`.
        Overwrite it if you'd like a different data loader.
        """
        if is_val:
            as_test = cfg.TEST.VAL_AS_TEST
        else:
            as_test = True
        return build_recon_val_loader(cfg, dataset_name, as_test=as_test)

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, is_val=False):
        """
        Returns:
            DatasetEvaluator

        It is not implemented by default.
        """
        if is_val and len(cfg.TEST.VAL_METRICS.RECON) > 0:
            metrics = cfg.TEST.VAL_METRICS.RECON
        else:
            metrics = None
        return ReconEvaluator(cfg, dataset_name=dataset_name, metrics=metrics)

    @classmethod
    def test(cls, cfg, model, evaluators=None, use_val: bool = False):
        """
        Args:
            cfg (CfgNode):
            model (nn.Module):
            evaluators (list[DatasetEvaluator] or None): if None, will call
                :meth:`build_evaluator`. Otherwise, must have the same length
                as `cfg.DATASETS.{VAL/TEST}` depending on `use_val` flag.
            use_val (bool, optional): If `True`, un inference on validation data.
                Should be `True` during training.

        Returns:
            dict: a dict of result metrics
        """
        logger = logging.getLogger(__name__)
        if isinstance(evaluators, DatasetEvaluator):
            evaluators = [evaluators]

        # To use validation data, flag must be enabled and val datasets must be available.
        if use_val and cfg.DATASETS.VAL:
            inf_dataset = cfg.DATASETS.VAL
        else:
            inf_dataset = cfg.DATASETS.TEST
        if evaluators is not None:
            assert len(inf_dataset) == len(evaluators), "{} != {}".format(
                len(inf_dataset), len(evaluators)
            )

        results = OrderedDict()
        for idx, dataset_name in enumerate(inf_dataset):
            data_loader = cls.build_test_loader(cfg, dataset_name)
            # When evaluators are passed in as arguments,
            # implicitly assume that evaluators can be created
            # before data_loader.
            if evaluators is not None:
                evaluator = evaluators[idx]
            else:
                try:
                    evaluator = cls.build_evaluator(cfg, dataset_name, is_val=use_val)
                except NotImplementedError:
                    logger.warning(
                        "No evaluator found. "
                        "Use `DefaultTrainer.test(evaluators=)`, "
                        "or implement its `build_evaluator` method."
                    )
                    results[dataset_name] = {}
                    continue
            results_i = inference_on_dataset(model, data_loader, evaluator)
            results[dataset_name] = results_i
            assert isinstance(results_i, dict), (
                "Evaluator must return a dict or list on the main process. "
                "Got {} instead.".format(results_i)
            )
            logger.info("Evaluation results for {} in csv format:".format(dataset_name))
            print_csv_format(results_i)

        # if len(results) == 1:
        #     results = list(results.values())[0]
        return results
