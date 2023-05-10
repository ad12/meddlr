"""Training loop.

Adapted from
https://github.com/facebookresearch/detectron2
"""

import logging
import time
import weakref

import torch

from meddlr.utils.events import EventStorage
from meddlr.utils.general import flatten_dict

__all__ = ["HookBase", "TrainerBase", "SimpleTrainer"]


class HookBase:
    """
    Base class for hooks that can be registered with :class:`TrainerBase`.

    Each hook can implement 4 methods. The way they are called is demonstrated
    in the following snippet:

    .. code-block:: python

        hook.before_train()
        for iter in range(start_iter, max_iter):
            hook.before_step()
            trainer.run_step()
            hook.after_step()
        hook.after_train()

    Notes:
        1. In the hook method, users can access `self.trainer` to access more
           properties about the context (e.g., current iteration).

        2. A hook that does something in :meth:`before_step` can often be
           implemented equivalently in :meth:`after_step`.
           If the hook takes non-trivial time, it is strongly recommended to
           implement the hook in :meth:`after_step` instead of
           :meth:`before_step`.
           The convention is that :meth:`before_step` should only take
           negligible time.

           Following this convention will allow hooks that do care about the
           difference
           between :meth:`before_step` and :meth:`after_step` (e.g., timer) to
           function properly.

    Attributes:
        trainer: A weak reference to the trainer object. Set by the trainer
            when the hook is registered.
    """

    def before_train(self):
        """
        Called before the first iteration.
        """
        pass

    def after_train(self):
        """
        Called after the last iteration.
        """
        pass

    def before_step(self):
        """
        Called before each iteration.
        """
        pass

    def after_step(self):
        """
        Called after each iteration.
        """
        pass


class TrainerBase:
    """
    Base class for iterative trainer with hooks.

    The only assumption we made here is: the training runs in a loop.
    A subclass can implement what the loop is.
    We made no assumptions about the existence of dataloader, optimizer, model,
    etc.

    Attributes:
        iter(int): the current iteration.

        start_iter(int): The iteration to start with.
            By convention the minimum possible value is 0.

        max_iter(int): The iteration to end training.

        storage(EventStorage): An EventStorage that's opened during the course
            of training.
    """

    def __init__(self):
        self._hooks = []

    def register_hooks(self, hooks):
        """
        Register hooks to the trainer. The hooks are executed in the order
        they are registered.

        Args:
            hooks (list[Optional[HookBase]]): list of hooks
        """
        hooks = [h for h in hooks if h is not None]
        for h in hooks:
            assert isinstance(h, HookBase)
            # To avoid circular reference, hooks and
            # trainer cannot own each other.
            # This normally does not matter, but will cause memory leak if the
            # involved objects contain __del__:
            # See http://engineering.hearsaysocial.com/2013/06/16/circular-references-in-python/  # noqa
            h.trainer = weakref.proxy(self)
        self._hooks.extend(hooks)

    def train(self, start_iter: int, max_iter: int):
        """
        Args:
            start_iter, max_iter (int): See docs above
        """
        logger = logging.getLogger(__name__)
        logger.info("Starting training from iteration {}".format(start_iter))

        self.iter = self.start_iter = start_iter
        self.max_iter = max_iter

        with EventStorage(start_iter) as self.storage:
            try:
                self.before_train()
                for self.iter in range(start_iter, max_iter):
                    self.before_step()
                    self.run_step()
                    self.after_step()
            except Exception as e:
                logger.error(e)
                raise e
            finally:
                self.after_train()

    def before_train(self):
        for h in self._hooks:
            h.before_train()

    def after_train(self):
        for h in self._hooks:
            h.after_train()

    def before_step(self):
        for h in self._hooks:
            h.before_step()

    def after_step(self):
        for h in self._hooks:
            h.after_step()
        # this guarantees, that in each hook's after_step,
        # storage.iter == trainer.iter
        self.storage.step()

    def run_step(self):
        raise NotImplementedError


class SimpleTrainer(TrainerBase):
    """
    A simple trainer for the most common type of task:
    single-cost single-optimizer single-data-source iterative optimization.
    It assumes that every step, you:

    1. Compute the loss with a data from the data_loader.
    2. Compute the gradients with the above loss.
    3. Update the model with the optimizer.

    If you want to do anything fancier than this,
    either subclass TrainerBase and implement your own `run_step`,
    or write your own training loop.
    """

    def __init__(self, model, data_loader, optimizer, loss_computer, metrics_computer=None):
        """
        Args:
            model: a torch Module. Takes a data from data_loader and returns a
                dict of losses.
            data_loader: an iterable. Contains data to be used to call model.
            optimizer: a torch optimizer.
            loss_computer: A callable that returns a dict of losses.
                All terms must have the word "loss" to be a valid loss.
            metrics_computer: A callable that returns a dict of metrics.
                Will not be used for loss.
        """
        super().__init__()

        """
        We set the model to training mode in the trainer.
        However it's valid to train a model that's in eval mode.
        If you want your model (or a submodule of it) to behave
        like evaluation during training, you can overwrite its train() method.
        """
        model.train()

        self.model = model
        self.data_loader = data_loader
        self._data_loader_iter = None
        self.optimizer = optimizer
        self.loss_computer = loss_computer
        self.metrics_computer = metrics_computer

    def before_train(self):
        out = super().before_train()
        self._data_loader_iter = iter(self.data_loader)
        return out

    def run_step(self):
        """
        Implement the standard training logic described above.
        """
        assert self.model.training, "[SimpleTrainer] model was changed to eval mode!"
        start = time.perf_counter()
        """
        If your want to do something with the data, you can wrap the dataloader.
        """
        try:
            inputs = next(self._data_loader_iter)
        except StopIteration:
            # Epoch has ended, reinitialize the iterator.
            self._data_loader_iter = iter(self.data_loader)
            inputs = next(self._data_loader_iter)

        data_time = time.perf_counter() - start

        # Pop keys that the model doesnt need.
        profiler = inputs.pop("_profiler", {})

        """
        If your want to do something with the losses, you can wrap the model.
        """
        forward_time = time.perf_counter()
        output_dict = self.model(inputs)
        forward_time = time.perf_counter() - forward_time

        output_dict.update({k: inputs[k] for k in ["mean", "std", "norm"] if k in inputs})
        loss_dict = {k: v for k, v in output_dict.items() if "loss" in k}
        loss_dict.update(self.loss_computer(inputs, output_dict))

        # losses = sum(v for k, v in loss_dict.items() if "loss" in k)
        losses = loss_dict["loss"]
        self._detect_anomaly(losses, loss_dict)

        metrics_dict = loss_dict
        metrics_dict.update(
            flatten_dict(
                {k: v for k, v in inputs.get("metrics", {}).items() if k not in metrics_dict}
            )
        )
        metrics_dict["data_time"] = data_time
        metrics_dict["forward_time"] = forward_time
        metrics_dict["total_loss"] = losses
        metrics_dict.update(self.metrics_computer(output_dict) if self.metrics_computer else {})

        if profiler is not None:
            metrics_dict.update(flatten_dict({"profiler": profiler}))
        # for k in ["_profiler", "profiler"]:
        #     if k in inputs:
        #         metrics_dict.update(flatten_dict({k.strip("_"): inputs[k]}))
        self._write_metrics(metrics_dict)

        """
        If you need accumulate gradients or something similar, you can
        wrap the optimizer with your custom `zero_grad()` method.
        """
        self.optimizer.zero_grad()
        losses.backward()

        """
        If you need gradient clipping/scaling or other processing, you can
        wrap the optimizer with your custom `step()` method.
        """
        self.optimizer.step()

    def _detect_anomaly(self, losses, loss_dict):
        if not torch.isfinite(losses).all():
            raise FloatingPointError(
                "Loss became infinite or NaN at iteration={}!\n"
                "loss_dict = {}".format(self.iter, loss_dict)
            )

    def _write_metrics(self, metrics_dict: dict):
        """
        Args:
            metrics_dict (dict): dict of scalar metrics
        """
        metrics_dict = {
            k: torch.mean(v.detach().cpu()).item() if isinstance(v, torch.Tensor) else float(v)
            for k, v in metrics_dict.items()
        }

        if len(metrics_dict) > 1:
            self.storage.put_scalars(**metrics_dict)
