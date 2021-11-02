import logging

from torch.optim.optimizer import Optimizer

__all__ = ["GradAccumOptimizer"]

logger = logging.getLogger(__name__)


class GradAccumOptimizer(object):
    """Zero grad must be called before step."""

    def __init__(self, optimizer: Optimizer, accumulation_iters: int):
        self.optimizer = optimizer
        self.accumulation_iters = accumulation_iters
        self.step_iters = 0

    def state_dict(self):
        # `step_iters` is not saved because if step_iters > 0,
        # gradients were being accumulated. However when the state dict is saved,
        # there is no way of restoring these gradients.
        # So a `step_iters`>0 would falsely indicate we have some gradients
        # accumulated when we do not.
        return {
            "optimizer": self.optimizer.state_dict(),
            "accumulation_iters": self.accumulation_iters,
        }

    def load_state_dict(self, state_dict):
        if isinstance(state_dict["accumulation_iters"], int):
            # Needed because of a bug in our saving the state dict
            self.accumulation_iters = state_dict["accumulation_iters"]
        else:
            logger.warning(
                f"`accumulation_iters` is of the wrong type due to saving issue. "
                f"Using initialized value of {self.accumulation_iters}."
            )
        return self.optimizer.load_state_dict(state_dict["optimizer"])

    def zero_grad(self):
        if self.step_iters % self.accumulation_iters == 0:
            self.optimizer.zero_grad()

    def step(self, closure=None):
        self.step_iters += 1
        if self.step_iters % self.accumulation_iters == 0:
            self._step(closure)

    def flush(self, closure=None):
        if self.step_iters > 0:
            self._step(closure)
        self.optimizer.zero_grad()

    def _step(self, closure=None):
        """Internal step method. Resets `step_iters` attribute to 0."""
        self.optimizer.step(closure)
        self.step_iters = 0

    def __getattr__(self, item):
        if hasattr(self.optimizer, item):
            return getattr(self.optimizer, item)
