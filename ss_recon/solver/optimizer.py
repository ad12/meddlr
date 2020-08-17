from torch.optim.optimizer import Optimizer

__all__ = ["GradAccumOptimizer"]


class GradAccumOptimizer(object):
    """Zero grad must be called before step."""
    def __init__(self, optimizer: Optimizer, accumulation_iters: int):
        self.optimizer = optimizer
        self.accumulation_iters = accumulation_iters
        self.step_iters = 0

    def state_dict(self):
        return {
            "optimizer": self.optimizer.state_dict(),
            "accumulation_iters": self.optimizer.state_dict(),
        }

    def load_state_dict(self, state_dict):
        self.optimizer.load_state_dict(state_dict["optimizer"])
        self.accumulation_iters = state_dict["accumulation_iters"]

    def zero_grad(self):
        if self.step_iters % self.accumulation_iters == 0:
            self.optimizer.zero_grad()

    def step(self, closure = None):
        self.step_iters += 1
        if self.step_iters % self.accumulation_iters == 0:
            self._step(closure)

    def flush(self, closure = None):
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
    
