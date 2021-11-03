import unittest

import torch
import torch.nn as nn
from torch.optim import Adam

from meddlr.solver.optimizer import GradAccumOptimizer


def _is_zero_grad(optimizer):
    for group in optimizer.param_groups:
        for p in group["params"]:
            if p.grad is not None:
                a = p.grad
                assert torch.allclose(a, torch.zeros(a.shape))


def build_mock_model():
    return nn.Sequential(
        *[nn.Conv2d(1, 16, kernel_size=(3, 3)), nn.Conv2d(16, 32, kernel_size=(3, 3))]
    )


class TestGradAccumOptimizer(unittest.TestCase):
    def test_basic_loop(self):
        """
        Pseudo-code:

            ```python
                for i in range(num_iterations):
                    ...

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
            ```

        Expected:
            i=0: [x] ZG, [-] S
            i=1: [-] ZG, [x] S
            i=2: [x] ZG, [-] S
            i=3: [-] ZG, [x] S
        """
        num_accum = 2
        model = build_mock_model()
        optimizer = Adam(model.parameters(), lr=1e-3)
        optimizer = GradAccumOptimizer(optimizer, num_accum)

        for i in range(4):
            input = torch.rand(1, 1, 16, 16)
            output = model(input)
            loss = torch.sum(torch.abs(output))

            optimizer.zero_grad()
            if i % num_accum == 0:
                _is_zero_grad(optimizer.optimizer)
            loss.backward()
            optimizer.step()
            if (i + 1) % num_accum == 0:
                assert optimizer.step_iters == 0, f"Iteration i={i}"

    def test_initial_zero_grad_outside_for_loop(self):
        """
        Pseudo-code:

            ```python
                optimizer.zero_grad()

                for i in range(num_iterations):
                    ...

                    loss.backward()
                    optimizer.step()
                    optimizer.zero_grad()
            ```
        """
        num_accum = 2
        model = build_mock_model()
        optimizer = Adam(model.parameters(), lr=1e-3)
        optimizer = GradAccumOptimizer(optimizer, num_accum)

        optimizer.zero_grad()
        for i in range(4):
            input = torch.rand(1, 1, 16, 16)
            output = model(input)
            loss = torch.sum(torch.abs(output))

            loss.backward()
            optimizer.step()
            if (i + 1) % num_accum == 0:
                assert optimizer.step_iters == 0, f"Iteration i={i}"

            optimizer.zero_grad()
            if (i + 1) % num_accum == 0:
                _is_zero_grad(optimizer.optimizer)

    def test_flush(self):
        num_accum = 10
        model = build_mock_model()
        optimizer = Adam(model.parameters(), lr=1e-3)
        optimizer = GradAccumOptimizer(optimizer, num_accum)

        optimizer.zero_grad()
        for _i in range(num_accum - 1):
            input = torch.rand(1, 1, 16, 16)
            output = model(input)
            loss = torch.sum(torch.abs(output))

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        assert optimizer.step_iters == num_accum - 1

        optimizer.flush()

        assert optimizer.step_iters == 0
        _is_zero_grad(optimizer.optimizer)

    def test_flush_zero_steps(self):
        """Flushing when step_iters == 0 should only zero out the gradient."""
        num_accum = 10
        model = build_mock_model()
        optimizer = Adam(model.parameters(), lr=1e-3)
        optimizer = GradAccumOptimizer(optimizer, num_accum)

        optimizer.zero_grad()
        for _i in range(num_accum):
            input = torch.rand(1, 1, 16, 16)
            output = model(input)
            loss = torch.sum(torch.abs(output))

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        assert optimizer.step_iters == 0

        optimizer.flush()

        assert optimizer.step_iters == 0
        _is_zero_grad(optimizer.optimizer)
