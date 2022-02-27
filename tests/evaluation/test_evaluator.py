import uuid

import pytest
import torch
from torch import nn

from meddlr.evaluation.evaluator import DatasetEvaluator, DatasetEvaluators, inference_on_dataset


class MockEvaluator(DatasetEvaluator):
    def __init__(self, name=None, func=None):
        if func is None:
            func = lambda input, output: torch.sum(torch.abs(input - output))  # noqa: E731
        if name is None:
            name = str(uuid.uuid4())
        self.name = name
        self.func = func
        self.values = []

    def reset(self):
        self.values = []
        return super().reset()

    def process(self, input, output):
        self.values.append(self.func(input, output))

    def evaluate(self):
        return {self.name: torch.mean(torch.as_tensor(self.values))}


class MultiplyModule(nn.Module):
    def __init__(self, factor=2):
        super().__init__()
        self.factor = factor

    def forward(self, x):
        return x * self.factor


@pytest.mark.parametrize("ltype", [list, dict])
def test_dataset_evaluators(ltype):
    a, b = MockEvaluator("a"), MockEvaluator("b")
    inputs = [torch.rand(10, 10), torch.rand(10, 10)]
    outputs = [torch.rand(10, 10), torch.rand(10, 10)]

    if ltype is list:
        evaluator = DatasetEvaluators([a, b])
    else:
        evaluator = DatasetEvaluators({"a": a, "b": b})

    assert len(evaluator) == 2
    assert a in evaluator
    assert b in evaluator

    evaluator.process(inputs[0], outputs[0])
    assert a.values
    assert b.values

    results = evaluator.evaluate()
    assert "a" in results and "b" in results

    evaluator.reset()
    assert not a.values
    assert not b.values


def test_inference_on_dataset():
    inputs = torch.rand(20, 10, 10)
    evaluator = MockEvaluator("a")
    results = inference_on_dataset(MultiplyModule(), inputs, evaluator)

    assert evaluator.values
    assert "a" in results
