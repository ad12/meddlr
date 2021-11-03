import unittest

import torch

from meddlr.metrics import Metric


def metric_func(preds, targets, alpha, beta=0.1):
    return (alpha / beta) * (targets - preds).mean(dim=tuple(range(2, len(preds) + 1)))


class MockMetric(Metric):
    def func(self, preds, targets, alpha, beta=0.1):
        # Return a BxC tensor
        return metric_func(preds, targets, alpha, beta)


class TestMetric(unittest.TestCase):
    def test_compute(self):
        preds = torch.randn((3, 4, 30, 20))
        targets = torch.randn((3, 4, 30, 20))

        alpha = 0.1
        beta = 0.1
        expected_out = metric_func(preds, targets, alpha, beta=0.1)

        metric = MockMetric()
        metric(preds, targets, alpha, beta)
        out = metric.compute()
        assert out.shape == (3, 4)
        assert torch.all(out == expected_out)

        metric = MockMetric()
        metric(preds, targets, alpha, beta=beta)
        out = metric.compute()
        assert out.shape == (3, 4)
        assert torch.all(out == expected_out)

        metric = MockMetric()
        metric(preds, targets, alpha, beta=beta)
        metric(preds, targets, alpha, beta=beta)
        metric(preds, targets, alpha, beta=beta)
        out = metric.compute()
        assert len(metric.ids) == 9
        assert out.shape == (9, 4)

    def test_to_pandas(self):
        preds = torch.randn((3, 4, 30, 20))
        targets = torch.randn((3, 4, 30, 20))
        alpha = 0.1
        beta = 0.1

        metric = MockMetric()
        metric(preds, targets, alpha, beta=beta)
        metric(preds, targets, alpha, beta=beta)
        metric(preds, targets, alpha, beta=beta)

        df = metric.to_pandas()
        assert len(df) == 9
