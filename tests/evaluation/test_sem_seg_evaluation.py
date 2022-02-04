import itertools
import os
import unittest

import pytest
import torch

from meddlr.config import get_cfg
from meddlr.evaluation.seg_evaluation import SemSegEvaluator


def _simulate_data(shape, num_scans=10, batch_size: int = None):
    inputs = []
    outputs = []
    for idx in range(num_scans):
        probs = torch.rand(*shape)
        target = torch.rand(*shape) >= 0.5
        metadata = [{"scan_id": f"scan_{idx}", "slice_id": i} for i in range(probs.shape[0])]
        inputs.append({"labels": target, "metadata": metadata})
        outputs.append({"probs": probs})

    assert len(inputs) == len(outputs)
    if not batch_size:
        return inputs, outputs

    assert batch_size > 0
    all_inputs = []
    for input in inputs:
        labels = torch.split(input["labels"], batch_size, dim=0)
        metadata = input["metadata"]
        metadata = [
            metadata[idx * batch_size : (idx + 1) * batch_size]
            for idx in range(len(metadata) // batch_size)
        ]
        assert len(labels) == len(metadata)
        assert (len(x_label) == len(x_metadata) for x_label, x_metadata in zip(labels, metadata))
        all_inputs.extend(
            [
                {"labels": x_label, "metadata": x_metadata}
                for x_label, x_metadata in zip(labels, metadata)
            ]
        )
    outputs = [
        {"probs": x} for out in outputs for x in torch.split(out["probs"], batch_size, dim=0)
    ]
    assert len(all_inputs) == len(outputs)
    return all_inputs, outputs


class TestSemSegEvaluator(unittest.TestCase):
    @pytest.fixture(autouse=True)
    def setup(self, tmpdir):
        self.tmpdir = tmpdir

    def test_basic(self):
        cfg = get_cfg()
        cfg.defrost()
        cfg.MODEL.SEG.CLASSES = ("class_1", "class_2", "class_3", "class_4")
        cfg.MODEL.SEG.ACTIVATION = "sigmoid"
        cfg.MODEL.SEG.INCLUDE_BACKGROUND = False

        evaluator = SemSegEvaluator("test", cfg, distributed=False, aggregate_scans=True)
        evaluator.reset()

        shape = (20, 4, 75, 75)  # N x # classes x Y x X

        for inputs, outputs in zip(*_simulate_data(shape, num_scans=10)):
            evaluator.process(inputs, outputs)

        metrics = evaluator.evaluate()
        # 7 metrics (DSC, VOE, CV, DSC_scan, VOE_scan, CV_scan, ASSD_scan) * 4 classes = 28
        assert len(metrics) == 28, len(metrics)
        expected_metric_names = [
            "/".join(x)
            for x in itertools.product(
                ["DSC", "VOE", "CV", "DSC_scan", "VOE_scan", "CV_scan", "ASSD_scan"],
                ["class_1", "class_2", "class_3", "class_4"],
            )
        ]
        assert all(x in metrics for x in expected_metric_names), metrics.keys()

    def test_flush(self):
        cfg = get_cfg()
        cfg.defrost()
        cfg.MODEL.SEG.CLASSES = ("class_1", "class_2", "class_3", "class_4")
        cfg.MODEL.SEG.ACTIVATION = "sigmoid"
        cfg.MODEL.SEG.INCLUDE_BACKGROUND = False

        # Test manual flushing.
        evaluator = SemSegEvaluator(
            "test", cfg, distributed=False, aggregate_scans=True, flush_period=None
        )
        evaluator.reset()

        shape = (20, 4, 75, 75)  # N x # classes x Y x X
        Z = shape[0]
        batch_size = 10
        assert shape[0] % batch_size == 0
        inputs, outputs = _simulate_data(shape, num_scans=2, batch_size=batch_size)

        num_steps = shape[0] // batch_size + 1
        for i in range(num_steps):
            evaluator.process(inputs[i], outputs[i])
        evaluator.flush(skip_last_scan=True)
        assert len(evaluator._predictions) == batch_size
        assert len({x["metadata"]["scan_id"] for x in evaluator._predictions}) == 1
        assert all(x["metadata"]["scan_id"] == "scan_1" for x in evaluator._predictions)
        assert len(evaluator.scan_metrics.ids()) == 1
        assert len(evaluator.slice_metrics.ids()) == Z

        for i in range(num_steps, len(inputs)):
            evaluator.process(inputs[i], outputs[i])
        evaluator.evaluate()
        assert len(evaluator.scan_metrics.ids()) == 2
        assert len(evaluator.slice_metrics.ids()) == 2 * Z

    def test_output_dir(self):
        cfg = get_cfg()
        cfg.defrost()
        cfg.MODEL.SEG.CLASSES = ("class_1", "class_2", "class_3", "class_4")
        cfg.MODEL.SEG.ACTIVATION = "sigmoid"
        cfg.MODEL.SEG.INCLUDE_BACKGROUND = False

        evaluator = SemSegEvaluator(
            "test", cfg, distributed=False, aggregate_scans=True, output_dir=self.tmpdir
        )
        evaluator.reset()

        shape = (20, 4, 75, 75)  # N x # classes x Y x X

        for inputs, outputs in zip(*_simulate_data(shape, num_scans=10)):
            evaluator.process(inputs, outputs)

        _ = evaluator.evaluate()

        assert os.path.exists(self.tmpdir / "results.txt")
        assert os.path.exists(self.tmpdir / "slice_metrics.csv")
        assert os.path.exists(self.tmpdir / "scan_metrics.csv")


if __name__ == "__main__":
    unittest.main()
