import logging
import unittest

import numpy as np
import torch

from ss_recon.evaluation.recon_evaluation import ReconEvaluator


class MockReconEvaluator(ReconEvaluator):
    def __init__(self):
        self._output_dir = None
        self._cpu_device = torch.device("cpu")
        self._logger = logging.getLogger(__name__)
        self._normalizer = None


class TestReconEvaluator(unittest.TestCase):
    def test_evaluation_metrics(self):
        # 2D
        evaluator = MockReconEvaluator()
        prediction = {
            "pred": torch.rand(384, 384, 1, 2),
            "target": torch.rand(384, 384, 1, 2),
        }
        vals = evaluator.evaluate_prediction(prediction)
        expected = evaluator.evaluate_prediction_old(prediction)

        # Maps from old strings to new strings
        key_mapping = {
            "l1": "l1",
            "l2": "l2",
            "psnr": "psnr",
            "ssim": "ssim_old",
        }

        assert all(np.allclose(vals[key_mapping[k]], expected[k]) for k in expected.keys()), (
            "\n".join("{}\tValue: {:.6f}\tExpected: {:.6f}".format(k, vals[key_mapping[k]], expected[k]) for k in expected)
        )


if __name__ == "__main__":
    unittest.main()