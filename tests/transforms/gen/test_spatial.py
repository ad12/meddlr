import unittest

from ss_recon.transforms.gen import RandomAffine
from ss_recon.transforms.tf_scheduler import WarmupTF

from ..mock import MockIterTracker


class TestRandomAffine(unittest.TestCase):
    def test_scheduler_basic(self):
        iter_tracker = MockIterTracker()
        affine = RandomAffine(angle=10, translate=1.0, scale=2.0)
        scheduler = WarmupTF(affine, warmup_iters=400, params=["angle", "translate"])
        scheduler.get_iteration = iter_tracker.get_iter
        affine.register_schedulers([scheduler])

        iter_tracker.step(100)
        params = affine._get_param_values(use_schedulers=True)
        assert params["angle"] == 2.5
        assert params["translate"] == 0.25
        assert params["scale"] == 2.0

    # def test_scheduler_p(self):
    #     iter_tracker = MockIterTracker()
    #     affine = RandomAffine(angle=10, translate=1.0, scale=2.)
    #     scheduler = WarmupTF(affine, warmup_iters=400, params=["p"])
    #     scheduler.get_iteration = iter_tracker.get_iter
    #     affine.register_schedulers([scheduler])

    #     iter_tracker.step(100)
    #     params = affine._get_param_values(use_schedulers=True)
    #     print(params)
