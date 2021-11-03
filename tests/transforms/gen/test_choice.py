import unittest

import numpy as np

from meddlr.transforms.base import AffineTransform
from meddlr.transforms.gen import RandomRot90, RandomTransformChoice
from meddlr.transforms.gen.spatial import RandomFlip, RandomTranslation
from meddlr.transforms.tf_scheduler import WarmupTF
from meddlr.utils.events import EventStorage


class TestRandomTransformChoice(unittest.TestCase):
    def test_basic(self):
        tfms_or_gens = [RandomRot90(p=1.0), AffineTransform(angle=12)]
        choice = RandomTransformChoice(tfms_or_gens=tfms_or_gens, tfm_ps="uniform", p=1.0)
        choice.seed(1)

        num_samples = 200
        tfms = [type(choice.get_transform()).__name__ for _ in range(num_samples)]
        unique, counts = np.unique(tfms, return_counts=True)
        assert len(unique) == 2
        assert all(0.45 * num_samples <= x <= 0.55 * num_samples for x in counts)
        assert len(choice.schedulers()) == 0

    def test_schedulers(self):
        tfm_rand_rot = RandomRot90(p=1.0)
        tfm_aff = AffineTransform(angle=12)
        tfms_or_gens = [tfm_rand_rot, tfm_aff]
        choice = RandomTransformChoice(tfms_or_gens=tfms_or_gens, tfm_ps="uniform", p=1.0)
        sch1 = WarmupTF(tfm_rand_rot, params=["p"], warmup_iters=100)
        tfm_rand_rot.register_schedulers(sch1)
        sch2 = WarmupTF(choice, params=["p"], warmup_iters=200)
        choice.register_schedulers(sch2)

        schedulers = choice.schedulers()
        assert sch1 in schedulers
        assert sch2 in schedulers

    def test_nested(self):
        tfms_or_gens = [
            RandomRot90(p=1.0),
            [RandomFlip(ndim=2, p=1.0), RandomTranslation(p=1.0, translate=[0.1, 0.2])],
        ]
        choice = RandomTransformChoice(tfms_or_gens=tfms_or_gens, tfm_ps="uniform", p=1.0)
        choice.seed(1)

        sch1 = WarmupTF(tfms_or_gens[-1][0], params=["p"], warmup_iters=100)
        tfms_or_gens[-1][0].register_schedulers(sch1)
        sch2 = WarmupTF(tfms_or_gens[0], params=["p"], warmup_iters=200)
        tfms_or_gens[0].register_schedulers(sch2)

        schedulers = choice.schedulers()
        assert sch1 in schedulers
        assert sch2 in schedulers

        with EventStorage():
            num_samples = 200
            tfms = [type(choice.get_transform()).__name__ for _ in range(num_samples)]
        unique, counts = np.unique(tfms, return_counts=True)
        assert len(unique) == 2
        assert all(0.45 * num_samples <= x <= 0.55 * num_samples for x in counts)
