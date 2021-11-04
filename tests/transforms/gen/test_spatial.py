import unittest

import numpy as np
import torch

from meddlr.transforms.base.spatial import AffineTransform, TranslationTransform
from meddlr.transforms.gen import RandomAffine
from meddlr.transforms.gen.spatial import RandomTranslation
from meddlr.transforms.tf_scheduler import WarmupTF

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

    def test_scheduler_p(self):
        iter_tracker = MockIterTracker()
        affine = RandomAffine(p=1.0, angle=10, translate=1.0, scale=2.0)

        scheduler = WarmupTF(affine, warmup_iters=400, params=["p"])
        scheduler.get_iteration = iter_tracker.get_iter
        affine.register_schedulers([scheduler])

        iter_tracker.step(100)
        params = affine._get_param_values(use_schedulers=True)
        p = params["p"]
        assert all(v == 0.25 for v in p.values())

        # Different probability for each.
        iter_tracker = MockIterTracker()
        affine = RandomAffine(
            p={"angle": 1.0, "translate": 0.5, "scale": 0.2}, angle=10, translate=1.0, scale=2.0
        )
        p = affine.p
        assert p["angle"] == 1.0
        assert p["translate"] == 0.5
        assert p["scale"] == 0.2
        assert p["shear"] == 0

        scheduler = WarmupTF(affine, warmup_iters=500, params=["p"])
        scheduler.get_iteration = iter_tracker.get_iter
        affine.register_schedulers([scheduler])

        iter_tracker.step(100)
        params = affine._get_param_values(use_schedulers=True)
        p = params["p"]
        assert np.allclose(p["angle"], 0.2)
        assert np.allclose(p["translate"], 0.1)
        assert np.allclose(p["scale"], 0.04)
        assert np.allclose(p["shear"], 0.0)

    def test_multi_arg_param_translate(self):
        h, w = 100, 100
        img = torch.randn(1, 1, h, w)

        t_h, t_w = 0.1, 0.8
        affine = RandomAffine(translate=(t_h, t_w), p=1.0)

        h_translate, w_translate = [], []
        for _ in range(100):
            tfm: AffineTransform = affine.get_transform(img)
            h_t, w_t = tuple(tfm.translate)
            h_translate.append(h_t)
            w_translate.append(w_t)

        assert any(x < 0 for x in h_translate) and any(x > 0 for x in h_translate)
        assert any(x < 0 for x in w_translate) and any(x > 0 for x in w_translate)
        assert all(abs(x) <= h * t_h for x in h_translate)
        assert all(abs(x) <= w * t_w for x in w_translate)

        # Fixed range of values.
        t_h1, t_h2, t_w1, t_w2 = 0.1, 0.2, -0.8, -0.6
        affine = RandomAffine(translate=((t_h1, t_h2), (t_w1, t_w2)), p=1.0)

        h_translate, w_translate = [], []
        for _ in range(100):
            tfm: AffineTransform = affine.get_transform(img)
            h_t, w_t = tuple(tfm.translate)
            h_translate.append(h_t)
            w_translate.append(w_t)

        assert all(x > 0 for x in h_translate)
        assert all(x < 0 for x in w_translate)
        assert all(h * t_h1 <= x <= h * t_h2 for x in h_translate)
        assert all(w * t_w1 <= x <= w * t_w2 for x in w_translate)

        # Mix of range and single value
        t_h, t_w1, t_w2 = 0.1, -0.8, -0.6
        affine = RandomAffine(translate=(t_h, (t_w1, t_w2)), p=1.0)

        h_translate, w_translate = [], []
        for _ in range(100):
            tfm: AffineTransform = affine.get_transform(img)
            h_t, w_t = tuple(tfm.translate)
            h_translate.append(h_t)
            w_translate.append(w_t)

        assert any(x < 0 for x in h_translate) and any(x > 0 for x in h_translate)
        assert all(x < 0 for x in w_translate)
        assert all(abs(x) <= h * t_h for x in h_translate)
        assert all(w * t_w1 <= x <= w * t_w2 for x in w_translate)

    def test_multi_arg_param_shear(self):
        h, w = 100, 100
        img = torch.randn(1, 1, h, w)

        s_h, s_w = 10, 80
        affine = RandomAffine(shear=(s_h, s_w), p=1.0)

        h_shear, w_shear = [], []
        for _ in range(100):
            tfm: AffineTransform = affine.get_transform(img)
            h_s, w_s = tuple(tfm.shear)
            h_shear.append(h_s)
            w_shear.append(w_s)

        assert any(x < 0 for x in h_shear) and any(x > 0 for x in h_shear)
        assert any(x < 0 for x in w_shear) and any(x > 0 for x in w_shear)
        assert all(abs(x) <= s_h for x in h_shear)
        assert all(abs(x) <= s_w for x in w_shear)

        # Fixed range of values.
        s_h1, s_h2, s_w1, s_w2 = 10, 20, -80, -60
        affine = RandomAffine(shear=((s_h1, s_h2), (s_w1, s_w2)), p=1.0)

        h_shear, w_shear = [], []
        for _ in range(100):
            tfm: AffineTransform = affine.get_transform(img)
            h_s, w_s = tuple(tfm.shear)
            h_shear.append(h_s)
            w_shear.append(w_s)

        assert all(x > 0 for x in h_shear)
        assert all(x < 0 for x in w_shear)
        assert all(s_h1 <= x <= s_h2 for x in h_shear)
        assert all(s_w1 <= x <= s_w2 for x in w_shear)

        # Mix of range and single value
        s_h, s_w1, s_w2 = 10, -80, -60
        affine = RandomAffine(shear=(s_h, (s_w1, s_w2)), p=1.0)

        h_shear, w_shear = [], []
        for _ in range(100):
            tfm: AffineTransform = affine.get_transform(img)
            h_s, w_s = tuple(tfm.shear)
            h_shear.append(h_s)
            w_shear.append(w_s)

        assert any(x < 0 for x in h_shear) and any(x > 0 for x in h_shear)
        assert all(x < 0 for x in w_shear)
        assert all(abs(x) <= s_h for x in h_shear)
        assert all(s_w1 <= x <= s_w2 for x in w_shear)


class TestRandomTranslation(unittest.TestCase):
    def test_scheduler_basic(self):
        iter_tracker = MockIterTracker()
        affine = RandomTranslation(translate=(0.1, 0.8))
        scheduler = WarmupTF(affine, warmup_iters=400, params=["translate"])
        scheduler.get_iteration = iter_tracker.get_iter
        affine.register_schedulers([scheduler])

        iter_tracker.step(100)
        params = affine._get_param_values(use_schedulers=True)
        assert tuple(params["translate"]) == (0.025, 0.2)

    def test_multi_arg_param_translate(self):
        h, w = 100, 100
        img = torch.randn(1, 1, h, w)

        t_h, t_w = 0.1, 0.8
        affine = RandomTranslation(translate=(t_h, t_w), p=1.0)

        h_translate, w_translate = [], []
        for _ in range(100):
            tfm: TranslationTransform = affine.get_transform(img)
            h_t, w_t = tuple(tfm.translate)
            h_translate.append(h_t)
            w_translate.append(w_t)

        assert any(x < 0 for x in h_translate) and any(x > 0 for x in h_translate)
        assert any(x < 0 for x in w_translate) and any(x > 0 for x in w_translate)
        assert all(abs(x) <= h * t_h for x in h_translate)
        assert all(abs(x) <= w * t_w for x in w_translate)

        # Fixed range of values.
        t_h1, t_h2, t_w1, t_w2 = 0.1, 0.2, -0.8, -0.6
        affine = RandomTranslation(translate=((t_h1, t_h2), (t_w1, t_w2)), p=1.0)

        h_translate, w_translate = [], []
        for _ in range(100):
            tfm: TranslationTransform = affine.get_transform(img)
            h_t, w_t = tuple(tfm.translate)
            h_translate.append(h_t)
            w_translate.append(w_t)

        assert all(x > 0 for x in h_translate)
        assert all(x < 0 for x in w_translate)
        assert all(h * t_h1 <= x <= h * t_h2 for x in h_translate)
        assert all(w * t_w1 <= x <= w * t_w2 for x in w_translate)

        # Mix of range and single value
        t_h, t_w1, t_w2 = 0.1, -0.8, -0.6
        affine = RandomTranslation(translate=(t_h, (t_w1, t_w2)), p=1.0)

        h_translate, w_translate = [], []
        for _ in range(100):
            tfm: TranslationTransform = affine.get_transform(img)
            h_t, w_t = tuple(tfm.translate)
            h_translate.append(h_t)
            w_translate.append(w_t)

        assert any(x < 0 for x in h_translate) and any(x > 0 for x in h_translate)
        assert all(x < 0 for x in w_translate)
        assert all(abs(x) <= h * t_h for x in h_translate)
        assert all(w * t_w1 <= x <= w * t_w2 for x in w_translate)
