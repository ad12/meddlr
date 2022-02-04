import multiprocessing as mp
import unittest
from collections import defaultdict

import numpy as np
import pytest

from meddlr.transforms.tf_scheduler import TFScheduler, WarmupMultiStepTF, WarmupTF

from .mock import MockIterTracker, MockSchedulable


def _run_simulation(scheduler: TFScheduler, iters):
    iter_tracker = MockIterTracker()
    scheduler.get_iteration = iter_tracker.get_iter
    params = defaultdict(list)
    for _ in range(iters):
        for k, v in scheduler.get_params().items():
            params[k].append(v)
        iter_tracker.step()
    return params


def _assert_fails_build_scheduler_mp():
    # Fail when initializing on worker thread.
    with pytest.raises(RuntimeError):
        WarmupTF(tfm=MockSchedulable(a=1.0, b=(0.0, 0.5)), params="a", warmup_iters=100)


class TestWarmupTF(unittest.TestCase):
    def test_init(self):
        with mp.Pool(1) as p:
            p.apply(_assert_fails_build_scheduler_mp)

        with pytest.raises(ValueError):
            WarmupTF(tfm=MockSchedulable(a=0.5), warmup_iters=100, params="foo")

    def test_warmup_basic(self):
        a = 0.5
        b = (0.0, 0.5)
        schedulable = MockSchedulable(a=a, b=b)

        warmup_iters = 5
        method = "linear"
        total_iters = warmup_iters + 2
        a_expected = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.5]
        b_expected = [(0, 0), (0, 0.1), (0, 0.2), (0, 0.3), (0, 0.4), (0, 0.5), (0, 0.5)]

        # Scalar
        scheduler = WarmupTF(
            tfm=schedulable, warmup_iters=warmup_iters, params="a", warmup_method=method
        )
        params = _run_simulation(scheduler, total_iters)
        assert len(params.keys()) == 1
        assert "a" in params
        assert np.allclose(params["a"], a_expected)
        assert params["a"]

        # Range
        scheduler = WarmupTF(
            tfm=schedulable, warmup_iters=warmup_iters, params="b", warmup_method=method
        )
        params = _run_simulation(scheduler, total_iters)
        assert len(params.keys()) == 1
        assert "b" in params
        assert np.allclose(params["b"], b_expected)

        # Both
        scheduler = WarmupTF(
            tfm=schedulable, warmup_iters=warmup_iters, params=("a", "b"), warmup_method=method
        )
        params = _run_simulation(scheduler, total_iters)
        assert len(params.keys()) == 2
        assert np.allclose(params["a"], a_expected)
        assert np.allclose(params["b"], b_expected)

    def test_warmup_nested_dict(self):
        a = 0.5
        b = {"v1": 0.5, "v2": (0, 0.5)}
        schedulable = MockSchedulable(a=a, b=b)

        warmup_iters = 5
        method = "linear"
        total_iters = warmup_iters + 2
        a_expected = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.5]
        b_v1_expected = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.5]
        b_v2_expected = [(0, 0), (0, 0.1), (0, 0.2), (0, 0.3), (0, 0.4), (0, 0.5), (0, 0.5)]

        scheduler = WarmupTF(
            tfm=schedulable,
            warmup_iters=warmup_iters,
            params=("a", "b.v1", "b.v2"),
            warmup_method=method,
        )
        params = _run_simulation(scheduler, total_iters)
        a_val = params["a"]
        b_v1_val = [x["v1"] for x in params["b"]]
        b_v2_val = [x["v2"] for x in params["b"]]
        assert len(params.keys()) == 2
        assert np.allclose(a_val, a_expected)
        assert np.allclose(b_v1_val, b_v1_expected)
        assert np.allclose(b_v2_val, b_v2_expected)

    def test_warmup_delay(self):
        a = 0.5
        b = (0.0, 0.5)
        schedulable = MockSchedulable(a=a, b=b)

        warmup_iters = 10
        delay_iters = 5
        method = "linear"
        total_iters = warmup_iters + 2
        # fmt: off
        a_expected = [0, 0, 0, 0, 0, 0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.5]
        b_expected = [
            (0, 0), (0, 0), (0, 0), (0, 0), (0, 0),
            (0, 0), (0, 0.1), (0, 0.2), (0, 0.3), (0, 0.4), (0, 0.5), (0, 0.5),
        ]
        # fmt: on

        scheduler = WarmupTF(
            tfm=schedulable,
            warmup_iters=warmup_iters,
            params=("a", "b"),
            warmup_method=method,
            delay_iters=delay_iters,
        )
        params = _run_simulation(scheduler, total_iters)
        assert len(params.keys()) == 2
        assert np.allclose(params["a"], a_expected)
        assert np.allclose(params["b"], b_expected)

    def test_unregister_params(self):
        a = 0.5
        b = (0.0, 0.5)
        schedulable = MockSchedulable(a=a, b=b)

        warmup_iters = 5
        method = "linear"
        total_iters = warmup_iters + 2
        a_expected = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.5]

        scheduler = WarmupTF(
            tfm=schedulable, warmup_iters=warmup_iters, params=["a", "b"], warmup_method=method
        )
        scheduler._unregister_parameters(["b"])
        params = _run_simulation(scheduler, total_iters)
        assert len(params.keys()) == 1
        assert np.allclose(params["a"], a_expected)


class TestWarmupMultiStepTF(unittest.TestCase):
    def test_warmup_basic(self):
        a = 1.0
        b = (0.0, 1.0)
        schedulable = MockSchedulable(a=a, b=b)

        warmup_milestones = (2, 4, 6, 8)
        method = "linear"
        total_iters = max(warmup_milestones) + 2
        # fmt: off
        a_expected = [0, 0, 0.25, 0.25, 0.5, 0.5, 0.75, 0.75, 1.0, 1.0]
        b_expected = [
            (0, 0), (0, 0), (0, 0.25), (0, 0.25), (0, 0.5), (0, 0.5),
            (0, 0.75), (0, 0.75), (0, 1.0), (0, 1.0),
        ]
        # fmt: on

        scheduler = WarmupMultiStepTF(
            tfm=schedulable,
            warmup_milestones=warmup_milestones,
            params=("a", "b"),
            warmup_method=method,
        )
        params = _run_simulation(scheduler, total_iters)
        assert len(params.keys()) == 2
        assert np.allclose(params["a"], a_expected)
        assert np.allclose(params["b"], b_expected)
