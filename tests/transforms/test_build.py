import unittest

import torch

from meddlr.config import get_cfg
from meddlr.engine.trainer import convert_cfg_time_to_iter
from meddlr.transforms.base import Rot90Transform
from meddlr.transforms.base.spatial import FlipTransform
from meddlr.transforms.build import build_transforms
from meddlr.transforms.gen import RandomNoise, RandomRot90
from meddlr.transforms.tf_scheduler import WarmupMultiStepTF, WarmupTF
from meddlr.transforms.transform_gen import TransformGen


class TestBuildTransforms(unittest.TestCase):
    def test_build_single(self):
        cfg = get_cfg()

        # Transform Gen
        ks = (1, 2)
        p = 0.2
        seed = 42
        tfm_cfg = {"name": "RandomRot90", "ks": ks}
        expected_tfm = RandomRot90(ks, p=p).seed(seed)
        tfm: TransformGen = build_transforms(cfg, tfm_cfg, p=p).seed(seed)
        assert tfm.ks == expected_tfm.ks == ks
        assert tfm.p == expected_tfm.p == p
        assert torch.all(tfm._generator.get_state() == expected_tfm._generator.get_state())

        # Transform
        k = 1
        dims = (-1, -2)
        seed = 42
        tfm_cfg = {"name": "Rot90Transform", "k": k, "dims": dims}
        expected_tfm = Rot90Transform(k, dims)
        tfm = build_transforms(cfg, tfm_cfg)
        assert tfm == expected_tfm

    def test_build_with_scheduler(self):
        cfg = get_cfg()

        ks = (1, 2)
        p = 0.2
        seed = 42
        warmup_iters = 100
        delay_iters = 40

        tfm_cfg = {
            "name": "RandomRot90",
            "ks": ks,
            "scheduler": {
                "name": "WarmupTF",
                "warmup_iters": warmup_iters,
                "delay_iters": delay_iters,
                "params": ["p"],
            },
        }
        expected_tfm = RandomRot90(ks, p=p).seed(seed)
        tfm: TransformGen = build_transforms(cfg, tfm_cfg, p=p).seed(seed)
        assert tfm.ks == expected_tfm.ks == ks
        assert tfm.p == expected_tfm.p == p
        assert torch.all(tfm._generator.get_state() == expected_tfm._generator.get_state())

        assert len(tfm._schedulers) == 1
        scheduler = tfm._schedulers[0]
        assert isinstance(scheduler, WarmupTF)
        assert scheduler.warmup_iters == warmup_iters
        assert scheduler.delay_iters == delay_iters
        assert tuple(scheduler._parameter_names()) == ("p",)

    def test_build_complex(self):
        cfg = get_cfg()
        cfg.SOLVER.MAX_ITERS = 1000

        tfm_cfg = [
            {
                "name": "RandomRot90",
                "p": 0.5,
                "scheduler": {
                    "name": "WarmupTF",
                    "warmup_iters": 800,
                    "delay_iters": 500,
                    "params": ["p"],
                },
            },
            {"name": "FlipTransform", "dims": (-1,)},
            {
                "name": "RandomNoise",
                "std_devs": (1, 2),
                "p": 0.2,
                "scheduler": [
                    {
                        "name": "WarmupStepTF",
                        "warmup_milestones": (100,),
                        "max_iter": 500,
                        "delay_iters": 100,
                        "params": ["p"],
                    },
                    {"name": "WarmupTF", "params": ("std_devs",), "warmup_iters": 600},
                ],
            },
        ]
        tfms = build_transforms(cfg, tfm_cfg)

        tfm = tfms[0]
        assert isinstance(tfm, RandomRot90)
        assert tfm.p == 0.5
        assert len(tfm._schedulers) == 1
        scheduler = tfm._schedulers[0]
        assert isinstance(scheduler, WarmupTF)
        assert scheduler.warmup_iters == 800
        assert scheduler.delay_iters == 500
        assert tuple(scheduler._params) == ("p",)

        tfm = tfms[1]
        assert isinstance(tfm, FlipTransform)
        assert tfm.dims == (-1,)

        tfm = tfms[2]
        assert isinstance(tfm, RandomNoise)
        assert tfm.std_devs == (1, 2)
        assert tfm.p == 0.2
        assert len(tfm._schedulers) == 2
        sch1 = tfm._schedulers[0]
        assert isinstance(sch1, WarmupMultiStepTF)
        assert sch1.warmup_milestones == (100, 200, 300, 400, 500)
        assert tuple(sch1._params) == ("p",)
        sch2 = tfm._schedulers[1]
        assert isinstance(sch2, WarmupTF)
        assert sch2.warmup_iters == 600
        assert tuple(sch2._params) == ("std_devs",)

    def test_to_iter(self):
        tfm_cfg = [
            {
                "name": "RandomRot90",
                "p": 0.5,
                "scheduler": {
                    "name": "WarmupTF",
                    "warmup_iters": -2,
                    "delay_iters": 500,
                    "params": ["p"],
                },
            },
            {"name": "FlipTransform", "dims": (-1,)},
            {
                "name": "RandomNoise",
                "std_devs": (1, 2),
                "p": 0.2,
                "scheduler": [
                    {
                        "name": "WarmupStepTF",
                        "warmup_milestones": (-1,),
                        "max_iter": -2,
                        "delay_iters": 500,
                        "params": ["p"],
                    },
                    {"name": "WarmupTF", "params": ("std_devs",), "warmup_iters": 600},
                ],
            },
        ]

        cfg = get_cfg()
        cfg.TIME_SCALE = "iter"
        cfg.AUG_TRAIN.MRI_RECON.TRANSFORMS = tfm_cfg

        iters_per_epoch = 100
        cfg = convert_cfg_time_to_iter(cfg, iters_per_epoch=iters_per_epoch)
        tfms = cfg.AUG_TRAIN.MRI_RECON.TRANSFORMS

        assert tfms[0]["scheduler"]["warmup_iters"] == 2 * iters_per_epoch
        assert tfms[0]["scheduler"]["delay_iters"] == 500
        assert tfms[2]["scheduler"][0]["warmup_milestones"] == (iters_per_epoch,)
        assert tfms[2]["scheduler"][0]["max_iter"] == 2 * iters_per_epoch
        assert tfms[2]["scheduler"][0]["delay_iters"] == 500
        assert tfms[2]["scheduler"][1]["warmup_iters"] == 600
