import unittest

import numpy as np
import torch

from meddlr.config.config import get_cfg
from meddlr.transforms.base import AffineTransform, FlipTransform, NoiseTransform, Rot90Transform
from meddlr.transforms.builtin.mri import MRIReconAugmentor
from meddlr.transforms.gen import (
    RandomAffine,
    RandomFlip,
    RandomMRIMotion,
    RandomNoise,
    RandomRot90,
)
from meddlr.transforms.tf_scheduler import WarmupMultiStepTF, WarmupTF
from meddlr.utils.events import EventStorage

from ..mock import generate_mock_mri_data


class TestMRIReconAugmentor(unittest.TestCase):
    def test_transform_separation(self):
        rot = Rot90Transform(k=1, dims=(-1, -2))
        noise = NoiseTransform(0.05, seed=42)
        tfms = [rot, noise]
        augmentor = MRIReconAugmentor(tfms)

        kspace, maps, target = generate_mock_mri_data()

        _, tfms_equivariant, tfms_invariant = augmentor(kspace, maps, target)
        assert rot in tfms_equivariant
        assert noise in tfms_invariant

    def test_forward(self):
        kspace, maps, target = generate_mock_mri_data(3, 3, 8, 1)
        tfms = [
            Rot90Transform(k=1, dims=(-2, -1)),
            FlipTransform(dims=(-1,)),
            AffineTransform(angle=12),
        ]

        augmentor = MRIReconAugmentor(tfms)
        out1, tfms_equivariant, tfms_invariant = augmentor(kspace, maps, target)

        augmentor = MRIReconAugmentor(tfms_equivariant + tfms_invariant)
        out2, _, _ = augmentor(kspace, maps, target)

        assert torch.allclose(out1["kspace"], out2["kspace"])
        assert torch.allclose(out1["maps"], out2["maps"])
        assert torch.allclose(out1["target"], out2["target"])

    # def test_inverse(self):
    #     kspace, maps, target = generate_mock_mri_data(3, 3, 8, 1)
    #     tfms = [Rot90Transform(k=1, dims=(-2, -1)), FlipTransform(dims=(-1,))]
    #     augmentor = MRIReconAugmentor(tfms)

    #     out, tfms_equivariant, tfms_invariant = augmentor(kspace, maps, target)
    #     assert len(tfms_equivariant) == 2
    #     assert len(tfms_invariant) == 0
    #     K, M, tgt = out["kspace"], out["maps"], out["target"]

    #     ops = lambda x: torch.flip(torch.rot90(x, k=1, dims=(1, 2)), dims=(2,))  # noqa: E731
    #     assert torch.allclose(K, ops(kspace))
    #     assert torch.allclose(M, ops(maps))
    #     assert torch.allclose(tgt, ops(target))

    #     inv_augmentor = MRIReconAugmentor(tfms_equivariant.inverse())
    #     out_inv, tfms_equivariant, tfms_invariant = inv_augmentor(K, M, tgt)

    #     assert torch.allclose(out_inv["kspace"], kspace)
    #     assert torch.allclose(out_inv["maps"], maps)
    #     assert torch.allclose(out_inv["target"], target)

    def test_random(self):
        kspace, maps, target = generate_mock_mri_data(100, 100, 8, 1, scale=100)

        p = 1.0
        tfms = [
            RandomRot90(p=p),
            RandomFlip(ndim=2, p=p),
            RandomAffine(p=p, angle=12, translate=None, scale=0.25, shear=30),
            RandomNoise(p=p, std_devs=(0.05, 1.0)),
            RandomMRIMotion(std_devs=(0.05, 1.0), p=p),
        ]

        augmentor = MRIReconAugmentor(tfms)
        out1, tfms_equivariant, tfms_invariant = augmentor(kspace, maps, target)
        assert len(tfms_equivariant) == 3
        assert len(tfms_invariant) == 2

        augmentor = MRIReconAugmentor(tfms_equivariant + tfms_invariant)
        out2, tfms_equivariant2, tfms_invariant2 = augmentor(kspace, maps, target)

        for e1, e2 in zip(tfms_equivariant, tfms_equivariant2):
            assert e1 == e2, f"{e1} vs {e2}"
        for i1, i2 in zip(tfms_invariant, tfms_invariant2):
            assert i1 == i2, f"{i1} vs {i2}"

        assert torch.allclose(out1["kspace"], out2["kspace"], atol=1e-3)
        assert torch.allclose(out1["maps"], out2["maps"], atol=1e-3)
        assert torch.allclose(out1["target"], out2["target"], atol=1e-3)

    def test_get_scheduler_params(self):
        cfg = get_cfg()

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
            {"name": "RandomFlip", "p": 0.2, "ndim": 2},
            {"name": "RandomAffine", "p": 0.4, "angle": 12.0, "scale": 2.0, "translate": 0.4},
            {
                "name": "RandomNoise",
                "std_devs": (1, 2),
                "p": 0.2,
                "scheduler": [
                    {
                        "name": "WarmupStepTF",
                        "warmup_milestones": (100,),
                        "max_iter": 500,
                        "params": ["p"],
                    },
                    {"name": "WarmupTF", "params": ("std_devs",), "warmup_iters": 600},
                ],
            },
        ]
        cfg.AUG_TRAIN.MRI_RECON.TRANSFORMS = tfm_cfg

        aug = MRIReconAugmentor.from_cfg(cfg, aug_kind="aug_train")

        expected_params_0 = {
            "RandomRot90/p": 0,
            "RandomFlip/p": 0.2,
            "RandomAffine/p.angle": 0.4,
            "RandomAffine/p.translate": 0.4,
            "RandomAffine/p.scale": 0.4,
            "RandomNoise/p": 0.0,
        }

        expected_params_200 = {
            "RandomRot90/p": 0,
            "RandomFlip/p": 0.2,
            "RandomAffine/p.angle": 0.4,
            "RandomAffine/p.translate": 0.4,
            "RandomAffine/p.scale": 0.4,
            "RandomNoise/p": 0.08,
        }

        with EventStorage(0) as e:
            # Iteration 0
            params = aug.get_tfm_gen_params()
            for k in expected_params_0:
                assert np.allclose(params[k], expected_params_0[k]), "{}: {}, {}".format(
                    k, params[k], expected_params_0[k]
                )

            # Iteration 200
            e._iter = 200
            params = aug.get_tfm_gen_params()
            for k in expected_params_200:
                assert np.allclose(params[k], expected_params_200[k]), "{}: {}, {}".format(
                    k, params[k], expected_params_200[k]
                )

    def test_from_cfg(self):
        cfg = get_cfg()

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
            {"name": "RandomFlip", "p": 0.2, "ndim": 2},
            {"name": "RandomAffine", "p": 0.2, "angle": 12.0, "scale": 2.0, "translate": 0.4},
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
        cfg.AUG_TRAIN.MRI_RECON.TRANSFORMS = tfm_cfg

        aug = MRIReconAugmentor.from_cfg(cfg, aug_kind="aug_train")
        tfms = aug.tfms_or_gens

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
        assert isinstance(tfm, RandomFlip)
        assert tfm.p == 0.2

        tfm = tfms[2]
        assert isinstance(tfm, RandomAffine)
        assert tfm.p == {"angle": 0.2, "scale": 0.2, "translate": 0.2, "shear": 0.2}
        assert tfm.angle == 12.0
        assert tfm.scale == 2.0
        assert tfm.translate == 0.4

        tfm = tfms[3]
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
