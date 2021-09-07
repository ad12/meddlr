import unittest

import torch

from ss_recon.transforms.base import AffineTransform, FlipTransform, NoiseTransform, Rot90Transform
from ss_recon.transforms.builtin.mri import MRIReconAugmentor
from ss_recon.transforms.gen import (
    RandomAffine,
    RandomFlip,
    RandomMRIMotion,
    RandomNoise,
    RandomRot90,
)

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
            RandomAffine(p=p, angle=12, translate=None, scale=2, shear=30),
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
