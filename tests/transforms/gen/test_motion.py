import unittest

import torch

from meddlr.transforms.gen import (
    RandomAffine,
    RandomMRIMotion,
    RandomMRIMultiShotMotion,
    RandomTranslation,
)

from ..mock import generate_mock_mri_data


class TestRandomMRIMotion(unittest.TestCase):
    def test_randomness(self):
        """Test if there is randomness in the generation process."""
        kspace, _, _ = generate_mock_mri_data()
        kspace = kspace.permute(0, 3, 1, 2)  # BxCxHxW

        motion_aug = RandomMRIMotion(p=1.0, std_devs=(0.6, 0.6))

        tfm = motion_aug.get_transform(kspace)
        out1 = tfm.apply_kspace(kspace)

        tfm = motion_aug.get_transform(kspace)
        out2 = tfm.apply_kspace(kspace)

        assert not torch.allclose(out1, out2)

    def test_reproducibility(self):
        kspace, _, _ = generate_mock_mri_data()
        kspace = kspace.permute(0, 3, 1, 2)  # BxCxHxW
        seed = 42

        aug1 = RandomMRIMotion(p=1.0, std_devs=(0.6, 0.6))
        aug1.seed(seed)
        aug2 = RandomMRIMotion(p=1.0, std_devs=(0.6, 0.6))
        aug2.seed(seed)
        assert torch.all(aug1._generator.get_state() == aug2._generator.get_state())

        tfm1 = aug1.get_transform(kspace)
        tfm2 = aug2.get_transform(kspace)
        assert torch.all(tfm1._generator_state == tfm2._generator_state)

        out1 = tfm1.apply_kspace(kspace)
        out2 = tfm2.apply_kspace(kspace)
        assert torch.all(out1 == out2)

        aug = RandomMRIMotion(p=1.0, std_devs=(0.6, 0.6))
        aug.seed(seed)
        tfm = aug.get_transform(kspace)
        out1 = tfm.apply_kspace(kspace)
        aug.seed(seed)
        tfm = aug.get_transform(kspace)
        out2 = tfm.apply_kspace(kspace)
        assert torch.all(out1 == out2)


class TestRandomMRIMultiShotMotion(unittest.TestCase):
    def test_reproducibility(self):
        kspace, _, _ = generate_mock_mri_data()
        kspace = kspace.permute(0, 3, 1, 2)  # BxCxHxW
        seed = 42

        tx, ty, angle = 0.1, 0.1, 10
        nshots = 2
        trajectory = "blocked"
        tfms = [
            RandomTranslation(translate=(tx, ty), p=1.0, pad_mode="reflect"),
            RandomAffine(angle=angle, p={"angle": 1.0}, pad_like="MRAugment"),
        ]

        aug1 = RandomMRIMultiShotMotion(
            p=1.0, tfms_or_gens=tfms, nshots=nshots, trajectory=trajectory
        )
        aug1.seed(seed)
        aug2 = RandomMRIMultiShotMotion(
            p=1.0, tfms_or_gens=tfms, nshots=nshots, trajectory=trajectory
        )
        aug2.seed(seed)
        assert torch.all(aug1._generator.get_state() == aug2._generator.get_state())

        tfm1 = aug1.get_transform(kspace)
        tfm2 = aug2.get_transform(kspace)
        assert torch.all(tfm1._generator_state == tfm2._generator_state)

        out1 = tfm1.apply_kspace(kspace)
        out2 = tfm2.apply_kspace(kspace)
        assert torch.all(out1 == out2)

        aug = RandomMRIMultiShotMotion(
            p=1.0, tfms_or_gens=tfms, nshots=nshots, trajectory=trajectory
        )
        aug.seed(seed)
        tfm = aug.get_transform(kspace)
        out1 = tfm.apply_kspace(kspace)
        aug.seed(seed)
        tfm = aug.get_transform(kspace)
        out2 = tfm.apply_kspace(kspace)
        assert torch.all(out1 == out2)
