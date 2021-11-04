import itertools
import unittest

import torch

from meddlr.data.transforms.motion import MotionModel
from meddlr.transforms.base.motion import MRIMotionTransform

from ..mock import generate_mock_mri_data


class TestMRIMotionTransform(unittest.TestCase):
    def test_reproducibility(self):
        """Test reproducibility with exisiting motion transform."""
        kspace, _, _ = generate_mock_mri_data()

        for std_dev, seed in itertools.product([0.2, 0.4, 0.6], [100, 200, 300]):
            motion_mdl = MotionModel((std_dev, std_dev))
            motion_tfm = MRIMotionTransform(std_dev=std_dev, seed=seed)

            kspace_mdl = motion_mdl(kspace, seed=seed)
            kspace_tfm = motion_tfm.apply_kspace(kspace, channel_first=False)

            assert torch.equal(kspace_tfm, kspace_mdl)
