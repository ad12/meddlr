import unittest

import torch

from meddlr.transforms.gen import RandomNoise

from ..mock import generate_mock_mri_data


class TestRandomNoise(unittest.TestCase):
    def test_randomness(self):
        """Test if there is randomness in the generation process."""
        kspace, _, _ = generate_mock_mri_data()
        noiser = RandomNoise(p=1.0, std_devs=(0.6, 0.6))

        tfm = noiser.get_transform(kspace)
        out1 = tfm.apply_kspace(kspace)

        tfm = noiser.get_transform(kspace)
        out2 = tfm.apply_kspace(kspace)

        assert not torch.allclose(out1, out2)

    def test_reproducibility(self):
        kspace, _, _ = generate_mock_mri_data()
        seed = 42

        noiser1 = RandomNoise(p=1.0, std_devs=(0.6, 0.6))
        noiser1.seed(seed)
        noiser2 = RandomNoise(p=1.0, std_devs=(0.6, 0.6))
        noiser2.seed(seed)
        assert torch.all(noiser1._generator.get_state() == noiser2._generator.get_state())

        tfm1 = noiser1.get_transform(kspace)
        tfm2 = noiser2.get_transform(kspace)
        assert torch.all(tfm1._generator_state == tfm2._generator_state)

        out1 = tfm1.apply_kspace(kspace)
        out2 = tfm2.apply_kspace(kspace)
        assert torch.all(out1 == out2)

        noiser = RandomNoise(p=1.0, std_devs=(0.6, 0.6))
        noiser.seed(seed)
        tfm = noiser.get_transform(kspace)
        out1 = tfm.apply_kspace(kspace)
        noiser.seed(seed)
        tfm = noiser.get_transform(kspace)
        out2 = tfm.apply_kspace(kspace)
        assert torch.all(out1 == out2)
