import unittest
from typing import Dict

import numpy as np
import torch
import torch.nn.functional as F
from parameterized import parameterized
from torch import nn

import meddlr.ops.complex as cplx
from meddlr.config.config import get_cfg
from meddlr.data.transforms.noise import NoiseModel
from meddlr.forward.mri import SenseModel
from meddlr.modeling.meta_arch.build import META_ARCH_REGISTRY
from meddlr.modeling.meta_arch.n2r import N2RModel
from meddlr.modeling.meta_arch.ssdu import SSDUModel
from meddlr.transforms.gen.mask import RandomKspaceMask
from meddlr.utils.general import flatten_dict
from tests import util
from tests.transforms.mock import generate_mock_mri_data


class DummyFCN(nn.Module):
    """A dummy fully convolutional network."""

    def __init__(self, nchannels: int = 2) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels=nchannels, out_channels=nchannels * 2, kernel_size=3, padding=1
        )
        self.conv2 = nn.Conv2d(
            in_channels=nchannels * 2, out_channels=nchannels, kernel_size=3, padding=1
        )

        # This should be set to -1 by N2R.
        self.vis_period = 10

    def forward(
        self, inputs: Dict[str, torch.Tensor], return_pp=True, vis_training: bool = False
    ) -> Dict[str, torch.Tensor]:
        kspace = inputs["kspace"]  # shape [batch, height, width, #coils]
        maps = inputs["maps"]  # shape [batch, height, width, #coils, #maps]
        mask = cplx.get_mask(kspace)
        target = inputs.get("target", None)  # shape [batch, height, width, #maps]

        x: torch.Tensor = SenseModel(maps, weights=mask)(kspace, adjoint=True)

        x = torch.view_as_real(x.squeeze(-1))
        x = x.permute(0, 3, 1, 2).contiguous()

        x = self.conv2(F.relu(self.conv1(x)))

        x = x.permute(0, 2, 3, 1).contiguous()
        x = torch.view_as_complex(x).unsqueeze(-1)

        return {"pred": x, "target": target}


class N2RModelTest(unittest.TestCase):
    def _build_model(
        self,
        model: nn.Module = None,
        noise_range=(0.2, 0.5),
        use_supervised_consistency: bool = False,
        vis_period: int = 10,
    ) -> N2RModel:
        if model is None:
            model = DummyFCN()
        return N2RModel(
            model,
            noiser=NoiseModel(noise_range),
            use_supervised_consistency=use_supervised_consistency,
            vis_period=vis_period,
        )

    def _build_inputs(
        self,
        supervised_bsz: int,
        unsupervised_bsz: int,
        ky: int = 10,
        kz: int = 20,
    ) -> Dict[str, Dict[str, torch.Tensor]]:
        inputs = {}
        if supervised_bsz > 0:
            inputs["supervised"] = generate_mock_mri_data(
                ky=ky, kz=kz, bsz=supervised_bsz, as_dict=True
            )
        if unsupervised_bsz > 0:
            data = generate_mock_mri_data(ky=ky, kz=kz, bsz=unsupervised_bsz, as_dict=True)
            data.pop("target")  # unsupervised data should not have target.
            inputs["unsupervised"] = data
        return inputs

    @parameterized.expand([(1, 2), (2, 1), (1, 0), (0, 1)])
    @util.temp_event_storage
    def test_forward(self, supervised_bsz: int, unsupervised_bsz: int):
        ky, kz = 10, 20

        inputs = self._build_inputs(supervised_bsz, unsupervised_bsz, ky=ky, kz=kz)
        n2r_model = self._build_model()
        outputs = n2r_model(inputs)

        assert n2r_model.model.vis_period == -1

        expected_shape = {}
        if supervised_bsz > 0:
            expected_shape["recon"] = {
                "pred": (supervised_bsz, ky, kz, 1),
                "target": (supervised_bsz, ky, kz, 1),
            }
        if unsupervised_bsz > 0:
            expected_shape["consistency"] = {
                "pred": (unsupervised_bsz, ky, kz, 1),
                "target": (unsupervised_bsz, ky, kz, 1),
            }
        util.assert_shape(outputs, expected_shape)

        outputs_flat = flatten_dict(outputs)
        for key in [k for k in outputs_flat.keys() if k.endswith("pred")]:
            assert outputs_flat[key].requires_grad, f"{key} should require gradients"
        for key in [k for k in outputs_flat.keys() if k.endswith("target")]:
            assert not outputs_flat[key].requires_grad, f"{key} should not require grad"

        if unsupervised_bsz > 0:
            with torch.no_grad():
                expected_targets = n2r_model.model(inputs["unsupervised"])["pred"]
            cons_target = outputs["consistency"]["target"]
            np.testing.assert_allclose(cons_target, expected_targets, atol=1e-5)

    @parameterized.expand([(1, 1), (1, 0)])
    @util.temp_event_storage
    def test_supervised_consistency(self, supervised_bsz: int, unsupervised_bsz: int):
        ky, kz = 10, 20

        inputs = self._build_inputs(supervised_bsz, unsupervised_bsz, ky=ky, kz=kz)
        n2r_model = self._build_model(use_supervised_consistency=True)
        outputs = n2r_model(inputs)

        expected_shape = {}
        if supervised_bsz > 0:
            expected_shape["recon"] = {
                "pred": (supervised_bsz, ky, kz, 1),
                "target": (supervised_bsz, ky, kz, 1),
            }
        cons_bsz = supervised_bsz + unsupervised_bsz
        expected_shape["consistency"] = {
            "pred": (cons_bsz, ky, kz, 1),
            "target": (cons_bsz, ky, kz, 1),
        }
        util.assert_shape(outputs, expected_shape)

        expected_targets = []
        with torch.no_grad():
            if unsupervised_bsz > 0:
                expected_targets.append(n2r_model.model(inputs["unsupervised"])["pred"])
            expected_targets.append(n2r_model.model(inputs["supervised"])["pred"])
            expected_targets = (
                torch.cat(expected_targets, dim=0)
                if len(expected_targets) > 1
                else expected_targets[0]
            )
        cons_target = outputs["consistency"]["target"]
        np.testing.assert_allclose(cons_target, expected_targets, atol=1e-5)

    @parameterized.expand(
        [
            ("UnetModel", False),
            ("UnetModel", True),
            ("GeneralizedUnrolledCNN", False),
            ("GeneralizedUnrolledCNN", True),
        ]
    )
    def test_from_config(self, meta_architecture: str, use_supervised_consistency: bool):
        cfg = get_cfg()
        cfg.MODEL.META_ARCHITECTURE = "N2RModel"
        cfg.MODEL.N2R.META_ARCHITECTURE = meta_architecture
        cfg.MODEL.N2R.USE_SUPERVISED_CONSISTENCY = use_supervised_consistency

        n2r_model = N2RModel(cfg)

        assert n2r_model.use_supervised_consistency == use_supervised_consistency
        assert isinstance(n2r_model.model, META_ARCH_REGISTRY.get(meta_architecture))

    def test_eval(self):
        inputs = self._build_inputs(1, 0)

        model = self._build_model().eval()
        assert not model.training

        with torch.no_grad():
            out1 = model(inputs)
            out2 = model(inputs["supervised"])
        np.testing.assert_allclose(out1["pred"], out2["pred"], atol=1e-5)

    @util.temp_event_storage
    def test_n2r_ssdu(self):
        """Test using self-supervised N2R (i.e. N2R+SSDU)."""
        inputs = self._build_inputs(0, 1, ky=10, kz=20)  # only undersampled inputs
        # SSDU requires an edge mask.
        kspace = inputs["unsupervised"]["kspace"]
        inputs["unsupervised"]["edge_mask"] = torch.zeros(
            kspace.shape[:3] + (1,), device=kspace.device
        )

        masker = RandomKspaceMask(p=1.0, rhos=0.4)
        model = SSDUModel(DummyFCN(), masker=masker, vis_period=10)
        model = self._build_model(model=model)

        assert model.model.vis_period == 10

        outputs = model(inputs)
        expected_shape = {
            "recon": {
                "pred": (1, 10, 20, 8),  # 8 channels for 8 coils
                "target": (1, 10, 20, 8),  # 8 channels for 8 coils
            },
            "consistency": {
                "pred": (1, 10, 20, 1),
                "target": (1, 10, 20, 1),
            },
        }
        util.assert_shape(outputs, expected_shape)
