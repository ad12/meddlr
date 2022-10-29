import pytest

from meddlr.config import get_cfg
from meddlr.data.transforms.subsample import build_mask_func
from meddlr.data.transforms.transform import DataTransform

from ...transforms.mock import generate_mock_mri_data
from ...util import assert_allclose


@pytest.mark.parametrize("tx", [0, 0.1])
@pytest.mark.parametrize("ty", [0, 0.1])
@pytest.mark.parametrize("angle", [0, 10])
@pytest.mark.parametrize("nshots", [1, 4])
@pytest.mark.parametrize("trajectory", ["interleaved", "blocked"])
def test_data_transform_affine_motion_reproducibility(
    tx: float, ty: float, angle: float, nshots: int, trajectory: str
):
    """Test that the DataTransform produces reproducibile augmentations."""
    # Create a transform that augments the data.
    tfms = [
        {"name": "RandomTranslation", "p": 1.0, "pad_mode": "reflect", "translate": (tx, ty)},
        {"name": "RandomAffine", "p": {"angle": 1.0}, "pad_like": "MRAugment", "angle": angle},
    ]
    motion_tfm = {
        "name": "RandomMRIMultiShotMotion",
        "p": 1.0,
        "nshots": nshots,
        "trajectory": trajectory,
        "tfms_or_gens": tfms,
    }

    cfg = get_cfg().defrost()
    # Motion augmentations.
    cfg.AUG_TRAIN.MRI_RECON.TRANSFORMS = [motion_tfm]
    cfg = _add_undersample_and_normalization_config(cfg, inplace=True)
    cfg = cfg.freeze()

    mask_func = build_mask_func(cfg.AUG_TRAIN)
    transform = DataTransform(cfg, mask_func=mask_func, is_test=True, use_augmentor=True)

    assert transform._normalizer is not None
    assert transform.augmentor is not None

    # Apply the transform to the data.
    kspace, maps, target = generate_mock_mri_data(remove_batch_dim=True)
    fname = "test-transform"
    slice_id = 0
    out1 = transform(kspace, maps, target, fname=fname, slice_id=slice_id, is_fixed=False)
    out2 = transform(kspace, maps, target, fname=fname, slice_id=slice_id, is_fixed=False)

    assert_allclose(out1, out2, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("sigma", [0, 0.1])
@pytest.mark.parametrize("alpha", [0, 0.1])
def test_data_transform_noise_motion_reproducibility(sigma: float, alpha: float):
    cfg = get_cfg().defrost()
    cfg = _add_undersample_and_normalization_config(cfg, inplace=True)
    cfg.MODEL.CONSISTENCY.AUG.NOISE.STD_DEV = (sigma,)  # Noise
    cfg.MODEL.CONSISTENCY.AUG.MOTION.RANGE = alpha  # Motion
    cfg = cfg.freeze()

    mask_func = build_mask_func(cfg.AUG_TRAIN)
    transform = DataTransform(
        cfg,
        mask_func=mask_func,
        add_noise=sigma > 0,
        add_motion=alpha > 0,
        is_test=True,
        use_augmentor=False,
    )

    assert transform._normalizer is not None
    assert transform.augmentor is None

    # Apply the transform to the data.
    kspace, maps, target = generate_mock_mri_data(remove_batch_dim=True)
    fname = "test-transform"
    slice_id = 0
    out1 = transform(kspace, maps, target, fname=fname, slice_id=slice_id, is_fixed=False)
    out2 = transform(kspace, maps, target, fname=fname, slice_id=slice_id, is_fixed=False)
    assert_allclose(out1, out2, rtol=1e-5, atol=1e-5)


def _add_undersample_and_normalization_config(cfg, inplace=False):
    if not inplace:
        cfg = cfg.clone()
    cfg = cfg.defrost()
    # Undersampling.
    cfg.AUG_TRAIN.UNDERSAMPLE.ACCELERATIONS = (2,)
    cfg.AUG_TRAIN.UNDERSAMPLE.NAME = "PoissonDiskMaskFunc"
    cfg.AUG_TRAIN.UNDERSAMPLE.CALIBRATION_SIZE = 4
    # Normalization.
    cfg.MODEL.NORMALIZER.NAME = "TopMagnitudeNormalizer"
    return cfg
