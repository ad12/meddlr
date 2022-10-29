import torch

from meddlr.transforms.functional.mri import add_affine_motion, get_multishot_trajectory

from ..mock import generate_mock_mri_data


def test_add_affine_motion_trivial():
    # Test that the transform is a no-op when the motion is zero.
    kspace, maps, target = generate_mock_mri_data()

    trajectory = get_multishot_trajectory(kind="blocked", nshots=1, shape=kspace.shape[1:3])
    out = add_affine_motion(
        target,
        transform_gens=[],
        trajectory=trajectory,
        maps=maps,
        is_batch=True,
        channels_first=False,
        xtype="image",
    )
    assert torch.allclose(out, target)

    trajectory = get_multishot_trajectory(kind="blocked", nshots=1, shape=kspace.shape[1:3])
    out = add_affine_motion(
        kspace,
        transform_gens=[],
        trajectory=trajectory,
        maps=None,
        is_batch=True,
        channels_first=False,
        xtype="kspace",
    )
    assert torch.allclose(out, kspace)
