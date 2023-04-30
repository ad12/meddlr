import inspect
import unittest

from torch import nn

from meddlr.modeling.blocks.conv_blocks import (
    SimpleConvBlock2d,
    SimpleConvBlock3d,
    SimpleConvBlockNd,
)
from meddlr.modeling.layers import ConvWS2d


class TestSimpleConvBlock(unittest.TestCase):
    def test_structure(self):
        # 2D
        block = SimpleConvBlockNd(16, 32, 3, 2, dropout=0.5)
        assert all(
            isinstance(x, cls)
            for x, cls in zip(block, [nn.Conv2d, nn.BatchNorm2d, nn.ReLU, nn.Dropout2d])
        ), f"Got {block}"
        assert block[0].in_channels == 16
        assert block[0].out_channels == 32
        assert block[0].kernel_size == (3, 3)

        # 3D
        block = SimpleConvBlockNd(16, 32, 3, 3, dropout=0.5)
        assert all(
            isinstance(x, cls)
            for x, cls in zip(block, [nn.Conv3d, nn.BatchNorm3d, nn.ReLU, nn.Dropout3d])
        ), f"Got {block}"
        assert block[0].in_channels == 16
        assert block[0].out_channels == 32
        assert block[0].kernel_size == (3, 3, 3)

    def test_order(self):
        # order
        block = SimpleConvBlockNd(
            in_channels=16,
            out_channels=32,
            kernel_size=3,
            dimension=2,
            order=("convws", ("groupnorm", {"num_groups": 8}), "relu"),
        )
        assert all(
            isinstance(x, cls) for x, cls in zip(block, [ConvWS2d, nn.GroupNorm, nn.ReLU])
        ), f"Got {block}"
        assert block[0].in_channels == 16
        assert block[0].out_channels == 32
        assert block[0].kernel_size == (3, 3)

    def test_nd_subclass_signature(self):
        """
        Verify that the signature for SimpleConvBlock2d and SimpleConvBlock3d are the same as
        SimpleConvBlockNd (except for `dimension`)
        """
        expected_signature = inspect.getfullargspec(SimpleConvBlockNd)
        expected_args = expected_signature.args
        expected_args.remove("dimension")
        expected_defaults = {
            arg: default
            for arg, default in zip(
                expected_signature.args[-len(expected_signature.defaults) :],
                expected_signature.defaults,
            )
        }
        for subclass in [SimpleConvBlock2d, SimpleConvBlock3d]:
            signature = inspect.getfullargspec(subclass)
            defaults = {
                arg: default
                for arg, default in zip(
                    signature.args[-len(signature.defaults) :], signature.defaults
                )
            }

            assert [x == e for x, e in zip(signature.args, expected_args)]
            assert expected_defaults == defaults
