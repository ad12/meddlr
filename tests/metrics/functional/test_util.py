import numpy as np
import torch

from meddlr.metrics.functional.util import flatten_non_category_dims, flatten_other_dims, to_bool


def test_to_bool():
    x = [1, 0]
    assert np.all(to_bool(x) == np.asarray([True, False]))

    x = np.array([1, 0])
    assert np.all(to_bool(x) == np.asarray([True, False]))

    x = torch.tensor([1, 0])
    assert torch.all(to_bool(x) == torch.as_tensor([True, False]))


def test_flatten_other_dims():
    x = np.random.rand(2, 3, 4)
    out = flatten_other_dims(x)
    assert np.all(out == x.flatten())

    x = torch.rand((2, 3, 4))
    out = flatten_other_dims(x, 0)
    assert torch.all(out == x.flatten(1, 2))


def test_flatten_non_category_dims():
    x = torch.rand((2, 3, 4))
    out = flatten_non_category_dims(x, 0)
    assert torch.all(out == x.flatten(1, 2))
