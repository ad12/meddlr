import numpy as np
import pytest
import torch
import torch.nn.functional as F

from meddlr.ops.categorical import categorical_to_one_hot, logits_to_prob, one_hot_to_categorical


@pytest.mark.parametrize("use_numpy", [False, True])
def test_categorical_to_one_hot(use_numpy):
    labels = torch.tensor([3, 0, 1, 0, 2, 0, 0, 3])
    expected = torch.tensor(
        [
            [0, 0, 0, 1],
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [1, 0, 0, 0],
            [0, 0, 1, 0],
            [1, 0, 0, 0],
            [1, 0, 0, 0],
            [0, 0, 0, 1],
        ]
    )

    if use_numpy:
        labels = labels.numpy()
        expected = expected.numpy()
        all_func = np.all
    else:
        all_func = torch.all
        dtype = torch.int64

    out = categorical_to_one_hot(labels, channel_dim=-1, background=None)
    assert all_func(out == expected)

    out = categorical_to_one_hot(labels, channel_dim=-1, background=0)
    assert all_func(out == expected[:, 1:])

    out = categorical_to_one_hot(labels, channel_dim=-1, background=None, num_categories=3)
    assert all_func(out == expected)

    if not use_numpy:
        out = categorical_to_one_hot(labels.T, channel_dim=0, background=0, dtype=dtype)
        assert out.dtype == dtype
        assert all_func(out.T == expected[:, 1:])


@pytest.mark.parametrize("use_numpy", [False, True])
def test_one_hot_to_categorical(use_numpy):
    one_hot = torch.tensor(
        [
            [0, 0, 0, 1],
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [1, 0, 0, 0],
            [0, 0, 1, 0],
            [1, 0, 0, 0],
            [1, 0, 0, 0],
            [0, 0, 0, 1],
        ]
    )
    expected = torch.tensor([3, 0, 1, 0, 2, 0, 0, 3])

    if use_numpy:
        one_hot = one_hot.numpy()
        expected = expected.numpy()
        all_func = np.all
    else:
        all_func = torch.all

    out = one_hot_to_categorical(one_hot, channel_dim=-1, background=None)
    assert all_func(out == expected + 1)

    out = one_hot_to_categorical(one_hot, channel_dim=-1, background=0)
    assert all_func(out == expected)


@pytest.mark.parametrize("use_numpy", [False, True])
def test_logits_to_prob(use_numpy):
    logits = torch.randn(10, 4)
    sigmoid = torch.sigmoid(logits)
    softmax = F.softmax(logits, dim=1)

    if use_numpy:
        logits = logits.numpy()
        sigmoid = sigmoid.numpy()
        softmax = softmax.numpy()
        all_func = np.all
    else:
        all_func = torch.all

    assert all_func(logits_to_prob(logits, "sigmoid") == sigmoid)
    assert all_func(logits_to_prob(logits, "softmax") == softmax)
    with pytest.raises(ValueError):
        logits_to_prob(logits, "invalid")
