import numpy as np
import torch
import torch.nn.functional as F

__all__ = [
    "one_hot_to_categorical",
    "categorical_to_one_hot",
    "logits_to_prob",
    "pred_to_categorical",
]


def one_hot_to_categorical(pred, channel_dim: int = 1, background=False):
    """Converts one-hot encoded predictions to categorical predictions.

    Args:
        pred: One-hot encoded predictions. Shape BxCx...
        background: If ``True``, assumes first channel is the background.

    Returns:
        torch.Tensor | np.ndarray: Categorical array or tensor. If ``background=False``,
            the output will be 1-indexed such that ``0`` corresponds to the background.
    """
    is_ndarray = isinstance(pred, np.ndarray)
    if is_ndarray:
        pred = torch.as_tensor(pred)

    if background is not None and background is not False:
        out = torch.argmax(pred, channel_dim)
    else:
        out = torch.argmax(pred.type(torch.long), dim=channel_dim) + 1
        out = torch.where(pred.sum(channel_dim) == 0, torch.tensor([0], device=pred.device), out)

    if is_ndarray:
        out = out.numpy()
    return out


def categorical_to_one_hot(
    tensor, channel_dim: int = 1, background=0, num_categories=None, dtype=None
):
    is_ndarray = isinstance(tensor, np.ndarray)
    if is_ndarray:
        tensor = torch.from_numpy(tensor)

    if num_categories is None:
        num_categories = torch.max(tensor).cpu().item()
    num_categories += 1

    shape = tensor.shape
    out_shape = (num_categories,) + shape

    if dtype is None:
        dtype = torch.bool
    default_value = True if dtype == torch.bool else 1
    if tensor.dtype != torch.long:
        tensor = tensor.type(torch.long)

    out = torch.zeros(out_shape, dtype=dtype, device=tensor.device)
    out.scatter_(0, tensor.reshape((1,) + tensor.shape), default_value)
    if background is not None:
        out = torch.cat([out[0:background], out[background + 1 :]], dim=0)
    if channel_dim != 0:
        if channel_dim < 0:
            channel_dim = out.ndim + channel_dim
        order = (channel_dim,) + tuple(d for d in range(out.ndim) if d != channel_dim)
        out = out.permute(tuple(np.argsort(order)))
        out = out.contiguous()

    if is_ndarray:
        out = out.numpy()
    return out


def logits_to_prob(logits, activation, channel_dim: int = 1):
    is_ndarray = isinstance(logits, np.ndarray)
    if is_ndarray:
        logits = torch.from_numpy(logits)

    if activation == "sigmoid":
        out = torch.sigmoid(logits)
    elif activation == "softmax":
        out = F.softmax(logits, dim=channel_dim)

    if is_ndarray:
        out = out.numpy()
    return out


def pred_to_categorical(pred_or_logits, activation, channel_dim: int = 1, threshold: float = 0.5):
    """Converts one-hot encoded predictions or logits to category.

    Args:
        pred_or_logits: One-hot encoded predictions or logits. Shape BxCx...
        activation (str): Activation to use.
            Either ``'sigmoid'`` or ``'softmax'`` if ``pred_or_logits`` are logits.
            If `None` or '', assumes that `pred` does not need to be passed through
            activation function. If 'softmax', should include a background class.
        include_background (bool): If `True`, the first slice of class dimension (``C``)
            will not be dropped.
    """
    if activation not in [None, "", "sigmoid", "softmax"]:
        raise ValueError(f"activation '{activation}' not supported'")

    pred = pred_or_logits
    is_ndarray = isinstance(pred, np.ndarray)
    if is_ndarray:
        pred = torch.from_numpy(pred)

    if activation == "sigmoid":
        # TODO: Validate this case.
        out = one_hot_to_categorical(torch.sigmoid(pred) > threshold, channel_dim=channel_dim)
    elif activation == "softmax":
        out = torch.argmax(pred, dim=channel_dim)
    else:
        # if not activation specified, assume it is one-hot encoded.
        out = one_hot_to_categorical(pred, channel_dim=channel_dim)

    if is_ndarray:
        out = out.numpy()
    return out
