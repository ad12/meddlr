import torch


def move_to_device(obj, device, non_blocking=False):
    """Given a structure (possibly) containing Tensors on the CPU, move all the Tensors
      to the specified GPU (or do nothing, if they should be on the CPU).
        device = -1 -> "cpu"
        device =  0 -> "cuda:0"

    Args:
      obj(Any): The object to convert.
      device(int): The device id, defaults to -1.

    Returns:
      Any: The converted object.
    """
    if not torch.cuda.is_available() or (isinstance(device, int) and device < 0):
        device = "cpu"
    elif isinstance(device, int):
        device = f"cuda:{device}"

    if isinstance(obj, torch.Tensor):
        return obj.to(device, non_blocking=non_blocking)  # type: ignore
    elif isinstance(obj, dict):
        return {key: move_to_device(value, device, non_blocking=non_blocking) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [move_to_device(item, device, non_blocking=non_blocking) for item in obj]
    elif isinstance(obj, tuple):
        return tuple([move_to_device(item, device, non_blocking=non_blocking) for item in obj])
    else:
        return obj
