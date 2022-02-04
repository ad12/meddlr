"""Utilities for testing."""
import os
from functools import wraps


def temp_env(func):
    """Allows func to temporarily set environment variables."""

    @wraps(func)
    def wrapper(self, *args, **kwargs):
        old_env = dict(os.environ)

        out = func(self, *args, **kwargs)

        os.environ.clear()
        os.environ.update(old_env)
        return out

    return wrapper


def get_cfg_path(cfg_filename):
    """Return path to config file."""
    cfg_filename = cfg_filename.split("/")
    cfg_filename = os.path.join(os.path.dirname(__file__), "..", "configs", *cfg_filename)
    if not os.path.exists(cfg_filename):
        raise FileNotFoundError(f"Config file {cfg_filename} not found.")
    return cfg_filename


class cplx_tensor_support:
    def __init__(self, value) -> None:
        self.prev_val = None
        self.value = value

    def __enter__(self):
        self.prev_val = os.environ.get("MEDDLR_ENABLE_CPLX_TENSORS", -1)
        os.environ.update({"MEDDLR_ENABLE_CPLX_TENSORS": str(self.value)})
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.prev_val == -1:
            os.environ.pop("MEDDLR_ENABLE_CPLX_TENSORS")
        else:
            os.environ["MEDDLR_ENABLE_CPLX_TENSORS"] = self.prev_val
