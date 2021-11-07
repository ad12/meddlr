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
