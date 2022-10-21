"""Utilities for testing."""
import inspect
import os
import pathlib
import re
import tempfile
import uuid
from functools import wraps
from typing import Any, Dict, Sequence, Union

import numpy as np

from meddlr.utils.events import EventStorage
from meddlr.utils.general import flatten_dict

TEST_MODEL_ZOOS = os.environ.get("MEDDLR_TEST_MODEL_ZOOS", "") == "True"

# Set cache directory to be non-conflicting with other tests.
_TEMP_CACHE_DIR = tempfile.TemporaryDirectory(f"meddlr-test-cache-{uuid.uuid4()}")
TEMP_CACHE_DIR = pathlib.Path(_TEMP_CACHE_DIR.name)


def temp_env(func):
    """Allows func to temporarily set environment variables."""

    @wraps(func)
    def wrapper(self, *args, **kwargs):
        old_env = dict(os.environ)

        try:
            out = func(self, *args, **kwargs)
        finally:
            os.environ.clear()
            os.environ.update(old_env)
        return out

    @wraps(func)
    def wrapper_func(*args, **kwargs):
        old_env = dict(os.environ)

        try:
            out = func(*args, **kwargs)
        finally:
            os.environ.clear()
            os.environ.update(old_env)
        return out

    return wrapper if "self" in inspect.signature(func).parameters else wrapper_func


def temp_event_storage(func):
    @wraps(func)
    def wrapper(self, *args, **kwargs):
        with EventStorage(start_iter=0):
            out = func(self, *args, **kwargs)
        return out

    @wraps(func)
    def wrapper_func(*args, **kwargs):
        with EventStorage(start_iter=0):
            out = func(*args, **kwargs)
        return out

    return wrapper if "self" in inspect.signature(func).parameters else wrapper_func


def get_cfg_path(cfg_filename):
    """Return path to config file."""
    cfg_filename = cfg_filename.split("/")
    cfg_filename = os.path.join(os.path.dirname(__file__), "..", "configs", *cfg_filename)
    if not os.path.exists(cfg_filename):
        raise FileNotFoundError(f"Config file {cfg_filename} not found.")
    return cfg_filename


class MarkdownNode:
    """Headings at same level must be uniquely named"""

    def __init__(self, name, content=(), children: Sequence["MarkdownNode"] = None) -> None:
        self.name = name
        self.content = content
        if children is None:
            children = []
        self.children = children

    def add_children(self, children: Union["MarkdownNode", Sequence["MarkdownNode"]]):
        if isinstance(children, MarkdownNode):
            children = [children]
        for c in children:
            assert c not in self.children
            self.children.append(c)

    def remove_children(self, children: Union["MarkdownNode", Sequence["MarkdownNode"]]):
        if isinstance(children, MarkdownNode):
            children = [children]
        for c in children:
            assert c in self.children
        self.children = [c for c in self.children if c not in children]

    def to_dict(self, flatten=False):
        if flatten:
            return self._flattened_dict_repr()
        out = {
            "_name": self.name,
            "content": self.content,
        }
        if self.children:
            out["_children"] = {c.name: c.to_dict() for c in self.children}

    def _flattened_dict_repr(self):
        if not self.children:
            return {self.name: self.content}

        out = {}
        if self.content:
            out[self.name] = self.content
        for c in self.children:
            out.update({f"{self.name}/{k}": v for k, v in c._flattened_dict_repr().items()})
        return out


def parse_markdown(lines: Sequence[str], node: MarkdownNode = None, level: int = 0):
    if node is None:
        node = MarkdownNode("_root", [])

    while len(lines) > 0:
        line = lines[0].strip()

        heading_tag = line.split(" ")[0].strip()
        is_heading = re.match("^#*$", heading_tag) is not None
        if not is_heading:
            node.content.append(line)
            lines = lines[1:]
            continue

        num_heading = len(heading_tag)
        if level >= num_heading:
            return node, lines
        else:
            child = MarkdownNode(line.split(" ", maxsplit=1)[1].strip(), [])
            child, lines = parse_markdown(lines[1:], child, level=num_heading)
            node.add_children(child)

    if not node.content and len(node.children) == 1:
        node = node.children[0]
    return node, []


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


def assert_shape(x: Dict[str, Any], expected_shape: Dict[str, Any]):
    """Check the shape of tensors in `x` are equivalent to `expected_shape`."""
    x_shape = {k: tuple(v.shape) for k, v in flatten_dict(x).items()}
    np.testing.assert_equal(x_shape, flatten_dict(expected_shape))
