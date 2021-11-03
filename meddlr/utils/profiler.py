import time
from collections import defaultdict
from typing import Dict, Mapping

from meddlr.utils import env

_CURRENT_TIMER_STACK = []


def get_timer() -> "AdvancedTimer":
    """
    Returns:
        The :class:`EventStorage` object that's currently being used.
        Throws an error if no :class`EventStorage` is currently enabled.
    """
    assert len(_CURRENT_TIMER_STACK), (
        "get_event_storage() has to be called inside a " "'with EventStorage(...)' context!"
    )
    return _CURRENT_TIMER_STACK[-1]


def time_profile(prefix=None, delimiter=".", disable=None, top_lvl_only=True):
    if disable is None:
        disable = not env.is_profiling_enabled()

    def _decorator(func):
        name = func.__qualname__
        if prefix is not None:
            name = f"{prefix}{delimiter}{name}"

        def _wrapper(*args, **kwargs):
            if disable:
                return func(*args, **kwargs)

            with AdvancedTimer(name, unroll=True) as timer:
                out = func(*args, **kwargs)
                time_args = timer.summarize()

            if (top_lvl_only and len(_CURRENT_TIMER_STACK) == 0) and isinstance(out, Dict):
                if "_profiler" not in out:
                    out["_profiler"] = {}
                out["_profiler"].update(time_args)
            return out

        return _wrapper

    return _decorator


class AdvancedTimer:
    def __init__(self, name, delimiter=".", unroll=False, data=None):
        self.name = name
        self.delimiter = delimiter
        self.unroll = unroll
        self.timer_started = defaultdict(bool)
        if data is None:
            data = {}
        self.data = data

    def start(self, key):
        self.data[key] = time.perf_counter()
        self.timer_started[key] = True

    def time(self, key):
        if not self.timer_started[key]:
            raise ValueError(f"Must start timer for key '{key}' before calling `time`.")
        self.data[key] = time.perf_counter() - self.data[key]
        self.timer_started[key] = False
        return self.data[key]

    def stop(self, key):
        return self.time(key)

    def update(self, data: Mapping):
        self.data.update(data)
        self.timer_started.update({k: False for k in data})

    def summarize(self) -> Dict:
        return {f"{self.name}{self.delimiter}{k}": v for k, v in self.data.items()}

    def __enter__(self):
        _CURRENT_TIMER_STACK.append(self)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        assert _CURRENT_TIMER_STACK[-1] == self
        if self.unroll and len(_CURRENT_TIMER_STACK) > 1:
            _CURRENT_TIMER_STACK[-2].update(self.summarize())
        _CURRENT_TIMER_STACK.pop()

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self.name})"
