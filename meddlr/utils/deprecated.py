import warnings
from typing import Any, Callable, TypeVar

TCallable = TypeVar("TCallable", bound=Callable[..., Any])


def deprecated(
    reason=None, vdeprecated=None, vremove=None, replacement=None
) -> Callable[[TCallable], TCallable]:
    def fn(func: TCallable) -> TCallable:
        msg = _get_deprecated_msg(func, reason, vdeprecated, vremove, replacement)
        warnings.warn(msg, DeprecationWarning)
        return func

    return fn


def _get_deprecated_msg(wrapped, reason, vdeprecated, vremoved, replacement=None):
    fmt = "{name} is deprecated"
    if vdeprecated:
        fmt += " since v{vdeprecated}"
    if vremoved:
        fmt += " and will be removed in v{vremoved}"
    fmt += "."

    if reason:
        fmt += " ({reason})"
    if replacement:
        fmt += " -- Use meddlr.{replacement} instead."

    return fmt.format(
        name=wrapped.__name__,
        reason=reason or "",
        vdeprecated=vdeprecated or "",
        vremoved=vremoved or "",
        replacement=replacement or "",
    )
