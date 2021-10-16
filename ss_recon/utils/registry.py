from typing import Optional

from fvcore.common.registry import Registry as _Registry


class Registry(_Registry):
    """Extension of fvcore's registry that supports aliases."""

    def register(self, obj: object = None) -> Optional[object]:
        out = super().register(obj=obj)
        if hasattr(obj, "_aliases"):
            for alias in obj._aliases:
                self._do_register(alias, obj)
        return out
