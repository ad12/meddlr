from typing import Optional

from fvcore.common.registry import Registry as _Registry


class Registry(_Registry):
    """Extension of fvcore's registry that supports aliases."""

    def register(self, obj: object = None) -> Optional[object]:
        if obj is None:
            # used as a decorator
            def deco(func_or_class: object) -> object:
                name = func_or_class.__name__  # pyre-ignore
                self._do_register(name, func_or_class)
                if hasattr(func_or_class, "_aliases"):
                    for alias in func_or_class._aliases:
                        self._do_register(alias, func_or_class)
                return func_or_class

            return deco

        name = obj.__name__  # pyre-ignore
        self._do_register(name, obj)
        if hasattr(obj, "_aliases"):
            for alias in obj._aliases:
                self._do_register(alias, obj)
