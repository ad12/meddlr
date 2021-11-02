import functools
from typing import Optional, Sequence

from fvcore.common.registry import Registry as _Registry


class Registry(_Registry):
    """Extension of fvcore's registry that supports aliases."""

    _ALIAS_KEYWORDS = ("_aliases", "_ALIASES")

    def _get_aliases(self, obj_func_or_class):
        for kw in self._ALIAS_KEYWORDS:
            if hasattr(obj_func_or_class, kw):
                return getattr(obj_func_or_class, kw)
        return []

    def register(self, obj: object = None, aliases: Sequence[str] = None) -> Optional[object]:
        if obj is None:
            # used as a decorator
            def deco(func_or_class: object, aliases=None) -> object:
                name = func_or_class.__name__
                self._do_register(name, func_or_class)
                if aliases is None:
                    aliases = self._get_aliases(func_or_class)
                if not isinstance(aliases, (list, tuple, set)):
                    aliases = [aliases]
                for alias in aliases:
                    self._do_register(alias, func_or_class)
                return func_or_class

            kwargs = {"aliases": aliases}
            if any(v is not None for v in kwargs.values()):
                return functools.partial(deco, **kwargs)
            else:
                return deco

        name = obj.__name__
        self._do_register(name, obj)
        if aliases is None:
            aliases = self._get_aliases(obj) if isinstance(obj, type) else []
        for alias in aliases:
            self._do_register(alias, obj)

    def clear(self):
        self._obj_map = {}
