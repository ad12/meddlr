import functools
from typing import Optional, Sequence

from fvcore.common.registry import Registry as _Registry
from tabulate import tabulate


class Registry(_Registry):
    """Extension of fvcore's registry that supports aliases."""

    _ALIAS_KEYWORDS = ("_aliases", "_ALIASES")

    def __init__(self, name: str):
        super().__init__(name=name)
        self._metadata_map = {}

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
                    self._do_register(alias, func_or_class, is_alias=True)
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
            self._do_register(alias, obj, is_alias=True)

    def _do_register(self, name: str, obj: object, **kwargs) -> None:
        docstring = obj.__doc__
        if docstring is None:
            docstring = ""

        aliases = self._get_aliases(obj) if isinstance(obj, type) else None
        if not aliases:
            aliases = None

        self._metadata_map[name] = {
            "name": name,
            "description": kwargs.pop("description", docstring.split("\n")[0]),
            "aliases": aliases,
            **kwargs,
        }
        return super()._do_register(name, obj)

    def clear(self):
        self._obj_map = {}
        self._metadata_map = {}

    def __repr__(self) -> str:
        metadata = [v for v in self._metadata_map.values() if not v.get("is_alias", False)]
        table = tabulate(metadata, headers="keys", tablefmt="fancy_grid")
        return "Registry of {}:\n{}".format(self._name, table)
