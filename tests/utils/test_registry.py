from meddlr.utils.registry import Registry

_MOCK_REGISTRY = Registry("MOCK_REGISTRY")


@_MOCK_REGISTRY.register(aliases="mock_fn")
def _mock_fn():
    pass


@_MOCK_REGISTRY.register(aliases="MockCls")
class _MockCls:
    pass


@_MOCK_REGISTRY.register()
class _MockAliasCls:
    _ALIASES = ["MockAliasCls"]


def test_registry_decorator():
    assert _MOCK_REGISTRY.get("_mock_fn") == _mock_fn
    assert _MOCK_REGISTRY.get("mock_fn") == _mock_fn

    assert _MOCK_REGISTRY.get("_MockCls") == _MockCls
    assert _MOCK_REGISTRY.get("MockCls") == _MockCls

    assert _MOCK_REGISTRY.get("_MockAliasCls") == _MockAliasCls
    assert _MOCK_REGISTRY.get("MockAliasCls") == _MockAliasCls


def test_registry_object():
    obj = _MockAliasCls()
    obj.__name__ = "Sample"

    registry = Registry("Mock")

    registry.register(obj)
    assert registry.get("Sample") == obj

    registry.clear()
    registry.register(obj, ["FooBar"])
    assert registry.get("Sample") == obj
    assert registry.get("FooBar") == obj
