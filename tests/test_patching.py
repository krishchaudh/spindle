from types import ModuleType

from spindle.patching import patch


def _module(name: str, **attributes: object) -> ModuleType:
    module = ModuleType(name)
    vars(module).update(attributes)
    return module


def test_dependents_repointed_only_while_bound_to_original() -> None:
    def f() -> int:
        return 1

    def new() -> int:
        return 2

    def local() -> int:
        return 3

    upstream = _module("upstream", f=f)
    stale = _module("stale", f=f)
    drifted = _module("drifted", f=local)
    empty = _module("empty")

    patch(f, owner=upstream, dependents=(stale, drifted, empty))(new)

    assert vars(upstream)["f"] is new
    assert vars(stale)["f"] is new
    assert vars(drifted)["f"] is local
    assert "f" not in vars(empty)
