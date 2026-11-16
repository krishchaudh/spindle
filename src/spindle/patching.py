"""Type-checked installation of replacements for pinned upstream functions.

``patch(old)(new)`` installs ``new`` over ``old`` under a shared signature
type, so ty fails the build when the pinned upstream signature drifts from a
replacement. CI installs the pinned upstream packages so ty sees the real
signatures.
"""

from __future__ import annotations

from collections.abc import Callable
from types import ModuleType


def patch[**P, R](
    old: Callable[P, R],
    *,
    owner: object,
    dependents: tuple[ModuleType, ...] = (),
) -> Callable[[Callable[P, R]], Callable[P, R]]:
    """Return an installer that replaces ``old`` with a same-signature callable.

    ``owner`` is the module or class holding ``old``. ``dependents`` are modules
    that bound ``old`` by name at import time (``from upstream import fn``);
    each is re-pointed only while it still refers to ``old``.
    """

    if not callable(old):
        raise TypeError(f"patch() replaces a callable, got {type(old).__name__}")

    def install(new: Callable[P, R]) -> Callable[P, R]:
        name = _attribute_name(owner, old)
        setattr(owner, name, new)
        for dependent in dependents:
            if getattr(dependent, name, None) is old:
                setattr(dependent, name, new)
        return new

    return install


def _attribute_name(owner: object, old: Callable) -> str:
    """The name ``owner`` holds ``old`` under; usually the function's own name.

    An earlier patch or a test fake may hold a function defined under another
    name, so fall back to an identity scan before trusting ``old.__name__``.
    """
    if getattr(owner, old.__name__, None) is old:
        return old.__name__
    for key, value in vars(owner).items():
        if value is old:
            return key
    return old.__name__
