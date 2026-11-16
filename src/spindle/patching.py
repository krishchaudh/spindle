"""Type-checked replacements for pinned upstream functions.

Wrap a replacement in ``same_signature(original)`` so ty fails when the pinned
upstream signature drifts from it. CI installs the pinned upstream packages so
ty sees the real signatures.
"""

from collections.abc import Callable


def same_signature[**P, R](
    original: Callable[P, R],
) -> Callable[[Callable[P, R]], Callable[P, R]]:
    """Return the replacement unchanged, typed to require ``original``'s signature."""
    return lambda new: new
