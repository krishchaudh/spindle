"""A replacement may add keyword-only parameters with defaults."""

from annotated_upstream import f
from spindle.patching import patch


@patch(f)
def f_patched(a: int, *, extra: int = 0) -> int:
    return a + extra
