"""Same parameters, incompatible return type: ty rejects it."""

from annotated_upstream import f
from spindle.patching import patch


@patch(f)
def f_patched(a: int) -> str:
    return str(a)
