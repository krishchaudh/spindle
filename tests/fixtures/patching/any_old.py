"""An Any-typed original (for example a module passed as an untyped parameter)
escapes the signature check: ty accepts even a drifted replacement. Kept as a
documented boundary of the helper, covered by runtime tests instead."""

from typing import Any

from annotated_upstream import f
from spindle.patching import patch

old: Any = f


@patch(old)
def f_patched(wrong_name, extra_required) -> str:
    return "drifted"
