"""A replacement that drifts from the upstream signature: ty rejects it."""

from spindle.patching import patch
from upstream_mod import slice_two


@patch(slice_two)
def slice_two_patched(
    value, total_length, response_length, qkv_format, max_seq_len=None
):
    return value
