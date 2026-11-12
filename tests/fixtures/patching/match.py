"""A replacement with the upstream signature: ty accepts it."""

from spindle.patching import patch
from upstream_mod import slice_two


@patch(slice_two)
def slice_two_patched(
    value, total_length, response_length, qkv_format="thd", max_token_len=None
):
    return value
