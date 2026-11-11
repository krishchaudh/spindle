"""The replacements Spindle installs over pinned Miles functions must accept every
call shape the pinned upstream makes."""

import importlib.util
import os
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).parents[2]
_spec = importlib.util.spec_from_file_location(
    "upstream_compat", REPO / "scripts/upstream_compat.py"
)
upstream_compat = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = upstream_compat
_spec.loader.exec_module(upstream_compat)
CallSite = upstream_compat.CallSite
Signature = upstream_compat.Signature

MILES_PIN = "fdd808c008175d4895fd8c57c95ef4fa5bb6a2a0"
MATH_UTILS = "miles/backends/training_utils/loss_hub/math_utils.py"
CP_UTILS = "miles/backends/training_utils/cp_utils.py"

# Known defect: the tinker_cp.py replacement's signature differs from pinned Miles,
# so the five marked call sites below would raise TypeError if reached. The runtime
# handling is an open question on the linked spindle issue; the strict xfails keep
# the defect visible (they fail as XPASS the moment a fix lands, forcing removal of
# the markers) without landing a red suite.
KNOWN_SLICE_DEFECT = pytest.mark.xfail(
    reason=(
        "known defect: tinker_cp.slice_log_prob_with_cp signature differs from "
        "pinned Miles fdd808c (tracked on the linked spindle issue); remove this "
        "marker together with the runtime fix"
    ),
    strict=True,
)

# Recorded from modal-projects/miles at MILES_PIN. test_recorded_call_sites_match_checkout
# re-derives them when a checkout is available.
MILES_SLICE_LOG_PROB_WITH_CP = Signature(
    file=CP_UTILS,
    line=405,
    positional=(
        "log_prob",
        "total_length",
        "response_length",
        "qkv_format",
        "max_token_len",
    ),
    defaults=("qkv_format", "max_token_len"),
    keyword_only=(),
    var_positional=False,
    var_keyword=False,
)
MILES_SLICE_LOG_PROB_WITH_CP_CALLS = (
    CallSite(CP_UTILS, 399, "allgather_cp_redistribute", (), 5, (), False),
    CallSite(
        "miles/backends/training_utils/data.py",
        98,
        "get_rollout_data",
        ("key in rollout_data",),
        5,
        (),
        False,
    ),
    CallSite(
        "miles/backends/training_utils/mm_data.py",
        272,
        "expand_multimodal_rollout_data_in_place",
        ("metadata_changed", "cp_size > 1 and qkv_format == 'thd'"),
        4,
        (),
        False,
    ),
    CallSite(
        MATH_UTILS,
        518,
        "get_reinforce_plus_plus_returns",
        ("cp_size > 1",),
        3,
        (),
        False,
    ),
    CallSite(
        MATH_UTILS, 606, "get_advantages_and_returns", ("cp_size > 1",), 3, (), False
    ),
    CallSite(
        MATH_UTILS, 607, "get_advantages_and_returns", ("cp_size > 1",), 3, (), False
    ),
    CallSite(
        MATH_UTILS,
        764,
        "get_advantages_and_returns_batch",
        ("cp_size > 1",),
        3,
        ("qkv_format", "max_token_len"),
        False,
    ),
    CallSite(
        MATH_UTILS,
        771,
        "get_advantages_and_returns_batch",
        ("cp_size > 1",),
        3,
        ("qkv_format", "max_token_len"),
        False,
    ),
)

# The pinned-Miles call sites the current replacement rejects (see KNOWN_SLICE_DEFECT).
KNOWN_BROKEN_SLICE_CALLS = frozenset(
    (MATH_UTILS, line) for line in (518, 606, 607, 764, 771)
)


def _slice_call_params():
    for site in MILES_SLICE_LOG_PROB_WITH_CP_CALLS:
        if (site.file, site.line) in KNOWN_BROKEN_SLICE_CALLS:
            yield pytest.param(
                site, marks=KNOWN_SLICE_DEFECT, id=f"{site.file}:{site.line}"
            )
        else:
            yield site


def _patch(name):
    return next(p for p in upstream_compat.PATCHES if p.name == name)


@pytest.mark.parametrize(
    "site", list(_slice_call_params()), ids=lambda s: f"{s.file}:{s.line}"
)
def test_slice_log_prob_with_cp_accepts_pinned_miles_calls(site) -> None:
    patch = _patch("slice_log_prob_with_cp")
    replacement = upstream_compat.signature(REPO, patch.replacement, patch.name)

    assert MILES_SLICE_LOG_PROB_WITH_CP.bind(site) is None, "recorded call is stale"
    assert replacement.bind(site) is None, (
        f"{patch.replacement}:{replacement.line} rejects {site.render()} "
        f"(miles@{MILES_PIN[:7]})"
    )


@KNOWN_SLICE_DEFECT
def test_slice_log_prob_with_cp_keeps_pinned_miles_signature() -> None:
    patch = _patch("slice_log_prob_with_cp")
    replacement = upstream_compat.signature(REPO, patch.replacement, patch.name)

    assert (
        upstream_compat.signature_diffs(MILES_SLICE_LOG_PROB_WITH_CP, replacement) == []
    )


UPSTREAM = """
def f(a, b, fmt="thd", max_len=None):
    return a
"""
CALLERS = """
def guarded(x, cp_size):
    if cp_size > 1:
        return f(x, 1)
    return x

def keyword(x):
    return f(x, 1, fmt="bshd", max_len=4)

def dotted(x):
    return mod.f(x, 1, 2)

def splat(x, extra):
    return f(x, *extra)

def wrong_upstream(x):
    return f(x, 1, 2, 3, 4, 5)
"""


@pytest.mark.parametrize(
    ("replacement", "broken", "diffs"),
    [
        (
            'def f(a, b, fmt="thd", max_len=None): ...',
            [],
            [],
        ),
        (
            "def f(a, b, fmt, max_seq_len=None): ...",
            [
                (4, "missing a required argument: 'fmt'"),
                (8, "got an unexpected keyword argument 'max_len'"),
            ],
            [
                "parameter 4 is max_len upstream, max_seq_len in the replacement",
                "max_len cannot be passed by keyword to the replacement",
                "fmt has a default upstream but not in the replacement",
            ],
        ),
        (
            "def f(a, b, *args, **kwargs): ...",
            [],
            [],
        ),
    ],
    ids=["same", "renamed_and_required", "varargs"],
)
def test_check_reports_breaking_call_sites(tmp_path, replacement, broken, diffs):
    miles = tmp_path / "miles"
    spindle = tmp_path / "spindle"
    for root, name, text in [
        (miles, "up.py", UPSTREAM),
        (miles, "callers.py", CALLERS),
        (spindle, "patch.py", replacement),
    ]:
        root.mkdir(exist_ok=True)
        (root / name).write_text(text)
    patch = upstream_compat.Patch("f", "patch.py", "up.py", ("up.py", "callers.py"))

    report = upstream_compat.check(patch, spindle, miles, "0" * 40)

    assert [(s.line, error) for s, error in report.broken] == broken
    assert report.signature_diffs == diffs
    sites = [s for s, _ in report.broken] + report.compatible
    assert [s.line for s in report.unknown] == [14]
    assert 17 not in [s.line for s in sites]
    assert {s.guards for s in sites} == {(), ("cp_size > 1",)}


@pytest.mark.skipif(
    "SPINDLE_MILES_SRC" not in os.environ,
    reason="set SPINDLE_MILES_SRC to a Miles checkout at MILES_COMMIT",
)
def test_recorded_call_sites_match_checkout() -> None:
    miles = Path(os.environ["SPINDLE_MILES_SRC"])
    assert upstream_compat.checkout_head(miles) == upstream_compat.pinned_commit()
    assert upstream_compat.pinned_commit() == MILES_PIN
    patch = _patch("slice_log_prob_with_cp")

    report = upstream_compat.check(patch, REPO, miles, MILES_PIN)

    assert report.upstream == MILES_SLICE_LOG_PROB_WITH_CP
    sites = [s for s, _ in report.broken] + report.compatible + report.unknown
    assert sorted(sites, key=lambda s: (s.file, s.line)) == sorted(
        MILES_SLICE_LOG_PROB_WITH_CP_CALLS, key=lambda s: (s.file, s.line)
    )
