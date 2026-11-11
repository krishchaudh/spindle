"""patch() installs replacements under a shared signature type and re-points
dependent modules that bound the original at import time."""

import shutil
import subprocess
import sys
from pathlib import Path
from types import ModuleType

import pytest

from spindle.patching import patch

REPO = Path(__file__).parents[1]
FIXTURES = Path(__file__).parent / "fixtures" / "patching"


def _module(name: str, source: str) -> ModuleType:
    module = ModuleType(name)
    exec(source, module.__dict__)
    sys.modules[name] = module
    return module


def test_install_replaces_module_function() -> None:
    upstream = _module("fake_upstream_install", "def f(a, b=1):\n    return a\n")
    original = upstream.f

    def new(a, b=1):
        return b

    assert patch(upstream.f)(new) is new
    assert upstream.f is new
    assert original(1) == 1


def test_dependents_repointed_only_while_bound_to_original() -> None:
    upstream = _module("fake_upstream_deps", "def f():\n    return 1\n")
    stale = _module("fake_dep_stale", "")
    stale.f = upstream.f
    drifted = _module("fake_dep_drifted", "")
    drifted.f = lambda: "local"
    empty = _module("fake_dep_empty", "")

    def new():
        return 2

    patch(upstream.f, dependents=(stale, drifted, empty))(new)  # type: ignore[attr-defined]

    assert stale.f is new
    assert drifted.f() == "local"
    assert not hasattr(empty, "f")


def test_owner_resolved_through_qualname_for_class_attributes() -> None:
    upstream = _module(
        "fake_upstream_method",
        "class Model:\n    def forward(self, x):\n        return x\n",
    )
    original = upstream.Model.forward

    def new(self, x):
        return original(self, x) + 1

    patch(upstream.Model.forward)(new)
    assert upstream.Model.forward is new


def test_owner_installs_onto_an_explicit_holder() -> None:
    # Stubbed tests attach test-defined functions to a fake module; the
    # function's own module would resolve to the test file, so the owner
    # is passed explicitly.
    holder = _module("fake_holder", "")

    def elsewhere():
        return 1

    holder.f = elsewhere

    def new():
        return 2

    patch(holder.f, owner=holder)(new)
    assert holder.f is new


def test_ty_rejects_a_signature_mismatch() -> None:
    result = _ty(FIXTURES / "mismatch.py")
    assert result.returncode != 0
    assert "invalid-argument-type" in result.stdout + result.stderr


def test_ty_accepts_a_matching_signature() -> None:
    result = _ty(FIXTURES / "match.py")
    assert result.returncode == 0, result.stdout + result.stderr


def _ty(path: Path) -> subprocess.CompletedProcess[str]:
    binary = shutil.which("ty")
    if binary is None:
        pytest.skip("ty is not installed")
    return subprocess.run(
        [
            binary,
            "check",
            "--extra-search-path",
            str(REPO / "src"),
            "--extra-search-path",
            str(FIXTURES),
            str(path),
        ],
        cwd=REPO,
        capture_output=True,
        text=True,
    )


def test_layered_patches_wrap_the_current_value() -> None:
    upstream = _module("fake_upstream_layered", "def f():\n    return 1\n")

    def first():
        return 2

    patch(upstream.f, owner=upstream)(first)

    def second():
        return first() + 1

    patch(upstream.f, owner=upstream)(second)

    assert upstream.f() == 3


def test_install_allows_a_manual_swap_back() -> None:
    upstream = _module("fake_upstream_swap", "def f():\n    return 1\n")
    original = upstream.f

    patch(upstream.f, owner=upstream)(lambda: 2)
    assert upstream.f() == 2

    upstream.f = original
    assert upstream.f() == 1


def test_replacing_a_non_callable_fails() -> None:
    holder = _module("fake_holder_value", "")
    holder.CONST = 3

    with pytest.raises(TypeError, match="callable"):
        patch(holder.CONST, owner=holder)(lambda: 3)


def test_ty_rejects_a_return_type_mismatch() -> None:
    result = _ty(FIXTURES / "return_mismatch.py")
    assert result.returncode != 0
    assert "invalid-argument-type" in result.stdout + result.stderr


def test_ty_accepts_a_keyword_only_superset() -> None:
    result = _ty(FIXTURES / "superset.py")
    assert result.returncode == 0, result.stdout + result.stderr


def test_ty_cannot_see_through_an_any_typed_original() -> None:
    # Documented boundary: modules handed to installers as untyped parameters
    # give patch() an Any-typed original, so drift there is caught by the
    # runtime tests, not by ty.
    result = _ty(FIXTURES / "any_old.py")
    assert result.returncode == 0, result.stdout + result.stderr


def test_ty_accepts_a_conforming_fake_at_an_installer_seam() -> None:
    result = _ty(FIXTURES / "seam_match.py")
    assert result.returncode == 0, result.stdout + result.stderr


def test_ty_rejects_a_drifted_fake_at_an_installer_seam() -> None:
    # Regression evidence for the annotated installer parameters: a module
    # whose patched member no longer matches the pinned signature is rejected
    # at the seam, where the old was previously Any-typed.
    result = _ty(FIXTURES / "seam_drift.py")
    assert result.returncode != 0
    assert "invalid-argument-type" in result.stdout + result.stderr
