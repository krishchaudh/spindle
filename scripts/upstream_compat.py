"""Check that Spindle's replacements for pinned Miles functions stay call-compatible
with the upstream call sites they are installed into.

Usage: uv run --no-sync python scripts/upstream_compat.py --miles /path/to/miles [--json]

The Miles checkout must be at MILES_COMMIT from miles_image.py; otherwise the
checker refuses to run, because every finding cites that commit.
"""

from __future__ import annotations

import argparse
import ast
import inspect
import json
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
MILES_IMAGE = REPO / "src/spindle/providers/modal/miles_image.py"
RUNTIME = "src/spindle/backends/miles_runtime"
TRAINING_UTILS = "miles/backends/training_utils"


@dataclass(frozen=True)
class Patch:
    """A Spindle function installed over an upstream one in the listed modules."""

    name: str
    replacement: str
    upstream: str
    installed_in: tuple[str, ...]


PATCHES = (
    Patch(
        "slice_log_prob_with_cp",
        f"{RUNTIME}/tinker_cp.py",
        f"{TRAINING_UTILS}/cp_utils.py",
        (
            f"{TRAINING_UTILS}/cp_utils.py",
            f"{TRAINING_UTILS}/data.py",
            f"{TRAINING_UTILS}/mm_data.py",
            f"{TRAINING_UTILS}/loss_hub/math_utils.py",
        ),
    ),
    Patch(
        "get_log_probs_and_entropy",
        f"{RUNTIME}/tinker_cp.py",
        f"{TRAINING_UTILS}/loss_hub/logit_processors.py",
        (
            f"{TRAINING_UTILS}/loss_hub/logit_processors.py",
            f"{TRAINING_UTILS}/loss.py",
            f"{TRAINING_UTILS}/loss_hub/tinker_losses.py",
            "miles/backends/megatron_utils/actor.py",
            "miles/backends/fsdp_utils/actor.py",
        ),
    ),
)


@dataclass(frozen=True)
class Signature:
    file: str
    line: int
    positional: tuple[str, ...]
    defaults: tuple[str, ...]
    keyword_only: tuple[str, ...]
    var_positional: bool
    var_keyword: bool

    def render(self) -> str:
        parts = [p + ("=…" if p in self.defaults else "") for p in self.positional]
        if self.var_positional:
            parts.append("*args")
        elif self.keyword_only:
            parts.append("*")
        parts += [k + ("=…" if k in self.defaults else "") for k in self.keyword_only]
        if self.var_keyword:
            parts.append("**kwargs")
        return "(" + ", ".join(parts) + ")"

    def bind(self, site: CallSite) -> str | None:
        """Return the TypeError message for calling this signature with `site`, or None."""
        default = inspect.Parameter.empty
        params = [
            inspect.Parameter(
                p,
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
                default=None if p in self.defaults else default,
            )
            for p in self.positional
        ]
        if self.var_positional:
            params.append(inspect.Parameter("args", inspect.Parameter.VAR_POSITIONAL))
        params += [
            inspect.Parameter(
                k,
                inspect.Parameter.KEYWORD_ONLY,
                default=None if k in self.defaults else default,
            )
            for k in self.keyword_only
        ]
        if self.var_keyword:
            params.append(inspect.Parameter("kwargs", inspect.Parameter.VAR_KEYWORD))
        try:
            inspect.Signature(params).bind(
                *[None] * site.positional, **dict.fromkeys(site.keywords)
            )
        except TypeError as error:
            return str(error)
        return None


@dataclass(frozen=True)
class CallSite:
    file: str
    line: int
    function: str
    guards: tuple[str, ...]
    positional: int
    keywords: tuple[str, ...]
    starred: bool

    def render(self) -> str:
        guard = f" [{' and '.join(self.guards)}]" if self.guards else ""
        return f"{self.file}:{self.line} in {self.function}{guard}"


@dataclass
class Report:
    patch: Patch
    pin: str
    replacement: Signature
    upstream: Signature
    signature_diffs: list[str]
    broken: list[tuple[CallSite, str]]
    compatible: list[CallSite]
    unknown: list[CallSite]

    def render(self) -> str:
        lines = [
            (
                f"{self.patch.name}: {self.replacement.file}:{self.replacement.line} "
                f"replaces {self.upstream.file}:{self.upstream.line} (miles@{self.pin[:7]})"
            ),
            f"  upstream    {self.upstream.render()}",
            f"  replacement {self.replacement.render()}",
        ]
        lines += [f"    - {diff}" for diff in self.signature_diffs]
        lines.append(f"  broken call sites: {len(self.broken)}")
        lines += [f"    {site.render()}: {error}" for site, error in self.broken]
        lines.append(f"  compatible call sites: {len(self.compatible)}")
        if self.unknown:
            lines.append(f"  call shapes not decidable statically: {len(self.unknown)}")
            lines += [f"    {site.render()}" for site in self.unknown]
        return "\n".join(lines)


def find_function(path: Path, name: str) -> ast.FunctionDef:
    """Return the single def of `name` in `path`, nested defs included."""
    found = [
        node
        for node in ast.walk(ast.parse(path.read_text()))
        if isinstance(node, ast.FunctionDef) and node.name == name
    ]
    if len(found) != 1:
        raise LookupError(f"{path}: {len(found)} definitions of {name}, need exactly 1")
    return found[0]


def signature(root: Path, file: str, name: str) -> Signature:
    node = find_function(root / file, name)
    args = node.args
    positional = args.posonlyargs + args.args
    with_default = positional[len(positional) - len(args.defaults) :]
    keyword_defaults = [
        k for k, d in zip(args.kwonlyargs, args.kw_defaults, strict=True) if d
    ]
    return Signature(
        file=file,
        line=node.lineno,
        positional=tuple(a.arg for a in positional),
        defaults=tuple(a.arg for a in with_default + keyword_defaults),
        keyword_only=tuple(a.arg for a in args.kwonlyargs),
        var_positional=args.vararg is not None,
        var_keyword=args.kwarg is not None,
    )


def call_sites(root: Path, file: str, name: str) -> list[CallSite]:
    """Every call of `name` in `file`, by bare name or attribute, with its enclosing
    function and the `if` tests that guard it."""
    sites: list[CallSite] = []

    def visit(node: ast.AST, function: str, guards: tuple[str, ...]) -> None:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            function = node.name
        if isinstance(node, ast.If):
            visit(node.test, function, guards)
            for child in node.body:
                visit(child, function, guards + (ast.unparse(node.test),))
            for child in node.orelse:
                visit(child, function, guards + (f"not ({ast.unparse(node.test)})",))
            return
        if isinstance(node, ast.Call):
            callee = node.func
            target = (
                callee.id
                if isinstance(callee, ast.Name)
                else callee.attr
                if isinstance(callee, ast.Attribute)
                else None
            )
            if target == name:
                sites.append(
                    CallSite(
                        file=file,
                        line=node.lineno,
                        function=function,
                        guards=guards,
                        positional=sum(
                            not isinstance(a, ast.Starred) for a in node.args
                        ),
                        keywords=tuple(k.arg for k in node.keywords if k.arg),
                        starred=any(isinstance(a, ast.Starred) for a in node.args)
                        or any(k.arg is None for k in node.keywords),
                    )
                )
        for child in ast.iter_child_nodes(node):
            visit(child, function, guards)

    visit(ast.parse((root / file).read_text()), "<module>", ())
    return sites


def signature_diffs(upstream: Signature, replacement: Signature) -> list[str]:
    """Differences that can break a call that was valid upstream."""
    diffs = []
    for index, (up, new) in enumerate(
        zip(upstream.positional, replacement.positional, strict=False), start=1
    ):
        if up != new:
            diffs.append(
                f"parameter {index} is {up} upstream, {new} in the replacement"
            )
    if len(replacement.positional) > len(upstream.positional):
        for extra in replacement.positional[len(upstream.positional) :]:
            if extra not in replacement.defaults:
                diffs.append(f"{extra} is required in the replacement, absent upstream")
    for lost in set(upstream.positional) - set(replacement.positional):
        if not replacement.var_keyword:
            diffs.append(f"{lost} cannot be passed by keyword to the replacement")
    for name in upstream.defaults:
        if name in replacement.positional + replacement.keyword_only:
            if name not in replacement.defaults:
                diffs.append(
                    f"{name} has a default upstream but not in the replacement"
                )
    for name in upstream.keyword_only:
        if name not in replacement.keyword_only + replacement.positional:
            if not replacement.var_keyword:
                diffs.append(f"keyword-only {name} is missing from the replacement")
    return diffs


def check(patch: Patch, spindle: Path, miles: Path, pin: str) -> Report:
    replacement = signature(spindle, patch.replacement, patch.name)
    upstream = signature(miles, patch.upstream, patch.name)
    report = Report(patch, pin, replacement, upstream, [], [], [], [])
    report.signature_diffs = signature_diffs(upstream, replacement)
    for file in patch.installed_in:
        for site in call_sites(miles, file, patch.name):
            if site.line == upstream.line and file == upstream.file:
                continue
            if site.starred:
                report.unknown.append(site)
            elif upstream.bind(site) is not None:
                continue
            elif (error := replacement.bind(site)) is not None:
                report.broken.append((site, error))
            else:
                report.compatible.append(site)
    return report


def pinned_commit(miles_image: Path = MILES_IMAGE) -> str:
    for node in ast.walk(ast.parse(miles_image.read_text())):
        if isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == "MILES_COMMIT" for t in node.targets
        ):
            return node.value.value
    raise LookupError(f"MILES_COMMIT not found in {miles_image}")


def checkout_head(path: Path) -> str:
    return subprocess.check_output(
        ["git", "-C", str(path), "rev-parse", "HEAD"], text=True
    ).strip()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--miles", type=Path, required=True)
    parser.add_argument("--spindle", type=Path, default=REPO)
    parser.add_argument("--json", action="store_true")
    options = parser.parse_args(argv)

    pin = pinned_commit(options.spindle / MILES_IMAGE.relative_to(REPO))
    head = checkout_head(options.miles)
    if head != pin:
        print(f"{options.miles} is at {head}, MILES_COMMIT is {pin}", file=sys.stderr)
        return 2
    try:
        reports = [check(p, options.spindle, options.miles, pin) for p in PATCHES]
    except (LookupError, FileNotFoundError) as error:
        print(f"cannot check: {error}", file=sys.stderr)
        return 2
    if options.json:
        print(json.dumps([asdict(r) for r in reports], indent=2))
    else:
        print("\n\n".join(r.render() for r in reports))
    return 1 if any(r.broken for r in reports) else 0


if __name__ == "__main__":
    sys.exit(main())
