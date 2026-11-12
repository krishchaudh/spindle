# Contributing to Spindle

These rules apply to everyone who changes this repo, people and coding agents alike. For how the code is laid out, see the `spindle-dev` skill (`src/spindle/skills/spindle-dev/SKILL.md`).

## Before you push

- Python 3.12 with uv: `uv sync --locked --python 3.12`.
- `uv run pre-commit run --all-files` (ruff format, ruff check and ty check).
- `uv run --no-sync pytest -q`.

## Code

- Put imports at the top of the file. Import inside a function only when the dependency is missing where the module gets imported (for example, a GPU-only package in a module that CPU tests import), and say why in one short comment.
- Make the code explain itself. Before you add a comment, try a clearer name, an extracted variable or a simpler structure. Keep a comment only when it says something the code can't, such as a non-obvious constraint or an upstream bug, and keep it to one line. Put the reasoning behind a change in the commit message or the PR description.
- Keep changes small. Reuse existing helpers instead of adding new ones. Remove dead branches, unused constants and flags left over from a refactor.
- Monkeypatch a pinned upstream function only through `spindle.patching.patch`:

  ```python
  patch(upstream_module.fn, owner=upstream_module, dependents=(...))(replacement)
  ```

  The helper binds the replacement to the upstream signature, so ty fails the build when a pinned upstream signature drifts. The check only works where ty sees the real package: install the patched package at its pinned commit in CI. When an installer receives the upstream module as a parameter (so tests can inject fakes), annotate it with the matching protocol from `spindle.backends.miles_runtime._upstream`; the trip-wire assignments there fail ty when a pin bump changes a member Spindle relies on.
- Don't delete existing comments or docstrings that a change doesn't touch.

## Tests

Add a test for each behavior that would hurt users if it broke: races, failure and restart paths, version compatibility, and messages users read. Skip tests that:

- only check that a mock was called;
- restate the implementation line for line;
- cover a trivial fallback.

Fold near-duplicate tests into one parametrized test.

## Compatibility

Spindle should work as a drop-in Tinker backend. Test and document changes with the stock Tinker SDK and tinker-cookbook pointed at a Spindle URL, not with Spindle-specific wrapper scripts.

## Public text

Spindle is open source, so docs, code comments, commit messages and PR descriptions are public. Write them for outside users:

- Use short, plain sentences. Say what the user sees and what to set, then why.
- Leave out marketing words, emoji and repeated sections.
- Don't mention internal environments, private links or people.
- Keep roadmap items and follow-up ideas in GitHub issues, not in docs or comments.

## Pull requests

- The description has a Summary, which says what changed and why for a reader new to the area, then how you tested it. Say what ran end to end, on which commit, and what didn't.
- Performance PRs report times, such as seconds per step, not throughput. They include before and after numbers in one small table, with links to the runs.
- Add a CHANGELOG.md entry for user-facing changes. Recipe tuning (replica counts, queue sizes, memory fractions) doesn't need one.

## Review

`.claude/agents/spindle-pr-reviewer.md` describes the review pass for every PR: race and lifecycle bugs first, then trimming to the code that matters. Claude Code users can run it as the `spindle-pr-reviewer` agent; other tools can follow it as a checklist.
