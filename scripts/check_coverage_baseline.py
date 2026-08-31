#!/usr/bin/env python3
"""Enforce the release coverage gate against the trusted Task 1 baseline.

Reads a pytest-cov JSON report (``--cov-branch --cov-report=json:<path>``)
and the quality baseline produced by ``scripts/collect_quality_baseline.py``
and fails when:

* the overall branch coverage falls below the baseline's ``branch-coverage``
  run percentage, or
* fewer than ``--changed-lines-min`` percent (default 95) of the changed
  measurable lines since the baseline commit are covered.

Changed lines are the lines added to ``fincore/**`` between the baseline's
``source.commit`` and the current worktree (``git diff --unified=0``).  A
changed line is classified against the coverage JSON data model:

* covered    — present in the file's ``executed_lines`` or ``excluded_lines``;
* uncovered  — present in the file's ``missing_lines``;
* unmeasured — absent from all three sets: blank/comment lines and any file
  the coverage run omits (``[tool.coverage.run] omit`` excludes
  ``*/__init__.py`` and ``*/deprecate.py``).  Unmeasured lines are reported
  but never enter the ratio.

Uncovered changed lines are listed in the failure output so the diff is
directly actionable.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

SCRIPT_ROOT = Path(__file__).resolve().parent
if str(SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_ROOT))

from _0042_r2_tooling import resolve_source_root

REPO_ROOT = resolve_source_root(SCRIPT_ROOT.parent)
DEFAULT_BASELINE = REPO_ROOT / "docs" / "quality" / "current-baseline.json"
GIT_TIMEOUT_SECONDS = 60
MAX_REPORTED_LINES = 100

_HUNK_RE = re.compile(r"^@@ -\d+(?:,\d+)? \+(\d+)(?:,(\d+))? @@")


class CoverageGateError(RuntimeError):
    """Raised when the coverage gate fails."""


def _load_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise CoverageGateError(f"cannot read JSON report {path}: {exc}") from exc


def _overall_percent(coverage: dict[str, Any]) -> float:
    totals = coverage.get("totals")
    if not isinstance(totals, dict):
        raise CoverageGateError("coverage JSON has no 'totals' section; run pytest-cov with --cov-report=json:<path>")
    if "percent_covered" in totals:
        return float(totals["percent_covered"])
    # Defensive fallback for coverage.py versions without percent_covered.
    covered = int(totals.get("covered_lines", 0)) + int(totals.get("covered_branches", 0))
    total = int(totals.get("num_statements", 0)) + int(totals.get("num_branches", 0))
    return 100.0 * covered / total if total else 100.0


def _baseline_percent(baseline: dict[str, Any]) -> float:
    runs = baseline.get("runs")
    if not isinstance(runs, list):
        raise CoverageGateError("baseline JSON has no 'runs' list")
    for run in runs:
        if run.get("label") == "branch-coverage":
            value = run.get("branch_coverage_percent")
            if value is None:
                raise CoverageGateError("baseline 'branch-coverage' run has no branch_coverage_percent")
            return float(value)
    raise CoverageGateError(
        "baseline JSON has no 'branch-coverage' run; regenerate with scripts/collect_quality_baseline.py"
    )


def _changed_lines(base_commit: str) -> dict[str, list[int]]:
    """Map each changed fincore file to the line numbers added since base_commit."""
    command = ["git", "diff", "--unified=0", base_commit, "--", "fincore"]
    try:
        result = subprocess.run(
            command,
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            timeout=GIT_TIMEOUT_SECONDS,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        raise CoverageGateError(f"git diff timed out after {GIT_TIMEOUT_SECONDS}s") from exc
    if result.returncode != 0:
        raise CoverageGateError(
            f"'git diff --unified=0 {base_commit} -- fincore' failed: {result.stderr.strip()}\n"
            "pass --changed-base with a resolvable commit or fetch full history"
        )
    lines: dict[str, list[int]] = defaultdict(list)
    current_file: str | None = None
    for raw in result.stdout.splitlines():
        if raw.startswith("+++ b/"):
            current_file = raw[len("+++ b/") :].strip()
        elif raw.startswith("+++ "):
            current_file = None
        elif raw.startswith("@@") and current_file is not None:
            match = _HUNK_RE.match(raw)
            if match is None:
                continue
            start = int(match.group(1))
            count = int(match.group(2)) if match.group(2) is not None else 1
            lines[current_file].extend(range(start, start + count))
    return {path: sorted(values) for path, values in lines.items()}


def _line_outcome(line: int, file_data: dict[str, Any]) -> str:
    executed = set(file_data.get("executed_lines", []))
    excluded = set(file_data.get("excluded_lines", []))
    missing = set(file_data.get("missing_lines", []))
    if line in executed or line in excluded:
        return "covered"
    if line in missing:
        return "uncovered"
    return "unmeasured"


def check_changed_lines(
    coverage: dict[str, Any],
    baseline: dict[str, Any],
    base_commit: str,
    minimum: float,
) -> None:
    files = coverage.get("files")
    if not isinstance(files, dict):
        raise CoverageGateError("coverage JSON has no 'files' section")
    covered = 0
    uncovered = 0
    unmeasured = 0
    uncovered_lines: list[str] = []
    omitted_files: list[str] = []
    for path, changed in _changed_lines(base_commit).items():
        file_data = files.get(path)
        if file_data is None:
            omitted_files.append(path)
            unmeasured += len(changed)
            continue
        for line in changed:
            outcome = _line_outcome(line, file_data)
            if outcome == "covered":
                covered += 1
            elif outcome == "uncovered":
                uncovered += 1
                uncovered_lines.append(f"{path}:{line}")
            else:
                unmeasured += 1
    measured = covered + uncovered
    percent = 100.0 * covered / measured if measured else 100.0
    details = [
        f"changed lines: {covered} covered, {uncovered} uncovered, {unmeasured} unmeasured "
        f"({percent:.2f}% of measured covered, minimum {minimum:.1f}%)",
    ]
    if omitted_files:
        details.append(f"unmeasured (omitted from coverage run): {', '.join(sorted(omitted_files))}")
    if percent + 1e-9 < minimum:
        shown = uncovered_lines[:MAX_REPORTED_LINES]
        more = len(uncovered_lines) - len(shown)
        details.append("uncovered changed lines:\n    " + "\n    ".join(shown))
        if more > 0:
            details.append(f"    ... and {more} more")
        raise CoverageGateError(
            f"changed-line coverage {percent:.2f}% is below the {minimum:.1f}% minimum "
            f"since baseline commit {base_commit}\n" + "\n".join(details)
        )
    print("\n".join(details))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--coverage-json", type=Path, required=True, help="pytest-cov JSON report")
    parser.add_argument("--baseline", type=Path, default=DEFAULT_BASELINE, help="quality baseline JSON")
    parser.add_argument("--changed-lines-min", type=float, default=95.0, help="minimum changed-line coverage percent")
    parser.add_argument("--changed-base", help="commit to diff changed lines against (default: baseline source.commit)")
    args = parser.parse_args()
    try:
        coverage = _load_json(args.coverage_json)
        baseline = _load_json(args.baseline)
        base_commit = args.changed_base
        if base_commit is None:
            source = baseline.get("source")
            base_commit = source.get("commit") if isinstance(source, dict) else None
            if not base_commit:
                raise CoverageGateError("baseline has no source.commit; pass --changed-base")
        baseline_percent = _baseline_percent(baseline)
        overall = _overall_percent(coverage)
        print(f"overall branch coverage: {overall:.2f}% (baseline: {baseline_percent:.2f}%)")
        if overall + 1e-9 < baseline_percent:
            raise CoverageGateError(
                f"overall branch coverage {overall:.2f}% is below the baseline {baseline_percent:.2f}%"
            )
        check_changed_lines(coverage, baseline, base_commit, args.changed_lines_min)
    except CoverageGateError as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return 1
    print("coverage gate OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
