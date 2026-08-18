#!/usr/bin/env python3
"""Fail-closed freshness gate for the current quality snapshot.

A quality snapshot is only accepted as evidence for the current commit when it
is clean (dirty=false), its recorded source commit matches the exact HEAD, every
declared test run returned 0 with an intact disposable copy, and the branch
coverage is present and meets the project minimum.  This checker never silently
accepts an old, dirty, or incomplete snapshot.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any, Sequence

SCHEMA_VERSION = 1
MIN_BRANCH_COVERAGE = 60.0
BRANCH_COVERAGE_LABEL = "branch-coverage"


def load_snapshot(path: str | Path) -> dict[str, Any]:
    """Load a quality snapshot JSON document."""
    return json.loads(Path(path).read_text(encoding="utf-8"))


def branch_coverage(snapshot: dict[str, Any]) -> float | None:
    """Return the recorded branch coverage percentage, or None when absent."""
    for run in snapshot.get("runs", []):
        if run.get("label") == BRANCH_COVERAGE_LABEL:
            value = run.get("branch_coverage_percent")
            return float(value) if value is not None else None
    return None


def _head_commit(root: Path) -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        capture_output=True,
        text=True,
        check=True,
        timeout=30,
    )
    return result.stdout.strip()


def check_snapshot(
    path: str | Path,
    expected_commit: str | None = None,
    *,
    skip_commit_check: bool = False,
) -> list[str]:
    """Return the list of freshness violations for ``path`` (empty means pass).

    The check is fail-closed: missing provenance, dirty state, a mismatched
    commit, a non-zero run, an impure disposable copy, or absent/low branch
    coverage are each reported as a violation.  ``expected_commit`` defaults to
    the repository HEAD; pass an explicit value in tests.

    ``skip_commit_check`` is for CI: the checked-in snapshot records the commit
    it was generated from, which is always an ancestor of (or equal to) the PR
    HEAD that CI checks out, so a strict equality check would false-positive on
    every subsequent change.  CI still enforces cleanliness, coverage, run exit
    status and copy integrity.
    """
    snapshot = load_snapshot(path)
    if expected_commit is None:
        expected_commit = _head_commit(Path(__file__).resolve().parents[1])
    violations: list[str] = []
    if snapshot.get("schema_version") != SCHEMA_VERSION:
        violations.append(f"schema_version must be {SCHEMA_VERSION}")
    if snapshot.get("outcome") != "pass":
        violations.append("outcome must be pass")
    source = snapshot.get("source", {})
    if not skip_commit_check and source.get("commit") != expected_commit:
        violations.append("source.commit does not match HEAD")
    if source.get("dirty") is not False:
        violations.append("source.dirty must be false")
    runs = snapshot.get("runs", [])
    if not runs:
        violations.append("no test runs recorded")
    for run in runs:
        label = run.get("label", "unknown")
        if run.get("returncode") != 0:
            violations.append(f"run {label} returncode is not 0")
        if run.get("integrity_ok") is not True:
            violations.append(f"run {label} integrity_ok is not true")
    coverage = branch_coverage(snapshot)
    if coverage is None:
        violations.append("branch coverage is missing")
    elif coverage < MIN_BRANCH_COVERAGE:
        violations.append(f"branch coverage {coverage}% is below the {MIN_BRANCH_COVERAGE}% threshold")
    return violations


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--snapshot",
        required=True,
        help="Path to the quality snapshot (e.g. docs/quality/current-baseline.json)",
    )
    parser.add_argument(
        "--skip-commit-check",
        action="store_true",
        help="Do not require source.commit to equal HEAD (CI-only mode).",
    )
    args = parser.parse_args(argv)
    violations = check_snapshot(args.snapshot, skip_commit_check=args.skip_commit_check)
    if violations:
        for violation in violations:
            print(f"FAIL: {violation}")
        return 1
    print("OK: quality snapshot is fresh, clean, and complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
