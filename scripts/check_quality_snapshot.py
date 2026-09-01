#!/usr/bin/env python3
"""Fail-closed freshness gate for the current quality snapshot.

A quality snapshot is only accepted as evidence for the current commit when it
is clean (dirty=false), its recorded source commit matches the exact HEAD (or
the only descendant delta is the snapshot's own two output files), every
declared test run returned 0 with an intact disposable copy, and the branch
coverage is present and meets the project minimum.  This checker never silently
accepts an old, dirty, or incomplete snapshot.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Sequence

SCRIPT_ROOT = Path(__file__).resolve().parent
if str(SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_ROOT))

from _0042_r2_tooling import resolve_source_root

SCHEMA_VERSION = 1
MIN_BRANCH_COVERAGE = 60.0
BRANCH_COVERAGE_LABEL = "branch-coverage"
SOURCE_ROOT = resolve_source_root(SCRIPT_ROOT.parent)
SNAPSHOT_OUTPUT_PATHS = frozenset(
    {
        "docs/quality/current-baseline.json",
        "docs/quality/current-baseline.md",
    }
)


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


def _snapshot_output_only_descendant(
    snapshot: dict[str, Any],
    snapshot_path: str | Path,
    source_commit: str,
    expected_commit: str,
) -> bool:
    """Return whether ``expected_commit`` differs only by snapshot outputs.

    A collector necessarily records the source commit before it writes the
    checked-in JSON and Markdown evidence.  The commit that adds those two
    outputs cannot have the same SHA as the source by construction.  This
    narrow exception keeps release tags verifiable without treating any later
    code, test, or documentation change as fresh evidence.
    """
    try:
        relative_snapshot = Path(snapshot_path).resolve().relative_to(SOURCE_ROOT).as_posix()
    except ValueError:
        return False
    if relative_snapshot != "docs/quality/current-baseline.json":
        return False

    declared_paths = snapshot.get("copy_manifest_excluded_paths")
    if not isinstance(declared_paths, list) or set(declared_paths) != SNAPSHOT_OUTPUT_PATHS:
        return False

    try:
        ancestor = subprocess.run(
            ["git", "merge-base", "--is-ancestor", source_commit, expected_commit],
            cwd=SOURCE_ROOT,
            capture_output=True,
            text=True,
            check=False,
            timeout=30,
        )
        if ancestor.returncode != 0:
            return False
        changed = subprocess.run(
            ["git", "diff", "--name-only", "--no-renames", f"{source_commit}..{expected_commit}"],
            cwd=SOURCE_ROOT,
            capture_output=True,
            text=True,
            check=False,
            timeout=30,
        )
    except (OSError, subprocess.SubprocessError):
        return False
    if changed.returncode != 0:
        return False
    changed_paths = {line for line in changed.stdout.splitlines() if line}
    return changed_paths <= SNAPSHOT_OUTPUT_PATHS


def check_snapshot(
    path: str | Path,
    expected_commit: str | None = None,
    *,
    skip_commit_check: bool = False,
    allow_snapshot_output_commit: bool = False,
    record_baseline: bool = False,
) -> list[str]:
    """Return the list of freshness violations for ``path`` (empty means pass).

    The check is fail-closed: missing provenance, dirty state, a mismatched
    commit, a non-zero run, an impure disposable copy, or absent/low branch
    coverage are each reported as a violation. ``record_baseline`` keeps all
    provenance, test-exit, integrity, and coverage-presence checks, but records
    a pre-refactor D0 measurement without pretending that it already meets the
    final 60% quality threshold. The final candidate gate must leave this flag
    unset. ``expected_commit`` defaults to the repository HEAD; pass an
    explicit value in tests.

    ``skip_commit_check`` is retained for historical baseline inspection only.
    Release CI instead uses ``allow_snapshot_output_commit``: it accepts a
    source commit only when it is an ancestor of ``expected_commit`` and every
    intervening file is one of the two declared snapshot outputs.  CI still
    enforces cleanliness, coverage, run exit status, and copy integrity.
    """
    snapshot = load_snapshot(path)
    if expected_commit is None:
        expected_commit = _head_commit(SOURCE_ROOT)
    violations: list[str] = []
    if snapshot.get("schema_version") != SCHEMA_VERSION:
        violations.append(f"schema_version must be {SCHEMA_VERSION}")
    if snapshot.get("outcome") != "pass":
        violations.append("outcome must be pass")
    source = snapshot.get("source", {})
    source_commit = source.get("commit")
    if (
        not skip_commit_check
        and source_commit != expected_commit
        and not (
            allow_snapshot_output_commit
            and isinstance(source_commit, str)
            and _snapshot_output_only_descendant(snapshot, path, source_commit, expected_commit)
        )
    ):
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
    elif not record_baseline and coverage < MIN_BRANCH_COVERAGE:
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
        help="Do not require source.commit to equal HEAD (historical-inspection mode).",
    )
    parser.add_argument(
        "--allow-snapshot-output-commit",
        action="store_true",
        help=(
            "Accept a snapshot-output-only descendant of source.commit: the source must be an ancestor of HEAD "
            "and the intervening paths must be exactly the declared baseline JSON/Markdown outputs."
        ),
    )
    parser.add_argument(
        "--record-baseline",
        action="store_true",
        help=(
            "Record a clean pre-refactor D0 coverage measurement without applying the final 60% threshold; "
            "coverage must still be present and all test/integrity checks must pass."
        ),
    )
    args = parser.parse_args(argv)
    violations = check_snapshot(
        args.snapshot,
        skip_commit_check=args.skip_commit_check,
        allow_snapshot_output_commit=args.allow_snapshot_output_commit,
        record_baseline=args.record_baseline,
    )
    if violations:
        for violation in violations:
            print(f"FAIL: {violation}")
        return 1
    if args.record_baseline:
        print("OK: quality baseline is fresh, clean, complete, and records its actual branch coverage.")
    else:
        print("OK: quality snapshot is fresh, clean, and complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
