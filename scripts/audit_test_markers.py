#!/usr/bin/env python3
"""Audit test markers against the release-required CI selectors.

Every release-required CI selector must collect at least one test, so a
selector typo or a drifted marker cannot silently empty a release gate and
ship untested code.  The selectors audited here mirror the exact pytest
invocations in ``.github/workflows/ci.yml``:

* ``fast-suite``         ``not slow and not integration and not serial``
* ``serial-suite``       ``serial and not slow and not integration``
* ``non-serial-suite``   ``not serial and not slow and not integration``
* ``integration-offline`` ``integration_offline``
* ``slow``               ``slow`` in ``tests/benchmarks/test_0042_r2_workloads.py``

The former ``tests/compat`` suite was retired with the 0.5 breaking API. Its
frozen inputs remain provenance fixtures, not executable compatibility tests.
The offline integration and slow selectors instead exercise canonical 0.5
workflows and registered workload profiling respectively.

Every test carrying ``integration`` must also carry exactly one subtype
(``integration_offline`` or ``integration_online``); a test with a subtype
but no parent ``integration`` marker fails as well.  This guarantees that
every ``not integration`` selector excludes both offline and online
integration tests and that network tests can never masquerade as the
offline release gate.

``--compare-junit A B`` verifies that two pytest JUnit XML reports agree on
collected, passed and skipped test counts (used to prove the single-process
and xdist non-serial runs are comparable).
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
COLLECT_TIMEOUT_SECONDS = 300

#: (label, marker expression or None for a pure path selector, pytest args).
#: Mirrors the release-gate invocations in .github/workflows/ci.yml exactly.
RELEASE_SELECTORS: tuple[tuple[str, str | None, list[str]], ...] = (
    ("fast-suite", "not slow and not integration and not serial", ["tests/", "--ignore=tests/benchmarks"]),
    ("serial-suite", "serial and not slow and not integration", ["tests/", "--ignore=tests/benchmarks"]),
    ("non-serial-suite", "not serial and not slow and not integration", ["tests/", "--ignore=tests/benchmarks"]),
    ("integration-offline", "integration_offline", ["tests/"]),
    ("slow", "slow", ["tests/benchmarks/test_0042_r2_workloads.py"]),
)


class MarkerAuditError(RuntimeError):
    """Raised when the marker audit finds a violation."""


def _collect(marker_expr: str | None, paths: list[str]) -> int:
    """Collect tests for one selector and return the number of collected tests."""
    command = [sys.executable, "-m", "pytest", "-o", "addopts=", "--collect-only", "-q", *paths]
    if marker_expr is not None:
        command.extend(["-m", marker_expr])
    try:
        result = subprocess.run(
            command,
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            timeout=COLLECT_TIMEOUT_SECONDS,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        raise MarkerAuditError(f"collection timed out after {COLLECT_TIMEOUT_SECONDS}s: {command}") from exc
    output = f"{result.stdout}\n{result.stderr}"
    # pytest exits with code 5 when a selector matches nothing; that is a
    # valid empty collection for the unowned-selector check.
    if "no tests collected" in output:
        return 0
    if result.returncode != 0:
        raise MarkerAuditError(f"collection failed (exit {result.returncode}): {command}\n{output[-2000:]}")
    match = re.search(r"(\d+)(?:/\d+)? tests? collected", output)
    if match is None:
        raise MarkerAuditError(f"could not parse collected count for selector {command}\n{output[-2000:]}")
    return int(match.group(1))


def _integration_violations() -> list[str]:
    """Return one message per test whose integration markers are inconsistent."""
    import pytest

    violations: list[str] = []

    class _AuditPlugin:
        @staticmethod
        def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
            for item in items:
                names = {marker.name for marker in item.iter_markers()}
                parent = "integration" in names
                subtypes = [name for name in ("integration_offline", "integration_online") if name in names]
                if not parent and not subtypes:
                    continue
                if not parent:
                    violations.append(f"{item.nodeid}: subtype {subtypes[0]!r} without parent 'integration' marker")
                elif not subtypes:
                    violations.append(
                        f"{item.nodeid}: missing integration subtype (add 'integration_offline' or 'integration_online')"
                    )
                elif len(subtypes) > 1:
                    violations.append(
                        f"{item.nodeid}: both 'integration_offline' and 'integration_online' markers present; "
                        "exactly one subtype is required"
                    )

    exit_code = pytest.main(
        ["-o", "addopts=", "--collect-only", "-q", "tests/"],
        plugins=[_AuditPlugin()],
    )
    if exit_code != 0:
        raise MarkerAuditError(f"integration marker collection failed (exit {exit_code})")
    return violations


def _junit_counts(path: Path) -> dict[str, int]:
    """Aggregate tests/errors/failures/skipped across every testsuite in a JUnit XML."""
    try:
        root = ET.parse(path).getroot()
    except (ET.ParseError, OSError) as exc:
        raise MarkerAuditError(f"cannot parse JUnit XML {path}: {exc}") from exc
    counts = {"tests": 0, "errors": 0, "failures": 0, "skipped": 0}
    for suite in root.iter("testsuite"):
        for key in counts:
            counts[key] += int(suite.get(key, "0"))
    counts["passed"] = counts["tests"] - counts["errors"] - counts["failures"] - counts["skipped"]
    return counts


def compare_junit(first: Path, second: Path) -> None:
    """Fail unless two JUnit XML reports agree on collected/passed/skipped counts."""
    left = _junit_counts(first)
    right = _junit_counts(second)
    for key in ("tests", "passed", "skipped"):
        if left[key] != right[key]:
            raise MarkerAuditError(
                f"JUnit mismatch on {key}: {first} has {left[key]}, {second} has {right[key]} "
                f"(full counts: {left} vs {right})"
            )
    print(
        f"JUnit comparison OK: collected={left['tests']} passed={left['passed']} "
        f"skipped={left['skipped']} ({first.name} == {second.name})"
    )


def audit_selectors() -> None:
    """Audit every release-required selector and the integration marker rules."""
    failures: list[str] = []
    for label, marker_expr, paths in RELEASE_SELECTORS:
        count = _collect(marker_expr, paths)
        print(f"{label}: {count} test(s) collected")
        if count < 1:
            failures.append(f"release-required selector {label!r} collected zero tests; fix the marker or the selector")
    violations = _integration_violations()
    failures.extend(violations)
    print(f"integration markers: {len(violations)} violation(s)")
    if failures:
        raise MarkerAuditError("marker audit failed:\n  " + "\n  ".join(failures))
    print("marker audit OK")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--compare-junit", nargs=2, type=Path, metavar=("A", "B"), help="compare two JUnit XML reports")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        if args.compare_junit is not None:
            compare_junit(args.compare_junit[0], args.compare_junit[1])
        else:
            audit_selectors()
    except MarkerAuditError as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
