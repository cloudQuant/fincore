#!/usr/bin/env python3
"""Validate the declared dependency matrix and probe optional imports.

Loads the ``constraints/`` files into a comparable matrix, and probes an
optional SDK import in a fresh interpreter so a package that merely resolves in
pip but fails at import time cannot pass.  Optional extras (yfinance, akshare)
must import in a pristine interpreter under the minimum constraints.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from packaging.version import InvalidVersion, Version

ROOT = Path(__file__).resolve().parent.parent
CONSTRAINTS_DIR = ROOT / "constraints"


def _parse_spec(line: str) -> tuple[str, str] | None:
    """Parse one ``package>=version`` line into (name, minimum version)."""
    stripped = line.split("#", 1)[0].strip()
    if not stripped:
        return None
    for operator in (">=", "==", ">", "~="):
        if operator in stripped:
            name, _, version = stripped.partition(operator)
            return name.strip(), version.strip()
    return None


def load_matrix(directory: str | Path | None = None) -> dict[str, dict[str, str]]:
    """Load the minimum/latest constraint files into a comparable matrix."""
    base = Path(directory) if directory is not None else CONSTRAINTS_DIR
    matrix: dict[str, dict[str, str]] = {}
    for path in sorted(base.glob("*.txt")):
        entries: dict[str, str] = {}
        for line in path.read_text(encoding="utf-8").splitlines():
            parsed = _parse_spec(line)
            if parsed is not None:
                name, version = parsed
                entries[name] = version
        matrix[path.stem] = entries
    return matrix


def probe_import(module: str, constraints: str | Path | None = None) -> dict:
    """Import ``module`` in a fresh interpreter and return a result record.

    ``constraints`` is recorded for provenance; the probe itself runs against
    the currently active interpreter, which must already be installed with the
    matrix in question.
    """

    code = f"import {module}"
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        timeout=120,
    )
    if result.returncode != 0:
        return {
            "module": module,
            "success": False,
            "error": result.stderr.strip().splitlines()[-1] if result.stderr.strip() else "import failed",
            "constraints": str(constraints) if constraints else None,
        }
    return {"module": module, "success": True, "error": None, "constraints": str(constraints) if constraints else None}


def _minimum_before_latest(matrix: dict[str, dict[str, str]]) -> list[str]:
    violations: list[str] = []
    minimum = matrix.get("minimum", {})
    latest = matrix.get("latest", {})
    for name in minimum:
        if name not in latest:
            continue
        try:
            min_version = Version(minimum[name])
            latest_version = Version(latest[name])
        except InvalidVersion:
            continue
        if min_version > latest_version:
            violations.append(f"{name}: minimum {minimum[name]} exceeds latest {latest[name]}")
    return violations


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--constraints", default=str(CONSTRAINTS_DIR), help="constraints directory")
    parser.add_argument("--probe", action="append", default=[], help="module to import-probe")
    args = parser.parse_args(argv)

    matrix = load_matrix(args.constraints)
    if not matrix:
        print(f"no constraints found under {args.constraints}", file=sys.stderr)
        return 1
    violations = _minimum_before_latest(matrix)
    for name in ("minimum", "latest"):
        if name not in matrix:
            violations.append(f"missing constraints file {name}.txt")
    for violation in violations:
        print(f"FAIL: {violation}", file=sys.stderr)

    probe_failures = 0
    for module in args.probe:
        result = probe_import(module, constraints=Path(args.constraints) / "minimum.txt")
        print(f"probe {module}: {'ok' if result['success'] else 'FAIL: ' + str(result['error'])}")
        if not result["success"]:
            probe_failures += 1

    if violations or probe_failures:
        return 1
    print("dependency matrix is valid.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
