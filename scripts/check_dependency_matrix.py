#!/usr/bin/env python3
"""Validate the declared dependency matrix and probe optional imports.

Loads the ``constraints/`` files into a comparable matrix, and probes an
optional SDK import in a fresh interpreter so a package that merely resolves in
pip but fails at import time cannot pass.  Optional extras (yfinance, akshare)
must import in a pristine interpreter under the minimum constraints.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

from packaging.version import InvalidVersion, Version

ROOT = Path(__file__).resolve().parent.parent
CONSTRAINTS_DIR = ROOT / "constraints"
CORE_DISTRIBUTIONS = ("fincore", "numpy", "pandas", "scipy", "pytz", "packaging")


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


def _venv_python(venv_directory: Path) -> Path:
    """Return the platform-specific interpreter path for an isolated venv."""
    return venv_directory / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python")


def _run(command: list[str], *, timeout: int = 600) -> subprocess.CompletedProcess[str]:
    """Run a checked subprocess while retaining a concise diagnostic on failure."""
    result = subprocess.run(command, capture_output=True, text=True, timeout=timeout)
    if result.returncode != 0:
        details = result.stderr.strip() or result.stdout.strip() or "no subprocess output"
        raise RuntimeError(f"{' '.join(command)} failed: {details}")
    return result


def _installed_versions(python: Path, distributions: tuple[str, ...]) -> dict[str, str]:
    """Read installed distribution versions from the candidate's own venv."""
    script = (
        "import importlib.metadata as metadata, json; "
        f"print(json.dumps({{name: metadata.version(name) for name in {distributions!r}}}, sort_keys=True))"
    )
    result = _run([str(python), "-c", script])
    payload = json.loads(result.stdout)
    if not isinstance(payload, dict):
        raise RuntimeError("isolated interpreter returned invalid installed-version evidence")
    versions: dict[str, str] = {}
    for name, version in payload.items():
        if not isinstance(name, str) or not isinstance(version, str):
            raise RuntimeError("isolated interpreter returned invalid installed-version evidence")
        versions[name] = version
    return versions


def check_installed_versions(
    installed: dict[str, str],
    expected: dict[str, str],
    *,
    lane: str,
) -> list[str]:
    """Return version-contract violations for an installed dependency lane.

    The ``minimum`` lane is an exact, reproducible oldest-resolvable install;
    the ``latest`` lane must be at least the declared floor.  This pure helper
    is intentionally separate from pip/venv orchestration so the policy has a
    fast unit-test oracle.
    """
    violations: list[str] = []
    for name, expected_version in expected.items():
        actual = installed.get(name)
        if actual is None:
            violations.append(f"{lane}: {name} is not installed")
            continue
        try:
            actual_version = Version(actual)
            required_version = Version(expected_version)
        except InvalidVersion:
            violations.append(f"{lane}: cannot compare {name} installed={actual!r} expected={expected_version!r}")
            continue
        if lane == "minimum" and actual_version != required_version:
            violations.append(f"{lane}: {name} installed {actual} != pinned {expected_version}")
        elif lane == "latest" and actual_version < required_version:
            violations.append(f"{lane}: {name} installed {actual} < floor {expected_version}")
    return violations


def verify_isolated_wheel(
    wheel: str | Path,
    *,
    lane: str,
    constraints_directory: str | Path | None = None,
    probes: tuple[str, ...] = (),
) -> dict[str, Any]:
    """Install one wheel in a brand-new venv and return reproducible evidence.

    ``pip check``, import probes, and version inspection all execute through
    the venv interpreter.  Nothing is imported from the developer's checkout
    or current environment, making the result suitable for a release-candidate
    gate rather than a constraints-file formatting check.
    """
    wheel_path = Path(wheel).resolve()
    if not wheel_path.is_file() or wheel_path.suffix != ".whl":
        raise ValueError(f"wheel must name an existing .whl file: {wheel_path}")
    if lane not in {"minimum", "latest"}:
        raise ValueError("lane must be 'minimum' or 'latest'")

    constraints_root = Path(constraints_directory) if constraints_directory is not None else CONSTRAINTS_DIR
    constraints_path = constraints_root / f"{lane}.txt"
    if not constraints_path.is_file():
        raise ValueError(f"missing constraints file: {constraints_path}")
    matrix = load_matrix(constraints_root)
    expected = matrix.get(lane)
    if expected is None:
        raise ValueError(f"constraints do not define lane {lane!r}")

    with tempfile.TemporaryDirectory(prefix=f"fincore-{lane}-") as temporary_directory:
        venv_directory = Path(temporary_directory) / "venv"
        _run([sys.executable, "-m", "venv", str(venv_directory)])
        python = _venv_python(venv_directory)
        _run([str(python), "-m", "pip", "install", "--disable-pip-version-check", "--upgrade", "pip"])
        _run(
            [
                str(python),
                "-m",
                "pip",
                "install",
                "--disable-pip-version-check",
                "--only-binary=:all:",
                "--constraint",
                str(constraints_path.resolve()),
                str(wheel_path),
            ]
        )
        _run([str(python), "-m", "pip", "check"])

        checked_probes = ("fincore", "fincore.metrics", *probes)
        probe_results: list[dict[str, str | bool | None]] = []
        for module in dict.fromkeys(checked_probes):
            result = subprocess.run(
                [str(python), "-c", f"import {module}"], capture_output=True, text=True, timeout=120
            )
            probe_results.append(
                {
                    "module": module,
                    "success": result.returncode == 0,
                    "error": None
                    if result.returncode == 0
                    else (result.stderr.strip().splitlines()[-1] if result.stderr.strip() else "import failed"),
                }
            )

        installed = _installed_versions(python, CORE_DISTRIBUTIONS)
        # A constraints file can also pin transitive alternatives used by an
        # optional extra.  ``pip --constraint`` deliberately does not install
        # an unrelated package, so only the wheel's mandatory runtime set is
        # expected in this core candidate proof.  Optional SDK installs have a
        # separate clean-environment smoke job.
        expected_runtime = {name: version for name, version in expected.items() if name in CORE_DISTRIBUTIONS}
        version_violations = check_installed_versions(installed, expected_runtime, lane=lane)
        probe_failures = [result for result in probe_results if not result["success"]]
        if version_violations or probe_failures:
            messages = [*version_violations, *(f"{item['module']}: {item['error']}" for item in probe_failures)]
            raise RuntimeError("isolated dependency verification failed: " + "; ".join(messages))

    return {
        "lane": lane,
        "wheel": wheel_path.name,
        "constraints": str(constraints_path),
        "installed": installed,
        "probes": probe_results,
    }


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


def _check_pin_policy(directory: str | Path) -> list[str]:
    """``minimum.txt`` must pin exact versions (``==``); ``latest.txt`` must use floors.

    Two ``>=`` files are not an isolation-install proof: the minimum lane would
    silently resolve to the same packages as the latest lane.  Requiring exact
    pins in the minimum file forces the matrix to be a real, reproducible
    oldest-resolvable combination.
    """
    base = Path(directory)
    violations: list[str] = []
    for filename, require_pin in (("minimum.txt", True), ("latest.txt", False)):
        path = base / filename
        if not path.exists():
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            spec = line.split("#", 1)[0].strip()
            if not spec:
                continue
            parsed = _parse_spec(line)
            if parsed is None:
                continue
            name, _ = parsed
            uses_pin = "==" in spec
            if require_pin and not uses_pin:
                violations.append(f"{filename}: {name} must use an exact == pin")
            if not require_pin and uses_pin:
                violations.append(f"{filename}: {name} must use a >= floor (not ==)")
    return violations


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--constraints", default=str(CONSTRAINTS_DIR), help="constraints directory")
    parser.add_argument("--probe", action="append", default=[], help="module to import-probe")
    parser.add_argument("--wheel", help="candidate wheel to install into a fresh venv")
    parser.add_argument("--lane", choices=("minimum", "latest"), help="isolated dependency lane for --wheel")
    args = parser.parse_args(argv)

    matrix = load_matrix(args.constraints)
    if not matrix:
        print(f"no constraints found under {args.constraints}", file=sys.stderr)
        return 1
    violations = _minimum_before_latest(matrix)
    violations.extend(_check_pin_policy(args.constraints))
    for name in ("minimum", "latest"):
        if name not in matrix:
            violations.append(f"missing constraints file {name}.txt")
    for violation in violations:
        print(f"FAIL: {violation}", file=sys.stderr)

    probe_failures = 0
    if args.wheel:
        if not args.lane:
            parser.error("--wheel requires --lane minimum or --lane latest")
        try:
            evidence = verify_isolated_wheel(
                args.wheel,
                lane=args.lane,
                constraints_directory=args.constraints,
                probes=tuple(args.probe),
            )
        except (RuntimeError, ValueError, subprocess.TimeoutExpired) as exc:
            print(f"FAIL: {exc}", file=sys.stderr)
            probe_failures += 1
        else:
            print(json.dumps(evidence, sort_keys=True))
    else:
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
