#!/usr/bin/env python
"""Validate a reviewed isolated Alphalens oracle tuple before writing a candidate.

This command deliberately does not create Conda environments, install packages,
checkout source, or import the sibling Alphalens package.  It is a gate around
an externally prepared, reviewed execution tuple.  In particular, a base
environment observation with an incomplete pip hash lock is rejected before an
output file can be created.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import locale
import os
import platform
import re
import subprocess
import sys
import sysconfig
import tempfile
import time
from importlib import metadata
from pathlib import Path
from typing import Any

ALPHALENS_PROFILE = "cloudquant-local-3fa17ad"
GIT_TIMEOUT_SECONDS = 30
NONINTERACTIVE_ENV_OVERRIDES = {"GIT_TERMINAL_PROMPT": "0", "GIT_ASKPASS": ""}
EXPLICIT_HASH = re.compile(r"https://[^\s]+#[0-9a-fA-F]{32}$")
TABLE_FIELDS = {
    "kind",
    "index",
    "index_names",
    "timezone",
    "columns",
    "dtypes",
    "values",
    "nan_mask",
}


def _error(message: str) -> ValueError:
    return ValueError(f"Alphalens oracle validation failed: {message}")


def _run_git(root: Path, arguments: list[str], operation: str) -> bytes:
    """Use only bounded, noninteractive Git commands against the supplied source."""
    environment = os.environ.copy()
    environment.update(NONINTERACTIVE_ENV_OVERRIDES)
    try:
        result = subprocess.run(
            ["git", *arguments],
            cwd=root,
            check=True,
            capture_output=True,
            env=environment,
            stdin=subprocess.DEVNULL,
            timeout=GIT_TIMEOUT_SECONDS,
        )
    except subprocess.TimeoutExpired as exc:
        raise _error(f"{operation} timed out after {GIT_TIMEOUT_SECONDS}s") from exc
    except subprocess.CalledProcessError as exc:
        detail = (exc.stderr or exc.stdout or b"").decode(errors="replace").strip()
        raise _error(f"{operation} failed: {detail or exc}") from exc
    return result.stdout


def _read_json(path: Path, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise _error(f"cannot read {label} JSON {path}") from exc
    if not isinstance(payload, dict):
        raise _error(f"{label} JSON must be an object")
    return payload


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path, label: str) -> str:
    try:
        return _sha256_bytes(path.read_bytes())
    except OSError as exc:
        raise _error(f"cannot read {label} {path}") from exc


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise _error(message)


def _source_blob_records(root: Path, commit: str, source_files: list[dict[str, Any]]) -> list[dict[str, str]]:
    records: list[dict[str, str]] = []
    for item in source_files:
        path = item.get("path")
        expected_blob = item.get("git_blob")
        expected_sha = item.get("sha256")
        _require(isinstance(path, str) and path and not Path(path).is_absolute(), "source file path is not portable")
        _require(isinstance(expected_blob, str) and isinstance(expected_sha, str), "source evidence lacks hashes")
        blob = _run_git(root, ["rev-parse", f"{commit}:{path}"], f"resolve pinned blob {path}").decode().strip()
        contents = _run_git(root, ["show", f"{commit}:{path}"], f"read pinned blob {path}")
        actual_sha = _sha256_bytes(contents)
        _require(blob == expected_blob, f"pinned blob mismatch for {path}")
        _require(actual_sha == expected_sha, f"pinned SHA256 mismatch for {path}")
        records.append({"path": path, "git_blob": blob, "sha256": actual_sha})
    return records


def _validate_source(root: Path, commit: str, environment: dict[str, Any]) -> list[dict[str, str]]:
    _require(root.is_dir(), f"source is not a directory: {root}")
    expected_source = environment.get("source")
    _require(isinstance(expected_source, dict), "environment has no source section")
    _require(expected_source.get("commit") == commit, "environment source commit does not match --commit")
    _run_git(root, ["cat-file", "-e", f"{commit}^{{commit}}"], "validate requested pinned commit")
    actual_head = _run_git(root, ["rev-parse", "HEAD"], "resolve source HEAD").decode().strip()
    _require(actual_head == commit, f"source HEAD is {actual_head}, expected {commit}")
    if expected_source.get("worktree_required_clean") is True:
        status = _run_git(root, ["status", "--porcelain=v1", "--untracked-files=all"], "check source dirty state")
        _require(not status.strip(), "source checkout is dirty")
    files = expected_source.get("source_files")
    _require(isinstance(files, list) and files, "environment source_files is missing")
    return _source_blob_records(root, commit, files)


def _validate_environment_static(
    environment: dict[str, Any], explicit_lock: Path, requirements: Path, commit: str
) -> None:
    _require(environment.get("profile") == ALPHALENS_PROFILE, "environment profile is not the pinned profile")
    _require(environment.get("source", {}).get("commit") == commit, "environment source commit is wrong")
    lock = environment.get("explicit_lock")
    requirement_lock = environment.get("requirements")
    _require(isinstance(lock, dict), "environment explicit_lock section is missing")
    _require(isinstance(requirement_lock, dict), "environment requirements section is missing")
    _require(lock.get("sha256") == _sha256_file(explicit_lock, "explicit lock"), "explicit lock SHA256 mismatch")
    _require(
        requirement_lock.get("sha256") == _sha256_file(requirements, "requirements lock"),
        "requirements lock SHA256 mismatch",
    )


def _validate_explicit_lock(path: Path, environment: dict[str, Any]) -> None:
    lines = path.read_text(encoding="utf-8").splitlines()
    _require("@EXPLICIT" in lines, "explicit lock lacks @EXPLICIT")
    package_lines = [line for line in lines if line and not line.startswith("#") and line != "@EXPLICIT"]
    _require(package_lines, "explicit lock contains no package URLs")
    _require(all(EXPLICIT_HASH.fullmatch(line) for line in package_lines), "explicit lock has unhashed package URL")
    metadata_lock = environment["explicit_lock"]
    _require(metadata_lock.get("package_url_count") == len(package_lines), "explicit lock package count mismatch")
    _require(
        metadata_lock.get("status") == "complete-reviewed-conda-packages",
        "explicit lock is not marked as a reviewed complete Conda lock",
    )


def _validate_requirements_lock(path: Path, environment: dict[str, Any]) -> None:
    lines = [line.strip() for line in path.read_text(encoding="utf-8").splitlines()]
    active = [line for line in lines if line and not line.startswith("#")]
    _require(active, "requirements lock contains no reviewed pip requirements")
    _require("--require-hashes" in active, "requirements lock does not enable --require-hashes")
    requirements = [line for line in active if not line.startswith("--")]
    _require(requirements, "requirements lock contains no pip requirement")
    _require(
        all(" --hash=sha256:" in line for line in requirements),
        "requirements lock has an unhashed pip requirement",
    )
    _require(
        environment["requirements"].get("status") == "complete-reviewed-pip-hash-lock",
        "requirements lock is not marked as reviewed with wheel hashes",
    )


def _runtime_observation(environment: dict[str, Any]) -> dict[str, Any]:
    """Collect a deliberately portable runtime fingerprint without importing Alphalens."""
    important = environment.get("distribution_inventory", {}).get("important_distributions", [])
    versions: dict[str, str | None] = {}
    for package in important:
        name = package.get("name")
        if not isinstance(name, str):
            raise _error("important distribution has no name")
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = None
    return {
        "python": {
            "implementation": platform.python_implementation(),
            "version": platform.python_version(),
            "soabi": sysconfig.get_config_var("SOABI"),
        },
        "platform": {
            "os": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "byteorder": sys.byteorder,
        },
        "locale": locale.setlocale(locale.LC_ALL, None),
        "timezone": {"TZ": os.environ.get("TZ"), "tzname": list(time.tzname)},
        "distributions": versions,
    }


def _validate_runtime(environment: dict[str, Any]) -> dict[str, Any]:
    observed = _runtime_observation(environment)
    expected = environment.get("runtime")
    _require(isinstance(expected, dict), "environment runtime fingerprint is missing")
    expected_python = expected.get("python", {})
    expected_platform = expected.get("platform", {})
    _require(
        observed["python"]["implementation"] == expected_python.get("implementation"), "Python implementation mismatch"
    )
    _require(observed["python"]["version"] == expected_python.get("version"), "Python version mismatch")
    _require(observed["platform"] == expected_platform, "OS/architecture fingerprint mismatch")
    _require(observed["locale"] == expected.get("locale"), "locale fingerprint mismatch")
    _require(observed["timezone"] == expected.get("timezone"), "timezone fingerprint mismatch")
    important = environment["distribution_inventory"]["important_distributions"]
    for item in important:
        name = item["name"]
        _require(observed["distributions"][name] == item.get("version"), f"distribution version mismatch for {name}")
    return observed


def _is_matrix(value: Any) -> bool:
    return isinstance(value, list) and all(isinstance(row, list) for row in value)


def _same_matrix_shape(left: Any, right: Any) -> bool:
    return (
        _is_matrix(left)
        and _is_matrix(right)
        and len(left) == len(right)
        and all(len(left_row) == len(right_row) for left_row, right_row in zip(left, right, strict=True))
    )


def _validate_cases(cases: dict[str, Any], commit: str) -> None:
    _require(cases.get("profile") == ALPHALENS_PROFILE, "case fixture profile is wrong")
    _require(cases.get("commit") == commit, "case fixture commit is wrong")
    _require(cases.get("serializer", {}).get("name") == "fincore-compat-json-table-v1", "case serializer is wrong")
    records = cases.get("cases")
    _require(isinstance(records, list) and records, "case fixture has no cases")
    identifiers = set()
    for case in records:
        _require(isinstance(case, dict), "case record is not an object")
        identifier = case.get("case_id")
        _require(isinstance(identifier, str) and identifier not in identifiers, "case IDs must be unique")
        identifiers.add(identifier)
        _require(case.get("serializer") == "fincore-compat-json-table-v1", f"case {identifier} has wrong serializer")
        _require("expected_output" not in case, f"case {identifier} invents an unreviewed output")
        tables = case.get("tables")
        _require(isinstance(tables, dict) and tables, f"case {identifier} has no tables")
        for table_name, table in tables.items():
            _require(
                isinstance(table, dict) and set(table) >= TABLE_FIELDS,
                f"case {identifier} table {table_name} lacks serializer fields",
            )
            _require(
                _same_matrix_shape(table["values"], table["nan_mask"]),
                f"case {identifier} table {table_name} has invalid NaN mask",
            )
            for row, mask_row in zip(table["values"], table["nan_mask"], strict=True):
                for value, is_nan in zip(row, mask_row, strict=True):
                    _require(isinstance(is_nan, bool), f"case {identifier} table {table_name} NaN mask is not boolean")
                    _require(
                        (value is None) == is_nan, f"case {identifier} table {table_name} has inconsistent NaN encoding"
                    )


def _validate_output_target(output: Path, source: Path, inputs: tuple[Path, ...]) -> Path:
    """Keep candidate writes outside the source checkout and immutable inputs."""
    resolved = output.resolve()
    try:
        resolved.relative_to(source)
    except ValueError:
        pass
    else:
        raise _error("output must not be inside the source checkout")
    _require(resolved not in {path.resolve() for path in inputs}, "output must not replace an oracle input")
    return resolved


def _candidate_payload(
    *,
    commit: str,
    source_files: list[dict[str, str]],
    environment_path: Path,
    explicit_lock: Path,
    requirements: Path,
    cases_path: Path,
    runtime: dict[str, Any],
) -> dict[str, Any]:
    payload = {
        "schema_version": 1,
        "profile": ALPHALENS_PROFILE,
        "commit": commit,
        "source_files": source_files,
        "environment": {
            "path": environment_path.name,
            "sha256": _sha256_file(environment_path, "environment metadata"),
            "explicit_lock_sha256": _sha256_file(explicit_lock, "explicit lock"),
            "requirements_sha256": _sha256_file(requirements, "requirements lock"),
        },
        "cases": {"path": cases_path.name, "sha256": _sha256_file(cases_path, "case fixture")},
        "runtime": runtime,
        "execution": "metadata-validation-only; no sibling import or source execution",
        "reviewed": False,
    }
    payload["candidate_digest"] = _sha256_bytes(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    )
    return payload


def _write_candidate(path: Path, candidate: dict[str, Any]) -> None:
    """Write only the requested candidate path, atomically after every validation."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.", delete=False
        ) as temporary:
            temporary_name = temporary.name
            json.dump(candidate, temporary, indent=2, sort_keys=True)
            temporary.write("\n")
        Path(temporary_name).replace(path)
    finally:
        if temporary_name is not None:
            Path(temporary_name).unlink(missing_ok=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--commit", required=True)
    parser.add_argument("--environment", type=Path, required=True)
    parser.add_argument("--explicit-lock", type=Path, required=True)
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    requirements = args.explicit_lock.with_name("requirements-alphalens-0.4.0-cloudquant.txt")
    output = _validate_output_target(
        args.output,
        args.source.resolve(),
        (args.environment, args.explicit_lock, requirements, args.cases),
    )
    environment = _read_json(args.environment, "environment")
    cases = _read_json(args.cases, "case fixture")
    source_files = _validate_source(args.source.resolve(), args.commit, environment)
    _validate_environment_static(
        environment,
        args.explicit_lock,
        requirements,
        args.commit,
    )
    _validate_cases(cases, args.commit)
    oracle = environment.get("oracle_verification", {})
    if environment.get("execution_status") != "reviewed-executable-tuple" or oracle.get("reviewed") is not True:
        raise _error("unreviewed environment metadata; refusing to create an oracle candidate")
    _validate_explicit_lock(args.explicit_lock, environment)
    _validate_requirements_lock(requirements, environment)
    runtime = _validate_runtime(environment)
    _write_candidate(
        output,
        _candidate_payload(
            commit=args.commit,
            source_files=source_files,
            environment_path=args.environment,
            explicit_lock=args.explicit_lock,
            requirements=requirements,
            cases_path=args.cases,
            runtime=runtime,
        ),
    )


if __name__ == "__main__":
    try:
        main()
    except ValueError as exc:
        print(exc, file=sys.stderr)
        raise SystemExit(2) from exc
