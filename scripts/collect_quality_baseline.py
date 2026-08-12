#!/usr/bin/env python3
"""Collect a reproducible, write-safe quality baseline in a disposable copy."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import subprocess
import sys
import tempfile
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Iterable


CACHE_PARTS = {".git", ".mypy_cache", ".pytest_cache", ".ruff_cache", "__pycache__", "build", "dist", "htmlcov"}
NON_SERIAL_SELECTOR = "not serial and not slow and not integration"
TRUSTED_SELECTOR = "not slow and not integration"
BENCHMARKS_IGNORE = "--ignore=tests/benchmarks"


class PackageWriteError(RuntimeError):
    """Raised when a baseline test writes to its disposable package copy."""


def _is_excluded(relative: Path) -> bool:
    return any(part in CACHE_PARTS for part in relative.parts) or relative.name.startswith(".coverage")


def _copy_ignore(directory: str, names: list[str]) -> set[str]:
    parent = Path(directory)
    return {name for name in names if _is_excluded(parent.joinpath(name).relative_to(parent))}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _inventory(root: Path) -> dict[str, str]:
    return {
        path.relative_to(root).as_posix(): _sha256(path)
        for path in sorted(root.rglob("*"))
        if path.is_file() and not _is_excluded(path.relative_to(root))
    }


def _tracked_package_snapshot(copy_root: Path, tracked_package_files: Iterable[str]) -> dict[str, str | None]:
    snapshot: dict[str, str | None] = {}
    for relative_name in tracked_package_files:
        path = copy_root / relative_name
        snapshot[relative_name] = _sha256(path) if path.is_file() else None
    return snapshot


def _copy_manifest_sha256(inventory: dict[str, str]) -> str:
    manifest = json.dumps(inventory, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(manifest).hexdigest()


def _git_lines(source_root: Path, *args: str) -> list[str]:
    result = subprocess.run(
        ["git", *args], cwd=source_root, capture_output=True, text=True, check=True
    )
    return [line for line in result.stdout.splitlines() if line]


def _parse_count(output: str, name: str) -> int:
    matches = re.findall(rf"(?:^|, |\s)(\d+) {name}(?:,|\s|$)", output, flags=re.MULTILINE)
    return int(matches[-1]) if matches else 0


def _parse_collected(output: str) -> int:
    matches = re.findall(r"collected (\d+) items?(?: / (\d+) deselected)?", output)
    if matches:
        collected, deselected = matches[-1]
        return int(collected) - int(deselected or 0)
    xdist_matches = re.findall(r"\[(\d+) items\]|(?:\d+/)?(\d+) tests collected", output)
    return int(next(value for value in xdist_matches[-1] if value)) if xdist_matches else 0


def _parse_warnings(output: str) -> int:
    matches = re.findall(r"(?:^|, |\s)(\d+) warnings?(?:,|\s|$)", output, flags=re.MULTILINE)
    return int(matches[-1]) if matches else 0


def _parse_branch_coverage(output: str) -> float | None:
    matches = re.findall(r"^TOTAL\s+.*?(\d+(?:\.\d+)?)%\s*$", output, flags=re.MULTILINE)
    return float(matches[-1]) if matches else None


def _run_checked(
    copy_root: Path,
    tracked_package_files: list[str],
    label: str,
    selector: str,
    pytest_args: list[str],
    coverage: bool = False,
) -> dict[str, Any]:
    package_before = _tracked_package_snapshot(copy_root, tracked_package_files)
    inventory_before = _inventory(copy_root)
    command = [sys.executable, "-m", "pytest", "-o", "addopts=", *pytest_args]
    started = time.perf_counter()
    result = subprocess.run(command, cwd=copy_root, capture_output=True, text=True, check=False)
    duration_seconds = time.perf_counter() - started
    package_after = _tracked_package_snapshot(copy_root, tracked_package_files)
    inventory_after = _inventory(copy_root)
    changed_package_files = sorted(
        name for name in package_before if package_before[name] != package_after[name]
    )
    added_files = sorted(set(inventory_after) - set(inventory_before))
    removed_files = sorted(set(inventory_before) - set(inventory_after))
    changed_files = sorted(
        name for name in set(inventory_before) & set(inventory_after) if inventory_before[name] != inventory_after[name]
    )
    integrity_ok = not (changed_package_files or added_files or removed_files or changed_files)
    record: dict[str, Any] = {
        "label": label,
        "command": command,
        "selector": selector,
        "returncode": result.returncode,
        "passed": _parse_count(result.stdout, "passed"),
        "skipped": _parse_count(result.stdout, "skipped"),
        "warnings": _parse_warnings(result.stdout),
        "duration_seconds": round(duration_seconds, 3),
        "integrity_ok": integrity_ok,
        "write_check": {
            "changed_tracked_package_files": changed_package_files,
            "added_non_cache_files": added_files,
            "removed_non_cache_files": removed_files,
            "changed_non_cache_files": changed_files,
        },
        "output_tail": (result.stdout + result.stderr)[-6000:],
    }
    record["collected"] = _parse_collected(result.stdout) or (
        record["passed"] + record["skipped"] if result.returncode == 0 else 0
    )
    if coverage:
        record["branch_coverage_percent"] = _parse_branch_coverage(result.stdout)
    if not integrity_ok:
        raise PackageWriteError(f"{label} modified disposable-copy files: {record['write_check']}")
    return record


def _dependency_versions() -> dict[str, str]:
    import matplotlib
    import numpy
    import pandas
    import scipy

    return {
        "python": sys.version.replace("\n", " "),
        "numpy": numpy.__version__,
        "pandas": pandas.__version__,
        "scipy": scipy.__version__,
        "matplotlib": matplotlib.__version__,
    }


def _render_markdown(data: dict[str, Any]) -> str:
    lines = [
        "# Current Quality Baseline",
        "",
        f"Generated: `{data['generated_at']}`",
        "",
        "## Provenance",
        "",
        f"- Source commit: `{data['source']['commit']}`",
        f"- Dirty state: `{data['source']['dirty']}`",
        f"- Disposable-copy manifest SHA256: `{data['copy_manifest_sha256']}`",
        "",
        "## Environment",
        "",
    ]
    lines.extend(f"- {name}: `{value}`" for name, value in data["environment"].items())
    lines.extend(["", "## Test Runs", "", "| Run | Selector | Collected | Passed | Skipped | Warnings | Duration | Exit |", "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |"])
    lines.extend(
        f"| {run['label']} | `{run['selector']}` | {run['collected']} | {run['passed']} | {run['skipped']} | {run['warnings']} | {run['duration_seconds']:.3f}s | {run['returncode']} |"
        for run in data["runs"]
    )
    coverage = next(run for run in data["runs"] if run["label"] == "branch-coverage")
    lines.extend(["", "## Branch Coverage", "", f"- Total: `{coverage['branch_coverage_percent']}%`", "", "## Integrity", ""])
    lines.extend(f"- {run['label']}: `{run['integrity_ok']}`" for run in data["runs"])
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", type=Path, required=True, help="JSON output path, relative to the source root")
    parser.add_argument("--markdown", type=Path, required=True, help="Markdown output path, relative to the source root")
    args = parser.parse_args()
    source_root = Path(__file__).resolve().parents[1]
    json_path = args.json if args.json.is_absolute() else source_root / args.json
    markdown_path = args.markdown if args.markdown.is_absolute() else source_root / args.markdown
    tracked_package_files = _git_lines(source_root, "ls-files", "fincore")
    data: dict[str, Any] = {
        "generated_at": datetime.now(UTC).isoformat(),
        "source": {
            "commit": _git_lines(source_root, "rev-parse", "HEAD")[0],
            "dirty": bool(_git_lines(source_root, "status", "--short")),
        },
        "copy_manifest_sha256": "",
        "environment": _dependency_versions(),
        "runs": [],
    }
    failure: str | None = None
    with tempfile.TemporaryDirectory(prefix="fincore-quality-baseline-") as temp_dir:
        copy_root = Path(temp_dir) / "fincore"
        shutil.copytree(source_root, copy_root, ignore=_copy_ignore)
        data["copy_manifest_sha256"] = _copy_manifest_sha256(_inventory(copy_root))
        specifications = [
            ("trusted-baseline", TRUSTED_SELECTOR, [BENCHMARKS_IGNORE, "-m", TRUSTED_SELECTOR]),
            ("serial", "serial", [BENCHMARKS_IGNORE, "-m", "serial"]),
            ("non-serial-single", NON_SERIAL_SELECTOR, [BENCHMARKS_IGNORE, "-m", NON_SERIAL_SELECTOR]),
            ("non-serial-xdist", NON_SERIAL_SELECTOR, [BENCHMARKS_IGNORE, "-m", NON_SERIAL_SELECTOR, "-n", "auto", "--dist=loadscope"]),
            ("branch-coverage", TRUSTED_SELECTOR, [BENCHMARKS_IGNORE, "-m", TRUSTED_SELECTOR, "--cov=fincore", "--cov-branch", "--cov-report=term-missing"]),
        ]
        try:
            for label, selector, pytest_args in specifications:
                data["runs"].append(
                    _run_checked(
                        copy_root,
                        tracked_package_files,
                        label,
                        selector,
                        pytest_args,
                        coverage=label == "branch-coverage",
                    )
                )
            single = next(run for run in data["runs"] if run["label"] == "non-serial-single")
            xdist = next(run for run in data["runs"] if run["label"] == "non-serial-xdist")
            data["non_serial_counts_match"] = all(
                single[key] == xdist[key] for key in ("collected", "passed", "skipped")
            )
            if not data["non_serial_counts_match"]:
                failure = "non-serial single-process and xdist counts differ"
        except PackageWriteError as exc:
            failure = str(exc)
    data["outcome"] = "pass" if failure is None and all(run["returncode"] == 0 for run in data["runs"]) else "fail"
    if failure:
        data["failure"] = failure
    json_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
    markdown_path.write_text(_render_markdown(data))
    return 0 if data["outcome"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
