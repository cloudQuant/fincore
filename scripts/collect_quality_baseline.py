#!/usr/bin/env python3
"""Collect a reproducible, write-safe quality baseline in a disposable copy."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
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
COMMAND_TIMEOUT_SECONDS = 900
NON_SERIAL_SELECTOR = "not serial and not slow and not integration"
TRUSTED_SELECTOR = "not slow and not integration"
BENCHMARKS_IGNORE = "--ignore=tests/benchmarks"


class PackageWriteError(RuntimeError):
    """Raised when a baseline test writes to its disposable package copy."""

    def __init__(self, message: str, record: dict[str, Any]) -> None:
        super().__init__(message)
        self.record = record


def _is_excluded(relative: Path, excluded_paths: set[Path] | None = None) -> bool:
    return (
        relative in (excluded_paths or set())
        or any(part in CACHE_PARTS for part in relative.parts)
        or relative.name.startswith(".coverage")
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _inventory(root: Path, excluded_paths: set[Path] | None = None) -> dict[str, str]:
    return {
        path.relative_to(root).as_posix(): _sha256(path)
        for path in sorted(root.rglob("*"))
        if path.is_file() and not _is_excluded(path.relative_to(root), excluded_paths)
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
        ["git", *args],
        cwd=source_root,
        capture_output=True,
        text=True,
        check=True,
        timeout=30,
    )
    return [line for line in result.stdout.splitlines() if line]


def _source_file_manifest(source_root: Path, excluded_paths: set[Path]) -> dict[str, str]:
    result = subprocess.run(
        ["git", "ls-files", "-z", "--cached", "--others", "--exclude-standard"],
        cwd=source_root,
        capture_output=True,
        check=True,
        timeout=30,
    )
    paths = (Path(os.fsdecode(item)) for item in result.stdout.split(b"\0") if item)
    return {
        relative.as_posix(): _sha256(source_root / relative)
        for relative in paths
        if (source_root / relative).is_file() and not _is_excluded(relative, excluded_paths)
    }


def _copy_source_tree(source_root: Path, copy_root: Path, manifest: dict[str, str]) -> None:
    for relative_name in manifest:
        source = source_root / relative_name
        destination = copy_root / relative_name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)


def _parse_count(output: str, name: str) -> int:
    matches = re.findall(rf"(?:^|, |\s)(\d+) {name}(?:,|\s|$)", output, flags=re.MULTILINE)
    return int(matches[-1]) if matches else 0


def _parse_test_counts(output: str) -> dict[str, int | None]:
    matches = re.findall(r"collected (\d+) items?(?: / (\d+) deselected)?", output)
    if matches:
        collected, deselected = matches[-1]
        return {"discovered": int(collected), "selected": int(collected) - int(deselected or 0)}
    xdist_matches = re.findall(r"\[(\d+) items\]|(?:\d+/)?(\d+) tests collected", output)
    if xdist_matches:
        return {"discovered": None, "selected": int(next(value for value in xdist_matches[-1] if value))}
    return {"discovered": None, "selected": None}


def _dirty_provenance(source_root: Path) -> dict[str, Any]:
    diff = subprocess.run(
        ["git", "diff", "--binary", "HEAD"],
        cwd=source_root,
        capture_output=True,
        check=True,
        timeout=30,
    ).stdout
    untracked = _git_lines(source_root, "ls-files", "--others", "--exclude-standard")
    inventory = {name: _sha256(source_root / name) for name in untracked if (source_root / name).is_file()}
    return {
        "dirty": bool(_git_lines(source_root, "status", "--short")),
        "tracked_diff_sha256": hashlib.sha256(diff).hexdigest(),
        "untracked_inventory": inventory,
        "untracked_manifest_sha256": _copy_manifest_sha256(inventory),
    }


def _parse_warnings(output: str) -> int:
    matches = re.findall(r"(?:^|, |\s)(\d+) warnings?(?:,|\s|$)", output, flags=re.MULTILINE)
    return int(matches[-1]) if matches else 0


def _parse_branch_coverage(output: str) -> float | None:
    matches = re.findall(r"^TOTAL\s+.*?(\d+(?:\.\d+)?)%\s*$", output, flags=re.MULTILINE)
    return float(matches[-1]) if matches else None


def _timeout_output(error: subprocess.TimeoutExpired) -> str:
    def as_text(value: bytes | str | None) -> str:
        if value is None:
            return ""
        return value.decode(errors="replace") if isinstance(value, bytes) else value

    return as_text(error.stdout) + as_text(error.stderr)


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
    try:
        result = subprocess.run(
            command,
            cwd=copy_root,
            capture_output=True,
            text=True,
            check=False,
            timeout=COMMAND_TIMEOUT_SECONDS,
        )
    except subprocess.TimeoutExpired as exc:
        output = _timeout_output(exc)
        result = subprocess.CompletedProcess(command, 124, output, "baseline command timed out")
    duration_seconds = time.perf_counter() - started
    package_after = _tracked_package_snapshot(copy_root, tracked_package_files)
    inventory_after = _inventory(copy_root)
    changed_package_files = sorted(name for name in package_before if package_before[name] != package_after[name])
    added_files = sorted(set(inventory_after) - set(inventory_before))
    removed_files = sorted(set(inventory_before) - set(inventory_after))
    changed_files = sorted(
        name for name in set(inventory_before) & set(inventory_after) if inventory_before[name] != inventory_after[name]
    )
    integrity_ok = not (changed_package_files or added_files or removed_files or changed_files)
    counts = _parse_test_counts(result.stdout)
    record: dict[str, Any] = {
        "label": label,
        "command": command,
        "selector": selector,
        "returncode": result.returncode,
        "discovered": counts["discovered"],
        "selected": counts["selected"],
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
    if record["selected"] is None and result.returncode == 0:
        record["selected"] = record["passed"] + record["skipped"]
    if coverage:
        record["branch_coverage_percent"] = _parse_branch_coverage(result.stdout)
    if not integrity_ok:
        message = f"{label} modified disposable-copy files: {record['write_check']}"
        record["failure"] = message
        raise PackageWriteError(message, record)
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
        f"- Source commit: `{data['source'].get('commit', 'unknown')}`",
        f"- Dirty state: `{data['source'].get('dirty', 'unknown')}`",
        f"- Tracked diff SHA256: `{data['source'].get('tracked_diff_sha256', 'unknown')}`",
        f"- Untracked manifest SHA256: `{data['source'].get('untracked_manifest_sha256', 'unknown')}`",
        f"- Disposable-copy manifest SHA256: `{data.get('copy_manifest_sha256', 'unknown')}`",
        "- Manifest exclusions: `" + ", ".join(data.get("copy_manifest_excluded_paths", [])) + "`",
        "",
        "## Environment",
        "",
    ]
    lines.extend(f"- {name}: `{value}`" for name, value in data["environment"].items())
    lines.extend(
        [
            "",
            "## Test Runs",
            "",
            "| Run | Selector | Discovered | Selected | Passed | Skipped | Warnings | Duration | Exit |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    lines.extend(
        f"| {run.get('label', 'unknown')} | `{run.get('selector', '')}` | {run.get('discovered', 'N/A') if run.get('discovered') is not None else 'N/A'} | {run.get('selected', 'N/A') if run.get('selected') is not None else 'N/A'} | {run.get('passed', 0)} | {run.get('skipped', 0)} | {run.get('warnings', 0)} | {run.get('duration_seconds', 0):.3f}s | {run.get('returncode', 'N/A')} |"
        for run in data["runs"]
    )
    coverage = next((run for run in data["runs"] if run.get("label") == "branch-coverage"), None)
    total_coverage = coverage.get("branch_coverage_percent", "N/A") if coverage else "N/A"
    lines.extend(["", "## Branch Coverage", "", f"- Total: `{total_coverage}%`", "", "## Integrity", ""])
    lines.extend(f"- {run.get('label', 'unknown')}: `{run.get('integrity_ok', False)}`" for run in data["runs"])
    if data.get("outcome") != "pass":
        lines.extend(["", "## Incomplete Baseline", "", str(data.get("failure", "baseline did not complete"))])
    lines.append("")
    return "\n".join(lines)


def _stage_artifact(path: Path, content: str) -> Path:
    with tempfile.NamedTemporaryFile("w", dir=path.parent, delete=False, encoding="utf-8") as stream:
        stream.write(content)
        return Path(stream.name)


def _backup_artifact(path: Path) -> Path | None:
    if not path.exists():
        return None
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as stream:
        backup_path = Path(stream.name)
    shutil.copy2(path, backup_path)
    return backup_path


def _replace_file(source: Path, destination: Path) -> None:
    source.replace(destination)


def _restore_artifact(path: Path, backup: Path | None) -> None:
    if backup is None:
        path.unlink(missing_ok=True)
    else:
        _replace_file(backup, path)


def _write_artifacts(data: dict[str, Any], json_path: Path, markdown_path: Path) -> None:
    """Replace the paired artifacts with rollback if either replacement fails.

    Cross-file filesystem atomicity is not available, so this stages both files
    and restores both prior versions if the second replacement raises.
    """
    json_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    json_stage = _stage_artifact(json_path, json.dumps(data, indent=2, sort_keys=True) + "\n")
    markdown_stage = _stage_artifact(markdown_path, _render_markdown(data))
    json_backup = _backup_artifact(json_path)
    markdown_backup = _backup_artifact(markdown_path)
    try:
        _replace_file(json_stage, json_path)
        _replace_file(markdown_stage, markdown_path)
    except Exception:
        _restore_artifact(json_path, json_backup)
        _restore_artifact(markdown_path, markdown_backup)
        raise
    else:
        for backup in (json_backup, markdown_backup):
            if backup is not None:
                backup.unlink(missing_ok=True)
    finally:
        json_stage.unlink(missing_ok=True)
        markdown_stage.unlink(missing_ok=True)


def _append_failure_run(data: dict[str, Any], error: PackageWriteError) -> None:
    data["runs"].append(error.record)
    data["failure"] = str(error)


def _normalize_nonserial_counts(single: dict[str, Any], xdist: dict[str, Any]) -> bool:
    if xdist.get("discovered") is None:
        xdist["discovered"] = single.get("discovered")
        xdist["collection_source"] = "non-serial-single pytest collection"
    else:
        xdist["collection_source"] = "xdist pytest collection"
    return all(single.get(key) == xdist.get(key) for key in ("selected", "passed", "skipped"))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", type=Path, required=True, help="JSON output path, relative to the source root")
    parser.add_argument(
        "--markdown", type=Path, required=True, help="Markdown output path, relative to the source root"
    )
    args = parser.parse_args()
    source_root = Path(__file__).resolve().parents[1]
    json_path = args.json if args.json.is_absolute() else source_root / args.json
    markdown_path = args.markdown if args.markdown.is_absolute() else source_root / args.markdown
    output_paths = {
        path.relative_to(source_root) for path in (json_path, markdown_path) if path.is_relative_to(source_root)
    }
    included_manifest = _source_file_manifest(source_root, output_paths)
    tracked_package_files = [name for name in included_manifest if name.startswith("fincore/")]
    dirty_provenance = _dirty_provenance(source_root)
    data: dict[str, Any] = {
        "generated_at": datetime.now(UTC).isoformat(),
        "source": {
            "commit": _git_lines(source_root, "rev-parse", "HEAD")[0],
            **dirty_provenance,
        },
        "copy_manifest_sha256": "",
        "tested_tree": {
            "included_file_manifest": included_manifest,
            "included_file_manifest_sha256": _copy_manifest_sha256(included_manifest),
        },
        "environment": _dependency_versions(),
        "runs": [],
    }
    failure: str | None = None
    with tempfile.TemporaryDirectory(prefix="fincore-quality-baseline-") as temp_dir:
        copy_root = Path(temp_dir) / "fincore"
        _copy_source_tree(source_root, copy_root, included_manifest)
        data["copy_manifest_sha256"] = _copy_manifest_sha256(_inventory(copy_root))
        specifications = [
            ("trusted-baseline", TRUSTED_SELECTOR, [BENCHMARKS_IGNORE, "-m", TRUSTED_SELECTOR]),
            ("serial", "serial", [BENCHMARKS_IGNORE, "-m", "serial"]),
            ("non-serial-single", NON_SERIAL_SELECTOR, [BENCHMARKS_IGNORE, "-m", NON_SERIAL_SELECTOR]),
            (
                "non-serial-xdist",
                NON_SERIAL_SELECTOR,
                [BENCHMARKS_IGNORE, "-m", NON_SERIAL_SELECTOR, "-n", "auto", "--dist=loadscope"],
            ),
            (
                "branch-coverage",
                TRUSTED_SELECTOR,
                [
                    BENCHMARKS_IGNORE,
                    "-m",
                    TRUSTED_SELECTOR,
                    "--cov=fincore",
                    "--cov-branch",
                    "--cov-report=term-missing",
                ],
            ),
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
            data["non_serial_counts_match"] = _normalize_nonserial_counts(single, xdist)
            if not data["non_serial_counts_match"]:
                failure = "non-serial single-process and xdist counts differ"
        except PackageWriteError as exc:
            _append_failure_run(data, exc)
            failure = data["failure"]
        except Exception as exc:  # pragma: no cover - defensive artifact path
            failure = f"baseline collection error: {exc}"
    data.setdefault("non_serial_counts_match", False)
    data["outcome"] = "pass" if failure is None and all(run["returncode"] == 0 for run in data["runs"]) else "fail"
    if failure:
        data["failure"] = failure
    data["copy_manifest_excluded_paths"] = sorted(path.as_posix() for path in output_paths)
    _write_artifacts(data, json_path, markdown_path)
    return 0 if data["outcome"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
