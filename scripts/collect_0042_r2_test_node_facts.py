#!/usr/bin/env python3
"""Collect deterministic, raw 0042-R2 pytest test-node facts from a clean Git HEAD.

The resulting artifact is intentionally not a test-node disposition and cannot
serve as D0 evidence. It only records the actual non-online functional pytest
collection from one clean initial Git commit. Source bytes are materialized
from ``git archive`` into a temporary snapshot; this command never uses the
mutable caller worktree as pytest's source tree.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import re
import subprocess
import sys
import tarfile
import tempfile
from collections import Counter
from collections.abc import Mapping, Sequence
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

SCHEMA_VERSION = 1
_GIT_OBJECT_ID = re.compile(r"^[0-9a-f]{40,64}$")
_TEST_PREFIX = "tests/"
_BENCHMARK_PREFIX = "tests/benchmarks/"
_PLUGIN_NAME = "_fincore_0042_r2_test_node_plugin"
_COLLECTION_TIMEOUT_SECONDS = 600


class DiscoveryError(RuntimeError):
    """Raised when one raw test-node discovery cannot be completed safely."""


@dataclass(frozen=True)
class SourceBlob:
    """One regular Git blob used to make the isolated pytest snapshot."""

    path: str
    sha256: str
    payload: bytes


_PLUGIN_SOURCE = r'''
"""Temporary pytest plugin for 0042-R2 raw test-node discovery."""

from __future__ import annotations

import json
import os
from pathlib import Path, PurePosixPath


def _validate_relative_test_path(value: str) -> str:
    path = PurePosixPath(value)
    if (
        not value
        or value != str(path)
        or path.is_absolute()
        or "\\" in value
        or any(part in {"", ".", ".."} for part in path.parts)
        or not value.startswith("tests/")
    ):
        raise RuntimeError(f"collected test path is not repository-relative and safe: {value!r}")
    return value


def _test_path(item) -> str:
    snapshot_root = Path(os.environ["FINCORE_0042_R2_SNAPSHOT_ROOT"]).resolve()
    item_path = Path(str(item.path)).resolve()
    try:
        relative = item_path.relative_to(snapshot_root).as_posix()
    except ValueError as exc:
        raise RuntimeError(f"collected node escapes the initial-HEAD snapshot: {item.nodeid!r}") from exc
    return _validate_relative_test_path(relative)


def pytest_collection_finish(session) -> None:
    records = []
    for item in session.items:
        marker_names = sorted({marker.name for marker in item.iter_markers()})
        records.append(
            {
                "nodeid": item.nodeid,
                "test_path": _test_path(item),
                "markers": marker_names,
            }
        )
    records.sort(key=lambda record: record["nodeid"])
    report_path = Path(os.environ["FINCORE_0042_R2_NODE_REPORT"])
    report_path.write_text(json.dumps({"nodes": records}, sort_keys=True) + "\n", encoding="utf-8")
'''


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _git_bytes(source_root: Path, *arguments: str) -> bytes:
    try:
        result = subprocess.run(
            ["git", *arguments],
            cwd=source_root,
            capture_output=True,
            check=False,
            timeout=30,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise DiscoveryError(f"cannot inspect source Git worktree: {exc}") from exc
    if result.returncode != 0:
        detail = (result.stderr or result.stdout).decode("utf-8", errors="replace").strip()
        raise DiscoveryError(f"cannot inspect source Git worktree: {detail or 'git command failed'}")
    return result.stdout


def _git_text(source_root: Path, *arguments: str) -> str:
    return _git_bytes(source_root, *arguments).decode("utf-8", errors="strict").strip()


def _validate_repository_path(path: str) -> None:
    pure_path = PurePosixPath(path)
    if (
        not path
        or path != str(pure_path)
        or pure_path.is_absolute()
        or "\\" in path
        or any(part in {"", ".", ".."} for part in pure_path.parts)
    ):
        raise DiscoveryError(f"source path must be a repository-relative POSIX path: {path!r}")


def _require_worktree_root(source_root: Path) -> Path:
    top_level = Path(_git_text(source_root, "rev-parse", "--show-toplevel")).resolve()
    if source_root.resolve() != top_level:
        raise DiscoveryError("run this command from the clean Git worktree root")
    return top_level


def _provenance(source_root: Path) -> dict[str, Any]:
    _require_worktree_root(source_root)
    dirty = _git_text(source_root, "status", "--porcelain=v1", "--untracked-files=all")
    if dirty:
        raise DiscoveryError("source Git worktree must be clean before test-node discovery")
    commit = _git_text(source_root, "rev-parse", "HEAD")
    tree = _git_text(source_root, "rev-parse", "HEAD^{tree}")
    if not _GIT_OBJECT_ID.fullmatch(commit) or not _GIT_OBJECT_ID.fullmatch(tree):
        raise DiscoveryError("source Git HEAD and tree must resolve to object identifiers")
    return {"commit": commit, "tree": tree, "clean": True}


def _verify_provenance(source_root: Path, initial: Mapping[str, Any]) -> None:
    final = _provenance(source_root)
    if final != dict(initial):
        raise DiscoveryError("source Git provenance changed while test-node discovery was running")


def _require_external_output(source_root: Path, output: Path) -> None:
    try:
        output.relative_to(source_root)
    except ValueError:
        return
    raise DiscoveryError("output path must be outside the source Git worktree")


def _regular_git_blob_paths(source_root: Path, commit: str) -> set[str]:
    records = [record for record in _git_bytes(source_root, "ls-tree", "-r", "-z", commit).split(b"\0") if record]
    paths: set[str] = set()
    for record in records:
        try:
            metadata, raw_path = record.split(b"\t", 1)
            mode, object_type, object_id = metadata.decode("ascii").split()
            path = raw_path.decode("utf-8", errors="strict")
        except (UnicodeDecodeError, ValueError) as exc:
            raise DiscoveryError("cannot inspect initial HEAD tree") from exc
        _validate_repository_path(path)
        if path in paths:
            raise DiscoveryError(f"initial HEAD contains duplicate source path: {path}")
        if object_type != "blob" or not mode.startswith("100") or not _GIT_OBJECT_ID.fullmatch(object_id):
            raise DiscoveryError(f"initial HEAD path must be a regular Git blob, not a link or tree: {path}")
        paths.add(path)
    if not paths:
        raise DiscoveryError("initial HEAD contains no regular Git blobs")
    return paths


def _archive_regular_blobs(source_root: Path, commit: str, expected_paths: set[str]) -> dict[str, SourceBlob]:
    archive = _git_bytes(source_root, "archive", "--format=tar", commit)
    blobs: dict[str, SourceBlob] = {}
    try:
        with tarfile.open(fileobj=io.BytesIO(archive), mode="r:") as tar:
            for member in tar.getmembers():
                _validate_repository_path(member.name)
                if member.isdir():
                    continue
                if member.name not in expected_paths:
                    raise DiscoveryError(f"initial HEAD archive contains an unexpected source path: {member.name}")
                if not member.isfile():
                    raise DiscoveryError(f"source archive member must be regular, not a link or device: {member.name}")
                if member.name in blobs:
                    raise DiscoveryError(f"initial HEAD archive contains duplicate source path: {member.name}")
                source = tar.extractfile(member)
                if source is None:
                    raise DiscoveryError(f"cannot read source from initial HEAD archive: {member.name}")
                payload = source.read()
                blobs[member.name] = SourceBlob(member.name, _sha256_bytes(payload), payload)
    except tarfile.TarError as exc:
        raise DiscoveryError("cannot read initial HEAD Git archive") from exc
    if set(blobs) != expected_paths:
        missing = sorted(expected_paths - set(blobs))
        extra = sorted(set(blobs) - expected_paths)
        detail = ", ".join([*(f"missing {path}" for path in missing), *(f"unexpected {path}" for path in extra)])
        raise DiscoveryError(f"initial HEAD archive does not match regular Git blob paths: {detail}")
    return blobs


def _materialize_archive(snapshot_root: Path, blobs: Mapping[str, SourceBlob]) -> None:
    snapshot_root.mkdir(parents=True, exist_ok=False)
    for path in sorted(blobs):
        _validate_repository_path(path)
        destination = snapshot_root.joinpath(*PurePosixPath(path).parts)
        try:
            destination.relative_to(snapshot_root)
        except ValueError as exc:
            raise DiscoveryError(f"initial HEAD source path escapes isolated snapshot: {path}") from exc
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("xb") as stream:
            stream.write(blobs[path].payload)


def _is_selected_test_python_path(path: str) -> bool:
    return path.startswith(_TEST_PREFIX) and not path.startswith(_BENCHMARK_PREFIX) and path.endswith(".py")


def _source_test_blobs(blobs: Mapping[str, SourceBlob]) -> list[SourceBlob]:
    selected = [blob for path, blob in blobs.items() if _is_selected_test_python_path(path)]
    if not selected:
        raise DiscoveryError("initial HEAD contains no selected non-benchmark Python test blobs")
    return sorted(selected, key=lambda blob: blob.path)


def _collection_argv() -> list[str]:
    return [
        "<python>",
        "-m",
        "pytest",
        "--collect-only",
        "-o",
        "addopts=",
        "-p",
        "no:cacheprovider",
        "-p",
        _PLUGIN_NAME,
        "tests",
        "-q",
        "--tb=short",
        "--maxfail=0",
        "-m",
        "not integration_online",
        "--ignore=tests/benchmarks",
    ]


def _collection_environment(plugin_directory: Path, snapshot_root: Path, report_path: Path) -> dict[str, str]:
    environment = os.environ.copy()
    environment.pop("PYTHONPATH", None)
    environment.pop("PYTEST_ADDOPTS", None)
    environment.pop("PYTEST_DISABLE_PLUGIN_AUTOLOAD", None)
    environment.update(
        {
            "FINCORE_0042_R2_NODE_REPORT": str(report_path),
            "FINCORE_0042_R2_SNAPSHOT_ROOT": str(snapshot_root),
            "MPLBACKEND": "Agg",
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONPATH": str(plugin_directory),
        }
    )
    return environment


def _diagnostic_text(result: subprocess.CompletedProcess[str], snapshot_root: Path) -> str:
    detail = (result.stderr or result.stdout or "pytest returned no diagnostic output").strip()
    sanitized = detail.replace(str(snapshot_root), "<snapshot>")
    if len(sanitized) > 4000:
        sanitized = f"{sanitized[:4000]} ... [truncated]"
    return sanitized


def _run_pytest_collection(snapshot_root: Path, scratch_root: Path) -> list[dict[str, Any]]:
    plugin_directory = scratch_root / "plugin"
    plugin_directory.mkdir()
    (plugin_directory / f"{_PLUGIN_NAME}.py").write_text(_PLUGIN_SOURCE, encoding="utf-8", newline="\n")
    report_path = scratch_root / "collected-nodes.json"
    argv = _collection_argv()
    actual_argv = [sys.executable, *argv[1:]]
    try:
        result = subprocess.run(
            actual_argv,
            cwd=snapshot_root,
            capture_output=True,
            check=False,
            text=True,
            timeout=_COLLECTION_TIMEOUT_SECONDS,
            env=_collection_environment(plugin_directory, snapshot_root, report_path),
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise DiscoveryError(f"pytest collection failed before completion: {exc}") from exc
    if result.returncode != 0:
        diagnostic = _diagnostic_text(result, snapshot_root)
        raise DiscoveryError(f"pytest collection failed with exit code {result.returncode}: {diagnostic}")
    try:
        report = json.loads(report_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise DiscoveryError("pytest collection completed without a valid node report") from exc
    nodes = report.get("nodes") if isinstance(report, Mapping) else None
    if not isinstance(nodes, list):
        raise DiscoveryError("pytest collection node report must contain a nodes list")
    return nodes


def _legacy_family(test_path: str) -> str:
    if test_path.startswith(("tests/compat/empyrical/", "tests/test_empyrical/")):
        return "empyrical"
    if test_path.startswith(("tests/compat/pyfolio/", "tests/test_pyfolio/")):
        return "pyfolio"
    if test_path.startswith(("tests/compat/alphalens/", "tests/test_factor_analysis/")):
        return "alphalens"
    return "other"


def _directory_group(test_path: str) -> str:
    return PurePosixPath(test_path).parent.as_posix()


def _normalize_nodes(raw_nodes: Sequence[object], test_sha256_by_path: Mapping[str, str]) -> list[dict[str, Any]]:
    nodes: list[dict[str, Any]] = []
    seen_nodeids: set[str] = set()
    for index, raw in enumerate(raw_nodes):
        if not isinstance(raw, Mapping):
            raise DiscoveryError(f"pytest collection node {index} must be an object")
        nodeid = raw.get("nodeid")
        test_path = raw.get("test_path")
        marker_values = raw.get("markers")
        if not isinstance(nodeid, str) or not nodeid:
            raise DiscoveryError(f"pytest collection node {index} has no non-empty nodeid")
        if not isinstance(test_path, str):
            raise DiscoveryError(f"pytest collection node {nodeid!r} has no test path")
        _validate_repository_path(test_path)
        if not _is_selected_test_python_path(test_path) or test_path not in test_sha256_by_path:
            raise DiscoveryError(f"pytest collection node {nodeid!r} is outside the selected test blob set")
        if not isinstance(marker_values, list) or not all(
            isinstance(marker, str) and marker for marker in marker_values
        ):
            raise DiscoveryError(f"pytest collection node {nodeid!r} has malformed marker facts")
        marker_names = sorted(set(marker_values))
        if "integration_online" in marker_names:
            raise DiscoveryError(f"pytest selection retained integration_online node {nodeid!r}")
        if nodeid in seen_nodeids:
            raise DiscoveryError(f"pytest collection repeats nodeid {nodeid!r}")
        seen_nodeids.add(nodeid)
        nodes.append(
            {
                "nodeid": nodeid,
                "test_path": test_path,
                "test_blob_sha256": test_sha256_by_path[test_path],
                "directory_group": _directory_group(test_path),
                "legacy_family": _legacy_family(test_path),
                "markers": marker_names,
            }
        )
    if not nodes:
        raise DiscoveryError("pytest collection produced no selected non-online test nodes")
    return sorted(nodes, key=lambda node: str(node["nodeid"]))


def _group_counts(nodes: Sequence[Mapping[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    directory_counts: Counter[str] = Counter()
    legacy_family_counts: Counter[str] = Counter()
    marker_counts: Counter[str] = Counter()
    for node in nodes:
        directory_counts[str(node["directory_group"])] += 1
        legacy_family_counts[str(node["legacy_family"])] += 1
        markers = node["markers"]
        if not isinstance(markers, list):
            raise DiscoveryError(f"normalized node {node['nodeid']!r} has malformed markers")
        if markers:
            marker_counts.update(str(marker) for marker in markers)
        else:
            marker_counts["<unmarked>"] += 1

    def summarize(counts: Mapping[str, int]) -> list[dict[str, Any]]:
        return [{"group": group, "count": counts[group]} for group in sorted(counts)]

    return {
        "directory": summarize(directory_counts),
        "legacy_family": summarize(legacy_family_counts),
        "marker": summarize(marker_counts),
    }


def _collect_artifact(source_root: Path) -> dict[str, Any]:
    initial = _provenance(source_root)
    expected_paths = _regular_git_blob_paths(source_root, str(initial["commit"]))
    archived_blobs = _archive_regular_blobs(source_root, str(initial["commit"]), expected_paths)
    test_blobs = _source_test_blobs(archived_blobs)
    test_sha256_by_path = {blob.path: blob.sha256 for blob in test_blobs}
    with tempfile.TemporaryDirectory(prefix="fincore-0042-r2-test-node-") as temporary_directory:
        temporary_root = Path(temporary_directory)
        snapshot_root = temporary_root / "snapshot"
        _materialize_archive(snapshot_root, archived_blobs)
        raw_nodes = _run_pytest_collection(snapshot_root, temporary_root)
    nodes = _normalize_nodes(raw_nodes, test_sha256_by_path)
    artifact = {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": "test_node_facts_discovery",
        "discovery_status": "partial",
        "not_for_d0": True,
        "partial_reason": (
            "This raw pytest collection records no test-node disposition, capability mapping, owner, or oracle and "
            "cannot be D0 evidence. It excludes benchmarks and does not cover maintained documentation, examples, "
            "built distributions, installed-consumer behavior, or runtime-only plugin discovery."
        ),
        "source_provenance": initial,
        "source_archive": {
            "scope": "full_repository",
            "regular_blob_count": len(archived_blobs),
            "verified_against_regular_blobs": True,
        },
        "collection": {
            "status": "passed",
            "argv": _collection_argv(),
            "marker_expression": "not integration_online",
            "ignored_paths": ["tests/benchmarks"],
            "collection_errors": [],
        },
        "source_test_blob_count": len(test_blobs),
        "source_test_blobs": [{"path": blob.path, "sha256": blob.sha256} for blob in test_blobs],
        "node_count": len(nodes),
        "nodes": nodes,
        "group_counts": _group_counts(nodes),
    }
    _verify_provenance(source_root, initial)
    return artifact


def _atomic_write(output: Path, artifact: Mapping[str, Any]) -> None:
    serialized = json.dumps(artifact, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{output.name}.", suffix=".tmp", dir=output.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
            stream.write(serialized)
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(output)
    except Exception:
        with suppress(FileNotFoundError):
            temporary.unlink()
        raise


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, required=True, help="path outside the source tree for raw test-node JSON"
    )
    arguments = parser.parse_args(argv)
    source_root = Path.cwd().resolve()
    output = arguments.output if arguments.output.is_absolute() else source_root / arguments.output
    output = output.resolve()
    try:
        _require_worktree_root(source_root)
        _require_external_output(source_root, output)
        artifact = _collect_artifact(source_root)
        _atomic_write(output, artifact)
    except DiscoveryError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    except (OSError, ValueError) as exc:
        print(f"error: test-node discovery failed closed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
