#!/usr/bin/env python3
"""Collect complete-input 0042-R2 surface-union facts from one clean source.

This command is deliberately a *raw discovery* boundary.  It makes the
source categories required by the 0042-R2 D0 plan explicit—public
definitions, registries, manifests, documentation, examples, benchmarks,
extras and wheel contents—without assigning an owner, disposition, target
operation, scenario, or oracle.  Consequently, even a successful artifact is
``not_for_d0``; a reviewed complete inventory must bind these facts before a
baseline capture can proceed.

Every source-tree byte comes from the initial clean Git ``HEAD`` rather than
the mutable worktree.  Wheel bytes are read from one protected regular file,
with path traversal, duplicate members, and non-regular members rejected.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import re
import stat
import subprocess
import sys
import tempfile
import zipfile
from collections import Counter
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence


SCHEMA_VERSION = 1
_GIT_OBJECT_ID = re.compile(r"^[0-9a-f]{40,64}$")
_ROOT_DOCUMENTS = frozenset({"README.md", "CHANGELOG.md", "CONTRIBUTING.md", "CODE_OF_CONDUCT.md"})
_EXTRA_PATHS = frozenset({"pyproject.toml", "setup.py", "setup.cfg", "MANIFEST.in", "tox.ini"})
_FORBIDDEN_DECISION_FIELDS = frozenset({"owner", "disposition", "target_operation_id", "oracle"})


class SurfaceUnionDiscoveryError(RuntimeError):
    """Raised when a safe complete-input surface discovery is impossible."""


@dataclass(frozen=True)
class GitBlob:
    """One regular blob resolved from the initial clean Git tree."""

    path: str
    git_object_id: str
    payload: bytes

    @property
    def sha256(self) -> str:
        return _sha256_bytes(self.payload)


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _canonical_sha256(value: object) -> str:
    return _sha256_bytes(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8"))


def _controlled_git_environment() -> dict[str, str]:
    """Ignore inherited Git settings that could redirect source resolution."""
    environment = {key: value for key, value in os.environ.items() if not key.startswith("GIT_")}
    environment.update(
        {
            "GIT_ATTR_NOSYSTEM": "1",
            "GIT_CONFIG_GLOBAL": os.devnull,
            "GIT_CONFIG_NOSYSTEM": "1",
            "GIT_NO_REPLACE_OBJECTS": "1",
        }
    )
    return environment


def _git_bytes(source_root: Path, *arguments: str) -> bytes:
    try:
        result = subprocess.run(
            ["git", "--no-replace-objects", "-c", "core.fsmonitor=false", *arguments],
            cwd=source_root,
            capture_output=True,
            check=False,
            timeout=60,
            env=_controlled_git_environment(),
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise SurfaceUnionDiscoveryError(f"cannot inspect source Git worktree: {exc}") from exc
    if result.returncode != 0:
        detail = (result.stderr or result.stdout).decode("utf-8", errors="replace").strip()
        raise SurfaceUnionDiscoveryError(f"cannot inspect source Git worktree: {detail or 'git command failed'}")
    return result.stdout


def _git_text(source_root: Path, *arguments: str) -> str:
    return _git_bytes(source_root, *arguments).decode("utf-8", errors="strict").strip()


def _validate_repository_path(path: str, *, label: str) -> PurePosixPath:
    pure_path = PurePosixPath(path)
    if (
        not path
        or path != str(pure_path)
        or pure_path.is_absolute()
        or "\\" in path
        or any(part in {"", ".", ".."} for part in pure_path.parts)
    ):
        raise SurfaceUnionDiscoveryError(f"{label} must be a repository-relative POSIX path: {path!r}")
    return pure_path


def _require_worktree_root(source_root: Path) -> Path:
    top_level = Path(_git_text(source_root, "rev-parse", "--show-toplevel")).resolve()
    if source_root.resolve() != top_level:
        raise SurfaceUnionDiscoveryError("run this command from the clean Git worktree root")
    return top_level


def _provenance(source_root: Path) -> dict[str, Any]:
    _require_worktree_root(source_root)
    dirty = _git_text(source_root, "status", "--porcelain=v1", "--untracked-files=all")
    if dirty:
        raise SurfaceUnionDiscoveryError("source Git worktree must be clean before surface-union discovery")
    commit = _git_text(source_root, "rev-parse", "HEAD")
    tree = _git_text(source_root, "rev-parse", "HEAD^{tree}")
    if not _GIT_OBJECT_ID.fullmatch(commit) or not _GIT_OBJECT_ID.fullmatch(tree):
        raise SurfaceUnionDiscoveryError("source Git HEAD and tree must resolve to object identifiers")
    return {"commit": commit, "tree": tree, "clean": True}


def _verify_provenance(source_root: Path, initial: Mapping[str, Any]) -> None:
    if _provenance(source_root) != dict(initial):
        raise SurfaceUnionDiscoveryError("source Git provenance changed while surface-union discovery was running")


def _list_regular_blobs(source_root: Path, commit: str) -> list[GitBlob]:
    raw_records = [record for record in _git_bytes(source_root, "ls-tree", "-r", "-z", commit).split(b"\0") if record]
    blobs: list[GitBlob] = []
    seen_paths: set[str] = set()
    for record in raw_records:
        try:
            metadata, raw_path = record.split(b"\t", 1)
            mode, object_type, object_id = metadata.decode("ascii").split()
            path = raw_path.decode("utf-8", errors="strict")
        except (UnicodeDecodeError, ValueError) as exc:
            raise SurfaceUnionDiscoveryError("cannot parse initial HEAD tree record") from exc
        _validate_repository_path(path, label="source path")
        if path in seen_paths:
            raise SurfaceUnionDiscoveryError(f"initial HEAD contains duplicate source path: {path}")
        seen_paths.add(path)
        if object_type != "blob" or not _GIT_OBJECT_ID.fullmatch(object_id):
            raise SurfaceUnionDiscoveryError(f"initial HEAD source entry must be a blob: {path}")
        if mode == "120000":
            raise SurfaceUnionDiscoveryError(f"initial HEAD source entry must be regular, not a symbolic link: {path}")
        if not mode.startswith("100"):
            raise SurfaceUnionDiscoveryError(f"initial HEAD source entry must be a regular blob: {path}")
        payload = _git_bytes(source_root, "show", f"{commit}:{path}")
        blobs.append(GitBlob(path=path, git_object_id=object_id, payload=payload))
    if not blobs:
        raise SurfaceUnionDiscoveryError("initial HEAD contains no regular blobs")
    return sorted(blobs, key=lambda blob: blob.path)


def _source(blob: GitBlob, locator: str) -> dict[str, str]:
    return {
        "artifact_path": blob.path,
        "artifact_sha256": blob.sha256,
        "locator": locator,
    }


def _entry(entry_id: str, source_kind: str, source: dict[str, str]) -> dict[str, Any]:
    if any(field in source for field in _FORBIDDEN_DECISION_FIELDS):
        raise SurfaceUnionDiscoveryError("raw source must not contain an inventory decision field")
    return {"entry_id": entry_id, "source_kind": source_kind, "source": source}


def _public_definition_entries(blob: GitBlob) -> list[dict[str, Any]]:
    if not blob.path.startswith("fincore/") or not blob.path.endswith(".py"):
        return []
    try:
        tree = ast.parse(blob.payload.decode("utf-8"), filename=blob.path)
    except (UnicodeDecodeError, SyntaxError) as exc:
        raise SurfaceUnionDiscoveryError(f"cannot parse public-definition source: {blob.path}") from exc
    entries: list[dict[str, Any]] = []
    for statement in tree.body:
        if isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef)) and not statement.name.startswith("_"):
            kind = "async_function" if isinstance(statement, ast.AsyncFunctionDef) else "function"
            locator = f"{kind}:{statement.name}:{statement.lineno}"
            entries.append(
                _entry(f"public_definition:{blob.path}:{locator}", "public_definition", _source(blob, locator))
            )
        elif isinstance(statement, ast.ClassDef) and not statement.name.startswith("_"):
            class_locator = f"class:{statement.name}:{statement.lineno}"
            entries.append(
                _entry(
                    f"public_definition:{blob.path}:{class_locator}", "public_definition", _source(blob, class_locator)
                )
            )
            for member in statement.body:
                if isinstance(member, (ast.FunctionDef, ast.AsyncFunctionDef)) and not member.name.startswith("_"):
                    member_locator = f"method:{statement.name}.{member.name}:{member.lineno}"
                    entries.append(
                        _entry(
                            f"public_definition:{blob.path}:{member_locator}",
                            "public_definition",
                            _source(blob, member_locator),
                        )
                    )
    return entries


def _is_registry_path(path: str) -> bool:
    name = PurePosixPath(path).name.casefold()
    return path.startswith("fincore/") and any(token in name for token in ("registry", "catalog", "manifest", "spec"))


def _is_manifest_path(path: str) -> bool:
    lower = path.casefold()
    return (
        (path.startswith("tests/compat/fixtures/") or path.startswith("tests/contracts/fixtures/"))
        and lower.endswith(".json")
    ) or "manifest" in PurePosixPath(path).name.casefold()


def _is_documentation_path(path: str) -> bool:
    return path.startswith(("docs/", "mkdocs_docs/")) or path in _ROOT_DOCUMENTS


def _is_extra_path(path: str) -> bool:
    name = PurePosixPath(path).name
    return path in _EXTRA_PATHS or (name.startswith("requirements") and name.endswith(".txt"))


def _tracked_source_entries(blobs: Sequence[GitBlob]) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    for blob in blobs:
        entries.extend(_public_definition_entries(blob))
        source = _source(blob, "blob")
        if _is_registry_path(blob.path):
            entries.append(_entry(f"registry:{blob.path}", "registry", source))
        if _is_manifest_path(blob.path):
            entries.append(_entry(f"manifest:{blob.path}", "manifest", source))
        if _is_documentation_path(blob.path):
            entries.append(_entry(f"documentation:{blob.path}", "documentation", source))
        if blob.path.startswith("examples/"):
            entries.append(_entry(f"example:{blob.path}", "example", source))
        if blob.path.startswith("tests/benchmarks/"):
            entries.append(_entry(f"benchmark:{blob.path}", "benchmark", source))
        if _is_extra_path(blob.path):
            entries.append(_entry(f"extra:{blob.path}", "extra", source))
    return entries


def _validate_wheel_path(name: str) -> PurePosixPath:
    pure_path = PurePosixPath(name)
    if (
        not name
        or name != str(pure_path)
        or pure_path.is_absolute()
        or "\\" in name
        or any(part in {"", ".", ".."} for part in pure_path.parts)
    ):
        raise SurfaceUnionDiscoveryError(f"wheel member path must be a safe POSIX relative path: {name!r}")
    return pure_path


def _read_regular_wheel(path: Path) -> bytes:
    try:
        status = os.lstat(path)
    except OSError as exc:
        raise SurfaceUnionDiscoveryError(f"cannot inspect wheel: {exc}") from exc
    if stat.S_ISLNK(status.st_mode) or not stat.S_ISREG(status.st_mode):
        raise SurfaceUnionDiscoveryError("wheel must be one regular file, not a symbolic link or special file")
    try:
        return path.read_bytes()
    except OSError as exc:
        raise SurfaceUnionDiscoveryError(f"cannot read wheel: {exc}") from exc


def _wheel_entries(wheel_path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    payload = _read_regular_wheel(wheel_path)
    wheel_sha256 = _sha256_bytes(payload)
    try:
        archive = zipfile.ZipFile(Path(wheel_path))
    except (OSError, zipfile.BadZipFile) as exc:
        raise SurfaceUnionDiscoveryError(f"cannot read wheel archive: {exc}") from exc
    entries: list[dict[str, Any]] = []
    names: set[str] = set()
    try:
        for info in archive.infolist():
            if info.is_dir():
                continue
            name = info.filename
            _validate_wheel_path(name)
            if name in names:
                raise SurfaceUnionDiscoveryError(f"wheel contains duplicate member: {name}")
            names.add(name)
            mode = info.external_attr >> 16
            file_type = stat.S_IFMT(mode)
            if info.create_system == 3 and file_type not in {0, stat.S_IFREG}:
                raise SurfaceUnionDiscoveryError(f"wheel member must be regular, not a link or special file: {name}")
            try:
                member_payload = archive.read(info)
            except (OSError, RuntimeError, zipfile.BadZipFile) as exc:
                raise SurfaceUnionDiscoveryError(f"cannot read wheel member: {name}: {exc}") from exc
            entries.append(
                _entry(
                    f"wheel_content:{name}",
                    "wheel_content",
                    {
                        "artifact_path": wheel_path.name,
                        "artifact_sha256": wheel_sha256,
                        "locator": name,
                        "member_sha256": _sha256_bytes(member_payload),
                    },
                )
            )
    finally:
        archive.close()
    if not entries:
        raise SurfaceUnionDiscoveryError("wheel contains no regular file members")
    return {"filename": wheel_path.name, "sha256": wheel_sha256, "member_count": len(entries)}, entries


def _collect_artifact(source_root: Path, wheel_path: Path) -> dict[str, Any]:
    provenance = _provenance(source_root)
    blobs = _list_regular_blobs(source_root, str(provenance["commit"]))
    wheel, entries = _wheel_entries(wheel_path)
    entries.extend(_tracked_source_entries(blobs))
    entries.sort(key=lambda entry: str(entry["entry_id"]))
    entry_ids = [str(entry["entry_id"]) for entry in entries]
    if len(entry_ids) != len(set(entry_ids)):
        duplicates = sorted(entry_id for entry_id, count in Counter(entry_ids).items() if count > 1)
        raise SurfaceUnionDiscoveryError(f"surface union contains duplicate entry identifiers: {', '.join(duplicates)}")
    if any(set(entry) != {"entry_id", "source_kind", "source"} for entry in entries):
        raise SurfaceUnionDiscoveryError("raw surface union must contain facts only, not decisions")
    kind_counts = dict(sorted(Counter(str(entry["source_kind"]) for entry in entries).items()))
    required_kinds = {
        "public_definition",
        "registry",
        "manifest",
        "documentation",
        "example",
        "benchmark",
        "extra",
        "wheel_content",
    }
    missing_kinds = sorted(required_kinds - set(kind_counts))
    if missing_kinds:
        raise SurfaceUnionDiscoveryError(f"source tree is missing required surface kinds: {', '.join(missing_kinds)}")
    _verify_provenance(source_root, provenance)
    return {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": "surface_union_facts_discovery",
        "discovery_status": "complete",
        "not_for_d0": True,
        "does_not_assert": ["D-TECH", "D0", "installed_wheel_behavior", "legacy_zero"],
        "source_provenance": provenance,
        "wheel": wheel,
        "entry_count": len(entries),
        "kind_counts": kind_counts,
        "entries": entries,
        "canonical_entries_sha256": _canonical_sha256(entries),
    }


def _reject_unsafe_output(source_root: Path, output: Path) -> None:
    try:
        output.resolve().relative_to(source_root.resolve())
    except ValueError:
        pass
    else:
        raise SurfaceUnionDiscoveryError("output path must be outside the source Git worktree")
    if output.exists() and output.is_symlink():
        raise SurfaceUnionDiscoveryError("output path must not be a symbolic link")
    if output.parent.exists() and output.parent.is_symlink():
        raise SurfaceUnionDiscoveryError("output parent must not be a symbolic link")


def _atomic_write(output: Path, artifact: Mapping[str, Any]) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor: int | None = None
    temporary_name: str | None = None
    try:
        descriptor, temporary_name = tempfile.mkstemp(prefix=f".{output.name}.", suffix=".tmp", dir=output.parent)
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            descriptor = None
            json.dump(artifact, stream, ensure_ascii=False, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        Path(temporary_name).replace(output)
        temporary_name = None
    finally:
        if descriptor is not None:
            os.close(descriptor)
        if temporary_name is not None:
            with suppress(FileNotFoundError):
                Path(temporary_name).unlink()


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wheel", type=Path, required=True, help="one wheel whose members enter the raw union")
    parser.add_argument("--output", type=Path, required=True, help="external JSON output path")
    arguments = parser.parse_args(argv)
    source_root = Path.cwd().resolve()
    try:
        _reject_unsafe_output(source_root, arguments.output)
        artifact = _collect_artifact(source_root, arguments.wheel)
        _atomic_write(arguments.output, artifact)
    except SurfaceUnionDiscoveryError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
