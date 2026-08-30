#!/usr/bin/env python3
"""Collect deterministic, raw 0042-R2 repository-surface facts.

This command is a discovery boundary only.  It classifies a deliberately
limited set of tracked repository paths from an initial clean Git ``HEAD`` and
does not decide whether any path should be kept, moved, deleted, or retired.
Source content is read only from regular Git blobs and a matching Git archive;
the caller's worktree files are never used as collector input.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import io
import json
import os
import re
import subprocess
import sys
import tarfile
import tempfile
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

SCHEMA_VERSION = 1
_GIT_OBJECT_ID = re.compile(r"^[0-9a-f]{40,64}$")
_MARKDOWN_LINK = re.compile(r"\]\(([^)]+)\)")
_REFERENCE_TOKEN = re.compile(
    r"(?<![A-Za-z0-9_./-])(?:https?://[^\s<>()\[\]{}]+|"
    r"(?:fincore(?:\.[A-Za-z_][A-Za-z0-9_]*)+)|"
    r"(?:(?:docs|mkdocs_docs|examples|scripts|tests|\.github)/[A-Za-z0-9_./-]+))"
)
_WORKFLOW_USES = re.compile(r"(?:^|[-\s])uses:\s*([^\s#]+)")
_WORKFLOW_RUN = re.compile(r"(?:^|[-\s])run:\s*([^#\n]+)")
_HISTORICAL_FILENAME = re.compile(
    r"(?:^|[_-])(analysis|baseline|changelog|change_log|cleanup|history|historical|improvements|"
    r"migration|provenance|report|review|summary)(?:[_-]|\.|$)",
    re.IGNORECASE,
)
_HISTORICAL_ITERATION_DIRECTORY = re.compile(r"^docs/\d{4}[-_]")
_HISTORICAL_PATH_PREFIXES = (
    ".superpowers/sdd/",
    "_bmad-output/",
    "docs/plans/",
    "docs/quality/",
    "docs/architecture/adr/",
    "docs/迭代计划/",
)
_CATEGORY_ORDER = (
    "active_workflow",
    "packaging_release_script",
    "compat_generator_checker",
    "template",
    "type_stub",
    "historical_provenance_candidate",
    "example",
    "active_maintained_doc",
)
_PACKAGING_ROOT_PATHS = frozenset({"pyproject.toml", "setup.py", "MANIFEST.in"})
_PACKAGING_SCRIPT_MARKERS = frozenset(
    {
        "release",
        "installed_wheel",
        "dependency_matrix",
        "python_versions",
        "install_",
        "notices",
        "environment",
    }
)
_COMPAT_SCRIPT_MARKERS = frozenset({"compat", "alphalens", "api_diff", "public_api"})


class DiscoveryError(RuntimeError):
    """Raised when one source-consistent discovery artifact cannot be made."""


@dataclass(frozen=True)
class SourceBlob:
    """One selected regular Git blob from the initial source commit."""

    path: str
    git_mode: str
    git_blob_id: str
    blob_sha256: str
    category_tags: tuple[str, ...]
    classification_basis: tuple[dict[str, str], ...]


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


def _validate_repo_relative_path(path: str) -> None:
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


def _is_within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _git_control_paths(source_root: Path) -> tuple[Path, ...]:
    raw_paths = (
        source_root / ".git",
        Path(_git_text(source_root, "rev-parse", "--git-dir")),
        Path(_git_text(source_root, "rev-parse", "--git-common-dir")),
    )
    controls: list[Path] = []
    for raw_path in raw_paths:
        resolved = (raw_path if raw_path.is_absolute() else source_root / raw_path).resolve()
        if resolved not in controls:
            controls.append(resolved)
    return tuple(controls)


def _reject_unsafe_output(source_root: Path, output: Path) -> None:
    if _is_within(output, source_root):
        raise DiscoveryError("output must be outside the source worktree")
    for control_path in _git_control_paths(source_root):
        if _is_within(output, control_path):
            raise DiscoveryError("output must not target the Git control directory")


def _provenance(source_root: Path) -> dict[str, Any]:
    _require_worktree_root(source_root)
    dirty = _git_text(source_root, "status", "--porcelain=v1", "--untracked-files=all")
    if dirty:
        raise DiscoveryError("source Git worktree must be clean before repository-surface discovery")
    commit = _git_text(source_root, "rev-parse", "HEAD")
    tree = _git_text(source_root, "rev-parse", "HEAD^{tree}")
    if not _GIT_OBJECT_ID.fullmatch(commit) or not _GIT_OBJECT_ID.fullmatch(tree):
        raise DiscoveryError("source Git HEAD and tree must resolve to object identifiers")
    return {"commit": commit, "tree": tree, "clean": True}


def _verify_provenance(source_root: Path, initial: Mapping[str, Any]) -> None:
    if _provenance(source_root) != dict(initial):
        raise DiscoveryError("source Git provenance changed while repository-surface discovery was running")


def _is_historical_or_provenance_path(path: str) -> bool:
    name = PurePosixPath(path).name
    if path == "CHANGELOG.md" or path.startswith(_HISTORICAL_PATH_PREFIXES):
        return True
    if _HISTORICAL_ITERATION_DIRECTORY.match(path):
        return True
    if path.startswith("examples/") and ("/logs/" in path or "/output/" in path):
        return True
    return path == "docs/upstream-provenance.md" or (
        path.startswith("docs/") and bool(_HISTORICAL_FILENAME.search(name))
    )


def _is_document_path(path: str) -> bool:
    return (
        path in {"README.md", "CONTRIBUTING.md", "mkdocs.yml", ".readthedocs.yaml", ".github/PAGES_SETUP.md"}
        or path.startswith(("docs/", "mkdocs_docs/"))
    ) and PurePosixPath(path).suffix.lower() in {".md", ".txt", ".yml", ".yaml"}


def _is_template_path(path: str) -> bool:
    lower_path = path.lower()
    return "/templates/" in lower_path or "template" in PurePosixPath(lower_path).name


def _is_packaging_or_release_path(path: str) -> bool:
    if path in _PACKAGING_ROOT_PATHS:
        return True
    if path == ".github/workflows/publish.yml":
        return True
    if not path.startswith("scripts/"):
        return False
    lower_name = PurePosixPath(path).name.lower()
    return any(marker in lower_name for marker in _PACKAGING_SCRIPT_MARKERS)


def _is_compat_generator_or_checker(path: str) -> bool:
    if not path.startswith("scripts/") or PurePosixPath(path).suffix != ".py":
        return False
    lower_name = PurePosixPath(path).name.lower()
    return any(marker in lower_name for marker in _COMPAT_SCRIPT_MARKERS)


def _classify_path(path: str) -> tuple[tuple[str, ...], tuple[dict[str, str], ...]] | None:
    """Classify path evidence without inferring a lifecycle outcome."""
    _validate_repo_relative_path(path)
    tags: set[str] = set()
    basis: list[dict[str, str]] = []

    def add(tag: str, rule_id: str, evidence: str) -> None:
        tags.add(tag)
        basis.append({"rule_id": rule_id, "evidence": evidence})

    if path.startswith(".github/workflows/") and PurePosixPath(path).suffix.lower() in {".yml", ".yaml"}:
        add("active_workflow", "github_workflow_path", "tracked GitHub Actions workflow path")
    if _is_packaging_or_release_path(path):
        add(
            "packaging_release_script",
            "packaging_release_path",
            "tracked packaging or release configuration/script path",
        )
    if _is_compat_generator_or_checker(path):
        add(
            "compat_generator_checker",
            "compat_script_name",
            "script filename carries compatibility/API contract marker",
        )
    if path.endswith(".pyi") or path.endswith("/py.typed"):
        add("type_stub", "type_stub_or_marker_path", "tracked PEP 561 marker or stub path")
    if _is_template_path(path):
        add("template", "template_path", "tracked template filename or directory")

    historical = _is_historical_or_provenance_path(path)
    if historical:
        add(
            "historical_provenance_candidate",
            "historical_provenance_path",
            "path/name is a historical, generated-artifact, plan, quality, ADR, or provenance candidate",
        )
    if path.startswith("examples/") and not historical:
        add("example", "example_path", "tracked example source, configuration, or input path")
    if _is_document_path(path) and not historical and not _is_template_path(path):
        add(
            "active_maintained_doc",
            "maintained_document_path",
            "tracked documentation/configuration path without historical marker",
        )

    if not tags:
        return None
    ordered_tags = tuple(sorted(tags))
    ordered_basis = tuple(sorted(basis, key=lambda item: (item["rule_id"], item["evidence"])))
    return ordered_tags, ordered_basis


def _list_selected_regular_blobs(source_root: Path, commit: str) -> list[SourceBlob]:
    records = [record for record in _git_bytes(source_root, "ls-tree", "-r", "-z", commit).split(b"\0") if record]
    selected: list[SourceBlob] = []
    seen_paths: set[str] = set()
    for record in records:
        try:
            metadata, raw_path = record.split(b"\t", 1)
            git_mode, object_type, git_blob_id = metadata.decode("ascii").split()
            path = raw_path.decode("utf-8", errors="strict")
        except (UnicodeDecodeError, ValueError) as exc:
            raise DiscoveryError("cannot inspect initial HEAD repository tree") from exc
        classification = _classify_path(path)
        if classification is None:
            continue
        if path in seen_paths:
            raise DiscoveryError(f"initial HEAD contains duplicate selected path: {path}")
        if object_type != "blob" or git_mode not in {"100644", "100755"} or not _GIT_OBJECT_ID.fullmatch(git_blob_id):
            raise DiscoveryError(f"selected source must be a regular Git blob, not a link or tree: {path}")
        payload = _git_bytes(source_root, "show", f"{commit}:{path}")
        category_tags, classification_basis = classification
        selected.append(
            SourceBlob(
                path=path,
                git_mode=git_mode,
                git_blob_id=git_blob_id,
                blob_sha256=_sha256_bytes(payload),
                category_tags=category_tags,
                classification_basis=classification_basis,
            )
        )
        seen_paths.add(path)
    if not selected:
        raise DiscoveryError("initial HEAD contains no classified repository-surface blobs")
    return sorted(selected, key=lambda item: item.path)


def _archive_selected_payloads(source_root: Path, commit: str, blobs: Sequence[SourceBlob]) -> dict[str, bytes]:
    expected_paths = {blob.path for blob in blobs}
    archive = _git_bytes(source_root, "archive", "--format=tar", commit, "--", *sorted(expected_paths))
    payloads: dict[str, bytes] = {}
    try:
        with tarfile.open(fileobj=io.BytesIO(archive), mode="r:") as tar:
            for member in tar.getmembers():
                _validate_repo_relative_path(member.name)
                if member.isdir():
                    continue
                if member.name not in expected_paths:
                    raise DiscoveryError(f"initial HEAD archive contains unexpected selected source: {member.name}")
                if not member.isfile():
                    raise DiscoveryError(f"selected archive member must be regular: {member.name}")
                if member.name in payloads:
                    raise DiscoveryError(f"initial HEAD archive contains duplicate selected source: {member.name}")
                stream = tar.extractfile(member)
                if stream is None:
                    raise DiscoveryError(f"cannot read selected source from initial HEAD archive: {member.name}")
                payloads[member.name] = stream.read()
    except tarfile.TarError as exc:
        raise DiscoveryError("cannot read initial HEAD Git archive") from exc
    if set(payloads) != expected_paths:
        missing = sorted(expected_paths - set(payloads))
        unexpected = sorted(set(payloads) - expected_paths)
        detail = ", ".join([*(f"missing {path}" for path in missing), *(f"unexpected {path}" for path in unexpected)])
        raise DiscoveryError(f"initial HEAD Git archive does not match selected regular blob set: {detail}")
    return payloads


def _is_main_guard(node: ast.If) -> bool:
    test = node.test
    if not isinstance(test, ast.Compare) or len(test.ops) != 1 or len(test.comparators) != 1:
        return False
    left = test.left
    right = test.comparators[0]
    if not isinstance(test.ops[0], ast.Eq):
        return False
    return (
        isinstance(left, ast.Name)
        and left.id == "__name__"
        and isinstance(right, ast.Constant)
        and right.value == "__main__"
    ) or (
        isinstance(right, ast.Name)
        and right.id == "__name__"
        and isinstance(left, ast.Constant)
        and left.value == "__main__"
    )


def _python_import_tokens(text: str, path: str) -> tuple[list[str], list[str]]:
    try:
        tree = ast.parse(text, filename=path)
    except SyntaxError:
        return [], []
    imports: set[str] = set()
    executable: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            base = "." * node.level + (node.module or "")
            imports.update(f"{base}.{alias.name}" if base else alias.name for alias in node.names)
        elif isinstance(node, ast.If) and _is_main_guard(node):
            executable.add("python:__main__")
    return sorted(executable), sorted(imports)


def _workflow_executable_tokens(text: str) -> tuple[list[str], list[str]]:
    executable: set[str] = set()
    references: set[str] = set()
    for line in text.splitlines():
        uses_match = _WORKFLOW_USES.search(line)
        if uses_match:
            action = uses_match.group(1).strip()
            if action:
                executable.add(f"workflow:uses:{action}")
                references.add(f"action:{action}")
        run_match = _WORKFLOW_RUN.search(line)
        if run_match:
            command = run_match.group(1).strip()
            if command and command not in {"|", ">", ">-", "|-"}:
                executable.add(f"workflow:run:{command.split(maxsplit=1)[0]}")
    return sorted(executable), sorted(references)


def _shell_executable_tokens(text: str) -> list[str]:
    tokens: set[str] = set()
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith(("#", "::", "REM ")):
            continue
        command = stripped.split(maxsplit=1)[0]
        if command and not command.endswith(":"):
            tokens.add(f"shell:{command}")
    return sorted(tokens)


def _reference_tokens(text: str) -> list[str]:
    tokens = {match.group(0).rstrip(".,:;!?)]}") for match in _REFERENCE_TOKEN.finditer(text)}
    for match in _MARKDOWN_LINK.finditer(text):
        candidate = match.group(1).strip()
        if candidate and not candidate.startswith(("/", "#")) and ".." not in PurePosixPath(candidate).parts:
            tokens.add(candidate)
    return sorted(token for token in tokens if token)


def _token_facts(path: str, payload: bytes) -> dict[str, Any]:
    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError:
        return {"content_kind": "binary", "executable_tokens": [], "import_tokens": [], "reference_tokens": []}
    executable: set[str] = set()
    imports: set[str] = set()
    references = set(_reference_tokens(text))
    suffix = PurePosixPath(path).suffix.lower()
    if suffix in {".py", ".pyi"}:
        python_executable, python_imports = _python_import_tokens(text, path)
        executable.update(python_executable)
        imports.update(python_imports)
    if suffix in {".yml", ".yaml"} and path.startswith(".github/workflows/"):
        workflow_executable, workflow_references = _workflow_executable_tokens(text)
        executable.update(workflow_executable)
        references.update(workflow_references)
    if suffix in {".sh", ".bat"}:
        executable.update(_shell_executable_tokens(text))
    return {
        "content_kind": "text",
        "executable_tokens": sorted(executable),
        "import_tokens": sorted(imports),
        "reference_tokens": sorted(references),
    }


def _primary_kind(category_tags: Sequence[str]) -> str:
    for category in _CATEGORY_ORDER:
        if category in category_tags:
            return category
    raise DiscoveryError("selected source has no recognized category tag")


def _collect_artifact(source_root: Path) -> dict[str, Any]:
    initial = _provenance(source_root)
    blobs = _list_selected_regular_blobs(source_root, initial["commit"])
    payloads = _archive_selected_payloads(source_root, initial["commit"], blobs)
    records: list[dict[str, Any]] = []
    for blob in blobs:
        payload = payloads[blob.path]
        if _sha256_bytes(payload) != blob.blob_sha256:
            raise DiscoveryError(f"initial HEAD archive differs from regular Git blob: {blob.path}")
        records.append(
            {
                "path": blob.path,
                "git_mode": blob.git_mode,
                "git_blob_id": blob.git_blob_id,
                "blob_sha256": blob.blob_sha256,
                "kind": _primary_kind(blob.category_tags),
                "category_tags": list(blob.category_tags),
                "classification_basis": [dict(item) for item in blob.classification_basis],
                "token_facts": _token_facts(blob.path, payload),
            }
        )
    category_counts = {category: sum(category in blob.category_tags for blob in blobs) for category in _CATEGORY_ORDER}
    artifact = {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": "repository_surface_facts_discovery",
        "discovery_status": "partial",
        "not_for_d0": True,
        "partial_reason": (
            "This raw path-facts artifact records no maintenance or lifecycle decision and cannot be D0 evidence. "
            "It does not prove complete compatibility removal, runtime execution, installed-wheel behavior, "
            "or a complete repository inventory."
        ),
        "boundaries": {
            "included": [
                "tracked active workflow paths",
                "tracked packaging/release scripts and configuration",
                "tracked maintained-document/template path candidates",
                "tracked examples, type stubs, compatibility generators/checkers, and historical/provenance candidates",
            ],
            "excluded": "runtime execution, installed distributions, generic test nodes, source-module dependency analysis, "
            "human maintenance/lifecycle decisions, and untracked files",
        },
        "source_provenance": initial,
        "source_archive": {"path_scope": "classified repository paths", "verified_against_regular_blobs": True},
        "category_counts": category_counts,
        "record_count": len(records),
        "records": records,
    }
    _verify_provenance(source_root, initial)
    return artifact


def _atomic_write(output: Path, artifact: Mapping[str, Any]) -> None:
    serialized = json.dumps(artifact, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{output.name}.", suffix=".tmp", dir=output.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
            stream.write(serialized)
            stream.flush()
            os.fsync(stream.fileno())
        Path(temporary_name).replace(output)
    except Exception:
        with suppress(FileNotFoundError):
            Path(temporary_name).unlink()
        raise


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, required=True, help="external path for deterministic raw repository facts JSON"
    )
    arguments = parser.parse_args(argv)
    source_root = Path.cwd().resolve()
    output = arguments.output if arguments.output.is_absolute() else source_root / arguments.output
    output = output.resolve()
    try:
        _require_worktree_root(source_root)
        _reject_unsafe_output(source_root, output)
        artifact = _collect_artifact(source_root)
        _atomic_write(output, artifact)
    except DiscoveryError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    except (OSError, ValueError) as exc:
        print(f"error: repository-surface discovery failed closed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
