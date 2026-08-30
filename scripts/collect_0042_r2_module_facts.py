#!/usr/bin/env python3
"""Collect deterministic, raw 0042-R2 module facts from a clean Git HEAD.

This is deliberately a discovery artifact, not a module-disposition decision.
It reads every ``fincore/**/*.py`` source from an initial clean Git commit via
regular blobs and a Git archive, records static AST facts, and refuses to use
the caller's worktree files as input.
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
from collections import defaultdict
from collections.abc import Mapping, Sequence
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

SCHEMA_VERSION = 1
_GIT_OBJECT_ID = re.compile(r"^[0-9a-f]{40,64}$")
_DYNAMIC_IMPORT_CALLS = frozenset(
    {
        "__import__",
        "importlib.find_loader",
        "importlib.import_module",
        "importlib.util.find_spec",
    }
)


class DiscoveryError(RuntimeError):
    """Raised when source facts cannot be collected from one exact Git state."""


@dataclass(frozen=True)
class SourceModule:
    """A Python source blob read from the initial Git HEAD only."""

    path: str
    blob_sha256: str
    payload: bytes


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


def _provenance(source_root: Path) -> dict[str, Any]:
    _require_worktree_root(source_root)
    dirty = _git_text(source_root, "status", "--porcelain=v1", "--untracked-files=all")
    if dirty:
        raise DiscoveryError("source Git worktree must be clean before module-facts discovery")
    commit = _git_text(source_root, "rev-parse", "HEAD")
    tree = _git_text(source_root, "rev-parse", "HEAD^{tree}")
    if not _GIT_OBJECT_ID.fullmatch(commit) or not _GIT_OBJECT_ID.fullmatch(tree):
        raise DiscoveryError("source Git HEAD and tree must resolve to object identifiers")
    return {"commit": commit, "tree": tree, "clean": True}


def _verify_provenance(source_root: Path, initial: Mapping[str, Any]) -> None:
    final = _provenance(source_root)
    if final != dict(initial):
        raise DiscoveryError("source Git provenance changed while module-facts discovery was running")


def _list_regular_python_blobs(source_root: Path, commit: str) -> list[SourceModule]:
    records = [
        record
        for record in _git_bytes(source_root, "ls-tree", "-r", "-z", commit, "--", "fincore").split(b"\0")
        if record
    ]
    modules: list[SourceModule] = []
    seen_paths: set[str] = set()
    for record in records:
        try:
            metadata, raw_path = record.split(b"\t", 1)
            mode, object_type, object_id = metadata.decode("ascii").split()
            path = raw_path.decode("utf-8", errors="strict")
        except (UnicodeDecodeError, ValueError) as exc:
            raise DiscoveryError("cannot inspect initial HEAD Python source tree") from exc
        _validate_repo_relative_path(path)
        if not path.startswith("fincore/") or not path.endswith(".py"):
            continue
        if path in seen_paths:
            raise DiscoveryError(f"initial HEAD contains duplicate Python source path: {path}")
        if object_type != "blob" or not mode.startswith("100") or not _GIT_OBJECT_ID.fullmatch(object_id):
            raise DiscoveryError(f"Python source must be a regular Git blob, not a link or tree: {path}")
        payload = _git_bytes(source_root, "show", f"{commit}:{path}")
        modules.append(SourceModule(path=path, blob_sha256=_sha256_bytes(payload), payload=payload))
        seen_paths.add(path)
    if not modules:
        raise DiscoveryError("initial HEAD contains no fincore Python source blobs")
    return sorted(modules, key=lambda item: item.path)


def _archive_python_payloads(source_root: Path, commit: str, expected_paths: set[str]) -> dict[str, bytes]:
    """Read only expected Python files from a safe initial-HEAD Git archive."""
    archive = _git_bytes(source_root, "archive", "--format=tar", commit, "--", "fincore")
    payloads: dict[str, bytes] = {}
    try:
        with tarfile.open(fileobj=io.BytesIO(archive), mode="r:") as tar:
            for member in tar.getmembers():
                _validate_repo_relative_path(member.name)
                if member.isdir():
                    continue
                if not member.name.endswith(".py"):
                    continue
                if member.name not in expected_paths:
                    raise DiscoveryError(f"initial HEAD archive contains an unexpected Python source: {member.name}")
                if not member.isfile():
                    raise DiscoveryError(
                        f"Python source archive member must be regular, not a link or device: {member.name}"
                    )
                if member.name in payloads:
                    raise DiscoveryError(f"initial HEAD archive contains duplicate Python source: {member.name}")
                source = tar.extractfile(member)
                if source is None:
                    raise DiscoveryError(f"cannot read Python source from initial HEAD archive: {member.name}")
                payloads[member.name] = source.read()
    except tarfile.TarError as exc:
        raise DiscoveryError("cannot read initial HEAD Git archive") from exc
    if set(payloads) != expected_paths:
        missing = sorted(expected_paths - set(payloads))
        extra = sorted(set(payloads) - expected_paths)
        detail = ", ".join([*(f"missing {path}" for path in missing), *(f"unexpected {path}" for path in extra)])
        raise DiscoveryError(f"initial HEAD Git archive does not match regular Python blob set: {detail}")
    return payloads


def _module_name(path: str) -> tuple[str, bool]:
    _validate_repo_relative_path(path)
    parts = list(PurePosixPath(path).with_suffix("").parts)
    is_package = parts[-1] == "__init__"
    if is_package:
        parts.pop()
    if not parts or parts[0] != "fincore":
        raise DiscoveryError(f"Python source has no valid fincore module name: {path}")
    return ".".join(parts), is_package


def _dotted_name(node: ast.expr) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        base = _dotted_name(node.value)
        return f"{base}.{node.attr}" if base else None
    return None


def _import_aliases(tree: ast.Module) -> dict[str, str]:
    """Return lexical aliases as a conservative static aid for call facts."""
    aliases: dict[str, str] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for imported in node.names:
                bound_name = imported.asname or imported.name.split(".", 1)[0]
                aliases[bound_name] = imported.name if imported.asname else bound_name
        elif isinstance(node, ast.ImportFrom) and node.module:
            for imported in node.names:
                if imported.name == "*":
                    continue
                aliases[imported.asname or imported.name] = f"{node.module}.{imported.name}"
    return aliases


def _canonical_call_name(node: ast.expr, aliases: Mapping[str, str]) -> str | None:
    dotted = _dotted_name(node)
    if dotted is None:
        return None
    head, separator, tail = dotted.partition(".")
    replacement = aliases.get(head)
    if replacement is None:
        return dotted
    return f"{replacement}.{tail}" if separator else replacement


def _first_argument_fact(call: ast.Call) -> dict[str, str]:
    if not call.args:
        return {"kind": "missing", "value": ""}
    argument = call.args[0]
    if isinstance(argument, ast.Constant) and isinstance(argument.value, str):
        return {"kind": "literal_string", "value": argument.value}
    try:
        expression = ast.unparse(argument)
    except (TypeError, ValueError):
        expression = type(argument).__name__
    return {"kind": "expression", "value": expression}


def _dynamic_import_calls(tree: ast.Module) -> list[dict[str, Any]]:
    aliases = _import_aliases(tree)
    calls: list[dict[str, Any]] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        call_name = _canonical_call_name(node.func, aliases)
        if call_name not in _DYNAMIC_IMPORT_CALLS:
            continue
        calls.append(
            {
                "line": node.lineno,
                "column": node.col_offset,
                "call": call_name,
                "first_argument": _first_argument_fact(node),
            }
        )
    return sorted(calls, key=lambda item: (item["line"], item["column"], item["call"]))


def _static_string_sequence(node: ast.expr) -> tuple[list[str], bool]:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return [node.value], False
    if not isinstance(node, ast.List | ast.Tuple | ast.Set):
        return [], True
    values: list[str] = []
    has_dynamic_parts = False
    for item in node.elts:
        if isinstance(item, ast.Constant) and isinstance(item.value, str):
            values.append(item.value)
        else:
            has_dynamic_parts = True
    return values, has_dynamic_parts


def _module_all_assignments(tree: ast.Module) -> list[dict[str, Any]]:
    assignments: list[dict[str, Any]] = []
    for node in tree.body:
        target: ast.expr | None = None
        value: ast.expr | None = None
        assignment_kind: str | None = None
        if isinstance(node, ast.Assign):
            target = next((item for item in node.targets if isinstance(item, ast.Name) and item.id == "__all__"), None)
            value = node.value
            assignment_kind = "assign"
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name) and node.target.id == "__all__":
            target = node.target
            value = node.value
            assignment_kind = "ann_assign"
        elif isinstance(node, ast.AugAssign) and isinstance(node.target, ast.Name) and node.target.id == "__all__":
            target = node.target
            value = node.value
            assignment_kind = "aug_assign"
        if target is None or value is None or assignment_kind is None:
            continue
        static_values, has_dynamic_parts = _static_string_sequence(value)
        assignments.append(
            {
                "line": node.lineno,
                "assignment_kind": assignment_kind,
                "expression_kind": type(value).__name__,
                "static_values": static_values,
                "has_dynamic_parts": has_dynamic_parts,
            }
        )
    return assignments


def _is_type_checking_test(node: ast.expr) -> bool:
    return (isinstance(node, ast.Name) and node.id == "TYPE_CHECKING") or (
        isinstance(node, ast.Attribute) and node.attr == "TYPE_CHECKING" and _dotted_name(node.value) == "typing"
    )


def _package_name(module_name: str, is_package: bool) -> str:
    return module_name if is_package else module_name.rpartition(".")[0]


def _resolve_relative_base(module_name: str, is_package: bool, level: int, module: str | None) -> str | None:
    package_parts = _package_name(module_name, is_package).split(".")
    parent_count = level - 1
    if parent_count >= len(package_parts):
        return None
    base_parts = package_parts[: len(package_parts) - parent_count]
    if module:
        base_parts.extend(module.split("."))
    return ".".join(base_parts)


def _sorted_targets(targets: set[str]) -> list[str]:
    return sorted(target for target in targets if target.startswith("fincore"))


def _import_fact(
    node: ast.Import | ast.ImportFrom,
    *,
    module_name: str,
    is_package: bool,
    known_modules: set[str],
) -> dict[str, Any]:
    if isinstance(node, ast.Import):
        names = [{"name": item.name, "asname": item.asname} for item in node.names]
        definite_targets = {item.name for item in node.names if item.name in known_modules}
        category = (
            "absolute_internal"
            if any(item.name == "fincore" or item.name.startswith("fincore.") for item in node.names)
            else "external_or_unknown"
        )
        return {
            "kind": "import",
            "line": node.lineno,
            "column": node.col_offset,
            "module": None,
            "level": 0,
            "names": names,
            "resolution": {
                "category": category,
                "base_module": None,
                "definite_internal_targets": _sorted_targets(definite_targets),
                "candidate_internal_targets": [],
            },
        }

    names = [{"name": item.name, "asname": item.asname} for item in node.names]
    if node.level:
        base_module = _resolve_relative_base(module_name, is_package, node.level, node.module)
        category = "relative_internal" if base_module and base_module.startswith("fincore") else "relative_escaped"
    else:
        base_module = node.module
        category = (
            "absolute_internal"
            if base_module == "fincore" or (isinstance(base_module, str) and base_module.startswith("fincore."))
            else "external_or_unknown"
        )
    definite_targets = {base_module} if base_module in known_modules else set()
    candidate_targets = {
        f"{base_module}.{item.name}"
        for item in node.names
        if base_module and item.name != "*" and f"{base_module}.{item.name}" in known_modules
    }
    return {
        "kind": "from_import",
        "line": node.lineno,
        "column": node.col_offset,
        "module": node.module,
        "level": node.level,
        "names": names,
        "resolution": {
            "category": category,
            "base_module": base_module,
            "definite_internal_targets": _sorted_targets(definite_targets),
            "candidate_internal_targets": _sorted_targets(candidate_targets),
        },
    }


def _ast_facts(tree: ast.Module, module_name: str, is_package: bool, known_modules: set[str]) -> dict[str, Any]:
    imports = [
        _import_fact(node, module_name=module_name, is_package=is_package, known_modules=known_modules)
        for node in ast.walk(tree)
        if isinstance(node, ast.Import | ast.ImportFrom)
    ]
    imports.sort(key=lambda item: (item["line"], item["column"], item["kind"], str(item["module"])))
    type_checking_lines = sorted(
        node.lineno for node in ast.walk(tree) if isinstance(node, ast.If) and _is_type_checking_test(node.test)
    )
    return {"imports": imports, "type_checking_lines": type_checking_lines}


def _module_getattr_lines(tree: ast.Module) -> list[int]:
    return sorted(
        node.lineno
        for node in tree.body
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef) and node.name == "__getattr__"
    )


def _parse_module(module: SourceModule, known_modules: set[str]) -> dict[str, Any]:
    try:
        tree = ast.parse(module.payload.decode("utf-8"), filename=module.path)
    except (UnicodeDecodeError, SyntaxError) as exc:
        raise DiscoveryError(f"cannot parse initial HEAD Python source: {module.path}") from exc
    module_name, is_package = _module_name(module.path)
    return {
        "path": module.path,
        "module_name": module_name,
        "is_package": is_package,
        "blob_sha256": module.blob_sha256,
        "ast_facts": _ast_facts(tree, module_name, is_package, known_modules),
        "risk_facts": {
            "dynamic_import_calls": _dynamic_import_calls(tree),
            "all_assignments": _module_all_assignments(tree),
            "module_getattr_lines": _module_getattr_lines(tree),
        },
    }


def _static_consumers(modules: Sequence[Mapping[str, Any]]) -> dict[str, list[str]]:
    module_paths = {str(module["module_name"]): str(module["path"]) for module in modules}
    consumers: dict[str, set[str]] = defaultdict(set)
    for module in modules:
        consumer_path = str(module["path"])
        imports = module["ast_facts"]["imports"]
        if not isinstance(imports, list):
            raise DiscoveryError(f"module AST import facts are malformed: {consumer_path}")
        for import_fact in imports:
            if not isinstance(import_fact, Mapping):
                raise DiscoveryError(f"module AST import fact is malformed: {consumer_path}")
            resolution = import_fact.get("resolution")
            if not isinstance(resolution, Mapping):
                raise DiscoveryError(f"module AST import resolution is malformed: {consumer_path}")
            targets = [
                *resolution.get("definite_internal_targets", []),
                *resolution.get("candidate_internal_targets", []),
            ]
            for target in targets:
                if not isinstance(target, str) or target not in module_paths:
                    continue
                if module_paths[target] != consumer_path:
                    consumers[target].add(consumer_path)
    return {module_name: sorted(consumers[module_name]) for module_name in module_paths}


def _collect_artifact(source_root: Path) -> dict[str, Any]:
    initial = _provenance(source_root)
    blob_modules = _list_regular_python_blobs(source_root, initial["commit"])
    archive_payloads = _archive_python_payloads(
        source_root, initial["commit"], {module.path for module in blob_modules}
    )
    source_modules: list[SourceModule] = []
    for module in blob_modules:
        archive_payload = archive_payloads[module.path]
        if _sha256_bytes(archive_payload) != module.blob_sha256 or archive_payload != module.payload:
            raise DiscoveryError(f"initial HEAD archive differs from regular Git blob: {module.path}")
        source_modules.append(SourceModule(module.path, module.blob_sha256, archive_payload))

    module_names = {_module_name(module.path)[0] for module in source_modules}
    modules = [_parse_module(module, module_names) for module in source_modules]
    modules.sort(key=lambda item: item["path"])
    consumers = _static_consumers(modules)
    for module in modules:
        paths = consumers[str(module["module_name"])]
        module["static_consumer_paths"] = paths
        module["static_consumer_count"] = len(paths)

    artifact = {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": "module_facts_discovery",
        "discovery_status": "partial",
        "not_for_d0": True,
        "partial_reason": (
            "This source-only static artifact records no human module disposition and cannot be D0 evidence. "
            "It excludes maintained docs, examples, benchmarks, built distributions, runtime plugin discovery, "
            "and test-node collection."
        ),
        "source_provenance": initial,
        "source_archive": {"path_prefix": "fincore", "verified_against_regular_blobs": True},
        "module_count": len(modules),
        "modules": modules,
    }
    _verify_provenance(source_root, initial)
    return artifact


def _atomic_write(output: Path, artifact: Mapping[str, Any]) -> None:
    serialized = json.dumps(artifact, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    output.parent.mkdir(parents=True, exist_ok=True)
    file_descriptor, temporary_name = tempfile.mkstemp(prefix=f".{output.name}.", suffix=".tmp", dir=output.parent)
    try:
        with os.fdopen(file_descriptor, "w", encoding="utf-8", newline="\n") as stream:
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
    parser.add_argument("--output", type=Path, required=True, help="path for deterministic raw module-facts JSON")
    arguments = parser.parse_args(argv)
    source_root = Path.cwd().resolve()
    output = arguments.output if arguments.output.is_absolute() else source_root / arguments.output
    output = output.resolve()
    try:
        _require_worktree_root(source_root)
        artifact = _collect_artifact(source_root)
        _atomic_write(output, artifact)
    except DiscoveryError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    except (OSError, ValueError) as exc:
        print(f"error: module-facts discovery failed closed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
