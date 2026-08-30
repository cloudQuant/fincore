#!/usr/bin/env python3
"""Measure and validate a fail-closed 0042-R2 architecture contract.

This tool deliberately performs static source measurement only.  A successful
``--capture`` produces a deterministic ``measurement_only`` artifact, not a
D0/D-TECH/Task 8 verdict.  The source must be a clean Git worktree and every
Python input is read from regular blobs in the initially observed ``HEAD``
tree.  Capture outputs must be outside that source worktree.

The resulting schema records physical and logical LOC, normalized function
body fingerprints, internal static import edges/cycles, and unguarded imports
of optional dependencies.  An optional baseline comparison is intentionally
strict: a baseline must use this schema, identify itself as frozen, match the
current platform, and provide every threshold.  Missing, pending, malformed,
or platform-incompatible baselines fail closed.

``--require-legacy-zero`` is intentionally unavailable in this generic Task 0
collector.  Legacy surface removal needs a separately frozen contract and must
not be inferred from these source measurements.
"""

from __future__ import annotations

import argparse
import ast
import copy
import hashlib
import json
import os
import platform
import re
import subprocess
import sys
import tempfile
import tokenize
from collections import defaultdict
from contextlib import suppress
from dataclasses import dataclass
from io import StringIO
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping, Sequence
    from typing import Any

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python 3.11 is the supported contract.
    tomllib = None  # type: ignore[assignment]


SCHEMA_VERSION = 1
ARTIFACT_TYPE = "fincore_0042_r2_architecture_measurement"
MEASUREMENT_CONTRACT_VERSION = 1
_GIT_OBJECT_ID = re.compile(r"^[0-9a-f]{40,64}$")
_PACKAGE_NAME = re.compile(r"^[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*$")
_OPTIONAL_DEPENDENCY_NAME = re.compile(r"^([A-Za-z0-9][A-Za-z0-9_.-]*)")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_THRESHOLD_TO_SUMMARY = {
    "max_duplicate_body_occurrences": "duplicate_body_occurrences",
    "max_implementation_fingerprint_count": "implementation_fingerprint_count",
    "max_internal_cycle_count": "internal_cycle_count",
    "max_logical_loc": "logical_loc",
    "max_optional_import_leakage_count": "optional_import_leakage_count",
    "max_physical_loc": "physical_loc",
}
_MEASUREMENT_COLLECTION_KEYS = frozenset(
    {
        "files",
        "implementation_fingerprints",
        "internal_import_graph",
        "normalized_ast_duplication",
        "optional_import_leakage",
        "optional_import_policy",
        "optional_imports",
        "summary",
    }
)
_LOGICAL_IGNORED_TOKENS = frozenset(
    {
        tokenize.COMMENT,
        tokenize.DEDENT,
        tokenize.ENDMARKER,
        tokenize.INDENT,
        tokenize.NL,
        tokenize.NEWLINE,
    }
)


class ArchitectureContractError(RuntimeError):
    """Raised when architecture measurement cannot produce trusted evidence."""


@dataclass(frozen=True)
class SourceModule:
    """One regular Python blob from the immutable source tree."""

    path: str
    module: str
    is_package: bool
    git_object_id: str
    sha256: str
    payload: bytes
    tree: ast.Module


@dataclass(frozen=True)
class ImportOccurrence:
    """A static or literal dynamic import observed in one source module."""

    source_module: str
    path: str
    line: int
    kind: str
    target: str
    level: int
    names: tuple[str, ...]
    optional_guarded: bool


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _non_empty_string(value: object) -> bool:
    return isinstance(value, str) and bool(value.strip())


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
        raise ArchitectureContractError(f"cannot inspect source Git worktree: {exc}") from exc
    if result.returncode != 0:
        detail = (result.stderr or result.stdout).decode("utf-8", errors="replace").strip()
        raise ArchitectureContractError(f"cannot inspect source Git worktree: {detail or 'git command failed'}")
    return result.stdout


def _git_text(source_root: Path, *arguments: str) -> str:
    try:
        return _git_bytes(source_root, *arguments).decode("utf-8", errors="strict").strip()
    except UnicodeDecodeError as exc:
        raise ArchitectureContractError("Git metadata must be UTF-8 text") from exc


def _is_within(candidate: Path, root: Path) -> bool:
    try:
        candidate.relative_to(root)
    except ValueError:
        return False
    return True


def _validate_repo_relative_path(path: str) -> None:
    pure_path = PurePosixPath(path)
    if (
        not path
        or path != str(pure_path)
        or pure_path.is_absolute()
        or "\\" in path
        or any(part in {"", ".", ".."} for part in pure_path.parts)
    ):
        raise ArchitectureContractError(f"source path must be a repository-relative POSIX path: {path!r}")


def _source_provenance(source_root: Path) -> tuple[Path, dict[str, Any]]:
    try:
        top_level = Path(_git_text(source_root, "rev-parse", "--show-toplevel")).resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise ArchitectureContractError("source-root must identify an existing Git worktree root") from exc
    if source_root.resolve(strict=True) != top_level:
        raise ArchitectureContractError("source-root must be the Git worktree root, not a descendant")
    dirty = _git_text(source_root, "status", "--porcelain=v1", "--untracked-files=all")
    if dirty:
        raise ArchitectureContractError("source Git worktree must be clean before architecture measurement")
    commit = _git_text(source_root, "rev-parse", "HEAD")
    tree = _git_text(source_root, "rev-parse", "HEAD^{tree}")
    if not _GIT_OBJECT_ID.fullmatch(commit) or not _GIT_OBJECT_ID.fullmatch(tree):
        raise ArchitectureContractError("source Git HEAD and tree must resolve to object identifiers")
    return top_level, {
        "clean": True,
        "commit": commit,
        "platform": _platform_provenance(),
        "tree": tree,
    }


def _verify_source_provenance(source_root: Path, expected: Mapping[str, Any]) -> None:
    _, actual = _source_provenance(source_root)
    if actual != dict(expected):
        raise ArchitectureContractError("source Git provenance changed while architecture measurement was running")


def _platform_provenance() -> dict[str, str]:
    return {
        "machine": platform.machine(),
        "python_implementation": platform.python_implementation(),
        "python_version": platform.python_version(),
        "system": platform.system(),
    }


def _validate_package_name(package: str) -> str:
    if not _PACKAGE_NAME.fullmatch(package):
        raise ArchitectureContractError("package must be a dotted Python package name")
    return package


def _package_path(package: str) -> str:
    return package.replace(".", "/")


def _module_name(package: str, path: str) -> tuple[str, bool]:
    _validate_repo_relative_path(path)
    package_parts = _package_path(package).split("/")
    parts = list(PurePosixPath(path).with_suffix("").parts)
    if parts[: len(package_parts)] != package_parts:
        raise ArchitectureContractError(f"source path is outside requested package: {path}")
    is_package = parts[-1] == "__init__"
    if is_package:
        parts.pop()
    if not parts:
        raise ArchitectureContractError(f"source path has no valid module name: {path}")
    return ".".join(parts), is_package


def _parse_tree(payload: bytes, path: str) -> ast.Module:
    try:
        source = payload.decode("utf-8", errors="strict")
    except UnicodeDecodeError as exc:
        raise ArchitectureContractError(f"Python source must be UTF-8: {path}") from exc
    try:
        tree = ast.parse(source, filename=path, mode="exec", type_comments=False)
    except (SyntaxError, ValueError) as exc:
        raise ArchitectureContractError(f"cannot parse Python source {path}: {exc}") from exc
    if not isinstance(tree, ast.Module):  # pragma: no cover - ast.parse mode='exec' guarantees this.
        raise ArchitectureContractError(f"Python source did not parse as a module: {path}")
    return tree


def _list_source_modules(source_root: Path, commit: str, package: str) -> list[SourceModule]:
    package_path = _package_path(package)
    raw_records = _git_bytes(source_root, "ls-tree", "-r", "-z", commit, "--", package_path).split(b"\0")
    modules: list[SourceModule] = []
    module_paths: dict[str, str] = {}
    seen_paths: set[str] = set()
    for raw_record in raw_records:
        if not raw_record:
            continue
        try:
            metadata, raw_path = raw_record.split(b"\t", 1)
            mode, object_type, object_id = metadata.decode("ascii", errors="strict").split()
            path = raw_path.decode("utf-8", errors="strict")
        except (UnicodeDecodeError, ValueError) as exc:
            raise ArchitectureContractError("cannot inspect initial HEAD Python source tree") from exc
        _validate_repo_relative_path(path)
        if not path.startswith(f"{package_path}/") or not path.endswith(".py"):
            continue
        if path in seen_paths:
            raise ArchitectureContractError(f"initial HEAD contains duplicate Python source path: {path}")
        if object_type != "blob" or not mode.startswith("100") or not _GIT_OBJECT_ID.fullmatch(object_id):
            raise ArchitectureContractError(f"Python source must be a regular Git blob, not a link or tree: {path}")
        payload = _git_bytes(source_root, "cat-file", "blob", object_id)
        module, is_package = _module_name(package, path)
        existing_path = module_paths.get(module)
        if existing_path is not None:
            raise ArchitectureContractError(
                f"multiple source paths resolve to module {module}: {existing_path}, {path}"
            )
        modules.append(
            SourceModule(
                path=path,
                module=module,
                is_package=is_package,
                git_object_id=object_id,
                sha256=_sha256_bytes(payload),
                payload=payload,
                tree=_parse_tree(payload, path),
            )
        )
        module_paths[module] = path
        seen_paths.add(path)
    if not modules:
        raise ArchitectureContractError(f"initial HEAD contains no Python source blobs for package {package!r}")
    if package not in module_paths:
        raise ArchitectureContractError(f"requested package {package!r} must contain an __init__.py regular blob")
    return sorted(modules, key=lambda item: item.path)


def _logical_loc(payload: bytes, path: str) -> int:
    try:
        source = payload.decode("utf-8", errors="strict")
        tokens = tokenize.generate_tokens(StringIO(source).readline)
        lines = {
            line_number
            for token in tokens
            if token.type not in _LOGICAL_IGNORED_TOKENS
            for line_number in range(token.start[0], token.end[0] + 1)
        }
    except (UnicodeDecodeError, tokenize.TokenError) as exc:
        raise ArchitectureContractError(f"cannot tokenize Python source {path}: {exc}") from exc
    return len(lines)


def _physical_loc(payload: bytes, path: str) -> int:
    try:
        return len(payload.decode("utf-8", errors="strict").splitlines())
    except UnicodeDecodeError as exc:
        raise ArchitectureContractError(f"Python source must be UTF-8: {path}") from exc


def _dependency_root(value: str) -> str | None:
    match = _OPTIONAL_DEPENDENCY_NAME.match(value.strip())
    if match is None:
        return None
    name = match.group(1).split(".", 1)[0]
    return name.replace("-", "_").casefold()


def _read_head_optional_dependency_roots(source_root: Path, commit: str) -> list[str]:
    if tomllib is None:  # pragma: no cover - Python 3.11 is the project support floor.
        raise ArchitectureContractError("Python 3.11 tomllib is required for optional import policy discovery")
    records = [
        record
        for record in _git_bytes(source_root, "ls-tree", "-z", commit, "--", "pyproject.toml").split(b"\0")
        if record
    ]
    if not records:
        return []
    if len(records) != 1:
        raise ArchitectureContractError("initial HEAD contains multiple pyproject.toml records")
    try:
        metadata, raw_path = records[0].split(b"\t", 1)
        mode, object_type, object_id = metadata.decode("ascii", errors="strict").split()
        path = raw_path.decode("utf-8", errors="strict")
    except (UnicodeDecodeError, ValueError) as exc:
        raise ArchitectureContractError("cannot inspect initial HEAD pyproject.toml") from exc
    if (
        path != "pyproject.toml"
        or object_type != "blob"
        or not mode.startswith("100")
        or not _GIT_OBJECT_ID.fullmatch(object_id)
    ):
        raise ArchitectureContractError("initial HEAD pyproject.toml must be a regular Git blob")
    payload = _git_bytes(source_root, "cat-file", "blob", object_id)
    try:
        document = tomllib.loads(payload.decode("utf-8", errors="strict"))
    except (UnicodeDecodeError, tomllib.TOMLDecodeError) as exc:
        raise ArchitectureContractError("pyproject.toml from initial HEAD must be valid UTF-8 TOML") from exc
    project = document.get("project")
    if not isinstance(project, dict):
        return []
    optional_dependencies = project.get("optional-dependencies")
    if optional_dependencies is None:
        return []
    if not isinstance(optional_dependencies, dict):
        raise ArchitectureContractError("project.optional-dependencies must be a TOML table")
    roots: set[str] = set()
    for extra, dependencies in optional_dependencies.items():
        if not _non_empty_string(extra) or not isinstance(dependencies, list):
            raise ArchitectureContractError("project.optional-dependencies entries must be named lists")
        for dependency in dependencies:
            if not _non_empty_string(dependency):
                raise ArchitectureContractError("optional dependency names must be non-empty strings")
            root = _dependency_root(str(dependency))
            if root is None:
                raise ArchitectureContractError(f"cannot derive optional import root from dependency {dependency!r}")
            roots.add(root)
    return sorted(roots)


def _optional_import_policy(source_root: Path, commit: str, explicit_modules: Iterable[str]) -> dict[str, list[str]]:
    explicit_roots: set[str] = set()
    for module in explicit_modules:
        if not _PACKAGE_NAME.fullmatch(module):
            raise ArchitectureContractError("--optional-module must be a dotted Python import name")
        root = _dependency_root(module)
        if root is None:  # pragma: no cover - package-name validation makes this unreachable.
            raise ArchitectureContractError("--optional-module must have an import root")
        explicit_roots.add(root)
    derived_roots = _read_head_optional_dependency_roots(source_root, commit)
    return {
        "derived_from_pyproject": derived_roots,
        "effective_module_roots": sorted(set(derived_roots) | explicit_roots),
        "explicit_module_roots": sorted(explicit_roots),
    }


def _except_catches_import_error(handler: ast.ExceptHandler) -> bool:
    exception = handler.type
    if exception is None:
        return True
    if isinstance(exception, ast.Name):
        return exception.id in {"ImportError", "ModuleNotFoundError"}
    if isinstance(exception, ast.Tuple):
        return any(
            isinstance(element, ast.Name) and element.id in {"ImportError", "ModuleNotFoundError"}
            for element in exception.elts
        )
    return False


def _dotted_expression_name(node: ast.expr) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        parent = _dotted_expression_name(node.value)
        return f"{parent}.{node.attr}" if parent else None
    return None


class _ImportCollector(ast.NodeVisitor):
    """Collect imports while preserving whether an ImportError guard encloses them."""

    def __init__(self, module: SourceModule) -> None:
        self._module = module
        self._optional_guard_depth = 0
        self.occurrences: list[ImportOccurrence] = []

    def _add(self, *, line: int, kind: str, target: str, level: int = 0, names: Iterable[str] = ()) -> None:
        self.occurrences.append(
            ImportOccurrence(
                source_module=self._module.module,
                path=self._module.path,
                line=line,
                kind=kind,
                target=target,
                level=level,
                names=tuple(names),
                optional_guarded=self._optional_guard_depth > 0,
            )
        )

    def visit_Try(self, node: ast.Try) -> None:
        guarded = any(_except_catches_import_error(handler) for handler in node.handlers)
        if not guarded:
            self.generic_visit(node)
            return
        self._optional_guard_depth += 1
        try:
            for statement in [*node.body, *node.orelse, *node.finalbody]:
                self.visit(statement)
            for handler in node.handlers:
                for statement in handler.body:
                    self.visit(statement)
        finally:
            self._optional_guard_depth -= 1

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            self._add(line=node.lineno, kind="import", target=alias.name)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        self._add(
            line=node.lineno,
            kind="from_import",
            target=node.module or "",
            level=node.level,
            names=(alias.name for alias in node.names),
        )

    def visit_Call(self, node: ast.Call) -> None:
        function_name = _dotted_expression_name(node.func)
        if function_name in {"__import__", "importlib.import_module"} and node.args:
            first_argument = node.args[0]
            if isinstance(first_argument, ast.Constant) and isinstance(first_argument.value, str):
                self._add(line=node.lineno, kind="dynamic_import", target=first_argument.value)
        self.generic_visit(node)


def _collect_imports(modules: Iterable[SourceModule]) -> list[ImportOccurrence]:
    occurrences: list[ImportOccurrence] = []
    for module in modules:
        collector = _ImportCollector(module)
        collector.visit(module.tree)
        occurrences.extend(collector.occurrences)
    return sorted(
        occurrences,
        key=lambda item: (item.path, item.line, item.kind, item.target, item.level, item.names, item.optional_guarded),
    )


def _relative_import_base(source: SourceModule, level: int) -> str | None:
    current_package = source.module if source.is_package else source.module.rpartition(".")[0]
    parts = current_package.split(".") if current_package else []
    if level <= 0:
        return None
    strip_count = level - 1
    if strip_count >= len(parts):
        return None
    return ".".join(parts[: len(parts) - strip_count])


def _internal_import_targets(occurrence: ImportOccurrence, source_by_module: Mapping[str, SourceModule]) -> list[str]:
    modules = set(source_by_module)
    if occurrence.kind in {"import", "dynamic_import"}:
        return [occurrence.target] if occurrence.target in modules else []
    source = source_by_module[occurrence.source_module]
    if occurrence.level:
        base = _relative_import_base(source, occurrence.level)
        if base is None:
            return []
        target = f"{base}.{occurrence.target}" if occurrence.target else base
    else:
        target = occurrence.target
    if not target:
        return []
    candidates: set[str] = set()
    if occurrence.target and target in modules and target != source.module.split(".")[0]:
        candidates.add(target)
    for name in occurrence.names:
        if name == "*":
            continue
        child = f"{target}.{name}"
        if child in modules:
            candidates.add(child)
    return sorted(candidates)


def _internal_import_graph(modules: Iterable[SourceModule], occurrences: Iterable[ImportOccurrence]) -> dict[str, Any]:
    source_by_module = {module.module: module for module in modules}
    edge_facts: dict[tuple[str, str], dict[str, set[Any]]] = defaultdict(lambda: {"kinds": set(), "lines": set()})
    for occurrence in occurrences:
        for target in _internal_import_targets(occurrence, source_by_module):
            fact = edge_facts[(occurrence.source_module, target)]
            fact["kinds"].add(occurrence.kind)
            fact["lines"].add(occurrence.line)
    edges: list[dict[str, Any]] = [
        {
            "from": source,
            "kinds": sorted(str(kind) for kind in facts["kinds"]),
            "lines": sorted(int(line) for line in facts["lines"]),
            "to": target,
        }
        for (source, target), facts in sorted(edge_facts.items())
    ]
    nodes = sorted(source_by_module)
    adjacency: dict[str, set[str]] = {node: set() for node in nodes}
    for source, target in edge_facts:
        adjacency[source].add(target)
    cycles = _strongly_connected_components(nodes, adjacency)
    return {"cycles": cycles, "edges": edges, "nodes": nodes}


def _strongly_connected_components(nodes: Iterable[str], adjacency: Mapping[str, set[str]]) -> list[list[str]]:
    index = 0
    indexes: dict[str, int] = {}
    lowlinks: dict[str, int] = {}
    stack: list[str] = []
    on_stack: set[str] = set()
    components: list[list[str]] = []

    def visit(node: str) -> None:
        nonlocal index
        indexes[node] = index
        lowlinks[node] = index
        index += 1
        stack.append(node)
        on_stack.add(node)
        for target in sorted(adjacency[node]):
            if target not in indexes:
                visit(target)
                lowlinks[node] = min(lowlinks[node], lowlinks[target])
            elif target in on_stack:
                lowlinks[node] = min(lowlinks[node], indexes[target])
        if lowlinks[node] != indexes[node]:
            return
        component: list[str] = []
        while stack:
            member = stack.pop()
            on_stack.remove(member)
            component.append(member)
            if member == node:
                break
        if len(component) > 1 or node in adjacency[node]:
            components.append(sorted(component))

    for node in sorted(nodes):
        if node not in indexes:
            visit(node)
    return sorted(components)


def _import_root(occurrence: ImportOccurrence) -> str | None:
    if occurrence.level:
        return None
    target = occurrence.target.split(".", 1)[0]
    return target.replace("-", "_").casefold() if target else None


def _optional_import_facts(
    occurrences: Iterable[ImportOccurrence], policy: Mapping[str, list[str]]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    roots = set(policy["effective_module_roots"])
    all_optional: list[dict[str, Any]] = []
    leakage: list[dict[str, Any]] = []
    for occurrence in occurrences:
        imported_root = _import_root(occurrence)
        if imported_root not in roots:
            continue
        fact = {
            "guarded": occurrence.optional_guarded,
            "import_kind": occurrence.kind,
            "imported_root": imported_root,
            "line": occurrence.line,
            "optional_root": imported_root,
            "path": occurrence.path,
            "source_module": occurrence.source_module,
        }
        all_optional.append(fact)
        if not occurrence.optional_guarded:
            leakage.append(fact)
    return all_optional, leakage


def _is_docstring(statement: ast.stmt) -> bool:
    return (
        isinstance(statement, ast.Expr)
        and isinstance(statement.value, ast.Constant)
        and isinstance(statement.value.value, str)
    )


class _LocalNameCollector(ast.NodeVisitor):
    """Collect a function's local bindings in lexical encounter order."""

    def __init__(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
        self.names: list[str] = []
        self._seen: set[str] = set()
        for argument in [*node.args.posonlyargs, *node.args.args, *node.args.kwonlyargs]:
            self._add(argument.arg)
        if node.args.vararg is not None:
            self._add(node.args.vararg.arg)
        if node.args.kwarg is not None:
            self._add(node.args.kwarg.arg)
        for statement in node.body:
            self.visit(statement)

    def _add(self, name: str) -> None:
        if name not in self._seen:
            self._seen.add(name)
            self.names.append(name)

    def visit_Name(self, node: ast.Name) -> None:
        if isinstance(node.ctx, (ast.Store, ast.Del)):
            self._add(node.id)

    def visit_arg(self, node: ast.arg) -> None:
        self._add(node.arg)

    def visit_ExceptHandler(self, node: ast.ExceptHandler) -> None:
        if node.name:
            self._add(node.name)
        self.generic_visit(node)

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            self._add(alias.asname or alias.name.split(".", 1)[0])

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        for alias in node.names:
            if alias.name != "*":
                self._add(alias.asname or alias.name)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        return

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        return

    def visit_Lambda(self, node: ast.Lambda) -> None:
        return

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        return


class _BodyNormalizer(ast.NodeTransformer):
    """Normalize local names/literals while preserving semantic AST structure."""

    def __init__(self, local_names: Iterable[str]) -> None:
        self._local_names = {name: f"local_{index}" for index, name in enumerate(local_names)}

    def visit_Name(self, node: ast.Name) -> ast.AST:
        normalized = self._local_names.get(node.id)
        if normalized is None:
            return node
        return ast.copy_location(ast.Name(id=normalized, ctx=node.ctx), node)

    def visit_arg(self, node: ast.arg) -> ast.AST:
        normalized = self._local_names.get(node.arg)
        if normalized is None:
            return node
        replacement = copy.copy(node)
        replacement.arg = normalized
        return ast.copy_location(replacement, node)

    def visit_Constant(self, node: ast.Constant) -> ast.AST:
        replacement = ast.Constant(value=f"<{type(node.value).__name__}>")
        return ast.copy_location(replacement, node)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.AST:
        return ast.copy_location(ast.Expr(value=ast.Constant(value="<nested-function>")), node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> ast.AST:
        return ast.copy_location(ast.Expr(value=ast.Constant(value="<nested-async-function>")), node)

    def visit_Lambda(self, node: ast.Lambda) -> ast.AST:
        return ast.copy_location(ast.Constant(value="<lambda>"), node)

    def visit_ClassDef(self, node: ast.ClassDef) -> ast.AST:
        return ast.copy_location(ast.Expr(value=ast.Constant(value="<nested-class>")), node)


def _normalized_function_fingerprint(node: ast.FunctionDef | ast.AsyncFunctionDef) -> str:
    body = [statement for index, statement in enumerate(node.body) if index != 0 or not _is_docstring(statement)]
    copied_module = ast.Module(body=copy.deepcopy(body), type_ignores=[])
    collector = _LocalNameCollector(node)
    normalized = _BodyNormalizer(collector.names).visit(copied_module)
    ast.fix_missing_locations(normalized)
    serialized = ast.dump(normalized, annotate_fields=True, include_attributes=False)
    return _sha256_bytes(serialized.encode("utf-8"))


class _FunctionFactCollector(ast.NodeVisitor):
    """Collect canonical facts for every sync/async implementation body."""

    def __init__(self, module: SourceModule) -> None:
        self._module = module
        self._context: list[str] = []
        self.facts: list[dict[str, Any]] = []

    def _record(self, node: ast.FunctionDef | ast.AsyncFunctionDef, kind: str) -> None:
        qualname = ".".join([*self._context, node.name])
        self.facts.append(
            {
                "end_line": getattr(node, "end_lineno", node.lineno),
                "fingerprint": _normalized_function_fingerprint(node),
                "kind": kind,
                "line": node.lineno,
                "module": self._module.module,
                "path": self._module.path,
                "qualname": qualname,
            }
        )

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        self._context.append(node.name)
        try:
            self.generic_visit(node)
        finally:
            self._context.pop()

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._record(node, "sync")
        self._context.append(node.name)
        try:
            self.generic_visit(node)
        finally:
            self._context.pop()

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self._record(node, "async")
        self._context.append(node.name)
        try:
            self.generic_visit(node)
        finally:
            self._context.pop()


def _implementation_fingerprints(modules: Iterable[SourceModule]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    facts: list[dict[str, Any]] = []
    for module in modules:
        collector = _FunctionFactCollector(module)
        collector.visit(module.tree)
        facts.extend(collector.facts)
    facts = sorted(facts, key=lambda item: (item["path"], item["line"], item["qualname"], item["kind"]))
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for fact in facts:
        grouped[fact["fingerprint"]].append(fact)
    groups = [
        {"fingerprint": fingerprint, "occurrences": occurrences}
        for fingerprint, occurrences in sorted(grouped.items())
        if len(occurrences) > 1
    ]
    duplication = {
        "duplicate_body_occurrences": sum(len(group["occurrences"]) - 1 for group in groups),
        "duplicate_group_count": len(groups),
        "groups": groups,
    }
    return facts, duplication


def _tool_provenance() -> dict[str, Any]:
    try:
        payload = Path(__file__).read_bytes()
    except OSError as exc:  # pragma: no cover - script execution necessarily supplies this file.
        raise ArchitectureContractError(f"cannot read checker tool bytes: {exc}") from exc
    return {
        "measurement_contract_version": MEASUREMENT_CONTRACT_VERSION,
        "script": "check_architecture_convergence.py",
        "script_sha256": _sha256_bytes(payload),
    }


def _build_measurement(
    source_root: Path, package: str, explicit_optional_modules: Iterable[str]
) -> tuple[dict[str, Any], dict[str, Any]]:
    source_root, source_provenance = _source_provenance(source_root)
    modules = _list_source_modules(source_root, str(source_provenance["commit"]), package)
    policy = _optional_import_policy(source_root, str(source_provenance["commit"]), explicit_optional_modules)
    file_facts: list[dict[str, Any]] = [
        {
            "git_object_id": module.git_object_id,
            "logical_loc": _logical_loc(module.payload, module.path),
            "module": module.module,
            "path": module.path,
            "physical_loc": _physical_loc(module.payload, module.path),
            "sha256": module.sha256,
        }
        for module in modules
    ]
    occurrences = _collect_imports(modules)
    graph = _internal_import_graph(modules, occurrences)
    optional_imports, optional_leakage = _optional_import_facts(occurrences, policy)
    fingerprints, duplication = _implementation_fingerprints(modules)
    summary = {
        "duplicate_body_occurrences": duplication["duplicate_body_occurrences"],
        "implementation_fingerprint_count": len(fingerprints),
        "internal_cycle_count": len(graph["cycles"]),
        "logical_loc": sum(item["logical_loc"] for item in file_facts),
        "optional_import_leakage_count": len(optional_leakage),
        "physical_loc": sum(item["physical_loc"] for item in file_facts),
    }
    artifact: dict[str, Any] = {
        "artifact_type": ARTIFACT_TYPE,
        "baseline_state": "captured",
        "measurement_contract": {
            "logical_loc": "nonblank, non-comment-only tokenize lines; docstrings included",
            "normalized_ast": "positions, docstrings, type comments removed; local names and literals normalized",
            "physical_loc": "splitlines over regular package Python blobs",
            "version": MEASUREMENT_CONTRACT_VERSION,
        },
        "measurements": {
            "files": file_facts,
            "implementation_fingerprints": fingerprints,
            "internal_import_graph": graph,
            "normalized_ast_duplication": duplication,
            "optional_import_leakage": optional_leakage,
            "optional_import_policy": policy,
            "optional_imports": optional_imports,
            "summary": summary,
        },
        "package": package,
        "schema_version": SCHEMA_VERSION,
        "source_provenance": source_provenance,
        "tool_provenance": _tool_provenance(),
        "verdict": "measurement_only",
    }
    return artifact, source_provenance


def _validate_baseline_path(path: Path) -> Path:
    if path.is_symlink():
        raise ArchitectureContractError("baseline must not be a symbolic link")
    try:
        resolved = path.resolve(strict=True)
    except FileNotFoundError as exc:
        raise ArchitectureContractError("baseline is missing") from exc
    if not resolved.is_file():
        raise ArchitectureContractError("baseline must be a regular file")
    return resolved


def _load_baseline(path: Path) -> tuple[dict[str, Any], str]:
    resolved = _validate_baseline_path(path)
    try:
        payload = resolved.read_bytes()
        document = json.loads(payload.decode("utf-8", errors="strict"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ArchitectureContractError(f"cannot read baseline JSON: {exc}") from exc
    if not isinstance(document, dict):
        raise ArchitectureContractError("baseline must be a JSON object")
    return document, _sha256_bytes(payload)


def _require_mapping(value: object, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ArchitectureContractError(f"{label} must be a JSON object")
    return value


def _validate_thresholds(baseline: Mapping[str, Any]) -> dict[str, int]:
    raw_thresholds = _require_mapping(baseline.get("thresholds"), "baseline thresholds")
    if set(raw_thresholds) != set(_THRESHOLD_TO_SUMMARY):
        raise ArchitectureContractError("baseline thresholds must contain the complete frozen threshold schema")
    thresholds: dict[str, int] = {}
    for key in _THRESHOLD_TO_SUMMARY:
        value = raw_thresholds[key]
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            raise ArchitectureContractError(f"baseline threshold {key} must be a non-negative integer")
        thresholds[key] = value
    return thresholds


def _validate_measurement_shape(document: Mapping[str, Any], label: str) -> dict[str, int]:
    tool_provenance = _require_mapping(document.get("tool_provenance"), f"{label} tool_provenance")
    if set(tool_provenance) != {"measurement_contract_version", "script", "script_sha256"}:
        raise ArchitectureContractError(f"{label} tool_provenance has an incompatible schema")
    if tool_provenance.get("measurement_contract_version") != MEASUREMENT_CONTRACT_VERSION:
        raise ArchitectureContractError(
            f"{label} tool_provenance measurement contract version does not match this checker"
        )
    if not _non_empty_string(tool_provenance.get("script")):
        raise ArchitectureContractError(f"{label} tool_provenance script must be non-empty")
    script_sha256 = tool_provenance.get("script_sha256")
    if not isinstance(script_sha256, str) or not _SHA256.fullmatch(script_sha256):
        raise ArchitectureContractError(f"{label} tool_provenance script_sha256 must be a SHA256 digest")

    provenance = _require_mapping(document.get("source_provenance"), f"{label} source_provenance")
    commit = provenance.get("commit")
    tree = provenance.get("tree")
    if not isinstance(commit, str) or not _GIT_OBJECT_ID.fullmatch(commit):
        raise ArchitectureContractError(f"{label} source_provenance commit must be an object identifier")
    if not isinstance(tree, str) or not _GIT_OBJECT_ID.fullmatch(tree):
        raise ArchitectureContractError(f"{label} source_provenance tree must be an object identifier")
    if provenance.get("clean") is not True:
        raise ArchitectureContractError(f"{label} source_provenance clean must be true")
    platform_provenance = _require_mapping(provenance.get("platform"), f"{label} platform")
    if set(platform_provenance) != {"machine", "python_implementation", "python_version", "system"} or not all(
        _non_empty_string(value) for value in platform_provenance.values()
    ):
        raise ArchitectureContractError(f"{label} platform must contain the complete non-empty platform schema")

    contract = _require_mapping(document.get("measurement_contract"), f"{label} measurement_contract")
    if contract.get("version") != MEASUREMENT_CONTRACT_VERSION:
        raise ArchitectureContractError(f"{label} measurement contract version does not match this checker")
    measurements = _require_mapping(document.get("measurements"), f"{label} measurements")
    missing_collections = sorted(_MEASUREMENT_COLLECTION_KEYS - set(measurements))
    if missing_collections:
        raise ArchitectureContractError(
            f"{label} measurements are missing required collections: {', '.join(missing_collections)}"
        )
    for key in {"files", "implementation_fingerprints", "optional_import_leakage", "optional_imports"}:
        if not isinstance(measurements[key], list):
            raise ArchitectureContractError(f"{label} measurements {key} must be a list")
    graph = _require_mapping(measurements["internal_import_graph"], f"{label} internal_import_graph")
    if set(graph) != {"cycles", "edges", "nodes"} or not all(isinstance(graph[key], list) for key in graph):
        raise ArchitectureContractError(f"{label} internal_import_graph must contain list nodes, edges, and cycles")
    duplication = _require_mapping(measurements["normalized_ast_duplication"], f"{label} normalized_ast_duplication")
    if set(duplication) != {"duplicate_body_occurrences", "duplicate_group_count", "groups"} or not isinstance(
        duplication["groups"], list
    ):
        raise ArchitectureContractError(f"{label} normalized_ast_duplication has an incompatible schema")
    policy = _require_mapping(measurements["optional_import_policy"], f"{label} optional_import_policy")
    if set(policy) != {"derived_from_pyproject", "effective_module_roots", "explicit_module_roots"} or not all(
        isinstance(policy[key], list) for key in policy
    ):
        raise ArchitectureContractError(f"{label} optional_import_policy has an incompatible schema")
    raw_summary = _require_mapping(measurements["summary"], f"{label} measurement summary")
    if set(raw_summary) != set(_THRESHOLD_TO_SUMMARY.values()):
        raise ArchitectureContractError(f"{label} measurement summary has an incompatible schema")
    summary: dict[str, int] = {}
    for key in _THRESHOLD_TO_SUMMARY.values():
        value = raw_summary[key]
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            raise ArchitectureContractError(f"{label} measurement summary {key} must be a non-negative integer")
        summary[key] = value
    return summary


def _validate_baseline_contract(baseline: Mapping[str, Any], actual: Mapping[str, Any]) -> dict[str, int]:
    if baseline.get("schema_version") != SCHEMA_VERSION:
        raise ArchitectureContractError(f"baseline schema_version must be {SCHEMA_VERSION}")
    if baseline.get("artifact_type") != ARTIFACT_TYPE:
        raise ArchitectureContractError("baseline artifact_type does not match architecture measurement contract")
    if baseline.get("baseline_state") != "frozen":
        raise ArchitectureContractError(
            "baseline_state must be frozen; captured or pending baselines are not comparable"
        )
    if baseline.get("verdict") != "architecture_baseline":
        raise ArchitectureContractError("baseline verdict must explicitly be architecture_baseline")
    if baseline.get("package") != actual.get("package"):
        raise ArchitectureContractError("baseline package does not match the measured package")
    _validate_measurement_shape(baseline, "baseline")
    _validate_measurement_shape(actual, "actual")
    baseline_tool = _require_mapping(baseline.get("tool_provenance"), "baseline tool_provenance")
    actual_tool = _require_mapping(actual.get("tool_provenance"), "actual tool_provenance")
    if baseline_tool["script_sha256"] != actual_tool["script_sha256"]:
        raise ArchitectureContractError("baseline tooling script SHA256 does not match this checker")
    baseline_provenance = _require_mapping(baseline.get("source_provenance"), "baseline source_provenance")
    baseline_platform = _require_mapping(baseline_provenance.get("platform"), "baseline platform")
    actual_provenance = _require_mapping(actual.get("source_provenance"), "actual source_provenance")
    actual_platform = _require_mapping(actual_provenance.get("platform"), "actual platform")
    if baseline_platform != actual_platform:
        raise ArchitectureContractError("baseline platform does not match the current measurement platform")
    return _validate_thresholds(baseline)


def _validate_legacy_zero_contract(baseline: Mapping[str, Any]) -> None:
    del baseline
    raise ArchitectureContractError(
        "legacy-zero is outside the generic Task 0 architecture measurement contract and requires separate frozen evidence"
    )


def _validate_against_baseline(
    actual: dict[str, Any], baseline: Mapping[str, Any], *, require_no_cycles: bool, require_legacy_zero: bool
) -> dict[str, Any]:
    thresholds = _validate_baseline_contract(baseline, actual)
    if require_legacy_zero:
        _validate_legacy_zero_contract(baseline)
    measurements = _require_mapping(actual.get("measurements"), "actual measurements")
    summary = _require_mapping(measurements.get("summary"), "actual measurement summary")
    for threshold_name, summary_name in _THRESHOLD_TO_SUMMARY.items():
        value = summary.get(summary_name)
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            raise ArchitectureContractError(f"actual summary {summary_name} must be a non-negative integer")
        if value > thresholds[threshold_name]:
            raise ArchitectureContractError(
                f"architecture threshold failed: {summary_name}={value} exceeds {threshold_name}={thresholds[threshold_name]}"
            )
    if require_no_cycles and summary["internal_cycle_count"] != 0:
        raise ArchitectureContractError("architecture cycle requirement failed: internal_cycle_count must be 0")
    return {"status": "passed", "thresholds": thresholds}


def _validate_capture_path(source_root: Path, path: Path) -> Path:
    if path.is_symlink():
        raise ArchitectureContractError("capture output must not be a symbolic link")
    resolved = path.resolve(strict=False)
    if _is_within(resolved, source_root):
        raise ArchitectureContractError("capture output must be outside the clean source worktree")
    for git_control_path in _git_control_paths(source_root):
        if _is_within(resolved, git_control_path):
            raise ArchitectureContractError("capture output must not target Git control data")
    parent = resolved.parent
    if not parent.exists() or not parent.is_dir():
        raise ArchitectureContractError("capture output parent directory must already exist")
    if path.exists() and not path.is_file():
        raise ArchitectureContractError("capture output must be a regular file when it already exists")
    return resolved


def _git_control_paths(source_root: Path) -> tuple[Path, ...]:
    paths: list[Path] = []
    for raw_path in (
        source_root / ".git",
        Path(_git_text(source_root, "rev-parse", "--git-dir")),
        Path(_git_text(source_root, "rev-parse", "--git-common-dir")),
    ):
        candidate = raw_path if raw_path.is_absolute() else source_root / raw_path
        resolved = candidate.resolve(strict=False)
        if resolved not in paths:
            paths.append(resolved)
    return tuple(paths)


def _write_capture_atomically(
    output: Path, serialized: bytes, source_root: Path, provenance: Mapping[str, Any]
) -> None:
    temporary_name: str | None = None
    _verify_source_provenance(source_root, provenance)
    try:
        with tempfile.NamedTemporaryFile(
            mode="xb", dir=output.parent, prefix=f".{output.name}.", suffix=".tmp", delete=False
        ) as temporary:
            temporary_name = temporary.name
            temporary.write(serialized)
            temporary.flush()
            os.fsync(temporary.fileno())
        _verify_source_provenance(source_root, provenance)
        Path(temporary_name).replace(output)
        temporary_name = None
    except OSError as exc:
        raise ArchitectureContractError(f"cannot atomically write capture output: {exc}") from exc
    finally:
        if temporary_name is not None:
            with suppress(OSError):
                Path(temporary_name).unlink(missing_ok=True)


def _serialize_artifact(artifact: Mapping[str, Any]) -> bytes:
    return (json.dumps(artifact, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode("utf-8")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-root", type=Path, default=Path.cwd(), help="clean Git worktree root (defaults to cwd)"
    )
    parser.add_argument("--package", required=True, help="dotted package name to measure")
    parser.add_argument("--capture", type=Path, help="write deterministic measurement JSON outside source-root")
    parser.add_argument("--baseline", type=Path, help="frozen architecture baseline JSON to validate")
    parser.add_argument(
        "--optional-module",
        action="append",
        default=[],
        help="additional optional import root; may be repeated",
    )
    parser.add_argument(
        "--require-no-cycles", action="store_true", help="fail unless the measured internal cycle count is zero"
    )
    parser.add_argument(
        "--require-legacy-zero",
        action="store_true",
        help="reserved for a separate frozen legacy-removal contract; fails in this generic checker",
    )
    return parser


def _run(arguments: argparse.Namespace) -> dict[str, Any]:
    if arguments.capture is None and arguments.baseline is None:
        raise ArchitectureContractError("provide --capture, --baseline, or both")
    package = _validate_package_name(arguments.package)
    try:
        source_root = arguments.source_root.resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise ArchitectureContractError("source-root must be an existing directory") from exc
    if not source_root.is_dir():
        raise ArchitectureContractError("source-root must be a directory")
    artifact, provenance = _build_measurement(source_root, package, arguments.optional_module)
    if arguments.baseline is not None:
        baseline, baseline_sha256 = _load_baseline(arguments.baseline)
        artifact["baseline_validation"] = _validate_against_baseline(
            artifact,
            baseline,
            require_no_cycles=arguments.require_no_cycles,
            require_legacy_zero=arguments.require_legacy_zero,
        )
        artifact["baseline_validation"]["baseline_sha256"] = baseline_sha256
    elif arguments.require_no_cycles:
        summary = artifact["measurements"]["summary"]
        if summary["internal_cycle_count"] != 0:
            raise ArchitectureContractError("architecture cycle requirement failed: internal_cycle_count must be 0")
    elif arguments.require_legacy_zero:
        _validate_legacy_zero_contract({})
    serialized = _serialize_artifact(artifact)
    if arguments.capture is not None:
        output = _validate_capture_path(source_root, arguments.capture)
        _write_capture_atomically(output, serialized, source_root, provenance)
    return {
        "capture_sha256": _sha256_bytes(serialized),
        "package": package,
        "result": "validated" if arguments.baseline is not None else "captured",
        "source_commit": provenance["commit"],
    }


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    try:
        result = _run(arguments)
    except ArchitectureContractError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
