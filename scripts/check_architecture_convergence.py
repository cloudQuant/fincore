#!/usr/bin/env python3
"""Measure and validate a fail-closed 0042-R2 architecture contract.

This tool deliberately performs static source measurement only.  A plain
``--capture`` produces a deterministic ``measurement_only`` artifact; an
explicit ``--seal-baseline`` capture produces an architecture-baseline
component from an immutable threshold policy.  Neither result is a D0,
D-TECH, Task 8, or release verdict.  The source must be a clean Git worktree
and every Python input is read from regular blobs in the initially observed
``HEAD`` tree.  Capture outputs must be outside that source worktree.

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
import secrets
import stat
import subprocess
import sys
import tokenize
from collections import defaultdict
from contextlib import suppress
from dataclasses import dataclass
from decimal import ROUND_FLOOR, Decimal, InvalidOperation
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
THRESHOLD_POLICY_ARTIFACT_TYPE = "fincore_0042_r2_architecture_threshold_policy"
THRESHOLD_POLICY_SCHEMA_VERSION = 1
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
_THRESHOLD_POLICY_KEYS = frozenset({"artifact_type", "policy_state", "rules", "schema_version"})
_RELATIVE_REDUCTION_SUMMARIES = frozenset(
    {"duplicate_body_occurrences", "implementation_fingerprint_count", "logical_loc", "physical_loc"}
)
_ZERO_MAXIMUM_SUMMARIES = frozenset({"internal_cycle_count", "optional_import_leakage_count"})
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
    optional_root_hint: str | None
    level: int
    names: tuple[str, ...]
    optional_guarded: bool


@dataclass(frozen=True)
class CaptureOutput:
    """A capture path bound to the directory descriptor validated for writing."""

    path: Path
    parent_fd: int


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _non_empty_string(value: object) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _controlled_git_environment() -> dict[str, str]:
    """Return a Git environment that cannot redirect source provenance."""
    # A source collector does not need any inherited Git setting.  Stripping
    # them all rules out alternate object stores, a redirected index/worktree,
    # replacement-ref bases, and injected config while retaining normal OS
    # process configuration such as PATH and locale.
    environment = {key: value for key, value in os.environ.items() if not key.startswith("GIT_")}
    environment["GIT_CONFIG_GLOBAL"] = os.devnull
    environment["GIT_CONFIG_NOSYSTEM"] = "1"
    environment["GIT_NO_REPLACE_OBJECTS"] = "1"
    return environment


def _git_bytes(source_root: Path, *arguments: str) -> bytes:
    try:
        result = subprocess.run(
            ["git", "-c", "core.fsmonitor=false", "--no-replace-objects", *arguments],
            cwd=source_root,
            capture_output=True,
            check=False,
            env=_controlled_git_environment(),
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
    tree = _git_text(source_root, "rev-parse", f"{commit}^{{tree}}")
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


class _HandlerRaiseFinder(ast.NodeVisitor):
    """Conservatively detect a rethrow path without descending into deferred scopes."""

    def __init__(self) -> None:
        self.has_raise = False

    def visit_Raise(self, node: ast.Raise) -> None:
        del node
        self.has_raise = True

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        del node

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        del node

    def visit_Lambda(self, node: ast.Lambda) -> None:
        del node

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        # Class decorators, bases, keywords, and top-level body statements
        # execute immediately.  Only nested methods and lambdas are deferred.
        # Skipping the whole class would allow a rethrow in an except suite to
        # incorrectly mark the corresponding try imports as optional.
        for decorator in node.decorator_list:
            self.visit(decorator)
        for base in node.bases:
            self.visit(base)
        for keyword in node.keywords:
            self.visit(keyword.value)
        for statement in node.body:
            self.visit(statement)


def _handler_may_reraise(handler: ast.ExceptHandler) -> bool:
    finder = _HandlerRaiseFinder()
    for statement in handler.body:
        finder.visit(statement)
    return finder.has_raise


def _dotted_expression_name(node: ast.expr) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        parent = _dotted_expression_name(node.value)
        return f"{parent}.{node.attr}" if parent else None
    return None


def _literal_string(node: ast.expr) -> str | None:
    """Return a literal string expression without evaluating source code."""
    return node.value if isinstance(node, ast.Constant) and isinstance(node.value, str) else None


def _literal_call_argument(node: ast.Call, positional_index: int, keyword: str) -> str | None:
    """Read one literal positional-or-keyword call argument conservatively."""
    if len(node.args) > positional_index:
        literal = _literal_string(node.args[positional_index])
        if literal is not None:
            return literal
    for item in node.keywords:
        if item.arg == keyword:
            literal = _literal_string(item.value)
            if literal is not None:
                return literal
    return None


def _bound_target_names(node: ast.expr) -> set[str]:
    """Return the names bound by one assignment-like target."""
    if isinstance(node, ast.Name):
        return {node.id}
    if isinstance(node, (ast.List, ast.Tuple)):
        names: set[str] = set()
        for element in node.elts:
            names.update(_bound_target_names(element))
        return names
    if isinstance(node, ast.Starred):
        return _bound_target_names(node.value)
    return set()


class _ScopeBindingFinder(ast.NodeVisitor):
    """Collect function-local bindings without descending into nested scopes."""

    def __init__(self) -> None:
        self.names: set[str] = set()
        self.declared_external_names: set[str] = set()

    def visit_Name(self, node: ast.Name) -> None:
        if isinstance(node.ctx, (ast.Del, ast.Store)):
            self.names.add(node.id)

    def visit_arg(self, node: ast.arg) -> None:
        self.names.add(node.arg)

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            self.names.add(alias.asname or alias.name.split(".", maxsplit=1)[0])

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        for alias in node.names:
            if alias.name != "*":
                self.names.add(alias.asname or alias.name)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self.names.add(node.name)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self.names.add(node.name)

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        self.names.add(node.name)

    def visit_Lambda(self, node: ast.Lambda) -> None:
        del node

    def visit_Global(self, node: ast.Global) -> None:
        self.declared_external_names.update(node.names)

    def visit_Nonlocal(self, node: ast.Nonlocal) -> None:
        self.declared_external_names.update(node.names)


def _function_scope_bound_names(arguments: ast.arguments, body: Iterable[ast.stmt]) -> set[str]:
    """Find bindings that shadow parent dynamic-import aliases in a function."""
    finder = _ScopeBindingFinder()
    finder.visit(arguments)
    for statement in body:
        finder.visit(statement)
    return finder.names - finder.declared_external_names


class _ImportCollector(ast.NodeVisitor):
    """Collect imports while preserving whether an ImportError guard encloses them."""

    def __init__(self, module: SourceModule) -> None:
        self._module = module
        self._optional_guard_depth = 0
        # This closed mapping records source-visible ways to invoke literal
        # dynamic imports.  Ambiguous aliasing errs toward an extra finding,
        # never a false zero-leakage result.
        self._dynamic_import_alias_scopes: list[dict[str, str]] = [{}]
        self.occurrences: list[ImportOccurrence] = []

    @property
    def _dynamic_import_aliases(self) -> dict[str, str]:
        return self._dynamic_import_alias_scopes[-1]

    def _push_alias_scope(self, shadowed_names: Iterable[str] = ()) -> None:
        aliases = dict(self._dynamic_import_aliases)
        for name in shadowed_names:
            aliases.pop(name, None)
        self._dynamic_import_alias_scopes.append(aliases)

    def _pop_alias_scope(self) -> None:
        self._dynamic_import_alias_scopes.pop()

    def _preserve_ambiguous_alias_bindings(self, names: Iterable[str]) -> None:
        """Keep a possible alias after a non-import rebinding.

        A linear AST walk cannot prove that an assignment in a branch, loop,
        handler, or conditional definition executes before a later call.  A
        retained alias can make the static metric conservative, while clearing
        it could fabricate a false zero-leakage result.  Function-local names
        are still removed when a new lexical scope is entered above.
        """
        del names

    def _register_import_alias(self, name: str, kind: str | None) -> None:
        if kind is not None:
            self._dynamic_import_aliases[name] = kind
        else:
            self._preserve_ambiguous_alias_bindings((name,))

    def _add(
        self,
        *,
        line: int,
        kind: str,
        target: str,
        optional_root_hint: str | None = None,
        level: int = 0,
        names: Iterable[str] = (),
    ) -> None:
        self.occurrences.append(
            ImportOccurrence(
                source_module=self._module.module,
                path=self._module.path,
                line=line,
                kind=kind,
                target=target,
                optional_root_hint=optional_root_hint,
                level=level,
                names=tuple(names),
                optional_guarded=self._optional_guard_depth > 0,
            )
        )

    def visit_Try(self, node: ast.Try) -> None:
        import_error_handlers = [handler for handler in node.handlers if _except_catches_import_error(handler)]
        # A handler that can rethrow does not make imports in the try suite
        # reliably optional. Treat conditional/complex rethrow paths as
        # leakage too: false positives are safer than hiding a hard import.
        guarded = bool(import_error_handlers) and not any(
            _handler_may_reraise(handler) for handler in import_error_handlers
        )
        if not guarded:
            self.generic_visit(node)
            return
        self._optional_guard_depth += 1
        try:
            # An ImportError handler only protects the try suite.  Imports in
            # ``else``, ``finally``, or handler suites can still raise an
            # unhandled ImportError and must remain visible as leakage.
            for statement in node.body:
                self.visit(statement)
        finally:
            self._optional_guard_depth -= 1
        for handler in node.handlers:
            self.visit(handler)
        for statement in [*node.orelse, *node.finalbody]:
            self.visit(statement)

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            self._add(line=node.lineno, kind="import", target=alias.name)
            bound_name = alias.asname or alias.name.split(".", maxsplit=1)[0]
            if alias.name == "importlib" or alias.name.startswith("importlib."):
                self._register_import_alias(bound_name, "importlib_module")
            elif alias.name == "builtins" or alias.name.startswith("builtins."):
                self._register_import_alias(bound_name, "builtins_module")
            else:
                self._register_import_alias(bound_name, None)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        self._add(
            line=node.lineno,
            kind="from_import",
            target=node.module or "",
            level=node.level,
            names=(alias.name for alias in node.names),
        )
        for alias in node.names:
            if alias.name == "*":
                continue
            bound_name = alias.asname or alias.name
            if node.level == 0 and node.module == "importlib" and alias.name == "import_module":
                self._register_import_alias(bound_name, "import_module_function")
            elif node.level == 0 and node.module == "builtins" and alias.name == "__import__":
                self._register_import_alias(bound_name, "builtin_import_function")
            else:
                self._register_import_alias(bound_name, None)

    def _visit_deferred_scope(
        self,
        *,
        arguments: ast.arguments,
        body: Iterable[ast.stmt] | ast.expr,
        decorators: Iterable[ast.expr] = (),
        returns: ast.expr | None = None,
        scope_body: Iterable[ast.stmt] = (),
    ) -> None:
        """Visit definition-time expressions under the current import guard.

        Function/lambda bodies execute later, potentially long after an outer
        ``try/except ImportError`` has completed.  Their imports must not
        inherit that outer guard.  Decorators, annotations, and defaults are
        evaluated while the definition itself is executed, so they do retain
        the current guard state.
        """
        for decorator in decorators:
            self.visit(decorator)
        self.visit(arguments)
        if returns is not None:
            self.visit(returns)
        prior_depth = self._optional_guard_depth
        self._optional_guard_depth = 0
        self._push_alias_scope(_function_scope_bound_names(arguments, scope_body))
        try:
            if isinstance(body, ast.expr):
                self.visit(body)
            else:
                for statement in body:
                    self.visit(statement)
        finally:
            self._pop_alias_scope()
            self._optional_guard_depth = prior_depth

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._visit_deferred_scope(
            arguments=node.args,
            body=node.body,
            decorators=node.decorator_list,
            returns=node.returns,
            scope_body=node.body,
        )
        self._preserve_ambiguous_alias_bindings((node.name,))

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self._visit_deferred_scope(
            arguments=node.args,
            body=node.body,
            decorators=node.decorator_list,
            returns=node.returns,
            scope_body=node.body,
        )
        self._preserve_ambiguous_alias_bindings((node.name,))

    def visit_Lambda(self, node: ast.Lambda) -> None:
        self._visit_deferred_scope(arguments=node.args, body=node.body)

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        """Visit class-time expressions; nested methods remain deferred."""
        for decorator in node.decorator_list:
            self.visit(decorator)
        for base in node.bases:
            self.visit(base)
        for keyword in node.keywords:
            self.visit(keyword.value)
        self._push_alias_scope()
        try:
            for statement in node.body:
                self.visit(statement)
        finally:
            self._pop_alias_scope()
        self._preserve_ambiguous_alias_bindings((node.name,))

    def _visit_comprehension(
        self,
        generators: list[ast.comprehension],
        values: Iterable[ast.expr],
        *,
        deferred: bool,
    ) -> None:
        """Visit a comprehension with correct eager/deferred alias and guard scope."""
        if not generators:  # pragma: no cover - Python AST always has at least one generator.
            for value in values:
                self.visit(value)
            return
        # The outer iterable runs immediately.  In a generator expression the
        # remaining iterables, predicates, and element run later, beyond any
        # enclosing optional-import guard.
        self.visit(generators[0].iter)
        shadowed_names: set[str] = set()
        for generator in generators:
            shadowed_names.update(_bound_target_names(generator.target))
        prior_depth = self._optional_guard_depth
        self._push_alias_scope(shadowed_names)
        if deferred:
            self._optional_guard_depth = 0
        try:
            for index, generator in enumerate(generators):
                if index:
                    self.visit(generator.iter)
                self.visit(generator.target)
                for condition in generator.ifs:
                    self.visit(condition)
            for value in values:
                self.visit(value)
        finally:
            self._pop_alias_scope()
            self._optional_guard_depth = prior_depth

    def visit_GeneratorExp(self, node: ast.GeneratorExp) -> None:
        self._visit_comprehension(node.generators, (node.elt,), deferred=True)

    def visit_ListComp(self, node: ast.ListComp) -> None:
        self._visit_comprehension(node.generators, (node.elt,), deferred=False)

    def visit_SetComp(self, node: ast.SetComp) -> None:
        self._visit_comprehension(node.generators, (node.elt,), deferred=False)

    def visit_DictComp(self, node: ast.DictComp) -> None:
        self._visit_comprehension(node.generators, (node.key, node.value), deferred=False)

    def visit_Assign(self, node: ast.Assign) -> None:
        self.visit(node.value)
        for target in node.targets:
            self.visit(target)
            self._bind_assignment_target(target, node.value)

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        self.visit(node.annotation)
        if node.value is not None:
            self.visit(node.value)
        self.visit(node.target)
        self._bind_assignment_target(node.target, node.value)

    def visit_AugAssign(self, node: ast.AugAssign) -> None:
        self.visit(node.target)
        self.visit(node.value)
        self._preserve_ambiguous_alias_bindings(_bound_target_names(node.target))

    def visit_NamedExpr(self, node: ast.NamedExpr) -> None:
        self.visit(node.value)
        self.visit(node.target)
        self._bind_assignment_target(node.target, node.value)

    def _visit_for(self, node: ast.For | ast.AsyncFor) -> None:
        self.visit(node.iter)
        self.visit(node.target)
        self._preserve_ambiguous_alias_bindings(_bound_target_names(node.target))
        for statement in node.body:
            self.visit(statement)
        for statement in node.orelse:
            self.visit(statement)

    def visit_For(self, node: ast.For) -> None:
        self._visit_for(node)

    def visit_AsyncFor(self, node: ast.AsyncFor) -> None:
        self._visit_for(node)

    def _visit_with(self, node: ast.With | ast.AsyncWith) -> None:
        for item in node.items:
            self.visit(item.context_expr)
            if item.optional_vars is not None:
                self.visit(item.optional_vars)
                self._preserve_ambiguous_alias_bindings(_bound_target_names(item.optional_vars))
        for statement in node.body:
            self.visit(statement)

    def visit_With(self, node: ast.With) -> None:
        self._visit_with(node)

    def visit_AsyncWith(self, node: ast.AsyncWith) -> None:
        self._visit_with(node)

    def visit_ExceptHandler(self, node: ast.ExceptHandler) -> None:
        if node.type is not None:
            self.visit(node.type)
        if node.name is not None:
            self._preserve_ambiguous_alias_bindings((node.name,))
        for statement in node.body:
            self.visit(statement)

    def _bind_assignment_target(self, target: ast.expr, value: ast.expr | None) -> None:
        if (
            isinstance(target, (ast.List, ast.Tuple))
            and isinstance(value, (ast.List, ast.Tuple))
            and len(target.elts) == len(value.elts)
        ):
            for child_target, child_value in zip(target.elts, value.elts, strict=True):
                self._bind_assignment_target(child_target, child_value)
            return
        alias_kind = self._import_alias_kind_from_value(value) if value is not None else None
        for name in _bound_target_names(target):
            self._register_import_alias(name, alias_kind)

    def _import_alias_kind_from_value(self, node: ast.expr) -> str | None:
        """Resolve a known import module/callable value for assignment aliases."""
        if isinstance(node, ast.Name):
            return self._dynamic_import_aliases.get(node.id)
        callable_kind = self._dynamic_import_callable_alias_kind(node)
        if callable_kind is not None:
            return callable_kind
        if (
            isinstance(node, ast.Call)
            and self._dynamic_import_callable_alias_kind(node.func) == "builtin_import_function"
        ):
            import_name = _literal_call_argument(node, 0, "name")
            if import_name == "importlib":
                return "importlib_module"
            if import_name == "builtins":
                return "builtins_module"
        return None

    def _dynamic_import_callable_alias_kind(self, node: ast.expr) -> str | None:
        function_name = _dotted_expression_name(node)
        if function_name == "importlib.import_module":
            return "import_module_function"
        if function_name in {"__import__", "builtins.__import__", "__builtins__.__import__"}:
            return "builtin_import_function"
        if isinstance(node, ast.Name):
            alias_kind = self._dynamic_import_aliases.get(node.id)
            if alias_kind in {"import_module_function", "builtin_import_function"}:
                return alias_kind
        if isinstance(node, ast.Attribute):
            alias_kind = self._import_alias_kind_from_value(node.value)
            if node.attr == "import_module" and alias_kind == "importlib_module":
                return "import_module_function"
            if node.attr == "__import__" and alias_kind == "builtins_module":
                return "builtin_import_function"
        if (
            isinstance(node, ast.Subscript)
            and _dotted_expression_name(node.value) == "__builtins__"
            and _literal_string(node.slice) == "__import__"
        ):
            return "builtin_import_function"
        return None

    def visit_Call(self, node: ast.Call) -> None:
        callable_kind = self._dynamic_import_callable_alias_kind(node.func)
        target = _literal_call_argument(node, 0, "name") if callable_kind is not None else None
        if target is not None:
            optional_root_hint = None
            if callable_kind == "import_module_function" and target.startswith("."):
                package = _literal_call_argument(node, 1, "package")
                optional_root_hint = _dependency_root(package) if package is not None else None
            self._add(
                line=node.lineno,
                kind="dynamic_import",
                target=target,
                optional_root_hint=optional_root_hint,
            )
        self.generic_visit(node)


def _collect_imports(modules: Iterable[SourceModule]) -> list[ImportOccurrence]:
    occurrences: list[ImportOccurrence] = []
    for module in modules:
        collector = _ImportCollector(module)
        collector.visit(module.tree)
        occurrences.extend(collector.occurrences)
    return sorted(
        occurrences,
        key=lambda item: (
            item.path,
            item.line,
            item.kind,
            item.target,
            item.optional_root_hint or "",
            item.level,
            item.names,
            item.optional_guarded,
        ),
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
        return (
            [occurrence.target]
            if occurrence.target in modules and occurrence.target != occurrence.source_module
            else []
        )
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
    # ``from package import API`` is an edge to the package module itself and
    # can complete a real root-to-submodule cycle.  Suppress only a literal
    # self-edge such as ``package.__init__: from . import child``; child
    # module edges are added below from the imported names.
    if occurrence.target and target in modules and target != occurrence.source_module:
        candidates.add(target)
    for name in occurrence.names:
        if name == "*":
            continue
        child = f"{target}.{name}"
        if child in modules and child != occurrence.source_module:
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
    if occurrence.optional_root_hint is not None:
        return occurrence.optional_root_hint
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
            "target": occurrence.target,
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


def _read_initial_head_regular_blob(
    source_root: Path,
    commit: str,
    path: Path,
    *,
    label: str,
) -> tuple[str, str, bytes]:
    """Read one source-local regular blob from the initially observed tree."""
    if path.is_symlink():
        raise ArchitectureContractError(f"{label} must not be a symbolic link")
    try:
        resolved_path = path.resolve(strict=True)
        relative = resolved_path.relative_to(source_root)
    except (OSError, RuntimeError, ValueError) as exc:
        raise ArchitectureContractError(f"{label} must be inside the clean source worktree") from exc
    relative_path = relative.as_posix()
    _validate_repo_relative_path(relative_path)
    records = [
        record
        for record in _git_bytes(source_root, "ls-tree", "-z", commit, "--", relative_path).split(b"\0")
        if record
    ]
    if len(records) != 1:
        raise ArchitectureContractError(f"{label} must be one regular file in the initial HEAD tree")
    try:
        metadata, raw_path = records[0].split(b"\t", 1)
        mode, object_type, object_id = metadata.decode("ascii", errors="strict").split()
        discovered_path = raw_path.decode("utf-8", errors="strict")
    except (UnicodeDecodeError, ValueError) as exc:
        raise ArchitectureContractError(f"cannot inspect initial HEAD {label}") from exc
    if (
        discovered_path != relative_path
        or mode not in {"100644", "100755"}
        or object_type != "blob"
        or not _GIT_OBJECT_ID.fullmatch(object_id)
    ):
        raise ArchitectureContractError(f"{label} must be a regular Git blob in the initial HEAD tree")
    return relative_path, object_id, _git_bytes(source_root, "cat-file", "blob", object_id)


def _validate_frozen_threshold_policy_rules(policy: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    raw_rules = _require_mapping(policy.get("rules"), "threshold policy rules")
    if set(raw_rules) != set(_THRESHOLD_TO_SUMMARY):
        raise ArchitectureContractError("threshold policy rules must contain the complete frozen threshold schema")
    rules: dict[str, dict[str, Any]] = {}
    for threshold_name, summary_name in _THRESHOLD_TO_SUMMARY.items():
        rule = _require_mapping(raw_rules.get(threshold_name), f"threshold policy rule {threshold_name}")
        kind = rule.get("kind")
        if summary_name in _RELATIVE_REDUCTION_SUMMARIES:
            if set(rule) != {"kind", "reduction"} or kind != "relative_reduction":
                raise ArchitectureContractError(
                    f"threshold policy rule {threshold_name} must be a relative_reduction rule"
                )
            reduction = rule.get("reduction")
            if isinstance(reduction, bool) or not isinstance(reduction, (int, float)):
                raise ArchitectureContractError(
                    f"threshold policy rule {threshold_name} reduction must be a number between zero and one"
                )
            try:
                normalized_reduction = Decimal(str(reduction))
            except (InvalidOperation, ValueError) as exc:
                raise ArchitectureContractError(
                    f"threshold policy rule {threshold_name} reduction must be a number between zero and one"
                ) from exc
            if (
                not normalized_reduction.is_finite()
                or normalized_reduction < Decimal("0")
                or normalized_reduction > Decimal("1")
            ):
                raise ArchitectureContractError(
                    f"threshold policy rule {threshold_name} reduction must be a number between zero and one"
                )
            if summary_name in {"physical_loc", "logical_loc"} and normalized_reduction < Decimal("0.12"):
                raise ArchitectureContractError(
                    f"threshold policy rule {threshold_name} must require at least 0.12 reduction"
                )
            if summary_name == "duplicate_body_occurrences" and normalized_reduction < Decimal("0.60"):
                raise ArchitectureContractError(
                    f"threshold policy rule {threshold_name} must require at least 0.60 reduction"
                )
            rules[threshold_name] = {"kind": "relative_reduction", "reduction": str(normalized_reduction)}
            continue
        if summary_name in _ZERO_MAXIMUM_SUMMARIES:
            if set(rule) != {"kind", "maximum"} or kind != "absolute_maximum" or rule.get("maximum") != 0:
                raise ArchitectureContractError(
                    f"threshold policy rule {threshold_name} must require an absolute maximum of zero"
                )
            rules[threshold_name] = {"kind": "absolute_maximum", "maximum": 0}
            continue
        raise ArchitectureContractError(f"threshold policy has no rule policy for {threshold_name}")
    return rules


def _load_frozen_threshold_policy(
    source_root: Path,
    commit: str,
    path: Path,
) -> tuple[dict[str, dict[str, Any]], dict[str, str]]:
    relative_path, object_id, payload = _read_initial_head_regular_blob(
        source_root,
        commit,
        path,
        label="threshold policy",
    )
    try:
        policy = json.loads(payload.decode("utf-8", errors="strict"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ArchitectureContractError(f"cannot read threshold policy JSON: {exc}") from exc
    if not isinstance(policy, dict) or set(policy) != _THRESHOLD_POLICY_KEYS:
        raise ArchitectureContractError("threshold policy has an incompatible frozen schema")
    if policy.get("schema_version") != THRESHOLD_POLICY_SCHEMA_VERSION:
        raise ArchitectureContractError("threshold policy schema_version does not match this checker")
    if policy.get("artifact_type") != THRESHOLD_POLICY_ARTIFACT_TYPE:
        raise ArchitectureContractError("threshold policy artifact_type does not match this checker")
    if policy.get("policy_state") != "frozen":
        raise ArchitectureContractError("threshold policy must be frozen")
    rules = _validate_frozen_threshold_policy_rules(policy)
    return rules, {
        "git_object_id": object_id,
        "path": relative_path,
        "sha256": _sha256_bytes(payload),
    }


def _derive_candidate_thresholds(
    summary: Mapping[str, Any],
    threshold_rules: Mapping[str, Mapping[str, Any]],
) -> dict[str, int]:
    """Derive final-candidate limits from a raw D0 summary and frozen policy rules."""
    thresholds: dict[str, int] = {}
    for threshold_name, summary_name in _THRESHOLD_TO_SUMMARY.items():
        value = summary.get(summary_name)
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            raise ArchitectureContractError(f"captured summary {summary_name} must be a non-negative integer")
        rule = threshold_rules[threshold_name]
        if rule["kind"] == "absolute_maximum":
            thresholds[threshold_name] = int(rule["maximum"])
        else:
            reduction = Decimal(str(rule["reduction"]))
            thresholds[threshold_name] = int(
                (Decimal(value) * (Decimal("1") - reduction)).to_integral_value(rounding=ROUND_FLOOR)
            )
    return thresholds


def _seal_architecture_baseline(
    captured: Mapping[str, Any],
    threshold_rules: Mapping[str, Mapping[str, Any]],
    threshold_policy: Mapping[str, str],
) -> dict[str, Any]:
    """Turn a clean measurement into a threshold-governed baseline artifact."""
    sealed = copy.deepcopy(dict(captured))
    measurements = _require_mapping(sealed.get("measurements"), "captured measurements")
    optional_policy = _require_mapping(measurements.get("optional_import_policy"), "captured optional_import_policy")
    if optional_policy.get("explicit_module_roots") != []:
        raise ArchitectureContractError(
            "sealed baselines require no --optional-module roots until an explicit frozen invocation manifest exists"
        )
    summary = _require_mapping(measurements.get("summary"), "captured measurement summary")
    thresholds = _derive_candidate_thresholds(summary, threshold_rules)
    sealed["baseline_state"] = "frozen"
    sealed["threshold_policy"] = dict(threshold_policy)
    sealed["thresholds"] = dict(thresholds)
    sealed["verdict"] = "architecture_baseline"
    return sealed


def _verify_frozen_baseline_threshold_policy(
    baseline: Mapping[str, Any],
    baseline_source_root: Path,
) -> None:
    """Rebuild and bind all sealed baseline facts to its exact clean Git source."""
    _validate_threshold_policy_provenance(baseline)
    policy_provenance = _require_mapping(baseline.get("threshold_policy"), "baseline threshold_policy")
    provenance = _require_mapping(baseline.get("source_provenance"), "baseline source_provenance")
    commit = provenance.get("commit")
    if not isinstance(commit, str) or not _GIT_OBJECT_ID.fullmatch(commit):
        raise ArchitectureContractError("baseline source_provenance commit must be an object identifier")
    tree = provenance.get("tree")
    if not isinstance(tree, str) or not _GIT_OBJECT_ID.fullmatch(tree):
        raise ArchitectureContractError("baseline source_provenance tree must be an object identifier")
    _, baseline_root_provenance = _source_provenance(baseline_source_root)
    if baseline_root_provenance != dict(provenance):
        raise ArchitectureContractError(
            "baseline-source-root must be clean and checked out at the baseline commit/tree"
        )
    policy_path = policy_provenance["path"]
    assert isinstance(policy_path, str)
    rules, actual_policy_provenance = _load_frozen_threshold_policy(
        baseline_source_root,
        commit,
        baseline_source_root / policy_path,
    )
    if actual_policy_provenance != dict(policy_provenance):
        raise ArchitectureContractError("baseline threshold_policy does not match its source Git blob")
    package = baseline.get("package")
    if not isinstance(package, str):
        raise ArchitectureContractError("baseline package must be a dotted Python package name")
    measurements = _require_mapping(baseline.get("measurements"), "baseline measurements")
    summary = _require_mapping(measurements.get("summary"), "baseline measurement summary")
    optional_policy = _require_mapping(measurements.get("optional_import_policy"), "baseline optional_import_policy")
    raw_explicit_roots = optional_policy.get("explicit_module_roots")
    if not isinstance(raw_explicit_roots, list) or not all(isinstance(root, str) for root in raw_explicit_roots):
        raise ArchitectureContractError("baseline optional_import_policy explicit_module_roots must be a string list")
    if raw_explicit_roots:
        raise ArchitectureContractError(
            "baseline optional_import_policy explicit_module_roots must be empty without a frozen invocation manifest"
        )
    rebuilt, rebuilt_provenance = _build_measurement(baseline_source_root, package, raw_explicit_roots)
    if rebuilt_provenance != dict(provenance):  # Defensive; _source_provenance already compared above.
        raise ArchitectureContractError("baseline source provenance changed while rebuilding measurements")
    if baseline.get("measurement_contract") != rebuilt.get("measurement_contract"):
        raise ArchitectureContractError("baseline measurement_contract does not match its source Git measurement")
    if baseline.get("measurements") != rebuilt.get("measurements"):
        raise ArchitectureContractError("baseline measurements do not match their source Git measurement")
    if baseline.get("tool_provenance") != rebuilt.get("tool_provenance"):
        raise ArchitectureContractError("baseline tool_provenance does not match its source Git measurement")
    expected_thresholds = _derive_candidate_thresholds(summary, rules)
    if _validate_thresholds(baseline) != expected_thresholds:
        raise ArchitectureContractError("baseline thresholds are not derived from the frozen threshold policy")


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


def _validate_threshold_policy_provenance(baseline: Mapping[str, Any]) -> None:
    policy = _require_mapping(baseline.get("threshold_policy"), "baseline threshold_policy")
    if set(policy) != {"git_object_id", "path", "sha256"}:
        raise ArchitectureContractError("baseline threshold_policy has an incompatible schema")
    path = policy.get("path")
    if not isinstance(path, str):
        raise ArchitectureContractError("baseline threshold_policy path must be a repository-relative POSIX path")
    _validate_repo_relative_path(path)
    object_id = policy.get("git_object_id")
    if not isinstance(object_id, str) or not _GIT_OBJECT_ID.fullmatch(object_id):
        raise ArchitectureContractError("baseline threshold_policy git_object_id must be an object identifier")
    digest = policy.get("sha256")
    if not isinstance(digest, str) or not _SHA256.fullmatch(digest):
        raise ArchitectureContractError("baseline threshold_policy sha256 must be a SHA256 digest")


def _validate_baseline_contract(
    baseline: Mapping[str, Any],
    actual: Mapping[str, Any],
    baseline_source_root: Path,
) -> dict[str, int]:
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
    _verify_frozen_baseline_threshold_policy(baseline, baseline_source_root)
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
    baseline_measurements = _require_mapping(baseline.get("measurements"), "baseline measurements")
    baseline_optional_policy = _require_mapping(
        baseline_measurements.get("optional_import_policy"), "baseline optional_import_policy"
    )
    actual_measurements = _require_mapping(actual.get("measurements"), "actual measurements")
    actual_optional_policy = _require_mapping(
        actual_measurements.get("optional_import_policy"), "actual optional_import_policy"
    )
    baseline_effective_roots = baseline_optional_policy.get("effective_module_roots")
    actual_effective_roots = actual_optional_policy.get("effective_module_roots")
    if not isinstance(baseline_effective_roots, list) or not all(
        isinstance(root, str) for root in baseline_effective_roots
    ):
        raise ArchitectureContractError("baseline optional_import_policy effective_module_roots must be a string list")
    if not isinstance(actual_effective_roots, list) or not all(
        isinstance(root, str) for root in actual_effective_roots
    ):
        raise ArchitectureContractError("actual optional_import_policy effective_module_roots must be a string list")
    if not set(baseline_effective_roots).issubset(actual_effective_roots):
        raise ArchitectureContractError(
            "actual optional_import_policy effective_module_roots must cover every frozen baseline optional root"
        )
    return _validate_thresholds(baseline)


def _validate_legacy_zero_contract(baseline: Mapping[str, Any]) -> None:
    del baseline
    raise ArchitectureContractError(
        "legacy-zero is outside the generic Task 0 architecture measurement contract and requires separate frozen evidence"
    )


def _validate_against_baseline(
    actual: dict[str, Any],
    baseline: Mapping[str, Any],
    baseline_source_root: Path,
    *,
    require_no_cycles: bool,
    require_legacy_zero: bool,
) -> dict[str, Any]:
    thresholds = _validate_baseline_contract(baseline, actual, baseline_source_root)
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


def _open_capture_parent(parent: Path) -> int:
    """Open and identity-check the output parent without following a late symlink."""
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_CLOEXEC", 0)
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        expected = parent.stat(follow_symlinks=False)
        if not stat.S_ISDIR(expected.st_mode):
            raise ArchitectureContractError("capture output parent directory must be a real directory")
        parent_fd = os.open(parent, flags)
    except OSError as exc:
        raise ArchitectureContractError(f"cannot safely open capture output parent: {exc}") from exc
    try:
        actual = os.fstat(parent_fd)
        if (actual.st_dev, actual.st_ino) != (expected.st_dev, expected.st_ino):
            raise ArchitectureContractError("capture output parent changed while being opened")
        return parent_fd
    except BaseException:
        with suppress(OSError):
            os.close(parent_fd)
        raise


def _validate_capture_path(source_root: Path, path: Path) -> CaptureOutput:
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
    if resolved.exists() and not resolved.is_file():
        raise ArchitectureContractError("capture output must be a regular file when it already exists")
    return CaptureOutput(path=resolved, parent_fd=_open_capture_parent(parent))


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


def _close_capture_output(output: CaptureOutput) -> None:
    with suppress(OSError):
        os.close(output.parent_fd)


def _write_capture_atomically(
    output: CaptureOutput, serialized: bytes, source_root: Path, provenance: Mapping[str, Any]
) -> None:
    temporary_name: str | None = None
    temporary_fd: int | None = None
    _verify_source_provenance(source_root, provenance)
    try:
        for _ in range(128):
            candidate = f".{output.path.name}.{secrets.token_hex(16)}.tmp"
            try:
                temporary_fd = os.open(
                    candidate,
                    os.O_CREAT | os.O_EXCL | os.O_WRONLY | getattr(os, "O_CLOEXEC", 0),
                    0o600,
                    dir_fd=output.parent_fd,
                )
            except FileExistsError:
                continue
            temporary_name = candidate
            break
        if (
            temporary_name is None or temporary_fd is None
        ):  # pragma: no cover - cryptographic collisions are implausible.
            raise ArchitectureContractError("cannot reserve a unique capture temporary file")
        temporary = os.fdopen(temporary_fd, "wb")
        temporary_fd = None
        with temporary:
            temporary.write(serialized)
            temporary.flush()
            os.fsync(temporary.fileno())
        _verify_source_provenance(source_root, provenance)
        os.replace(
            temporary_name,
            output.path.name,
            src_dir_fd=output.parent_fd,
            dst_dir_fd=output.parent_fd,
        )
        os.fsync(output.parent_fd)
        temporary_name = None
    except OSError as exc:
        raise ArchitectureContractError(f"cannot atomically write capture output: {exc}") from exc
    finally:
        if temporary_fd is not None:
            with suppress(OSError):
                os.close(temporary_fd)
        if temporary_name is not None:
            with suppress(OSError):
                os.unlink(temporary_name, dir_fd=output.parent_fd)


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
        "--baseline-source-root",
        type=Path,
        help="clean Git root that owns the baseline's recorded policy blob; defaults to --source-root",
    )
    parser.add_argument(
        "--seal-baseline",
        action="store_true",
        help="seal --capture as a threshold-governed architecture baseline; requires --threshold-policy",
    )
    parser.add_argument(
        "--threshold-policy",
        type=Path,
        help="frozen source-tree threshold-policy JSON required by --seal-baseline",
    )
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
    if arguments.seal_baseline:
        if arguments.capture is None:
            raise ArchitectureContractError("--seal-baseline requires --capture")
        if arguments.baseline is not None:
            raise ArchitectureContractError("--seal-baseline cannot be combined with --baseline")
        if arguments.threshold_policy is None:
            raise ArchitectureContractError("--seal-baseline requires --threshold-policy")
        if arguments.optional_module:
            raise ArchitectureContractError(
                "--seal-baseline does not permit --optional-module without a frozen invocation manifest"
            )
    elif arguments.threshold_policy is not None:
        raise ArchitectureContractError("--threshold-policy is only valid with --seal-baseline")
    if arguments.baseline_source_root is not None and arguments.baseline is None:
        raise ArchitectureContractError("--baseline-source-root is only valid with --baseline")
    if arguments.require_legacy_zero:
        _validate_legacy_zero_contract({})
    package = _validate_package_name(arguments.package)
    try:
        source_root = arguments.source_root.resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise ArchitectureContractError("source-root must be an existing directory") from exc
    if not source_root.is_dir():
        raise ArchitectureContractError("source-root must be a directory")
    baseline_source_root = source_root
    if arguments.baseline_source_root is not None:
        try:
            requested_baseline_root = arguments.baseline_source_root.resolve(strict=True)
        except (OSError, RuntimeError) as exc:
            raise ArchitectureContractError("baseline-source-root must be an existing directory") from exc
        if not requested_baseline_root.is_dir():
            raise ArchitectureContractError("baseline-source-root must be a directory")
        baseline_source_root, _ = _source_provenance(requested_baseline_root)
    capture_output = _validate_capture_path(source_root, arguments.capture) if arguments.capture is not None else None
    try:
        baseline_path = _validate_baseline_path(arguments.baseline) if arguments.baseline is not None else None
        if capture_output is not None and baseline_path is not None:
            same_file = capture_output.path == baseline_path
            if capture_output.path.exists():
                try:
                    same_file = same_file or capture_output.path.samefile(baseline_path)
                except OSError as exc:
                    raise ArchitectureContractError(f"cannot compare capture output and baseline: {exc}") from exc
            if same_file:
                raise ArchitectureContractError("capture output must not replace the baseline input")
        artifact, provenance = _build_measurement(source_root, package, arguments.optional_module)
        if arguments.seal_baseline:
            assert arguments.threshold_policy is not None
            threshold_rules, threshold_policy = _load_frozen_threshold_policy(
                source_root,
                str(provenance["commit"]),
                arguments.threshold_policy,
            )
            artifact = _seal_architecture_baseline(artifact, threshold_rules, threshold_policy)
        elif baseline_path is not None:
            baseline, baseline_sha256 = _load_baseline(baseline_path)
            artifact["baseline_validation"] = _validate_against_baseline(
                artifact,
                baseline,
                baseline_source_root,
                require_no_cycles=arguments.require_no_cycles,
                require_legacy_zero=arguments.require_legacy_zero,
            )
            artifact["baseline_validation"]["baseline_sha256"] = baseline_sha256
        if arguments.require_no_cycles:
            summary = artifact["measurements"]["summary"]
            if summary["internal_cycle_count"] != 0:
                raise ArchitectureContractError("architecture cycle requirement failed: internal_cycle_count must be 0")
        serialized = _serialize_artifact(artifact)
        if capture_output is not None:
            _write_capture_atomically(capture_output, serialized, source_root, provenance)
        return {
            "capture_sha256": _sha256_bytes(serialized),
            "package": package,
            "result": "sealed" if arguments.seal_baseline else "validated" if baseline_path is not None else "captured",
            "source_commit": provenance["commit"],
        }
    finally:
        if capture_output is not None:
            _close_capture_output(capture_output)


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
