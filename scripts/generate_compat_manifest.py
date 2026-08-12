#!/usr/bin/env python
"""Generate portable compatibility manifests from pinned Git source blobs."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
import operator
import os
import subprocess
import tempfile
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

EMPYRICAL_COMMIT = "74655e974ed2935563820c548c339731f1fe0621"
PYFOLIO_COMMIT = "724bbd7dbed9a88bb47e1057f2ca29b3409d8e7a"
PYFOLIO_PROFILE = (
    "create_full_tear_sheet",
    "create_simple_tear_sheet",
    "create_returns_tear_sheet",
    "create_position_tear_sheet",
    "create_txn_tear_sheet",
    "create_round_trip_tear_sheet",
    "create_interesting_times_tear_sheet",
    "create_capacity_tear_sheet",
    "create_bayesian_tear_sheet",
    "create_risk_tear_sheet",
    "create_perf_attrib_tear_sheet",
)
COMPAT_PENDING = dict.fromkeys(("C0", "C1", "C2", "C3", "C4"), "not-verified")
MAX_RESOLVE_DEPTH = 32
MAX_NODE_VISITS = 2048
MAX_CONTAINER_ITEMS = 256
MAX_ABS_INTEGER = 10**12
MAX_ABS_FLOAT = 10**12
MAX_STRING_LENGTH = 4096
MAX_POWER_BASE = 10**6
MAX_POWER_EXPONENT = 12
LOCAL_GIT_TIMEOUT_SECONDS = 30
ORACLE_TIMEOUT_SECONDS = 120
NONINTERACTIVE_ENV_OVERRIDES = {"GIT_TERMINAL_PROMPT": "0", "GIT_ASKPASS": ""}
SAFE_BINARY_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
SAFE_UNARY_OPERATORS = {ast.UAdd: operator.pos, ast.USub: operator.neg}
UNRESOLVED = object()
ORACLE_SCRIPT = r"""
import hashlib
import importlib
import inspect
import json
import os
from pathlib import Path
import subprocess
import sys

GIT_TIMEOUT_SECONDS = 30
environment = os.environ.copy()
environment.update({"GIT_TERMINAL_PROMPT": "0", "GIT_ASKPASS": ""})

def run_git(arguments, operation, root):
    try:
        return subprocess.run(
            ["git", *arguments],
            cwd=root,
            check=True,
            capture_output=True,
            env=environment,
            stdin=subprocess.DEVNULL,
            timeout=GIT_TIMEOUT_SECONDS,
        )
    except subprocess.TimeoutExpired as error:
        raise RuntimeError(
            f"{operation} timed out after {GIT_TIMEOUT_SECONDS}s"
        ) from error
    except subprocess.CalledProcessError as error:
        detail = (error.stderr or error.stdout or b"").decode(errors="replace").strip()
        raise RuntimeError(f"{operation} failed: {detail or error}") from error

root = Path(sys.argv[1]).resolve()
package_name = sys.argv[2]
names = json.loads(sys.argv[3])
source_paths = json.loads(sys.argv[4])
sys.path.insert(0, str(root))
package = importlib.import_module(package_name)
module_path = Path(package.__file__).resolve()
try:
    relative_module = module_path.relative_to(root).as_posix()
except ValueError as error:
    raise RuntimeError(f"oracle imported outside pinned root: {module_path}") from error
commit = run_git(["rev-parse", "HEAD"], "resolve oracle commit", root).stdout.decode().strip()
source_files = []
for relative in source_paths:
    blob = run_git(
        ["show", f"{commit}:{package_name}/{relative}"],
        f"read oracle source blob {relative}",
        root,
    ).stdout
    source_files.append({
        "path": relative,
        "sha256": hashlib.sha256(blob).hexdigest(),
    })
print(json.dumps({
    "version": package.__version__,
    "commit": commit,
    "module_path": relative_module,
    "source_root": "isolated-pinned-git-checkout",
    "source_files": source_files,
    "signatures": {name: str(inspect.signature(getattr(package, name))) for name in names},
}, sort_keys=True))
"""


def _run_process(
    command: list[str],
    *,
    operation: str,
    cwd: Path | None = None,
    timeout: int,
    text: bool = False,
) -> subprocess.CompletedProcess[Any]:
    """Run one bounded, noninteractive Git/oracle subprocess."""
    environment = os.environ.copy()
    environment.update(NONINTERACTIVE_ENV_OVERRIDES)
    try:
        return subprocess.run(
            command,
            cwd=cwd,
            check=True,
            capture_output=True,
            text=text,
            env=environment,
            stdin=subprocess.DEVNULL,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired as error:
        raise ValueError(f"{operation} timed out after {timeout}s") from error
    except subprocess.CalledProcessError as error:
        stderr = error.stderr or error.stdout or ""
        if isinstance(stderr, bytes):
            stderr = stderr.decode(errors="replace")
        detail = stderr.strip()
        raise ValueError(f"{operation} failed: {detail or error}") from error


@dataclass(frozen=True)
class Export:
    """A public name and the source definition that provides it."""

    public_name: str
    source_name: str
    source_module: str


@dataclass(frozen=True)
class _ModuleReference:
    module: str


class PinnedGitSource:
    """Read bytes from one immutable Git commit, never from worktree files."""

    def __init__(self, root: Path, commit: str) -> None:
        self.root = root.resolve()
        self.commit = commit
        _run_process(
            ["git", "cat-file", "-e", f"{commit}^{{commit}}"],
            cwd=self.root,
            operation="validate pinned Git commit",
            timeout=LOCAL_GIT_TIMEOUT_SECONDS,
        )

    def read_bytes(self, path: str) -> bytes:
        result = _run_process(
            ["git", "show", f"{self.commit}:{path}"],
            cwd=self.root,
            operation=f"read pinned Git blob {path}",
            timeout=LOCAL_GIT_TIMEOUT_SECONDS,
        )
        return result.stdout

    def read_text(self, path: str) -> str:
        return self.read_bytes(path).decode("utf-8")

    def sha256(self, path: str) -> str:
        return hashlib.sha256(self.read_bytes(path)).hexdigest()


class StaticConstantResolver:
    """Resolve a deliberately small, import-free subset of Python constants."""

    def __init__(self, trees: dict[str, ast.Module]) -> None:
        self.trees = trees
        self.assignments: dict[str, dict[str, ast.expr]] = {}
        self.imports: dict[str, dict[str, tuple[str, str | None]]] = {}
        for module, tree in trees.items():
            assignments: dict[str, ast.expr] = {}
            imports: dict[str, tuple[str, str | None]] = {}
            for node in tree.body:
                if isinstance(node, (ast.Assign, ast.AnnAssign)):
                    targets = node.targets if isinstance(node, ast.Assign) else [node.target]
                    for target in targets:
                        if isinstance(target, ast.Name):
                            assignments[target.id] = node.value
                elif isinstance(node, ast.ImportFrom) and node.level:
                    target_module = node.module
                    for alias in node.names:
                        if alias.name == "*":
                            continue
                        local_name = alias.asname or alias.name
                        if target_module is None:
                            imports[local_name] = (alias.name, None)
                        else:
                            imports[local_name] = (target_module, alias.name)
            self.assignments[module] = assignments
            self.imports[module] = imports
        self.last_unresolved_reason = ""

    def resolve(self, node: ast.expr | None, module: str) -> tuple[Any, bool]:
        self.last_unresolved_reason = ""
        value = self._resolve(node, module, set(), depth=0, budget={"visits": 0})
        return (None, False) if value is UNRESOLVED else (value, True)

    def _fail(self, reason: str) -> object:
        if not self.last_unresolved_reason:
            self.last_unresolved_reason = reason
        return UNRESOLVED

    def _resolve(
        self,
        node: ast.expr | None,
        module: str,
        seen: set[tuple[str, str]],
        *,
        depth: int,
        budget: dict[str, int],
    ) -> Any:
        if depth > MAX_RESOLVE_DEPTH:
            return self._fail(f"maximum constant resolution depth {MAX_RESOLVE_DEPTH} exceeded")
        budget["visits"] += 1
        if budget["visits"] > MAX_NODE_VISITS:
            return self._fail(f"maximum constant AST node visits {MAX_NODE_VISITS} exceeded")
        if node is None:
            return None
        if isinstance(node, ast.Constant):
            if _bounded_json_scalar(node.value):
                return node.value
            return self._fail(f"unsupported or unbounded scalar constant: {type(node.value).__name__}")
        if isinstance(node, ast.Set):
            return self._fail("set literals are not JSON-safe")
        if isinstance(node, (ast.Tuple, ast.List)):
            if len(node.elts) > MAX_CONTAINER_ITEMS:
                return self._fail(f"container has {len(node.elts)} items; maximum is {MAX_CONTAINER_ITEMS}")
            values = []
            for item in node.elts:
                value = self._resolve(
                    item,
                    module,
                    seen.copy(),
                    depth=depth + 1,
                    budget=budget,
                )
                if value is UNRESOLVED:
                    return UNRESOLVED
                values.append(value)
            if isinstance(node, ast.Tuple):
                return tuple(values)
            return values
        if isinstance(node, ast.Dict):
            if len(node.keys) > MAX_CONTAINER_ITEMS:
                return self._fail(f"container has {len(node.keys)} items; maximum is {MAX_CONTAINER_ITEMS}")
            result: dict[str, Any] = {}
            for key_node, value_node in zip(node.keys, node.values, strict=True):
                key = self._resolve(
                    key_node,
                    module,
                    seen.copy(),
                    depth=depth + 1,
                    budget=budget,
                )
                if key is UNRESOLVED:
                    return UNRESOLVED
                if not isinstance(key, str):
                    return self._fail("dictionary keys must be bounded strings")
                value = self._resolve(
                    value_node,
                    module,
                    seen.copy(),
                    depth=depth + 1,
                    budget=budget,
                )
                if value is UNRESOLVED:
                    return UNRESOLVED
                result[key] = value
            return result
        if isinstance(node, ast.Name):
            key = (module, node.id)
            if key in seen:
                return self._fail(f"constant reference cycle at {module}.{node.id}")
            seen.add(key)
            if node.id in self.assignments.get(module, {}):
                return self._resolve(
                    self.assignments[module][node.id],
                    module,
                    seen,
                    depth=depth + 1,
                    budget=budget,
                )
            imported = self.imports.get(module, {}).get(node.id)
            if imported is None:
                return self._fail(f"unknown name {module}.{node.id}")
            target_module, target_name = imported
            if target_name is None:
                return _ModuleReference(target_module)
            return self._resolve(
                ast.Name(id=target_name),
                target_module,
                seen,
                depth=depth + 1,
                budget=budget,
            )
        if isinstance(node, ast.Attribute):
            owner = self._resolve(
                node.value,
                module,
                seen.copy(),
                depth=depth + 1,
                budget=budget,
            )
            if not isinstance(owner, _ModuleReference):
                return self._fail(f"attribute owner for {ast.unparse(node)} is not a local module")
            return self._resolve(
                ast.Name(id=node.attr),
                owner.module,
                seen,
                depth=depth + 1,
                budget=budget,
            )
        if isinstance(node, ast.BinOp) and type(node.op) in SAFE_BINARY_OPERATORS:
            left = self._resolve(
                node.left,
                module,
                seen.copy(),
                depth=depth + 1,
                budget=budget,
            )
            right = self._resolve(
                node.right,
                module,
                seen.copy(),
                depth=depth + 1,
                budget=budget,
            )
            if left is UNRESOLVED or right is UNRESOLVED:
                return UNRESOLVED
            if not (_bounded_number(left) and _bounded_number(right)):
                return self._fail("arithmetic operands must be bounded finite numbers")
            if isinstance(node.op, ast.Pow) and (
                not isinstance(right, int)
                or isinstance(right, bool)
                or right < 0
                or right > MAX_POWER_EXPONENT
                or abs(left) > MAX_POWER_BASE
            ):
                return self._fail(f"power base/exponent exceeds bounds ({MAX_POWER_BASE}, {MAX_POWER_EXPONENT})")
            try:
                value = SAFE_BINARY_OPERATORS[type(node.op)](left, right)
            except (ArithmeticError, TypeError, ValueError):
                return self._fail(f"unsafe arithmetic expression: {ast.unparse(node)}")
            if not _bounded_number(value):
                return self._fail("arithmetic result is non-finite or exceeds numeric bounds")
            return value
        if isinstance(node, ast.UnaryOp) and type(node.op) in SAFE_UNARY_OPERATORS:
            operand = self._resolve(
                node.operand,
                module,
                seen.copy(),
                depth=depth + 1,
                budget=budget,
            )
            if operand is UNRESOLVED:
                return UNRESOLVED
            if not _bounded_number(operand):
                return self._fail("unary arithmetic operand must be a bounded finite number")
            try:
                value = SAFE_UNARY_OPERATORS[type(node.op)](operand)
            except (ArithmeticError, TypeError, ValueError):
                return self._fail(f"unsafe unary arithmetic expression: {ast.unparse(node)}")
            return value if _bounded_number(value) else self._fail("unary result exceeds numeric bounds")
        return self._fail(f"unsupported constant expression: {type(node).__name__}")


def _bounded_json_scalar(value: Any) -> bool:
    if value is None or isinstance(value, bool):
        return True
    if isinstance(value, int):
        return abs(value) <= MAX_ABS_INTEGER
    if isinstance(value, float):
        return math.isfinite(value) and abs(value) <= MAX_ABS_FLOAT
    if isinstance(value, str):
        return len(value) <= MAX_STRING_LENGTH
    return False


def _bounded_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and _bounded_json_scalar(value)


def _portable_value(value: Any, *, depth: int = 0) -> bool:
    if depth > MAX_RESOLVE_DEPTH:
        return False
    if _bounded_json_scalar(value):
        return True
    if isinstance(value, (list, tuple)):
        return len(value) <= MAX_CONTAINER_ITEMS and all(_portable_value(item, depth=depth + 1) for item in value)
    if isinstance(value, dict):
        return len(value) <= MAX_CONTAINER_ITEMS and all(
            isinstance(key, str) and _bounded_json_scalar(key) and _portable_value(item, depth=depth + 1)
            for key, item in value.items()
        )
    return False


def _read_ast(text: str, filename: str) -> ast.Module:
    return ast.parse(text, filename=filename)


def _literal_assignment(tree: ast.Module, name: str) -> Any:
    for node in tree.body:
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            if any(isinstance(target, ast.Name) and target.id == name for target in targets):
                return ast.literal_eval(node.value)
    raise ValueError(f"static assignment {name!r} was not found")


def _relative_module(node: ast.ImportFrom, alias: ast.alias) -> str | None:
    if not node.level:
        return None
    return node.module if node.module is not None else alias.name


def _resolve_public_exports(
    module: str,
    trees: dict[str, ast.Module],
    active: frozenset[str] = frozenset(),
) -> list[Export]:
    """Mirror static alias/star namespace behavior for local modules."""
    if module in active or module not in trees:
        return []
    active = active | {module}
    namespace: dict[str, Export] = {}
    explicit_all: list[str] | None = None
    for node in trees[module].body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            namespace[node.name] = Export(node.name, node.name, module)
        elif isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            for target in targets:
                if isinstance(target, ast.Name):
                    namespace[target.id] = Export(target.id, target.id, module)
                    if target.id == "__all__":
                        try:
                            explicit_all = list(ast.literal_eval(node.value))
                        except (ValueError, TypeError):
                            explicit_all = None
        elif isinstance(node, ast.Import):
            for alias in node.names:
                public_name = alias.asname or alias.name.split(".")[0]
                namespace[public_name] = Export(public_name, alias.name, module)
        elif isinstance(node, ast.ImportFrom):
            for alias in node.names:
                target_module = _relative_module(node, alias)
                if alias.name == "*":
                    if target_module is None:
                        continue
                    for exported in _resolve_public_exports(target_module, trees, active):
                        namespace[exported.public_name] = exported
                    continue
                public_name = alias.asname or alias.name
                if target_module is None:
                    namespace[public_name] = Export(public_name, alias.name, module)
                    continue
                target_exports = {
                    exported.public_name: exported for exported in _resolve_public_exports(target_module, trees, active)
                }
                source = target_exports.get(alias.name)
                namespace[public_name] = (
                    Export(public_name, source.source_name, source.source_module)
                    if source is not None
                    else Export(public_name, alias.name, target_module)
                )
    names = explicit_all if explicit_all is not None else [name for name in namespace if not name.startswith("_")]
    return [
        Export(name, namespace[name].source_name, namespace[name].source_module)
        if name in namespace
        else Export(name, name, module)
        for name in names
    ]


def _definitions(tree: ast.Module) -> dict[str, ast.AST]:
    definitions: dict[str, ast.AST] = {}
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            definitions[node.name] = node
        elif isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            for target in targets:
                if isinstance(target, ast.Name):
                    definitions[target.id] = node
    return definitions


def _expression(node: ast.expr | None) -> str | None:
    return None if node is None else ast.unparse(node)


def _parameter_record(
    argument: ast.arg,
    kind: str,
    default: ast.expr | None,
    resolver: StaticConstantResolver,
    module: str,
) -> dict[str, Any]:
    required = default is None
    value, resolved = (None, True) if required else resolver.resolve(default, module)
    result = {
        "name": argument.arg,
        "kind": kind,
        "required": required,
        "default": value,
        "default_expression": _expression(default),
        "resolved": resolved,
        "annotation": _expression(argument.annotation),
    }
    if not resolved:
        result["unresolved_reason"] = resolver.last_unresolved_reason
    return result


def _parameters(
    node: ast.FunctionDef | ast.AsyncFunctionDef,
    resolver: StaticConstantResolver,
    module: str,
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    positional = [*node.args.posonlyargs, *node.args.args]
    defaults: list[ast.expr | None] = [None] * (len(positional) - len(node.args.defaults))
    defaults.extend(node.args.defaults)
    for index, (argument, default) in enumerate(zip(positional, defaults, strict=True)):
        kind = "POSITIONAL_ONLY" if index < len(node.args.posonlyargs) else "POSITIONAL_OR_KEYWORD"
        result.append(_parameter_record(argument, kind, default, resolver, module))
    if node.args.vararg is not None:
        result.append(_parameter_record(node.args.vararg, "VAR_POSITIONAL", None, resolver, module))
    for argument, default in zip(node.args.kwonlyargs, node.args.kw_defaults, strict=True):
        result.append(_parameter_record(argument, "KEYWORD_ONLY", default, resolver, module))
    if node.args.kwarg is not None:
        result.append(_parameter_record(node.args.kwarg, "VAR_KEYWORD", None, resolver, module))
    return result


def _canonical_signature(
    node: ast.FunctionDef | ast.AsyncFunctionDef,
    parameters: list[dict[str, Any]],
) -> str | None:
    if any(not parameter["resolved"] for parameter in parameters):
        return None
    parts: list[str] = []
    positional_only = sum(parameter["kind"] == "POSITIONAL_ONLY" for parameter in parameters)
    has_varargs = any(parameter["kind"] == "VAR_POSITIONAL" for parameter in parameters)
    keyword_marker_added = False
    positional_seen = 0
    for parameter in parameters:
        kind = parameter["kind"]
        if kind == "KEYWORD_ONLY" and not has_varargs and not keyword_marker_added:
            parts.append("*")
            keyword_marker_added = True
        prefix = "*" if kind == "VAR_POSITIONAL" else "**" if kind == "VAR_KEYWORD" else ""
        rendered = f"{prefix}{parameter['name']}"
        if parameter["annotation"] is not None:
            rendered += f": {parameter['annotation']}"
        if not parameter["required"]:
            rendered += f"={parameter['default']!r}"
        parts.append(rendered)
        if kind in {"POSITIONAL_ONLY", "POSITIONAL_OR_KEYWORD"}:
            positional_seen += 1
            if positional_seen == positional_only:
                parts.append("/")
    rendered = f"({', '.join(parts)})"
    if node.returns is not None:
        rendered += f" -> {ast.unparse(node.returns)}"
    return rendered


def _source_record(source: PinnedGitSource, package: str, path: str) -> dict[str, str]:
    repository_path = f"{package}/{path}"
    return {"path": path, "sha256": source.sha256(repository_path)}


def _entry(
    *,
    package: str,
    source: PinnedGitSource,
    export: Export,
    node: ast.AST | None,
    resolver: StaticConstantResolver,
    definitions: dict[str, dict[str, ast.AST]],
) -> dict[str, Any]:
    module_path = f"{export.source_module}.py"
    is_callable = isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
    parameters: list[dict[str, Any]] = []
    signature = None
    factory_name = None
    factory_template_line = None
    needs_review = node is None or isinstance(node, ast.ClassDef)
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
        parameters = _parameters(node, resolver, export.source_module)
        signature = _canonical_signature(node, parameters)
        needs_review = signature is None
    elif isinstance(node, (ast.Assign, ast.AnnAssign)):
        value = node.value
        if isinstance(value, ast.Call) and isinstance(value.func, ast.Name):
            factory = definitions[export.source_module].get(value.func.id)
            generated = next(
                (
                    child
                    for child in getattr(factory, "body", [])
                    if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef))
                ),
                None,
            )
            if value.func.id.startswith("_create_") and generated is not None:
                is_callable = True
                parameters = _parameters(generated, resolver, export.source_module)
                signature = _canonical_signature(generated, parameters)
                factory_name = value.func.id
                factory_template_line = generated.lineno
                needs_review = True
    compatibility = dict(COMPAT_PENDING)
    if not is_callable:
        compatibility["C1"] = "not-applicable"
    result: dict[str, Any] = {
        "symbol": export.public_name,
        "source_name": export.source_name,
        "public_path": f"{package}.{export.public_name}",
        "kind": "callable" if is_callable else "constant",
        "source": {
            **_source_record(source, package, module_path),
            "line": getattr(node, "lineno", None),
        },
        "parameters": parameters,
        "signature": signature,
        "signature_expression": (
            f"({ast.unparse(node.args)})" if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) else None
        ),
        "return_annotation": (
            _expression(node.returns) if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) else None
        ),
        "extraction": "static_ast_from_pinned_git_blob",
        "target_evidence": {
            "source_frozen": node is not None,
            "signature_frozen": signature is not None,
        },
        "needs_dynamic_review": needs_review,
        "reviewed": False,
        "compatibility": compatibility,
    }
    if factory_name is not None:
        result.update({"factory": factory_name, "factory_template_line": factory_template_line})
    if not is_callable and isinstance(node, (ast.Assign, ast.AnnAssign)):
        value, resolved = resolver.resolve(node.value, export.source_module)
        result.update(
            {
                "value": value,
                "value_expression": ast.unparse(node.value),
                "value_resolved": resolved,
            }
        )
        result["needs_dynamic_review"] = not resolved
    return result


def _package_trees(
    source: PinnedGitSource,
    package: str,
    modules: tuple[str, ...],
) -> dict[str, ast.Module]:
    return {
        module: _read_ast(
            source.read_text(f"{package}/{module}.py"),
            f"{package}/{module}.py@{source.commit}",
        )
        for module in modules
    }


def _manifest_base(
    project: str,
    version: str,
    commit: str,
    source_files: list[dict[str, str]],
) -> dict[str, Any]:
    return {
        "schema_version": 2,
        "project": project,
        "version": version,
        "commit": commit,
        "fixture_source": {
            "mode": "static_ast_from_pinned_git_blob",
            "generator": "scripts/generate_compat_manifest.py",
            "root": f"external Git object database supplied with --{project}-root",
        },
        "oracle_verification": {"status": "not_run", "reviewed": False},
        "source_files": source_files,
    }


def _generate_empyrical(root: Path) -> dict[str, Any]:
    source = PinnedGitSource(root, EMPYRICAL_COMMIT)
    _require_head(source, EMPYRICAL_COMMIT, "empyrical")
    modules = ("__init__", "stats", "periods", "perf_attrib", "utils")
    trees = _package_trees(source, "empyrical", modules)
    resolver = StaticConstantResolver(trees)
    definitions = {module: _definitions(tree) for module, tree in trees.items()}
    exports = _resolve_public_exports("__init__", trees)
    symbols = [
        _entry(
            package="empyrical",
            source=source,
            export=export,
            node=definitions.get(export.source_module, {}).get(export.source_name),
            resolver=resolver,
            definitions=definitions,
        )
        for export in exports
    ]
    public_symbols = [entry["symbol"] for entry in symbols]
    callables = [entry for entry in symbols if entry["kind"] == "callable"]
    if (len(public_symbols), len(callables)) != (54, 49):
        raise ValueError(f"unexpected empyrical surface: {len(public_symbols)} symbols, {len(callables)} callables")
    source_files = [_source_record(source, "empyrical", f"{module}.py") for module in modules]
    return {
        **_manifest_base(
            "empyrical", _literal_assignment(trees["__init__"], "__version__"), source.commit, source_files
        ),
        "public_symbols": public_symbols,
        "callables": callables,
        "symbols": symbols,
    }


def _generate_pyfolio(root: Path) -> dict[str, Any]:
    source = PinnedGitSource(root, PYFOLIO_COMMIT)
    _require_head(source, PYFOLIO_COMMIT, "pyfolio")
    modules = ("__init__", "tears", "utils")
    trees = _package_trees(source, "pyfolio", modules)
    resolver = StaticConstantResolver(trees)
    definitions = {module: _definitions(tree) for module, tree in trees.items()}
    exports = {export.public_name: export for export in _resolve_public_exports("__init__", trees)}
    profile = {}
    for name in PYFOLIO_PROFILE:
        export = exports.get(name, Export(name, name, "tears"))
        entry = _entry(
            package="pyfolio",
            source=source,
            export=export,
            node=definitions.get(export.source_module, {}).get(export.source_name),
            resolver=resolver,
            definitions=definitions,
        )
        if entry["kind"] != "callable":
            raise ValueError(f"pyfolio profile entry {name!r} is not a static callable")
        profile[name] = entry
    source_files = [_source_record(source, "pyfolio", f"{module}.py") for module in modules]
    return {
        **_manifest_base("pyfolio", _literal_assignment(trees["__init__"], "__version__"), source.commit, source_files),
        "compatibility_profile": profile,
    }


def _require_head(source: PinnedGitSource, expected: str, project: str) -> None:
    actual = _run_process(
        ["git", "rev-parse", "HEAD"],
        cwd=source.root,
        operation=f"resolve {project} checkout HEAD",
        timeout=LOCAL_GIT_TIMEOUT_SECONDS,
        text=True,
    ).stdout.strip()
    if actual != expected:
        raise ValueError(f"{project} root is at {actual}; expected pinned commit {expected}")


def _generate_flat_migrations(repo_root: Path) -> dict[str, Any]:
    init_path = repo_root / "fincore" / "__init__.py"
    tree = _read_ast(init_path.read_text(encoding="utf-8"), init_path.as_posix())
    entries = []
    for symbol, target in _literal_assignment(tree, "_FLAT_API").items():
        current_target = ".".join(target)
        entries.append(
            {
                "symbol": symbol,
                "current_target": current_target,
                "recommended_target": (
                    f"fincore.empyrical.{symbol}" if symbol != "information_ratio" else current_target
                ),
                "deprecate_in": None,
                "remove_or_switch_in": "next-major-not-scheduled",
                "status": "unchanged-in-0.3.x",
            }
        )
    return {
        "schema_version": 2,
        "fincore_version": _literal_assignment(tree, "__version__"),
        "policy": "preserve-0.3.x-flat-api",
        "source": {"path": "fincore/__init__.py", "sha256": hashlib.sha256(init_path.read_bytes()).hexdigest()},
        "entries": entries,
    }


def _execute_oracle(
    interpreter: Path,
    checkout: Path,
    package: str,
    names: list[str],
    source_paths: list[str],
) -> subprocess.CompletedProcess[Any]:
    return _run_process(
        [
            interpreter.as_posix(),
            "-I",
            "-c",
            ORACLE_SCRIPT,
            checkout.as_posix(),
            package,
            json.dumps(names),
            json.dumps(source_paths),
        ],
        operation=f"execute isolated oracle for {package}",
        timeout=ORACLE_TIMEOUT_SECONDS,
        text=True,
    )


def _run_oracle(
    interpreter: Path,
    root: Path,
    package: str,
    names: list[str],
    *,
    expected_version: str,
    expected_commit: str,
    expected_source_files: list[dict[str, str]],
) -> dict[str, Any]:
    """Import only an isolated checkout of the exact pinned commit."""
    with tempfile.TemporaryDirectory(prefix=f"{package}-oracle-") as temporary:
        checkout = Path(temporary) / "checkout"
        _run_process(
            ["git", "clone", "--quiet", "--no-checkout", root.resolve().as_posix(), checkout.as_posix()],
            operation=f"clone pinned {package} source for oracle",
            timeout=LOCAL_GIT_TIMEOUT_SECONDS,
        )
        _run_process(
            ["git", "checkout", "--quiet", expected_commit],
            cwd=checkout,
            operation=f"checkout pinned {package} oracle commit",
            timeout=LOCAL_GIT_TIMEOUT_SECONDS,
        )
        result = _execute_oracle(
            interpreter,
            checkout,
            package,
            names,
            [item["path"] for item in expected_source_files],
        )
    payload = json.loads(result.stdout)
    if payload["version"] != expected_version:
        raise ValueError(f"oracle version {payload['version']!r} != pinned version {expected_version!r}")
    if payload["commit"] != expected_commit:
        raise ValueError(f"oracle commit {payload['commit']} != pinned commit {expected_commit}")
    if payload["source_files"] != expected_source_files:
        raise ValueError("oracle source hashes differ from pinned manifest source hashes")
    if not payload["module_path"].startswith(f"{package}/"):
        raise ValueError("oracle module path is outside the requested pinned package")
    return {"status": "captured-unreviewed", "reviewed": False, **payload}


def _without_review(value: dict[str, Any]) -> dict[str, Any]:
    result = deepcopy(value)
    result.pop("reviewed", None)
    return result


def _entry_evidence(manifest: dict[str, Any], entry: dict[str, Any]) -> str:
    evidence = {
        "project": manifest["project"],
        "commit": manifest["commit"],
        "source_files": manifest["source_files"],
        "symbol": entry["symbol"],
        "signature": entry.get("signature"),
        "oracle": _without_review(manifest["oracle_verification"]),
    }
    return hashlib.sha256(json.dumps(evidence, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _merge_review_attestations(
    generated: dict[str, Any],
    previous: dict[str, Any] | None,
    section: str,
) -> dict[str, Any]:
    result = deepcopy(generated)
    generated_entries = result[section]
    iterable = generated_entries.values() if isinstance(generated_entries, dict) else generated_entries
    for entry in iterable:
        entry["evidence_key"] = _entry_evidence(result, entry)
    if previous is None:
        return result
    previous_entries = previous.get(section, {})
    if isinstance(previous_entries, dict):
        previous_by_symbol = {entry["symbol"]: entry for entry in previous_entries.values()}
    else:
        previous_by_symbol = {entry["symbol"]: entry for entry in previous_entries}
    for entry in iterable:
        old = previous_by_symbol.get(entry["symbol"])
        old_key = (
            old.get("evidence_key")
            if old and old.get("evidence_key")
            else _entry_evidence(previous, old)
            if old
            else None
        )
        if old and old.get("reviewed") is True and old_key == entry["evidence_key"]:
            entry["reviewed"] = True
    old_oracle = previous.get("oracle_verification", {})
    if old_oracle.get("reviewed") is True and _without_review(old_oracle) == _without_review(
        result["oracle_verification"]
    ):
        result["oracle_verification"]["reviewed"] = True
    return result


def _load_existing(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--empyrical-root", type=Path, required=True)
    parser.add_argument("--pyfolio-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--oracle-python",
        type=Path,
        help="optional isolated interpreter with pinned upstream dependencies installed",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    empyrical_path = args.output / "empyrical-0.6.0-api.json"
    pyfolio_path = args.output / "pyfolio-0.9.6-api.json"
    empyrical = _generate_empyrical(args.empyrical_root.resolve())
    pyfolio = _generate_pyfolio(args.pyfolio_root.resolve())
    if args.oracle_python is not None:
        empyrical["oracle_verification"] = _run_oracle(
            args.oracle_python,
            args.empyrical_root,
            "empyrical",
            [entry["symbol"] for entry in empyrical["callables"]],
            expected_version=empyrical["version"],
            expected_commit=empyrical["commit"],
            expected_source_files=empyrical["source_files"],
        )
        pyfolio["oracle_verification"] = _run_oracle(
            args.oracle_python,
            args.pyfolio_root,
            "pyfolio",
            list(pyfolio["compatibility_profile"]),
            expected_version=pyfolio["version"],
            expected_commit=pyfolio["commit"],
            expected_source_files=pyfolio["source_files"],
        )
    empyrical = _merge_review_attestations(empyrical, _load_existing(empyrical_path), "symbols")
    empyrical["callables"] = [entry for entry in empyrical["symbols"] if entry["kind"] == "callable"]
    pyfolio = _merge_review_attestations(pyfolio, _load_existing(pyfolio_path), "compatibility_profile")
    repo_root = Path(__file__).resolve().parents[1]
    _write_json(empyrical_path, empyrical)
    _write_json(pyfolio_path, pyfolio)
    _write_json(args.output / "fincore-flat-api-migrations.json", _generate_flat_migrations(repo_root))


if __name__ == "__main__":
    main()
