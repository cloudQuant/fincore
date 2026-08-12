#!/usr/bin/env python
"""Generate portable compatibility manifests from pinned upstream source trees.

The default path is deliberately static: sibling projects are parsed with the
standard-library AST and are never imported into this interpreter.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import subprocess
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
COMPAT_PENDING = {
    "C0": "not-verified",
    "C1": "not-verified",
    "C2": "not-verified",
    "C3": "not-verified",
    "C4": "not-verified",
}
ORACLE_SCRIPT = r"""
import importlib
import inspect
import json
import sys

package = importlib.import_module(sys.argv[1])
names = json.loads(sys.argv[2])
result = {}
for name in names:
    value = getattr(package, name)
    result[name] = str(inspect.signature(value))
print(json.dumps(result, sort_keys=True))
"""


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read_ast(path: Path) -> ast.Module:
    return ast.parse(path.read_text(encoding="utf-8"), filename=path.as_posix())


def _literal_assignment(tree: ast.Module, name: str) -> Any:
    for node in tree.body:
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            if any(isinstance(target, ast.Name) and target.id == name for target in targets):
                return ast.literal_eval(node.value)
    raise ValueError(f"static assignment {name!r} was not found")


def _git_commit(root: Path) -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _require_pin(root: Path, expected: str, project: str) -> str:
    actual = _git_commit(root)
    if actual != expected:
        raise ValueError(f"{project} root is at {actual}; expected pinned commit {expected}")
    return actual


def _expr(node: ast.expr | None) -> str | None:
    return None if node is None else ast.unparse(node)


def _signature(node: ast.FunctionDef | ast.AsyncFunctionDef) -> str:
    rendered = f"({ast.unparse(node.args)})"
    if node.returns is not None:
        rendered = f"{rendered} -> {ast.unparse(node.returns)}"
    return rendered


def _parameters(node: ast.FunctionDef | ast.AsyncFunctionDef) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    positional = [*node.args.posonlyargs, *node.args.args]
    defaults: list[ast.expr | None] = [None] * (len(positional) - len(node.args.defaults))
    defaults.extend(node.args.defaults)
    posonly_count = len(node.args.posonlyargs)
    for index, (argument, default) in enumerate(zip(positional, defaults, strict=True)):
        result.append(
            {
                "name": argument.arg,
                "kind": "POSITIONAL_ONLY" if index < posonly_count else "POSITIONAL_OR_KEYWORD",
                "required": default is None,
                "default": _expr(default),
                "annotation": _expr(argument.annotation),
            }
        )
    if node.args.vararg is not None:
        result.append(
            {
                "name": node.args.vararg.arg,
                "kind": "VAR_POSITIONAL",
                "required": False,
                "default": None,
                "annotation": _expr(node.args.vararg.annotation),
            }
        )
    for argument, default in zip(node.args.kwonlyargs, node.args.kw_defaults, strict=True):
        result.append(
            {
                "name": argument.arg,
                "kind": "KEYWORD_ONLY",
                "required": default is None,
                "default": _expr(default),
                "annotation": _expr(argument.annotation),
            }
        )
    if node.args.kwarg is not None:
        result.append(
            {
                "name": node.args.kwarg.arg,
                "kind": "VAR_KEYWORD",
                "required": False,
                "default": None,
                "annotation": _expr(node.args.kwarg.annotation),
            }
        )
    return result


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


def _source_record(package_root: Path, path: Path) -> dict[str, str]:
    return {"path": path.relative_to(package_root).as_posix(), "sha256": _sha256(path)}


def _symbol_entry(
    *,
    package: str,
    package_root: Path,
    source_path: Path,
    name: str,
    node: ast.AST | None,
) -> dict[str, Any]:
    is_callable = isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
    needs_review = node is None or isinstance(node, ast.ClassDef)
    compatibility = dict(COMPAT_PENDING)
    if not is_callable:
        compatibility["C1"] = "not-applicable"
    entry: dict[str, Any] = {
        "symbol": name,
        "public_path": f"{package}.{name}",
        "kind": "callable" if is_callable else "constant",
        "source": {
            **_source_record(package_root, source_path),
            "line": getattr(node, "lineno", None),
        },
        "parameters": [],
        "signature": None,
        "return_annotation": None,
        "extraction": "static_ast",
        "target_evidence": {
            "source_frozen": node is not None,
            "signature_frozen": isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)),
        },
        "needs_dynamic_review": needs_review,
        "reviewed": False,
        "compatibility": compatibility,
    }
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
        entry["parameters"] = _parameters(node)
        entry["signature"] = _signature(node)
        entry["return_annotation"] = _expr(node.returns)
    elif isinstance(node, (ast.Assign, ast.AnnAssign)):
        value = node.value
        try:
            entry["value"] = ast.literal_eval(value)
        except (ValueError, TypeError):
            entry["value_expression"] = ast.unparse(value)
            entry["needs_dynamic_review"] = True
    return entry


def _static_factory_entry(
    *,
    package_root: Path,
    source_path: Path,
    name: str,
    assignment: ast.Assign | ast.AnnAssign,
    definitions: dict[str, ast.AST],
) -> dict[str, Any] | None:
    value = assignment.value
    if not isinstance(value, ast.Call) or not isinstance(value.func, ast.Name):
        return None
    factory_name = value.func.id
    if not factory_name.startswith("_create_") or not factory_name.endswith("_function"):
        return None
    factory = definitions.get(factory_name)
    if not isinstance(factory, (ast.FunctionDef, ast.AsyncFunctionDef)):
        return None
    generated = next(
        (node for node in factory.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))),
        None,
    )
    if generated is None:
        return None
    return {
        "symbol": name,
        "public_path": f"empyrical.{name}",
        "kind": "callable",
        "source": {
            **_source_record(package_root, source_path),
            "line": assignment.lineno,
        },
        "parameters": _parameters(generated),
        "signature": _signature(generated),
        "return_annotation": _expr(generated.returns),
        "factory": factory_name,
        "factory_template_line": generated.lineno,
        "extraction": "static_ast",
        "target_evidence": {
            "source_frozen": True,
            "signature_frozen_from_factory_template": True,
        },
        "needs_dynamic_review": True,
        "reviewed": False,
        "compatibility": {
            **COMPAT_PENDING,
        },
    }


def _empyrical_exports(init_tree: ast.Module) -> list[tuple[str, str]]:
    exports: list[tuple[str, str]] = []
    seen: set[str] = set()
    allowed_modules = {"stats", "periods", "perf_attrib"}
    for node in init_tree.body:
        if not isinstance(node, ast.ImportFrom) or node.module not in allowed_modules:
            continue
        for alias in node.names:
            public_name = alias.asname or alias.name
            if alias.name == "*" or public_name in seen:
                continue
            seen.add(public_name)
            exports.append((public_name, node.module))
    return exports


def _generate_empyrical(root: Path) -> dict[str, Any]:
    package_root = root / "empyrical"
    init_path = package_root / "__init__.py"
    init_tree = _read_ast(init_path)
    version = _literal_assignment(init_tree, "__version__")
    commit = _require_pin(root, EMPYRICAL_COMMIT, "empyrical")
    module_cache: dict[str, tuple[Path, dict[str, ast.AST]]] = {}
    symbols: list[dict[str, Any]] = []
    for name, module in _empyrical_exports(init_tree):
        if module not in module_cache:
            source_path = package_root / f"{module}.py"
            module_cache[module] = (source_path, _definitions(_read_ast(source_path)))
        source_path, definitions = module_cache[module]
        node = definitions.get(name)
        factory_entry = (
            _static_factory_entry(
                package_root=package_root,
                source_path=source_path,
                name=name,
                assignment=node,
                definitions=definitions,
            )
            if isinstance(node, (ast.Assign, ast.AnnAssign))
            else None
        )
        symbols.append(
            factory_entry
            or _symbol_entry(
                package="empyrical",
                package_root=package_root,
                source_path=source_path,
                name=name,
                node=node,
            )
        )
    public_symbols = [entry["symbol"] for entry in symbols]
    callables = [entry for entry in symbols if entry["kind"] == "callable"]
    if (len(public_symbols), len(callables)) != (54, 49):
        raise ValueError(f"unexpected empyrical surface: {len(public_symbols)} symbols, {len(callables)} callables")
    source_paths = [init_path, *(path for path, _ in module_cache.values())]
    return {
        "schema_version": 1,
        "project": "empyrical",
        "version": version,
        "commit": commit,
        "fixture_source": {
            "mode": "static_ast",
            "generator": "scripts/generate_compat_manifest.py",
            "root": "external pinned checkout supplied with --empyrical-root",
        },
        "oracle_verification": {"status": "not_run", "reviewed": False},
        "source_files": [_source_record(package_root, path) for path in dict.fromkeys(source_paths)],
        "public_symbols": public_symbols,
        "callables": callables,
        "symbols": symbols,
    }


def _generate_pyfolio(root: Path) -> dict[str, Any]:
    package_root = root / "pyfolio"
    init_path = package_root / "__init__.py"
    tears_path = package_root / "tears.py"
    init_tree = _read_ast(init_path)
    version = _literal_assignment(init_tree, "__version__")
    commit = _require_pin(root, PYFOLIO_COMMIT, "pyfolio")
    definitions = _definitions(_read_ast(tears_path))
    profile: dict[str, dict[str, Any]] = {}
    for name in PYFOLIO_PROFILE:
        entry = _symbol_entry(
            package="pyfolio",
            package_root=package_root,
            source_path=tears_path,
            name=name,
            node=definitions.get(name),
        )
        if entry["kind"] != "callable":
            raise ValueError(f"pyfolio profile entry {name!r} is not a static callable")
        profile[name] = entry
    return {
        "schema_version": 1,
        "project": "pyfolio",
        "version": version,
        "commit": commit,
        "fixture_source": {
            "mode": "static_ast",
            "generator": "scripts/generate_compat_manifest.py",
            "root": "external pinned checkout supplied with --pyfolio-root",
        },
        "oracle_verification": {"status": "not_run", "reviewed": False},
        "source_files": [
            _source_record(package_root, init_path),
            _source_record(package_root, tears_path),
        ],
        "compatibility_profile": profile,
    }


def _generate_flat_migrations(repo_root: Path) -> dict[str, Any]:
    init_path = repo_root / "fincore" / "__init__.py"
    tree = _read_ast(init_path)
    version = _literal_assignment(tree, "__version__")
    flat_api = _literal_assignment(tree, "_FLAT_API")
    entries = []
    for symbol, target in flat_api.items():
        current_target = ".".join(target)
        recommended_target = f"fincore.empyrical.{symbol}" if symbol != "information_ratio" else current_target
        entries.append(
            {
                "symbol": symbol,
                "current_target": current_target,
                "recommended_target": recommended_target,
                "deprecate_in": None,
                "remove_or_switch_in": "next-major-not-scheduled",
                "status": "unchanged-in-0.3.x",
            }
        )
    return {
        "schema_version": 1,
        "fincore_version": version,
        "policy": "preserve-0.3.x-flat-api",
        "source": {"path": "fincore/__init__.py", "sha256": _sha256(init_path)},
        "entries": entries,
    }


def _run_oracle(interpreter: Path, package: str, names: list[str]) -> dict[str, str]:
    result = subprocess.run(
        [interpreter.as_posix(), "-I", "-c", ORACLE_SCRIPT, package, json.dumps(names)],
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(result.stdout)


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
        help="optional isolated interpreter with pinned upstream packages installed",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    empyrical = _generate_empyrical(args.empyrical_root.resolve())
    pyfolio = _generate_pyfolio(args.pyfolio_root.resolve())
    if args.oracle_python is not None:
        empyrical["oracle_verification"] = {
            "status": "captured-unreviewed",
            "reviewed": False,
            "signatures": _run_oracle(
                args.oracle_python, "empyrical", [entry["symbol"] for entry in empyrical["callables"]]
            ),
        }
        pyfolio["oracle_verification"] = {
            "status": "captured-unreviewed",
            "reviewed": False,
            "signatures": _run_oracle(args.oracle_python, "pyfolio", list(pyfolio["compatibility_profile"])),
        }
    repo_root = Path(__file__).resolve().parents[1]
    _write_json(args.output / "empyrical-0.6.0-api.json", empyrical)
    _write_json(args.output / "pyfolio-0.9.6-api.json", pyfolio)
    _write_json(args.output / "fincore-flat-api-migrations.json", _generate_flat_migrations(repo_root))


if __name__ == "__main__":
    main()
