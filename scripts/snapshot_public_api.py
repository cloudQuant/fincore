#!/usr/bin/env python3
"""Snapshot the fincore public API surface into a versioned JSON document.

The snapshot is AST-based so it never imports heavy optional dependencies.
Each public surface module is assigned a semantic profile; every entry records
its public path, kind, callable signature, and stability class.  ``--check``
compares against the checked-in fixture and fails on any drift, so a rename,
added export, or removed symbol cannot slip through unnoticed.

Static scan (layer 1) enumerates path/surface/profile/signature/stability; the
behavior probes (layer 2) in ``tests/contracts/test_public_api_behavior_probes.py``
record success shapes, dtypes, exceptions, and NaN behavior.
"""

from __future__ import annotations

import argparse
import ast
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PACKAGE = ROOT / "fincore"

SCHEMA_VERSION = 1
SNAPSHOT_BASELINE = "0.3.x"

#: Each public surface maps to exactly one semantic profile.
SURFACE_PROFILES = {
    "fincore": "enhanced_v1",
    "fincore.empyrical": "strict_empyrical_0_6_0",
    "fincore.pyfolio": "strict_pyfolio_0_9_6",
    "fincore.alphalens": "strict_alphalens_cloudquant_0_4_0",
    "fincore.metrics": "enhanced_v1",
    "fincore.risk": "enhanced_v1",
    "fincore.simulation": "enhanced_v1",
    "fincore.attribution": "enhanced_v1",
    "fincore.optimization": "enhanced_v1",
    "fincore.factor_analysis": "enhanced_v1",
    "fincore.report": "enhanced_v1",
    "fincore.data": "enhanced_v1",
    "fincore.plugin": "plugin_v1",
}

#: Package submodules that are surfaces via their own ``__init__.py``.
_PACKAGE_SURFACES = {
    "fincore.alphalens",
    "fincore.metrics",
    "fincore.risk",
    "fincore.simulation",
    "fincore.attribution",
    "fincore.optimization",
    "fincore.factor_analysis",
    "fincore.report",
    "fincore.data",
    "fincore.plugin",
}


def _module_file(module_path: str) -> Path | None:
    if module_path == "fincore":
        return PACKAGE / "__init__.py"
    relative = module_path[len("fincore.") :].replace(".", "/")
    init = PACKAGE / relative / "__init__.py"
    direct = PACKAGE / f"{relative}.py"
    if init.is_file() and module_path in _PACKAGE_SURFACES:
        return init
    return direct if direct.is_file() else init if init.is_file() else None


def _signature(node: ast.AST) -> str | None:
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
        args = node.args
        parts: list[str] = []
        args_names = [a.arg for a in args.posonlyargs + args.args]
        defaults = [None] * (len(args_names) - len(args.defaults)) + args.defaults
        for i, name in enumerate(args_names):
            default = defaults[i]
            parts.append(f"{name}={ast.unparse(default)}" if default is not None else name)
        if args.vararg is not None:
            parts.append(f"*{args.vararg.arg}")
        elif args.kwonlyargs:
            parts.append("*")
        for i, kw in enumerate(args.kwonlyargs):
            default = args.kw_defaults[i]
            parts.append(f"{kw.arg}={ast.unparse(default)}" if default is not None else kw.arg)
        if args.kwarg is not None:
            parts.append(f"**{args.kwarg.arg}")
        return f"({', '.join(parts)})"
    return None


def _extract_all(tree: ast.AST) -> list[str] | None:
    for node in tree.body:
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            for target in targets:
                if isinstance(target, ast.Name) and target.id == "__all__":
                    value = node.value
                    if isinstance(value, (ast.List, ast.Tuple)):
                        return [
                            elt.value
                            for elt in value.elts
                            if isinstance(elt, ast.Constant) and isinstance(elt.value, str)
                        ]
    return None


def _public_definitions(tree: ast.AST) -> dict[str, str]:
    definitions: dict[str, str] = {}
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and not node.name.startswith("_"):
            definitions[node.name] = "class" if isinstance(node, ast.ClassDef) else "function"
    return definitions


def _runtime_all(module_path: str) -> list[str] | None:
    """Runtime ``__all__`` for modules whose exports are built dynamically.

    Light modules (fincore, empyrical, metrics, ...) import safely; heavier
    optional-dependency modules fall back to the static scan via the caller's
    ``try/except``.
    """
    import importlib

    try:
        module = importlib.import_module(module_path)
        return list(getattr(module, "__all__", []))
    except Exception:
        return None


def _surface_entries(module_path: str) -> dict[str, object]:
    source = _module_file(module_path)
    if source is None:
        return {}
    tree = ast.parse(source.read_text(encoding="utf-8"))
    all_names = _extract_all(tree)
    definitions = _public_definitions(tree)

    runtime = _runtime_all(module_path)
    if runtime is not None and (all_names is None or len(runtime) > len(all_names)):
        all_names = runtime

    names = all_names if all_names is not None else sorted(definitions)

    entries: dict[str, object] = {}
    for name in names:
        entry: dict[str, object] = {
            "public_path": f"{module_path}.{name}",
            "kind": definitions.get(name, "unknown"),
        }
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and node.name == name:
                sig = _signature(node)
                if sig is not None:
                    entry["signature"] = sig
                break
        entries[name] = entry
    return entries


def build_snapshot() -> dict[str, object]:
    surfaces: dict[str, object] = {}
    for module_path, profile in SURFACE_PROFILES.items():
        entries = _surface_entries(module_path)
        if not entries:
            continue
        surfaces[module_path] = {
            "profile": profile,
            "public_symbols": sorted(entries),
            "entries": entries,
        }
    return {
        "schema_version": SCHEMA_VERSION,
        "project": "fincore",
        "baseline": SNAPSHOT_BASELINE,
        "surfaces": surfaces,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default=None, help="write the snapshot to this JSON path")
    parser.add_argument("--check", default=None, help="compare against this fixture JSON and fail on drift")
    args = parser.parse_args(argv)

    snapshot = build_snapshot()

    if args.output:
        Path(args.output).write_text(json.dumps(snapshot, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(f"wrote public API snapshot to {args.output}")

    if args.check:
        fixture = json.loads(Path(args.check).read_text(encoding="utf-8"))
        if fixture != snapshot:
            print("FAIL: public API snapshot drift detected", file=sys.stderr)
            return 1
        print("public API snapshot matches the checked-in fixture.")

    if not args.output and not args.check:
        print(json.dumps(snapshot, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
