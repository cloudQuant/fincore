#!/usr/bin/env python3
"""Create fail-closed static snapshots of a fincore public API surface.

The frozen 0.4 ``--check`` projection remains available until its atomic
cutover.  New source/wheel modes emit schema v2 and parse source syntax only:
they never import optional plotting, data-provider, or scientific packages.
"""

from __future__ import annotations

import argparse
import ast
import json
import sys
import zipfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Iterable

ROOT = Path(__file__).resolve().parent.parent


def _is_checkout_path(entry: str) -> bool:
    try:
        return Path(entry or ".").resolve() == ROOT
    except (OSError, RuntimeError, ValueError):
        return False


# Retained solely for the schema-v1 compatibility projection below.  Schema-v2
# source/wheel modes do not import any scanned module.
sys.path[:] = [entry for entry in sys.path if not _is_checkout_path(entry)]
sys.path.insert(0, str(ROOT))
PACKAGE = ROOT / "fincore"

SCHEMA_VERSION = 1
STRICT_SCHEMA_VERSION = 2
SNAPSHOT_BASELINE = "0.4.0.dev0"
DEFAULT_FIXTURE = ROOT / "tests" / "contracts" / "fixtures" / "public-api-0.4.0.dev0.json"

SURFACE_PROFILES = {
    "fincore": "enhanced_v1",
    "fincore.empyrical": "strict_empyrical_0_6_0",
    "fincore.pyfolio": "strict_pyfolio_0_9_6",
    "fincore.alphalens": "strict_alphalens_cloudquant_0_4_0",
    "fincore.metrics": "enhanced_v1",
    "fincore.performance": "enhanced_v1",
    "fincore.risk": "enhanced_v1",
    "fincore.simulation": "enhanced_v1",
    "fincore.attribution": "enhanced_v1",
    "fincore.optimization": "enhanced_v1",
    "fincore.factor_analysis": "enhanced_v1",
    "fincore.report": "enhanced_v1",
    "fincore.data": "enhanced_v1",
    "fincore.plugin": "plugin_v1",
}

_PACKAGE_SURFACES = {
    "fincore.alphalens",
    "fincore.metrics",
    "fincore.performance",
    "fincore.risk",
    "fincore.simulation",
    "fincore.attribution",
    "fincore.optimization",
    "fincore.factor_analysis",
    "fincore.report",
    "fincore.data",
    "fincore.plugin",
}


# ---------------------------------------------------------------------------
# Frozen schema-v1 projection.  This is intentionally isolated from the new
# static contract so the existing fixture is not rewritten to fabricate D0.
# ---------------------------------------------------------------------------


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
    if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
        return None
    args = node.args
    parts: list[str] = []
    argument_names = [argument.arg for argument in args.posonlyargs + args.args]
    defaults = [None] * (len(argument_names) - len(args.defaults)) + args.defaults
    for index, name in enumerate(argument_names):
        default = defaults[index]
        parts.append(f"{name}={ast.unparse(default)}" if default is not None else name)
    if args.vararg is not None:
        parts.append(f"*{args.vararg.arg}")
    elif args.kwonlyargs:
        parts.append("*")
    for index, keyword_only in enumerate(args.kwonlyargs):
        default = args.kw_defaults[index]
        parts.append(f"{keyword_only.arg}={ast.unparse(default)}" if default is not None else keyword_only.arg)
    if args.kwarg is not None:
        parts.append(f"**{args.kwarg.arg}")
    return f"({', '.join(parts)})"


def _extract_all(tree: ast.Module) -> list[str] | None:
    for node in tree.body:
        if not isinstance(node, (ast.Assign, ast.AnnAssign)):
            continue
        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
        for target in targets:
            if (
                isinstance(target, ast.Name)
                and target.id == "__all__"
                and isinstance(node.value, (ast.List, ast.Tuple))
            ):
                return [
                    element.value
                    for element in node.value.elts
                    if isinstance(element, ast.Constant) and isinstance(element.value, str)
                ]
    return None


def _public_definitions(tree: ast.Module) -> dict[str, str]:
    definitions: dict[str, str] = {}
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and not node.name.startswith("_"):
            definitions[node.name] = "class" if isinstance(node, ast.ClassDef) else "function"
    return definitions


def _runtime_all(module_path: str) -> list[str] | None:
    """Return legacy runtime exports only for the frozen schema-v1 fixture."""

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
        entry: dict[str, object] = {"public_path": f"{module_path}.{name}", "kind": definitions.get(name, "unknown")}
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and node.name == name:
                signature = _signature(node)
                if signature is not None:
                    entry["signature"] = signature
                break
        entries[name] = entry
    return entries


def build_snapshot() -> dict[str, object]:
    """Build the current schema-v1 fixture projection without changing it."""

    surfaces: dict[str, object] = {}
    for module_path, profile in SURFACE_PROFILES.items():
        entries = _surface_entries(module_path)
        if not entries:
            continue
        surfaces[module_path] = {"profile": profile, "public_symbols": sorted(entries), "entries": entries}
    return {
        "schema_version": SCHEMA_VERSION,
        "project": "fincore",
        "baseline": SNAPSHOT_BASELINE,
        "surfaces": surfaces,
    }


# ---------------------------------------------------------------------------
# Schema-v2 static source/wheel contract
# ---------------------------------------------------------------------------


class SnapshotContractError(ValueError):
    """Raised when the selected public-API input cannot be trusted."""


@dataclass(frozen=True)
class _ModuleText:
    text: str
    locator: str
    is_package: bool


@dataclass(frozen=True)
class _ResolvedExport:
    kind: str
    signature: dict[str, object] | None


@dataclass(frozen=True)
class _Binding:
    kind: str
    resolved: _ResolvedExport | None = None
    module_path: str | None = None
    symbol: str | None = None


class _SourceReader:
    """Read a source tree without following package or module symlinks."""

    def __init__(self, source_root: Path) -> None:
        if source_root.is_symlink():
            raise SnapshotContractError("source root must not be a symlink")
        try:
            root = source_root.resolve(strict=True)
        except (OSError, RuntimeError) as error:
            raise SnapshotContractError(f"source root is not readable: {source_root}") from error
        if not root.is_dir():
            raise SnapshotContractError(f"source root is not a directory: {source_root}")
        package = root / "fincore"
        if package.is_symlink():
            raise SnapshotContractError("fincore package must not be a symlink")
        if not package.is_dir():
            raise SnapshotContractError(f"source root has no fincore package: {source_root}")
        self.root = root
        self.package = package

    def module_text(self, module_path: str) -> _ModuleText:
        parts = _validated_module_parts(module_path)
        relative = Path(*parts[1:])
        candidates: tuple[Path, ...]
        if module_path == "fincore":
            candidates = (self.package / "__init__.py",)
        else:
            candidates = (self.package / relative / "__init__.py", self.package / relative.with_suffix(".py"))
        for candidate in candidates:
            if candidate.exists() or candidate.is_symlink():
                return self._read_source_file(candidate)
        raise SnapshotContractError(f"public surface source is missing: {module_path}")

    def _read_source_file(self, path: Path) -> _ModuleText:
        if path.is_symlink():
            raise SnapshotContractError(f"source module must not be a symlink: {path}")
        try:
            resolved = path.resolve(strict=True)
        except (OSError, RuntimeError) as error:
            raise SnapshotContractError(f"source module is not readable: {path}") from error
        if not _is_within(resolved, self.package):
            raise SnapshotContractError(f"source module must stay inside source root: {path}")
        if not resolved.is_file():
            raise SnapshotContractError(f"source module is not a regular file: {path}")
        try:
            return _ModuleText(
                text=resolved.read_text(encoding="utf-8"),
                locator=resolved.relative_to(self.root).as_posix(),
                is_package=resolved.name == "__init__.py",
            )
        except UnicodeDecodeError as error:
            raise SnapshotContractError(f"source module is not UTF-8: {path}") from error
        except OSError as error:
            raise SnapshotContractError(f"source module is not readable: {path}") from error


class _WheelReader:
    """Read a validated wheel directly; it never extracts member paths."""

    def __init__(self, wheel: Path) -> None:
        if wheel.is_symlink() or wheel.suffix != ".whl" or not wheel.is_file():
            raise SnapshotContractError(f"invalid wheel: {wheel}")
        try:
            self.archive = zipfile.ZipFile(wheel)
        except (OSError, zipfile.BadZipFile) as error:
            raise SnapshotContractError(f"invalid wheel: {wheel}") from error
        self._members: dict[str, zipfile.ZipInfo] = {}
        try:
            self._validate_members()
        except Exception:
            self.archive.close()
            raise

    def _validate_members(self) -> None:
        for member in self.archive.infolist():
            name = member.filename
            _validate_wheel_member_name(name)
            if name in self._members:
                raise SnapshotContractError(f"wheel has duplicate member: {name}")
            if member.flag_bits & 0x1:
                raise SnapshotContractError(f"wheel has encrypted member: {name}")
            self._members[name] = member
        try:
            corrupt_member = self.archive.testzip()
        except (OSError, RuntimeError, zipfile.BadZipFile) as error:
            raise SnapshotContractError("invalid wheel: archive cannot be read") from error
        if corrupt_member is not None:
            raise SnapshotContractError(f"invalid wheel: corrupt member {corrupt_member}")

    def module_text(self, module_path: str) -> _ModuleText:
        parts = _validated_module_parts(module_path)
        relative = "/".join(parts[1:])
        candidates: tuple[str, ...]
        if module_path == "fincore":
            candidates = ("fincore/__init__.py",)
        else:
            candidates = (f"fincore/{relative}/__init__.py", f"fincore/{relative}.py")
        for candidate in candidates:
            if candidate not in self._members:
                continue
            try:
                payload = self.archive.read(candidate)
                return _ModuleText(
                    text=payload.decode("utf-8"),
                    locator=candidate,
                    is_package=candidate.endswith("/__init__.py"),
                )
            except UnicodeDecodeError as error:
                raise SnapshotContractError(f"wheel module is not UTF-8: {candidate}") from error
            except (OSError, RuntimeError, zipfile.BadZipFile) as error:
                raise SnapshotContractError(f"invalid wheel: cannot read {candidate}") from error
        raise SnapshotContractError(f"public surface source is missing from wheel: {module_path}")

    def close(self) -> None:
        self.archive.close()


def _is_within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _validated_module_parts(module_path: str) -> tuple[str, ...]:
    parts = tuple(module_path.split("."))
    if not parts or parts[0] != "fincore" or any(not part.isidentifier() for part in parts):
        raise SnapshotContractError(f"invalid public surface path: {module_path!r}")
    return parts


def _validate_wheel_member_name(name: str) -> None:
    path = PurePosixPath(name)
    if not name or name.startswith(("/", "\\")) or "\\" in name or "\x00" in name:
        raise SnapshotContractError(f"unsafe wheel member: {name!r}")
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise SnapshotContractError(f"unsafe wheel member: {name!r}")


def _unparse(node: ast.AST | None) -> str | None:
    return None if node is None else ast.unparse(node)


def _signature_contract(
    arguments: ast.arguments, returns: ast.AST | None, *, drop_receiver: bool = False
) -> dict[str, object]:
    positional = list(arguments.posonlyargs) + list(arguments.args)
    defaults: list[ast.expr | None] = [None] * (len(positional) - len(arguments.defaults)) + list(arguments.defaults)
    parameters: list[dict[str, object]] = []
    for index, argument in enumerate(positional):
        if drop_receiver and index == 0 and argument.arg in {"self", "cls"}:
            continue
        kind = "POSITIONAL_ONLY" if index < len(arguments.posonlyargs) else "POSITIONAL_OR_KEYWORD"
        parameters.append(
            {
                "name": argument.arg,
                "kind": kind,
                "default": _unparse(defaults[index]),
                "annotation": _unparse(argument.annotation),
            }
        )
    if arguments.vararg is not None:
        parameters.append(
            {
                "name": arguments.vararg.arg,
                "kind": "VAR_POSITIONAL",
                "default": None,
                "annotation": _unparse(arguments.vararg.annotation),
            }
        )
    for index, argument in enumerate(arguments.kwonlyargs):
        parameters.append(
            {
                "name": argument.arg,
                "kind": "KEYWORD_ONLY",
                "default": _unparse(arguments.kw_defaults[index]),
                "annotation": _unparse(argument.annotation),
            }
        )
    if arguments.kwarg is not None:
        parameters.append(
            {
                "name": arguments.kwarg.arg,
                "kind": "VAR_KEYWORD",
                "default": None,
                "annotation": _unparse(arguments.kwarg.annotation),
            }
        )
    return {"parameters": parameters, "return_annotation": _unparse(returns)}


def _class_signature(node: ast.ClassDef) -> dict[str, object]:
    for child in node.body:
        if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)) and child.name == "__init__":
            return _signature_contract(child.args, None, drop_receiver=True)
    return {"parameters": [], "return_annotation": None}


def _direct_binding(node: ast.AST) -> _Binding | None:
    if isinstance(node, ast.FunctionDef):
        return _Binding("direct", _ResolvedExport("function", _signature_contract(node.args, node.returns)))
    if isinstance(node, ast.AsyncFunctionDef):
        return _Binding("direct", _ResolvedExport("async_function", _signature_contract(node.args, node.returns)))
    if isinstance(node, ast.ClassDef):
        return _Binding("direct", _ResolvedExport("class", _class_signature(node)))
    return None


class _StaticResolver:
    """Resolve declared public symbols using syntax only, never imports."""

    def __init__(self, reader: _SourceReader | _WheelReader) -> None:
        self.reader = reader
        self._modules: dict[str, tuple[ast.Module, _ModuleText]] = {}
        self._bindings: dict[str, dict[str, _Binding]] = {}
        self._exports: dict[str, list[str]] = {}

    def snapshot(self, surfaces: Iterable[str]) -> dict[str, object]:
        surface_map: dict[str, object] = {}
        for surface in surfaces:
            names = self.exports(surface)
            if not names:
                raise SnapshotContractError(f"public surface {surface!r} has no public exports")
            entries: dict[str, object] = {}
            for name in names:
                resolved = self.resolve(surface, name, ())
                entry: dict[str, object] = {"public_path": f"{surface}.{name}", "kind": resolved.kind}
                if resolved.signature is not None:
                    entry["signature"] = resolved.signature
                entries[name] = entry
            surface_map[surface] = {
                "profile": SURFACE_PROFILES[surface],
                "public_symbols": sorted(entries),
                "entries": entries,
            }
        if not surface_map:
            raise SnapshotContractError("at least one public surface is required")
        return {"schema_version": STRICT_SCHEMA_VERSION, "project": "fincore", "surfaces": surface_map}

    def exports(self, module_path: str) -> list[str]:
        if module_path in self._exports:
            return self._exports[module_path]
        tree, _ = self.module(module_path)
        static_all: list[str] | None = None
        has_all_assignment = False
        for node in tree.body:
            if not isinstance(node, (ast.Assign, ast.AnnAssign)):
                continue
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            if not any(isinstance(target, ast.Name) and target.id == "__all__" for target in targets):
                continue
            has_all_assignment = True
            if not isinstance(node.value, (ast.List, ast.Tuple)):
                raise SnapshotContractError(f"public surface {module_path!r} has a non-static __all__")
            names_from_all: list[str] = []
            for element in node.value.elts:
                if not isinstance(element, ast.Constant) or not isinstance(element.value, str):
                    raise SnapshotContractError(f"public surface {module_path!r} has a non-static __all__")
                names_from_all.append(element.value)
            static_all = names_from_all
        if has_all_assignment:
            assert static_all is not None
            if len(static_all) != len(set(static_all)):
                raise SnapshotContractError(f"public surface {module_path!r} has duplicate __all__ exports")
            names = static_all
        else:
            names = sorted(name for name in self.bindings(module_path) if not name.startswith("_"))
        self._exports[module_path] = names
        return names

    def module(self, module_path: str) -> tuple[ast.Module, _ModuleText]:
        if module_path not in self._modules:
            source = self.reader.module_text(module_path)
            try:
                tree = ast.parse(source.text, filename=source.locator)
            except SyntaxError as error:
                raise SnapshotContractError(f"cannot parse public surface {module_path!r}: {error.msg}") from error
            self._modules[module_path] = (tree, source)
        return self._modules[module_path]

    def bindings(self, module_path: str) -> dict[str, _Binding]:
        if module_path in self._bindings:
            return self._bindings[module_path]
        tree, source = self.module(module_path)
        bindings: dict[str, _Binding] = {}
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                direct = _direct_binding(node)
                assert direct is not None
                bindings[node.name] = direct
                continue
            if isinstance(node, ast.Import):
                for alias in node.names:
                    name = alias.asname or alias.name.split(".", 1)[0]
                    bindings[name] = _Binding("module", module_path=alias.name)
                continue
            if isinstance(node, ast.ImportFrom):
                imported_module = _resolve_import_module(module_path, source, node.level, node.module)
                for alias in node.names:
                    if alias.name == "*":
                        raise SnapshotContractError(f"public surface {module_path!r} uses star imports")
                    name = alias.asname or alias.name
                    module_candidate = imported_module + "." + alias.name
                    if node.module is None and self._module_exists(module_candidate):
                        bindings[name] = _Binding("module", module_path=module_candidate)
                    else:
                        bindings[name] = _Binding("import", module_path=imported_module, symbol=alias.name)
                continue
            if isinstance(node, (ast.Assign, ast.AnnAssign)):
                targets = node.targets if isinstance(node, ast.Assign) else [node.target]
                for target in targets:
                    if isinstance(target, ast.Name) and not target.id.startswith("_"):
                        bindings.setdefault(target.id, _Binding("direct", _ResolvedExport("value", None)))
        bindings.update(self._dynamic_bindings(module_path, tree, source))
        self._bindings[module_path] = bindings
        return bindings

    def resolve(self, module_path: str, name: str, chain: tuple[tuple[str, str], ...]) -> _ResolvedExport:
        location = (module_path, name)
        if location in chain:
            raise SnapshotContractError(f"cyclic static export resolution at {module_path}.{name}")
        binding = self.bindings(module_path).get(name)
        if binding is None:
            raise SnapshotContractError(f"cannot statically resolve public export {module_path}.{name}")
        if binding.kind == "direct":
            assert binding.resolved is not None
            return binding.resolved
        if binding.kind == "module":
            return _ResolvedExport("module", None)
        if binding.kind == "import":
            assert binding.module_path is not None and binding.symbol is not None
            return self.resolve(binding.module_path, binding.symbol, (*chain, location))
        raise SnapshotContractError(f"cannot statically resolve public export {module_path}.{name}")

    def _module_exists(self, module_path: str) -> bool:
        try:
            self.reader.module_text(module_path)
        except SnapshotContractError:
            return False
        return True

    def _dynamic_bindings(self, module_path: str, tree: ast.Module, source: _ModuleText) -> dict[str, _Binding]:
        bindings: dict[str, _Binding] = {}
        for node in tree.body:
            if isinstance(node, ast.Assign) and isinstance(node.value, ast.Dict):
                bindings.update(self._literal_export_map(node.value))
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == "__getattr__":
                bindings.update(self._getattr_bindings(module_path, source, node))
        return bindings

    def _literal_export_map(self, node: ast.Dict) -> dict[str, _Binding]:
        bindings: dict[str, _Binding] = {}
        for key, value in zip(node.keys, node.values, strict=True):
            if not isinstance(key, ast.Constant) or not isinstance(key.value, str):
                continue
            if isinstance(value, ast.Constant) and isinstance(value.value, str):
                bindings[key.value] = _Binding("module", module_path=value.value)
            elif (
                isinstance(value, ast.Tuple)
                and len(value.elts) >= 2
                and isinstance(value.elts[0], ast.Constant)
                and isinstance(value.elts[0].value, str)
                and isinstance(value.elts[1], ast.Constant)
                and isinstance(value.elts[1].value, str)
            ):
                bindings[key.value] = _Binding("import", module_path=value.elts[0].value, symbol=value.elts[1].value)
        return bindings

    def _getattr_bindings(
        self, module_path: str, source: _ModuleText, node: ast.FunctionDef | ast.AsyncFunctionDef
    ) -> dict[str, _Binding]:
        bindings: dict[str, _Binding] = {}
        for candidate in ast.walk(node):
            if not isinstance(candidate, ast.If):
                continue
            export_name = _name_comparison(candidate.test)
            if export_name is None:
                continue
            imported = self._binding_from_getattr_body(module_path, source, candidate.body)
            if imported is not None:
                bindings[export_name] = imported
        return bindings

    def _binding_from_getattr_body(
        self, module_path: str, source: _ModuleText, body: list[ast.stmt]
    ) -> _Binding | None:
        for candidate in ast.walk(ast.Module(body=body, type_ignores=[])):
            if isinstance(candidate, ast.ImportFrom):
                imported_module = _resolve_import_module(module_path, source, candidate.level, candidate.module)
                for alias in candidate.names:
                    if alias.name != "*":
                        return _Binding("import", module_path=imported_module, symbol=alias.name)
            if isinstance(candidate, ast.Assign) and isinstance(candidate.value, ast.Call):
                call = candidate.value
                if (
                    isinstance(call.func, ast.Attribute)
                    and call.func.attr == "import_module"
                    and call.args
                    and isinstance(call.args[0], ast.Constant)
                    and isinstance(call.args[0].value, str)
                ):
                    return _Binding("module", module_path=call.args[0].value)
        return None


def _resolve_import_module(module_path: str, source: _ModuleText, level: int, target: str | None) -> str:
    if level == 0:
        if not target:
            raise SnapshotContractError(f"invalid import in {module_path!r}")
        return target
    package = module_path.split(".") if source.is_package else module_path.split(".")[:-1]
    parent_levels = level - 1
    if parent_levels > len(package):
        raise SnapshotContractError(f"relative import escapes fincore package in {module_path!r}")
    base = package[: len(package) - parent_levels]
    if target:
        base.extend(target.split("."))
    resolved = ".".join(base)
    if not resolved.startswith("fincore"):
        raise SnapshotContractError(f"relative import escapes fincore package in {module_path!r}")
    return resolved


def _name_comparison(node: ast.AST) -> str | None:
    if not isinstance(node, ast.Compare) or len(node.ops) != 1 or not isinstance(node.ops[0], ast.Eq):
        return None
    if len(node.comparators) != 1:
        return None
    left, right = node.left, node.comparators[0]
    if (
        isinstance(left, ast.Name)
        and left.id == "name"
        and isinstance(right, ast.Constant)
        and isinstance(right.value, str)
    ):
        return right.value
    if (
        isinstance(right, ast.Name)
        and right.id == "name"
        and isinstance(left, ast.Constant)
        and isinstance(left.value, str)
    ):
        return left.value
    return None


def _selected_surfaces(values: list[str] | None) -> tuple[str, ...]:
    surfaces = tuple(values) if values else tuple(SURFACE_PROFILES)
    if not surfaces:
        raise SnapshotContractError("at least one public surface is required")
    if len(surfaces) != len(set(surfaces)):
        raise SnapshotContractError("public surfaces must be unique")
    unknown = [surface for surface in surfaces if surface not in SURFACE_PROFILES]
    if unknown:
        raise SnapshotContractError(f"unknown public surface: {unknown[0]}")
    return surfaces


def build_static_snapshot(
    *, source_root: Path | None = None, wheel: Path | None = None, surfaces: Iterable[str] | None = None
) -> dict[str, object]:
    """Build one schema-v2 static snapshot from exactly one source type."""

    if (source_root is None) == (wheel is None):
        raise SnapshotContractError("provide exactly one of source_root or wheel")
    selected = _selected_surfaces(list(surfaces) if surfaces is not None else None)
    reader: _SourceReader | _WheelReader
    if source_root is not None:
        reader = _SourceReader(source_root)
    else:
        assert wheel is not None
        reader = _WheelReader(wheel)
    try:
        return _StaticResolver(reader).snapshot(selected)
    finally:
        if isinstance(reader, _WheelReader):
            reader.close()


def _write_json(path: str, payload: dict[str, object]) -> None:
    destination = Path(path)
    if destination.is_symlink():
        raise SnapshotContractError(f"output path must not be a symlink: {destination}")
    destination.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote public API snapshot to {destination}")


def _run_static_mode(args: argparse.Namespace, parser: argparse.ArgumentParser) -> int:
    wheels: list[str] = args.wheel or []
    if len(wheels) > 1:
        parser.error("provide exactly one wheel")
    if args.check is not None:
        parser.error("--check is only available for the frozen schema-v1 fixture")
    if args.compare:
        if args.source_root is None or len(wheels) != 1:
            parser.error("--compare requires --source-root and exactly one --wheel")
        source_snapshot = build_static_snapshot(source_root=Path(args.source_root), surfaces=args.surface)
        wheel_snapshot = build_static_snapshot(wheel=Path(wheels[0]), surfaces=args.surface)
        if source_snapshot != wheel_snapshot:
            print("FAIL: source/wheel public API snapshot mismatch", file=sys.stderr)
            return 1
        if args.output:
            _write_json(args.output, source_snapshot)
        print("source/wheel public API snapshots match.")
        return 0
    if args.source_root is not None and wheels:
        parser.error("use --compare when both --source-root and --wheel are supplied")
    if args.source_root is None and len(wheels) != 1:
        parser.error("provide exactly one of --source-root or --wheel")
    snapshot = build_static_snapshot(
        source_root=Path(args.source_root) if args.source_root is not None else None,
        wheel=Path(wheels[0]) if wheels else None,
        surfaces=args.surface,
    )
    if args.output:
        _write_json(args.output, snapshot)
    else:
        print(json.dumps(snapshot, indent=2, sort_keys=True))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default=None, help="write the snapshot to this JSON path")
    parser.add_argument(
        "--check",
        nargs="?",
        const=str(DEFAULT_FIXTURE),
        default=None,
        help="compare the frozen schema-v1 projection against a fixture JSON",
    )
    parser.add_argument("--source-root", default=None, help="source checkout root for a static schema-v2 snapshot")
    parser.add_argument("--wheel", action="append", default=None, help="one explicit wheel archive for schema-v2")
    parser.add_argument(
        "--surface", action="append", default=None, help="explicit public surface; repeat to select more"
    )
    parser.add_argument("--compare", action="store_true", help="fail unless source and wheel schema-v2 snapshots match")
    args = parser.parse_args(argv)

    if args.source_root is not None or args.wheel is not None or args.compare or args.surface is not None:
        try:
            return _run_static_mode(args, parser)
        except SnapshotContractError as error:
            print(f"FAIL: {error}", file=sys.stderr)
            return 2

    snapshot = build_snapshot()
    if args.output:
        _write_json(args.output, snapshot)
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
