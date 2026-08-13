#!/usr/bin/env python3
"""Audit the static and later executed migration of pinned Alphalens tests.

Task 1.5 intentionally runs this tool in static mode: inventory and migration
map are checked without inventing future Task 3/4/8 target tests.  Passing
``--nodeids`` activates the deferred target source, collection, marker, and
non-xdist result audit once those tests exist.
"""

from __future__ import annotations

import argparse
import ast
import json
import re
import sys
from collections import Counter
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
PINNED_COMMIT = "3fa17ad4c3edb025d1410de7aeba9673cba7791c"
EXPECTED_COUNTS = {
    "active_declared_cases": 117,
    "diagnostic_collectible_cases": 116,
    "active_methods": 22,
    "dormant_tear_rows": 24,
    "dormant_tear_workflows": 7,
    "dormant_tear_invocations": 96,
}
EXPECTED_CASE_COUNT = 141
EXPECTED_SOURCE_FILES = {
    "tests/test_utils.py": {
        "git_blob": "22480c305a07b8ccd83e15ed7b6d1b06be08307e",
        "sha256": "0f476933684b1eae8f86c3ce9dcf3806b840cc69a1005e19f43a52d4bdf31334",
    },
    "tests/test_performance.py": {
        "git_blob": "5f38d92b936f3b7f0afb0b4d63a84edd347766a1",
        "sha256": "278ecc858a228e686edd6e8aa4ef30d42fe7258a9af5da14263de61607474917",
    },
    "tests/test_tears.py": {
        "git_blob": "8c1b74705e89ae3fe090049120c06d34fe7f13fd",
        "sha256": "227d23e8eebb3585b29f5f953e67f817517d802148f3e72c0cf8b27087853b86",
    },
}
ALLOWED_DISPOSITIONS = {"rewritten_strict", "rewritten_invariant", "rebuilt_c4"}
FORBIDDEN_DISPOSITIONS = {"skip", "xfail", "smoke_only", "raw_copy", "unmapped"}
GROUP_PATHS = {
    "utils": {
        "tests/compat/alphalens/test_forward_returns.py",
        "tests/compat/alphalens/test_factor_cleaning.py",
    },
    "performance": {
        "tests/compat/alphalens/test_performance.py",
        "tests/test_factor_analysis/test_information.py",
        "tests/test_factor_analysis/test_weights_returns.py",
        "tests/test_factor_analysis/test_turnover.py",
        "tests/test_factor_analysis/test_events.py",
    },
    "tears": {
        "tests/compat/alphalens/test_tearsheets_e2e.py",
        "tests/test_factor_analysis/test_tears.py",
    },
}
GROUP_GRADES = {"utils": {"C2", "C3"}, "performance": {"C2", "C3"}, "tears": {"C4"}}
RESULT_OUTCOMES = ("setup", "call", "teardown")
RESULT_SCHEMA_VERSION = "alphalens-upstream-case-results-v2"
SUMMARY_LINE = re.compile(r"^\d+(?:/\d+)? tests? collected")
SOURCE_TEST_MODULES = frozenset({"test_utils", "test_performance", "test_tears"})
SYS_PATH_MUTATION_METHODS = frozenset({"append", "extend", "insert", "remove", "pop", "clear"})
C4_FIGURE_HELPERS = frozenset(
    {
        "assert_figure_axes",
        "assert_figure_artifacts",
        "assert_axes_artifacts",
        "assert_rendered_figure",
        "assert_tear_sheet_figures",
    }
)
C4_SHOW_HELPERS = frozenset({"show_figure", "show_owned_figures", "assert_show_called"})
C4_CLOSE_HELPERS = frozenset(
    {"close_figure", "close_owned_figures", "assert_figures_closed", "assert_no_open_figures"}
)
C4_OWNERSHIP_HELPERS = frozenset(
    {
        "assert_artifact_ownership",
        "assert_owned_artifacts",
        "assert_figure_ownership",
        "assert_no_figure_leaks",
        "assert_no_open_figures",
        "close_owned_figures",
    }
)


class MigrationAuditError(RuntimeError):
    """Raised when an inventory, map, collection, or result proof is invalid."""


def _load_json(path: Path, label: str) -> dict[str, Any]:
    """Load one object-shaped JSON fixture with useful audit failures."""
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise MigrationAuditError(f"cannot read {label} {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise MigrationAuditError(f"{label} must be a JSON object: {path}")
    return value


def _group(case: dict[str, Any]) -> str:
    """Classify an inventory record by its frozen upstream source path."""
    path = case.get("source_path")
    if path == "tests/test_utils.py":
        return "utils"
    if path == "tests/test_performance.py":
        return "performance"
    if path == "tests/test_tears.py":
        return "tears"
    raise MigrationAuditError(f"unknown source path in inventory record: {path!r}")


def _scope_groups(scope: str) -> set[str]:
    """Expand the CLI scope to inventory group names."""
    return set(GROUP_PATHS) if scope == "all" else {scope}


def _selector_path(selector: str) -> str:
    """Extract the repository-relative test path from a pytest node selector."""
    path, separator, _ = selector.partition("::")
    if not separator or not path:
        raise MigrationAuditError(f"target selector is not an exact pytest nodeid: {selector!r}")
    return path


def _selector_function(selector: str) -> str:
    """Extract the function or class-method portion of a pytest selector."""
    _, separator, remainder = selector.partition("::")
    if not separator or not remainder:
        raise MigrationAuditError(f"target selector is not an exact pytest nodeid: {selector!r}")
    return remainder.split("[", 1)[0]


def _validate_static(
    inventory: dict[str, Any], migration: dict[str, Any], groups: set[str]
) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    """Validate the complete map, returning only records selected by scope."""
    cases = inventory.get("cases")
    mapped = migration.get("cases")
    if not isinstance(cases, list):
        raise MigrationAuditError("inventory.cases must be a list")
    if not isinstance(mapped, dict):
        raise MigrationAuditError("migration.cases must be an object keyed by source_case_id")
    if inventory.get("commit") != PINNED_COMMIT or migration.get("commit") != PINNED_COMMIT:
        raise MigrationAuditError("inventory and migration must both bind the pinned Alphalens commit")
    if inventory.get("counts") != EXPECTED_COUNTS:
        raise MigrationAuditError("inventory counts differ from the frozen 117/116/22/24/7/96 contract")
    if len(cases) != EXPECTED_CASE_COUNT:
        raise MigrationAuditError(f"inventory must contain {EXPECTED_CASE_COUNT} source rows, found {len(cases)}")
    if inventory.get("source_files") != EXPECTED_SOURCE_FILES:
        raise MigrationAuditError("inventory source blobs or SHA256 values differ from the pinned evidence")
    inventory_by_id: dict[str, dict[str, Any]] = {}
    for case in cases:
        if not isinstance(case, dict) or not isinstance(case.get("source_case_id"), str):
            raise MigrationAuditError("every inventory case needs a string source_case_id")
        source_case_id = case["source_case_id"]
        if source_case_id in inventory_by_id:
            raise MigrationAuditError(f"inventory duplicates source_case_id {source_case_id}")
        inventory_by_id[source_case_id] = case
    if set(mapped) != set(inventory_by_id):
        missing = sorted(set(inventory_by_id) - set(mapped))
        extra = sorted(set(mapped) - set(inventory_by_id))
        raise MigrationAuditError(f"inventory/map case-ID mismatch: missing={missing[:3]} extra={extra[:3]}")

    all_invocation_targets: dict[str, str] = {}
    ordinary_selectors: list[str] = []
    for source_case_id, case in inventory_by_id.items():
        record = mapped[source_case_id]
        if not isinstance(record, dict):
            raise MigrationAuditError(f"migration record must be an object: {source_case_id}")
        group = _group(case)
        disposition = record.get("disposition")
        if disposition in FORBIDDEN_DISPOSITIONS or disposition not in ALLOWED_DISPOSITIONS:
            raise MigrationAuditError(f"forbidden or unknown disposition for {source_case_id}: {disposition!r}")
        if group == "tears" and disposition != "rebuilt_c4":
            raise MigrationAuditError(f"tear row must be rebuilt_c4: {source_case_id}")
        if group != "tears" and disposition == "rebuilt_c4":
            raise MigrationAuditError(f"non-tear row cannot be rebuilt_c4: {source_case_id}")
        if record.get("source_collection_state") != case.get("source_collection_state"):
            raise MigrationAuditError(f"source collection state is not preserved for {source_case_id}")
        if record.get("source_assertion_quality") != case.get("assertion_quality"):
            raise MigrationAuditError(f"source assertion quality is not preserved for {source_case_id}")
        grade = record.get("assertion_grade")
        if grade not in GROUP_GRADES[group]:
            raise MigrationAuditError(f"assertion grade {grade!r} is incompatible with {group}: {source_case_id}")
        selectors = record.get("target_selectors")
        if not isinstance(selectors, list) or not selectors or not all(isinstance(value, str) for value in selectors):
            raise MigrationAuditError(f"target_selectors must be a non-empty string list: {source_case_id}")
        for selector in selectors:
            if _selector_path(selector) not in GROUP_PATHS[group]:
                raise MigrationAuditError(f"selector escapes Task 3/4/8 target paths: {selector}")
        if group != "tears":
            if len(selectors) != 1:
                raise MigrationAuditError(f"ordinary source row must have one exact target selector: {source_case_id}")
            if not any(source_case_id in selector for selector in selectors):
                raise MigrationAuditError(f"ordinary source ID missing from parametrized selector: {source_case_id}")
            ordinary_selectors.extend(selectors)
            if record.get("invocation_targets"):
                raise MigrationAuditError(f"ordinary source row cannot carry invocation_targets: {source_case_id}")
            continue
        invocation_ids = case.get("invocation_ids")
        targets = record.get("invocation_targets")
        if not isinstance(invocation_ids, list) or not all(isinstance(value, str) for value in invocation_ids):
            raise MigrationAuditError(f"tear inventory row lacks invocation_ids: {source_case_id}")
        if not isinstance(targets, dict) or set(targets) != set(invocation_ids):
            raise MigrationAuditError(f"tear invocation key mismatch: {source_case_id}")
        for invocation_id, selector in targets.items():
            if not isinstance(selector, str) or _selector_path(selector) not in GROUP_PATHS[group]:
                raise MigrationAuditError(f"tear invocation selector escapes Task 8 paths: {invocation_id}")
            if invocation_id not in selector:
                raise MigrationAuditError(f"tear invocation selector omits exact invocation ID: {invocation_id}")
            if invocation_id in all_invocation_targets:
                raise MigrationAuditError(f"duplicate tear invocation ID: {invocation_id}")
            all_invocation_targets[invocation_id] = selector
    duplicate_selectors = [selector for selector, count in Counter(ordinary_selectors).items() if count > 1]
    if duplicate_selectors:
        raise MigrationAuditError(f"ordinary source cases share target selectors: {duplicate_selectors[:3]}")
    target_counts = Counter(all_invocation_targets.values())
    duplicate_targets = [target for target, count in target_counts.items() if count > 1]
    if duplicate_targets:
        raise MigrationAuditError(f"tear invocations reuse target nodeids: {duplicate_targets[:3]}")
    all_tear_ids = {
        invocation_id
        for case in inventory_by_id.values()
        if _group(case) == "tears"
        for invocation_id in case.get("invocation_ids", [])
    }
    if set(all_invocation_targets) != all_tear_ids:
        raise MigrationAuditError("inventory/map tear invocation-ID sets differ")
    if len(all_invocation_targets) != 96:
        raise MigrationAuditError(f"expected 96 unique tear invocation targets, found {len(all_invocation_targets)}")

    selected_inventory = {
        source_case_id: case
        for source_case_id, case in inventory_by_id.items()
        if _group(case) in groups
    }
    selected_map = {source_case_id: mapped[source_case_id] for source_case_id in selected_inventory}
    return selected_inventory, selected_map


def _read_nodeids(path: Path) -> list[str]:
    """Extract exact nodeids from a ``pytest --collect-only -q`` transcript."""
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise MigrationAuditError(f"cannot read --nodeids file {path}: {exc}") from exc
    nodeids = [line.strip() for line in lines if "::" in line and not SUMMARY_LINE.match(line.strip())]
    if not nodeids:
        raise MigrationAuditError(f"--nodeids contains no pytest nodeids: {path}")
    return nodeids


def _target_files(selected_map: dict[str, dict[str, Any]]) -> set[str]:
    """Get every mapped target file implied by the selected source scope."""
    paths = {
        _selector_path(selector)
        for record in selected_map.values()
        for selector in record["target_selectors"]
    }
    paths.update(
        _selector_path(selector)
        for record in selected_map.values()
        for selector in record.get("invocation_targets", {}).values()
    )
    return paths


def _call_is_bare_equals(call: ast.Call, parents: dict[ast.AST, ast.AST]) -> bool:
    """Return whether ``.equals()`` is discarded rather than asserted."""
    if not (isinstance(call.func, ast.Attribute) and call.func.attr == "equals"):
        return False
    parent = parents.get(call)
    return not isinstance(parent, ast.Assert)


def _is_testing_call(call: ast.Call) -> bool:
    """Recognize standard pandas/numpy assertion-call shapes in target AST."""
    function = call.func
    if isinstance(function, ast.Name):
        return function.id.startswith("assert_") and function.id.endswith("_equal")
    if not isinstance(function, ast.Attribute) or not function.attr.startswith("assert_"):
        return False
    owner = function.value
    return isinstance(owner, ast.Attribute) and owner.attr == "testing"


def _call_name(call: ast.Call) -> str | None:
    """Return the terminal callable name without resolving or importing source code."""
    if isinstance(call.func, ast.Name):
        return call.func.id
    if isinstance(call.func, ast.Attribute):
        return call.func.attr
    return None


def _is_upstream_module(module: str) -> bool:
    """Recognize the pinned Alphalens package name or a source-side test module."""
    return module == "alphalens" or module.startswith("alphalens.") or _is_source_test_module(module)


def _is_source_test_module(module: str) -> bool:
    """Recognize the three frozen upstream test modules without banning local tests."""
    parts = module.split(".")
    return (
        module in SOURCE_TEST_MODULES
        or module.startswith("tests.test_")
        or (len(parts) >= 2 and parts[-2] == "tests" and parts[-1] in SOURCE_TEST_MODULES)
    )


def _import_aliases(tree: ast.Module) -> tuple[set[str], set[str], set[str], set[str]]:
    """Collect AST-visible aliases for sys.path and importlib dynamic imports."""
    sys_modules: set[str] = set()
    sys_paths: set[str] = set()
    importlib_modules: set[str] = set()
    import_module_functions: set[str] = {"__import__"}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == "sys":
                    sys_modules.add(alias.asname or "sys")
                elif alias.name == "importlib":
                    importlib_modules.add(alias.asname or "importlib")
        elif isinstance(node, ast.ImportFrom):
            if node.module == "sys":
                sys_paths.update(alias.asname or alias.name for alias in node.names if alias.name == "path")
            elif node.module == "importlib":
                import_module_functions.update(
                    alias.asname or alias.name for alias in node.names if alias.name == "import_module"
                )
    return sys_modules, sys_paths, importlib_modules, import_module_functions


def _literal_string(node: ast.AST | None) -> str | None:
    """Return a statically visible string literal, if one is present."""
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    return None


def _is_forbidden_absolute_source_path(value: str) -> bool:
    """Reject absolute sibling Alphalens/test-source paths while allowing normal relative paths."""
    for path_type in (PurePosixPath, PureWindowsPath):
        candidate = path_type(value)
        if not candidate.is_absolute():
            continue
        parts = tuple(part.lower() for part in candidate.parts)
        for index, part in enumerate(parts):
            if part == "alphalens" and (index == 0 or parts[index - 1] != "fincore"):
                return True
        if parts and parts[-1] in {f"{module}.py" for module in SOURCE_TEST_MODULES} and "alphalens" in parts:
            return True
    return False


def _is_sys_path_reference(node: ast.AST, sys_modules: set[str], sys_paths: set[str]) -> bool:
    """Return whether an expression denotes ``sys.path`` or a directly imported alias."""
    if isinstance(node, ast.Name):
        return node.id in sys_paths
    return (
        isinstance(node, ast.Attribute)
        and node.attr == "path"
        and isinstance(node.value, ast.Name)
        and node.value.id in sys_modules
    )


def _is_sys_path_mutation(call: ast.Call, sys_modules: set[str], sys_paths: set[str]) -> bool:
    """Return whether a call mutates a sys.path alias, which target tests may not do."""
    function = call.func
    return (
        isinstance(function, ast.Attribute)
        and function.attr in SYS_PATH_MUTATION_METHODS
        and _is_sys_path_reference(function.value, sys_modules, sys_paths)
    )


def _assignment_mutates_sys_path(node: ast.AST, sys_modules: set[str], sys_paths: set[str]) -> bool:
    """Detect assignment and slice-assignment variants in addition to mutation method calls."""
    if isinstance(node, ast.Assign):
        targets = node.targets
    elif isinstance(node, (ast.AnnAssign, ast.AugAssign)):
        targets = [node.target]
    else:
        return False
    for target in targets:
        reference = target.value if isinstance(target, ast.Subscript) else target
        if _is_sys_path_reference(reference, sys_modules, sys_paths):
            return True
    return False


def _is_forbidden_dynamic_import(
    call: ast.Call, importlib_modules: set[str], import_module_functions: set[str]
) -> bool:
    """Detect literal importlib/__import__ access to frozen upstream source modules."""
    function = call.func
    uses_import_module = isinstance(function, ast.Name) and function.id in import_module_functions
    uses_import_module = uses_import_module or (
        isinstance(function, ast.Attribute)
        and function.attr == "import_module"
        and isinstance(function.value, ast.Name)
        and function.value.id in importlib_modules
    )
    if not uses_import_module:
        return False
    return bool(call.args and (module := _literal_string(call.args[0])) and _is_upstream_module(module))


def _validate_target_provenance(tree: ast.Module, relative: str) -> None:
    """Reject static paths and imports that could execute the pinned upstream test package."""
    sys_modules, sys_paths, importlib_modules, import_module_functions = _import_aliases(tree)
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == "alphalens" or alias.name.startswith("alphalens."):
                    raise MigrationAuditError(f"mapped target imports upstream alphalens: {relative}")
                if _is_source_test_module(alias.name):
                    raise MigrationAuditError(f"mapped target imports upstream source-side test module: {relative}")
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            if module == "alphalens" or module.startswith("alphalens."):
                raise MigrationAuditError(f"mapped target imports upstream alphalens: {relative}")
            if _is_source_test_module(module) or (
                module == "tests" and any(alias.name in SOURCE_TEST_MODULES for alias in node.names)
            ):
                raise MigrationAuditError(f"mapped target imports upstream source-side test module: {relative}")
        elif isinstance(node, ast.Call):
            if _is_sys_path_mutation(node, sys_modules, sys_paths):
                raise MigrationAuditError(f"mapped target mutates sys.path: {relative}")
            if _is_forbidden_dynamic_import(node, importlib_modules, import_module_functions):
                raise MigrationAuditError(f"mapped target dynamically imports upstream source: {relative}")
        elif _assignment_mutates_sys_path(node, sys_modules, sys_paths):
            raise MigrationAuditError(f"mapped target mutates sys.path: {relative}")
        elif isinstance(node, ast.Constant) and isinstance(node.value, str) and _is_forbidden_absolute_source_path(node.value):
            raise MigrationAuditError(f"mapped target contains an absolute upstream/source path: {relative}")


def _c4_evidence(function: ast.FunctionDef) -> dict[str, bool]:
    """Collect explicit Task-8 C4 figure lifecycle and artifact ownership signals."""
    figure = False
    show = False
    close = False
    ownership = False
    for node in ast.walk(function):
        if isinstance(node, ast.Name) and node.id.lower() in {"fig", "figure", "ax", "axes"}:
            figure = True
        if isinstance(node, ast.Attribute) and node.attr.lower() in {"axes", "figure", "get_axes", "get_figure"}:
            figure = True
        if not isinstance(node, ast.Call):
            continue
        name = _call_name(node)
        if name is None:
            continue
        lowered = name.lower()
        if name in C4_FIGURE_HELPERS or lowered in {"gca", "gcf"}:
            figure = True
        if name in C4_SHOW_HELPERS or lowered == "show":
            show = True
        if name in C4_CLOSE_HELPERS or lowered in {"close", "close_all"}:
            close = True
        if name in C4_OWNERSHIP_HELPERS:
            ownership = True
    return {"figure_or_axes": figure, "show": show, "close": close, "ownership": ownership}


def _validate_c4_evidence(function: ast.FunctionDef, selector: str) -> None:
    """Require each C4 target to prove rendered figures, lifecycle, and owned cleanup."""
    evidence = _c4_evidence(function)
    missing: list[str] = []
    if not evidence["figure_or_axes"]:
        missing.append("Figure/Axes return or inspection")
    if not evidence["show"] or not evidence["close"]:
        missing.append("show and close handling")
    if not evidence["ownership"]:
        missing.append("artifact/resource ownership or cleanup")
    if missing:
        raise MigrationAuditError(f"C4 target lacks {', '.join(missing)}: {selector}")


def _is_pytest_upstream_marker(call: ast.Call) -> bool:
    """Return whether a call is the concrete ``pytest.mark`` migration marker."""
    function = call.func
    return (
        isinstance(function, ast.Attribute)
        and function.attr == "alphalens_upstream_case"
        and isinstance(function.value, ast.Attribute)
        and function.value.attr == "mark"
        and isinstance(function.value.value, ast.Name)
        and function.value.value.id == "pytest"
    )


def _decorator_marker_ids(nodes: list[ast.expr]) -> set[str]:
    """Read literal upstream-case marker IDs attached to a definition decorator."""
    ids: set[str] = set()
    for decorator in nodes:
        for node in ast.walk(decorator):
            if not (
                isinstance(node, ast.Call)
                and _is_pytest_upstream_marker(node)
                and node.args
                and isinstance(node.args[0], ast.Constant)
                and isinstance(node.args[0].value, str)
            ):
                continue
            ids.add(node.args[0].value)
    return ids


def _target_definitions(tree: ast.Module) -> dict[str, tuple[ast.FunctionDef, set[str]]]:
    """Map qualified target definitions to marker IDs attached to their decorators."""
    definitions: dict[str, tuple[ast.FunctionDef, set[str]]] = {}

    def visit(statements: list[ast.stmt], prefix: tuple[str, ...], inherited: set[str]) -> None:
        for node in statements:
            if isinstance(node, ast.ClassDef):
                visit(node.body, (*prefix, node.name), inherited | _decorator_marker_ids(node.decorator_list))
            elif isinstance(node, ast.FunctionDef):
                qualified = "::".join((*prefix, node.name))
                definitions[qualified] = (node, inherited | _decorator_marker_ids(node.decorator_list))

    visit(tree.body, (), set())
    return definitions


def _validate_target_ast(
    selected_inventory: dict[str, dict[str, Any]], selected_map: dict[str, dict[str, Any]]
) -> None:
    """Audit later target files for marker, import, and assertion anti-patterns."""
    files = _target_files(selected_map)
    functions: dict[tuple[str, str], tuple[ast.FunctionDef, set[str]]] = {}
    for relative in files:
        path = REPO_ROOT / relative
        try:
            text = path.read_text(encoding="utf-8")
        except OSError as exc:
            raise MigrationAuditError(f"mapped target source is unavailable: {relative}: {exc}") from exc
        try:
            tree = ast.parse(text, filename=str(path))
        except SyntaxError as exc:
            raise MigrationAuditError(f"cannot parse mapped target {relative}: {exc}") from exc
        parents = {child: node for node in ast.walk(tree) for child in ast.iter_child_nodes(node)}
        if any(_call_is_bare_equals(call, parents) for call in ast.walk(tree) if isinstance(call, ast.Call)):
            raise MigrationAuditError(f"mapped target contains discarded .equals(): {relative}")
        _validate_target_provenance(tree, relative)
        functions.update({(relative, name): value for name, value in _target_definitions(tree).items()})
    for source_case_id, case in selected_inventory.items():
        record = selected_map[source_case_id]
        if _group(case) == "tears":
            expected_ids = case["invocation_ids"]
            targets = record["invocation_targets"]
            pairs = [(invocation_id, targets[invocation_id]) for invocation_id in expected_ids]
        else:
            pairs = [(source_case_id, selector) for selector in record["target_selectors"]]
        for case_id, selector in pairs:
            relative = _selector_path(selector)
            function_name = _selector_function(selector)
            target = functions.get((relative, function_name))
            if target is None:
                raise MigrationAuditError(f"mapped target function is absent: {selector}")
            function, marker_ids = target
            if case_id not in marker_ids:
                raise MigrationAuditError(f"mapped target lacks a bound alphalens_upstream_case marker: {case_id}")
            if _group(case) == "tears":
                _validate_c4_evidence(function, selector)
                continue
            has_assertion = any(isinstance(node, ast.Assert) for node in ast.walk(function)) or any(
                _is_testing_call(node) for node in ast.walk(function) if isinstance(node, ast.Call)
            )
            if not has_assertion:
                raise MigrationAuditError(f"mapped target has no assertion or testing call: {selector}")


def _validate_nodeids(
    nodeids: list[str], selected_inventory: dict[str, dict[str, Any]], selected_map: dict[str, dict[str, Any]]
) -> dict[str, str]:
    """Prove each selected case/invocation is collected exactly once at its map target."""
    expected: dict[str, str] = {}
    for source_case_id, case in selected_inventory.items():
        record = selected_map[source_case_id]
        if _group(case) == "tears":
            expected.update(record["invocation_targets"])
        else:
            selectors = record["target_selectors"]
            if len(selectors) != 1:
                raise MigrationAuditError(f"ordinary source map must have one exact target selector: {source_case_id}")
            expected[source_case_id] = selectors[0]
    for case_id, expected_nodeid in expected.items():
        matching_ids = [nodeid for nodeid in nodeids if case_id in nodeid]
        if matching_ids != [expected_nodeid]:
            raise MigrationAuditError(
                f"collection does not contain exactly the mapped nodeid for {case_id}: {matching_ids} != {[expected_nodeid]}"
            )
    duplicate_nodeids = [nodeid for nodeid, count in Counter(nodeids).items() if count > 1 and nodeid in expected.values()]
    if duplicate_nodeids:
        raise MigrationAuditError(f"collection repeats mapped nodeids: {duplicate_nodeids[:3]}")
    return expected


def _validate_results(path: Path, expected: dict[str, str]) -> None:
    """Validate non-xdist all-attempt pass results written by the shared marker hook."""
    result_document = _load_json(path, "--results")
    if result_document.get("schema_version") != RESULT_SCHEMA_VERSION:
        raise MigrationAuditError(
            f"--results schema must be {RESULT_SCHEMA_VERSION}: {result_document.get('schema_version')!r}"
        )
    if result_document.get("xdist") is not False:
        raise MigrationAuditError("--results was produced under xdist; non-xdist proof is required")
    if result_document.get("pytest_exitstatus") != 0:
        raise MigrationAuditError(f"--results pytest session did not pass: {result_document.get('pytest_exitstatus')!r}")
    entries = result_document.get("results")
    if not isinstance(entries, list):
        raise MigrationAuditError("--results requires a results list")
    by_nodeid: dict[str, dict[str, Any]] = {}
    for entry in entries:
        if not isinstance(entry, dict) or not isinstance(entry.get("nodeid"), str):
            raise MigrationAuditError("--results has a malformed marked-item entry")
        nodeid = entry["nodeid"]
        if nodeid in by_nodeid:
            raise MigrationAuditError(f"--results duplicates nodeid: {nodeid}")
        by_nodeid[nodeid] = entry
    for case_id, nodeid in expected.items():
        entry = by_nodeid.get(nodeid)
        if entry is None:
            raise MigrationAuditError(f"--results omits mapped target: {nodeid}")
        if entry.get("case_id") != case_id:
            raise MigrationAuditError(f"--results marker case ID mismatch for {nodeid}")
        outcomes = entry.get("outcomes")
        if not isinstance(outcomes, dict) or any(outcomes.get(phase) != "passed" for phase in RESULT_OUTCOMES):
            raise MigrationAuditError(f"--results is not all-passed for {nodeid}: {outcomes!r}")
        attempts = entry.get("attempts")
        if not isinstance(attempts, list) or not attempts:
            raise MigrationAuditError(f"--results lacks append-only attempt history for {nodeid}")
        for attempt_number, attempt in enumerate(attempts, start=1):
            attempt_outcomes = attempt.get("outcomes") if isinstance(attempt, dict) else None
            if not isinstance(attempt_outcomes, dict) or any(
                attempt_outcomes.get(phase) != "passed" for phase in RESULT_OUTCOMES
            ):
                raise MigrationAuditError(
                    f"--results has a non-passing attempt {attempt_number} for {nodeid}: {attempt_outcomes!r}"
                )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse static or later execution-audit inputs."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--migration", type=Path, required=True)
    parser.add_argument("--nodeids", type=Path, help="pytest --collect-only -q output from later target tests")
    parser.add_argument("--results", type=Path, help="non-xdist marker-hook result JSON from later target tests")
    parser.add_argument("--scope", choices=("utils", "performance", "tears", "all"), default="all")
    arguments = parser.parse_args(argv)
    if arguments.results is not None and arguments.nodeids is None:
        parser.error("--results requires --nodeids")
    if arguments.nodeids is not None and arguments.results is None:
        parser.error("--nodeids requires non-xdist --results proof")
    return arguments


def main(argv: list[str] | None = None) -> int:
    """Run static audit now, and collection/result proof when inputs are supplied."""
    arguments = parse_args(argv)
    try:
        inventory = _load_json(arguments.inventory, "inventory")
        migration = _load_json(arguments.migration, "migration")
        selected_inventory, selected_map = _validate_static(inventory, migration, _scope_groups(arguments.scope))
        if arguments.nodeids is None:
            print(
                "static migration audit OK: "
                f"scope={arguments.scope} cases={len(selected_inventory)} "
                "(target collection/results deferred until Tasks 3/4/8)"
            )
            return 0
        nodeids = _read_nodeids(arguments.nodeids)
        _validate_target_ast(selected_inventory, selected_map)
        expected = _validate_nodeids(nodeids, selected_inventory, selected_map)
        assert arguments.results is not None
        _validate_results(arguments.results, expected)
        print(f"executed migration audit OK: scope={arguments.scope} targets={len(expected)}")
    except MigrationAuditError as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
