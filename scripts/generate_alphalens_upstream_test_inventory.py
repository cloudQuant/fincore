#!/usr/bin/env python3
"""Build the static Alphalens upstream-test inventory from pinned Git blobs.

The generator deliberately never imports or executes the upstream package or
its tests.  Every source byte is obtained with ``git show`` from the exact
commit recorded below, then parsed with the Python standard-library AST.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

PINNED_COMMIT = "3fa17ad4c3edb025d1410de7aeba9673cba7791c"
PROFILE = "cloudquant-local-3fa17ad"
SOURCE_PATHS = (
    "tests/test_utils.py",
    "tests/test_performance.py",
    "tests/test_tears.py",
)
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
EXPECTED_COUNTS = {
    "active_declared_cases": 117,
    "diagnostic_collectible_cases": 116,
    "active_methods": 22,
    "dormant_tear_rows": 24,
    "dormant_tear_workflows": 7,
    "dormant_tear_invocations": 96,
}
GIT_TIMEOUT_SECONDS = 30


class InventoryError(RuntimeError):
    """Raised when a pinned source or the inventory contract is invalid."""


@dataclass(frozen=True)
class SourceBlob:
    """Immutable Git-object metadata used by every inventory row."""

    path: str
    git_blob: str
    sha256: str
    text: str


def _run_git(source: Path, arguments: list[str], operation: str) -> bytes:
    """Run one bounded, noninteractive Git command against the source repo."""
    environment = {"GIT_TERMINAL_PROMPT": "0", "GIT_ASKPASS": ""}
    try:
        result = subprocess.run(
            ["git", *arguments],
            cwd=source,
            capture_output=True,
            check=True,
            env=environment,
            stdin=subprocess.DEVNULL,
            timeout=GIT_TIMEOUT_SECONDS,
        )
    except FileNotFoundError as exc:
        raise InventoryError("Git executable is required for pinned source extraction") from exc
    except subprocess.TimeoutExpired as exc:
        raise InventoryError(f"{operation} timed out after {GIT_TIMEOUT_SECONDS}s") from exc
    except subprocess.CalledProcessError as exc:
        detail = (exc.stderr or exc.stdout or b"").decode(errors="replace").strip()
        raise InventoryError(f"{operation} failed: {detail or exc}") from exc
    return result.stdout


def _read_pinned_source(source: Path, commit: str, path: str) -> SourceBlob:
    """Read one source file from its Git object, never from the worktree."""
    raw = _run_git(source, ["show", f"{commit}:{path}"], f"read pinned source {path}")
    git_blob = _run_git(source, ["rev-parse", f"{commit}:{path}"], f"resolve pinned blob {path}").decode().strip()
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise InventoryError(f"pinned source {path} is not UTF-8") from exc
    return SourceBlob(path=path, git_blob=git_blob, sha256=hashlib.sha256(raw).hexdigest(), text=text)


def _direct_test_methods(tree: ast.Module, class_name: str) -> list[ast.FunctionDef]:
    """Return direct ``test_*`` methods from one named source class."""
    classes = [node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == class_name]
    if len(classes) != 1:
        raise InventoryError(f"expected exactly one class {class_name}, found {len(classes)}")
    return [
        node
        for node in classes[0].body
        if isinstance(node, ast.FunctionDef) and node.name.startswith("test_")
    ]


def _parameterized_rows(method: ast.FunctionDef) -> list[ast.expr]:
    """Return AST-only ``parameterized.expand`` row nodes for one method."""
    for decorator in method.decorator_list:
        if not (
            isinstance(decorator, ast.Call)
            and isinstance(decorator.func, ast.Attribute)
            and isinstance(decorator.func.value, ast.Name)
            and decorator.func.value.id == "parameterized"
            and decorator.func.attr == "expand"
        ):
            continue
        if len(decorator.args) != 1 or decorator.keywords:
            raise InventoryError(f"{method.name} has a non-literal parameterized.expand decorator")
        row_container = decorator.args[0]
        if not isinstance(row_container, (ast.List, ast.Tuple)):
            raise InventoryError(f"{method.name} parameterized.expand rows are not a literal sequence")
        if not all(isinstance(row, (ast.List, ast.Tuple)) for row in row_container.elts):
            raise InventoryError(f"{method.name} parameterized.expand contains a non-literal row")
        return list(row_container.elts)
    return []


def _assertion_quality(method: ast.FunctionDef) -> str:
    """Classify the source assertion form without evaluating source code."""
    names = {node.id for node in ast.walk(method) if isinstance(node, ast.Name)}
    calls = [node for node in ast.walk(method) if isinstance(node, ast.Call)]
    if any(name.startswith("assert_") and name.endswith("_equal") for name in names):
        return "pandas_assertion"
    if any(
        isinstance(call.func, ast.Attribute) and call.func.attr == "equals"
        for call in calls
    ):
        return "discarded_equals"
    return "smoke_only"


def _case_id(path: str, class_name: str, method: ast.FunctionDef, ordinal: int) -> str:
    """Build the stable, source-oriented identifier used by future test nodes."""
    return f"{path}::{class_name}::{method.name}#{ordinal:02d}"


def _active_cases(blob: SourceBlob, class_name: str) -> list[dict[str, Any]]:
    """Inventory active Utils/Performance methods and static parameter rows."""
    try:
        tree = ast.parse(blob.text, filename=blob.path)
    except SyntaxError as exc:
        raise InventoryError(f"cannot parse pinned source {blob.path}: {exc}") from exc
    methods = _direct_test_methods(tree, class_name)
    direct_method_names = {method.name for method in methods}
    cases: list[dict[str, Any]] = []
    for method in methods:
        rows = _parameterized_rows(method)
        row_count = len(rows) if rows else 1
        for ordinal in range(row_count):
            generated_method_name = f"{method.name}_{ordinal}"
            state = "active_declared"
            if rows and generated_method_name in direct_method_names:
                state = "shadowed_by_generated_method_name"
            case: dict[str, Any] = {
                "source_case_id": _case_id(blob.path, class_name, method, ordinal),
                "source_path": blob.path,
                "source_class": class_name,
                "source_method": method.name,
                "source_line": method.lineno,
                "parameter_ordinal": f"#{ordinal:02d}",
                "source_collection_state": state,
                "assertion_quality": _assertion_quality(method),
                "source_git_blob": blob.git_blob,
                "source_sha256": blob.sha256,
            }
            if state == "shadowed_by_generated_method_name":
                case["shadowed_generated_method_name"] = generated_method_name
            cases.append(case)
    return cases


def _uncomment_tear_class(source: str) -> str:
    """Remove comment prefixes only within the dormant ``TearsTestCase`` block."""
    lines = source.splitlines(keepends=True)
    start = next(
        (index for index, line in enumerate(lines) if line.startswith("# class TearsTestCase(")),
        None,
    )
    if start is None:
        raise InventoryError("could not find the commented TearsTestCase source block")
    end = start
    while end < len(lines) and lines[end].startswith("#"):
        end += 1
    if end == start:
        raise InventoryError("commented TearsTestCase source block is empty")
    for index in range(start, end):
        uncommented = lines[index][1:]
        lines[index] = uncommented[1:] if uncommented.startswith(" ") else uncommented
    return "".join(lines)


def _class_list_lengths(class_node: ast.ClassDef) -> dict[str, int]:
    """Read literal list sizes from class attributes used by tear workflows."""
    lengths: dict[str, int] = {}
    for statement in class_node.body:
        if not isinstance(statement, ast.Assign) or not isinstance(statement.value, (ast.List, ast.Tuple)):
            continue
        for target in statement.targets:
            if isinstance(target, ast.Name):
                lengths[target.id] = len(statement.value.elts)
    return lengths


def _loop_input_count(iterable: ast.expr, list_lengths: dict[str, int]) -> int:
    """Determine a literal class-list loop cardinality without executing it."""
    if isinstance(iterable, ast.Call) and isinstance(iterable.func, ast.Name) and iterable.func.id == "zip":
        candidates = [
            argument.attr
            for argument in iterable.args
            if isinstance(argument, ast.Attribute)
            and isinstance(argument.value, ast.Name)
            and argument.value.id == "self"
            and argument.attr in list_lengths
        ]
        sizes = {list_lengths[name] for name in candidates}
        if len(sizes) == 1:
            return sizes.pop()
    if (
        isinstance(iterable, ast.Attribute)
        and isinstance(iterable.value, ast.Name)
        and iterable.value.id == "self"
        and iterable.attr in list_lengths
    ):
        return list_lengths[iterable.attr]
    raise InventoryError("tear workflow loop is not over a pinned literal class list")


class _TearCallGroups(ast.NodeVisitor):
    """Collect tear-sheet calls grouped by their static input-loop cardinality."""

    def __init__(self, list_lengths: dict[str, int]) -> None:
        self.list_lengths = list_lengths
        self.groups: list[tuple[int, list[str]]] = []
        self._active_group: list[str] | None = None
        self._active_group_inputs: int | None = None

    def visit_For(self, node: ast.For) -> None:
        if self._active_group is not None:
            raise InventoryError("nested tear workflow loops are outside the restricted AST contract")
        self._active_group = []
        self._active_group_inputs = _loop_input_count(node.iter, self.list_lengths)
        for child in [*node.body, *node.orelse]:
            self.visit(child)
        if self._active_group:
            self.groups.append((self._active_group_inputs, self._active_group))
        self._active_group = None
        self._active_group_inputs = None

    def visit_Call(self, node: ast.Call) -> None:
        if isinstance(node.func, ast.Name) and node.func.id.startswith("create_") and node.func.id.endswith("_tear_sheet"):
            if self._active_group is None:
                self.groups.append((1, [node.func.id]))
            else:
                self._active_group.append(node.func.id)
        self.generic_visit(node)


def _tear_invocations(
    method: ast.FunctionDef,
    source_case_id: str,
    list_lengths: dict[str, int],
) -> tuple[list[str], dict[str, str]]:
    """Build one stable ID per original tear-sheet call over each input row."""
    collector = _TearCallGroups(list_lengths)
    collector.visit(method)
    if not collector.groups:
        raise InventoryError(f"{method.name} has no static tear-sheet calls")
    invocation_ids: list[str] = []
    call_names: dict[str, str] = {}
    input_offset = 0
    for input_count, calls in collector.groups:
        for input_index in range(input_count):
            for call_index, call_name in enumerate(calls):
                invocation_id = (
                    f"{source_case_id}/input-{input_offset + input_index:02d}/call-{call_index:02d}"
                )
                invocation_ids.append(invocation_id)
                call_names[invocation_id] = call_name
        input_offset += input_count
    return invocation_ids, call_names


def _tear_cases(blob: SourceBlob) -> list[dict[str, Any]]:
    """Inventory the intentionally commented tear workflows with restricted AST."""
    try:
        tree = ast.parse(_uncomment_tear_class(blob.text), filename=blob.path)
    except SyntaxError as exc:
        raise InventoryError(f"cannot parse uncommented TearsTestCase source: {exc}") from exc
    classes = [node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "TearsTestCase"]
    if len(classes) != 1:
        raise InventoryError(f"expected exactly one TearsTestCase, found {len(classes)}")
    class_node = classes[0]
    list_lengths = _class_list_lengths(class_node)
    methods = [
        node
        for node in class_node.body
        if isinstance(node, ast.FunctionDef) and node.name.startswith("test_")
    ]
    cases: list[dict[str, Any]] = []
    for method in methods:
        rows = _parameterized_rows(method)
        if not rows:
            raise InventoryError(f"dormant tear workflow {method.name} lacks literal parameter rows")
        for ordinal in range(len(rows)):
            source_case_id = _case_id(blob.path, class_node.name, method, ordinal)
            invocation_ids, invocation_calls = _tear_invocations(method, source_case_id, list_lengths)
            cases.append(
                {
                    "source_case_id": source_case_id,
                    "source_path": blob.path,
                    "source_class": class_node.name,
                    "source_method": method.name,
                    "source_line": method.lineno,
                    "parameter_ordinal": f"#{ordinal:02d}",
                    "source_collection_state": "commented_out",
                    "assertion_quality": "smoke_only",
                    "source_git_blob": blob.git_blob,
                    "source_sha256": blob.sha256,
                    "invocation_ids": invocation_ids,
                    "invocation_calls": invocation_calls,
                }
            )
    return cases


def _source_files(blobs: dict[str, SourceBlob]) -> dict[str, dict[str, str]]:
    """Return the compact source evidence shape kept in the checked-in JSON."""
    return {
        path: {"git_blob": blob.git_blob, "sha256": blob.sha256}
        for path, blob in blobs.items()
    }


def _validate_inventory(inventory: dict[str, Any]) -> None:
    """Fail closed unless the frozen source evidence and counts exactly agree."""
    problems: list[str] = []
    if inventory.get("commit") != PINNED_COMMIT:
        problems.append("commit does not match the fixed Alphalens source identity")
    if inventory.get("source_files") != EXPECTED_SOURCE_FILES:
        problems.append("pinned source blob IDs or SHA256 values differ from the reviewed inputs")
    cases = inventory.get("cases")
    if not isinstance(cases, list):
        problems.append("cases is not a list")
    else:
        source_ids = [case.get("source_case_id") for case in cases]
        if len(source_ids) != len(set(source_ids)):
            problems.append("source_case_id values are not unique")
        active_cases = [case for case in cases if case.get("source_path") != "tests/test_tears.py"]
        tear_cases = [case for case in cases if case.get("source_path") == "tests/test_tears.py"]
        observed = {
            "active_declared_cases": len(active_cases),
            "diagnostic_collectible_cases": sum(
                case.get("source_collection_state") == "active_declared" for case in active_cases
            ),
            "active_methods": len({(case.get("source_path"), case.get("source_method")) for case in active_cases}),
            "dormant_tear_rows": len(tear_cases),
            "dormant_tear_workflows": len({case.get("source_method") for case in tear_cases}),
            "dormant_tear_invocations": sum(len(case.get("invocation_ids", [])) for case in tear_cases),
        }
        if observed != EXPECTED_COUNTS:
            problems.append(f"inventory counts differ: observed {observed}, expected {EXPECTED_COUNTS}")
        shadowed = [
            case
            for case in active_cases
            if case.get("source_collection_state") == "shadowed_by_generated_method_name"
        ]
        if len(shadowed) != 1:
            problems.append(f"expected one shadowed generated performance row, found {len(shadowed)}")
        if any(case.get("source_collection_state") != "commented_out" for case in tear_cases):
            problems.append("all dormant tear rows must remain commented_out")
    if inventory.get("counts") != EXPECTED_COUNTS:
        problems.append("serialized counts do not match the fixed inventory contract")
    if problems:
        raise InventoryError("invalid Alphalens upstream-test inventory:\n- " + "\n- ".join(problems))


def build_inventory(source: Path, commit: str = PINNED_COMMIT) -> dict[str, Any]:
    """Build and validate the deterministic inventory from the pinned Git tree."""
    if commit != PINNED_COMMIT:
        raise InventoryError(f"only pinned commit {PINNED_COMMIT} is accepted, got {commit}")
    source = source.resolve()
    _run_git(source, ["cat-file", "-e", f"{commit}^{{commit}}"], "validate pinned Git commit")
    blobs = {path: _read_pinned_source(source, commit, path) for path in SOURCE_PATHS}
    cases = [
        *_active_cases(blobs["tests/test_utils.py"], "UtilsTestCase"),
        *_active_cases(blobs["tests/test_performance.py"], "PerformanceTestCase"),
        *_tear_cases(blobs["tests/test_tears.py"]),
    ]
    cases.sort(key=lambda case: (case["source_path"], case["source_line"], case["parameter_ordinal"]))
    inventory: dict[str, Any] = {
        "schema_version": "alphalens-upstream-test-inventory-v1",
        "profile": PROFILE,
        "commit": commit,
        "extraction": {
            "mode": "static-ast-only",
            "source_bytes": "git-show-pinned-blobs",
            "upstream_imported": False,
            "upstream_executed": False,
        },
        "source_files": _source_files(blobs),
        "counts": EXPECTED_COUNTS,
        "cases": cases,
    }
    _validate_inventory(inventory)
    return inventory


def _render(inventory: dict[str, Any]) -> bytes:
    """Render canonical, byte-stable JSON for checked-in fixture comparison."""
    return (json.dumps(inventory, indent=2, sort_keys=True, ensure_ascii=False) + "\n").encode("utf-8")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse the inventory generator command line."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True, help="Git checkout containing the pinned commit")
    parser.add_argument("--commit", default=PINNED_COMMIT, help="required pinned Git commit")
    output = parser.add_mutually_exclusive_group(required=True)
    output.add_argument("--output", type=Path, help="write the canonical inventory fixture")
    output.add_argument("--check", type=Path, help="fail unless this fixture is byte-identical")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Generate or verify the inventory fixture without importing upstream code."""
    arguments = parse_args(argv)
    try:
        rendered = _render(build_inventory(arguments.source, arguments.commit))
        if arguments.check is not None:
            try:
                expected = arguments.check.read_bytes()
            except OSError as exc:
                raise InventoryError(f"cannot read --check fixture {arguments.check}: {exc}") from exc
            if expected != rendered:
                raise InventoryError(f"fixture is not byte-identical: {arguments.check}")
            print(f"inventory check OK: {arguments.check}")
        else:
            assert arguments.output is not None
            arguments.output.parent.mkdir(parents=True, exist_ok=True)
            if not arguments.output.exists() or arguments.output.read_bytes() != rendered:
                arguments.output.write_bytes(rendered)
            print(f"inventory written: {arguments.output}")
    except InventoryError as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
