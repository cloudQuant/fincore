"""Static audit coverage for the pinned Alphalens upstream-test handoff."""

from __future__ import annotations

import ast
import json
import runpy
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

pytest_plugins = ("pytester",)


ROOT = Path(__file__).resolve().parents[2]
FIXTURES = Path(__file__).resolve().parent / "fixtures"
INVENTORY_PATH = FIXTURES / "alphalens-0.4.0-cloudquant-upstream-test-inventory.json"
MIGRATION_PATH = FIXTURES / "alphalens-0.4.0-cloudquant-upstream-test-migration.json"
CHECKER = ROOT / "scripts" / "check_alphalens_upstream_test_migration.py"
GENERATOR = ROOT / "scripts" / "generate_alphalens_upstream_test_inventory.py"

PINNED_COMMIT = "3fa17ad4c3edb025d1410de7aeba9673cba7791c"
EXPECTED_COUNTS = {
    "active_declared_cases": 117,
    "diagnostic_collectible_cases": 116,
    "active_methods": 22,
    "dormant_tear_rows": 24,
    "dormant_tear_workflows": 7,
    "dormant_tear_invocations": 96,
}
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


def _load(path: Path) -> dict[str, object]:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _single_target_context(
    tmp_path: Path, source_path: str
) -> tuple[dict[str, object], dict[str, object], dict[str, object], str, Path, str]:
    """Build one deferred target AST context without creating a repository target test."""
    inventory = _load(INVENTORY_PATH)
    migration = _load(MIGRATION_PATH)
    cases = inventory["cases"]
    mapped_cases = migration["cases"]
    assert isinstance(cases, list)
    assert isinstance(mapped_cases, dict)
    case = next(item for item in cases if item["source_path"] == source_path)
    case_id = case["source_case_id"]
    assert isinstance(case_id, str)
    record = mapped_cases[case_id]
    assert isinstance(record, dict)

    if source_path == "tests/test_tears.py":
        invocation_ids = case["invocation_ids"]
        invocation_targets = record["invocation_targets"]
        assert isinstance(invocation_ids, list) and isinstance(invocation_targets, dict)
        marker_id = invocation_ids[0]
        selector = invocation_targets[marker_id]
    else:
        marker_id = case_id
        selectors = record["target_selectors"]
        assert isinstance(selectors, list)
        selector = selectors[0]
    assert isinstance(marker_id, str) and isinstance(selector, str)

    relative, _, selected_name = selector.partition("::")
    function_name = selected_name.split("[", 1)[0]
    target_path = tmp_path / relative
    target_path.parent.mkdir(parents=True)
    checker = runpy.run_path(str(CHECKER))
    checker["_validate_target_ast"].__globals__["REPO_ROOT"] = tmp_path
    return checker, {case_id: case}, {case_id: record}, marker_id, target_path, function_name


def _write_marked_target(
    target_path: Path, function_name: str, marker_id: str, body: str, *, imports: str = "import pytest"
) -> None:
    """Write a parse-only deferred target fixture carrying the exact marker protocol."""
    body_block = textwrap.indent(textwrap.dedent(body).strip(), "    ")
    import_block = textwrap.dedent(imports).strip()
    target_path.write_text(
        f"{import_block}\n\n"
        "@pytest.mark.parametrize(\n"
        "    \"source_case_id\",\n"
        "    [\n"
        "        pytest.param(\n"
        f"            {marker_id!r},\n"
        f"            id={marker_id!r},\n"
        f"            marks=pytest.mark.alphalens_upstream_case({marker_id!r}),\n"
        "        ),\n"
        "    ],\n"
        ")\n"
        f"def {function_name}(source_case_id):\n"
        f"{body_block}\n",
        encoding="utf-8",
    )


def test_pinned_upstream_test_inventory_and_migration_map_are_complete() -> None:
    inventory = _load(INVENTORY_PATH)
    migration = _load(MIGRATION_PATH)

    assert inventory["commit"] == PINNED_COMMIT
    assert inventory["counts"] == EXPECTED_COUNTS
    assert inventory["source_files"] == EXPECTED_SOURCE_FILES

    cases = inventory["cases"]
    assert isinstance(cases, list)
    source_ids = {case["source_case_id"] for case in cases}
    assert len(source_ids) == 141  # 117 active rows + 24 dormant tear rows.
    assert len(source_ids) == len(cases)

    mapped_cases = migration["cases"]
    assert isinstance(mapped_cases, dict)
    assert set(mapped_cases) == source_ids
    assert all(
        item["disposition"]
        in {"rewritten_strict", "rewritten_invariant", "rebuilt_c4"}
        for item in mapped_cases.values()
    )
    assert all(item["target_selectors"] and item["assertion_grade"] for item in mapped_cases.values())

    tear_cases = [case for case in cases if case["source_path"] == "tests/test_tears.py"]
    assert len(tear_cases) == 24
    assert all(case["source_collection_state"] == "commented_out" for case in tear_cases)
    assert sum(len(case["invocation_ids"]) for case in tear_cases) == 96
    invocation_ids = {
        invocation_id
        for case in tear_cases
        for invocation_id in case["invocation_ids"]
    }
    invocation_targets = {
        invocation_id: nodeid
        for item in mapped_cases.values()
        for invocation_id, nodeid in item.get("invocation_targets", {}).items()
    }
    assert set(invocation_targets) == invocation_ids
    assert len(set(invocation_targets.values())) == len(invocation_ids)


def test_inventory_cases_have_stable_source_and_provenance_fields() -> None:
    inventory = _load(INVENTORY_PATH)
    source_files = inventory["source_files"]
    assert isinstance(source_files, dict)
    cases = inventory["cases"]
    assert isinstance(cases, list)

    for case in cases:
        assert case["source_path"] in source_files
        assert case["source_class"]
        assert case["source_method"]
        assert isinstance(case["source_line"], int) and case["source_line"] > 0
        assert case["parameter_ordinal"].startswith("#")
        assert case["source_collection_state"] in {
            "active_declared",
            "shadowed_by_generated_method_name",
            "commented_out",
        }
        assert case["assertion_quality"] in {"pandas_assertion", "discarded_equals", "smoke_only"}
        assert case["source_git_blob"] == source_files[case["source_path"]]["git_blob"]
        assert case["source_sha256"] == source_files[case["source_path"]]["sha256"]
        assert case["source_case_id"].endswith(case["parameter_ordinal"])

    shadowed = [
        case
        for case in cases
        if case["source_collection_state"] == "shadowed_by_generated_method_name"
    ]
    assert len(shadowed) == 1
    assert shadowed[0]["source_path"] == "tests/test_performance.py"


def test_migration_map_is_staticly_auditable_before_target_tests_exist() -> None:
    result = subprocess.run(
        [
            sys.executable,
            str(CHECKER),
            "--inventory",
            str(INVENTORY_PATH),
            "--migration",
            str(MIGRATION_PATH),
            "--scope",
            "all",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "static migration audit OK" in result.stdout


@pytest.mark.parametrize(
    ("case_id", "field", "value"),
    [
        (
            "tests/test_utils.py::UtilsTestCase::test_compute_forward_returns#00",
            "assertion_grade",
            "C4",
        ),
        (
            "tests/test_performance.py::PerformanceTestCase::test_information_coefficient#00",
            "target_selectors",
            ["tests/compat/alphalens/test_tearsheets_e2e.py::test_wrong_target[source-case]"],
        ),
    ],
)
def test_static_checker_rejects_incompatible_target_contract(
    tmp_path: Path, case_id: str, field: str, value: object
) -> None:
    migration = _load(MIGRATION_PATH)
    cases = migration["cases"]
    assert isinstance(cases, dict)
    cases[case_id][field] = value
    mutated = tmp_path / "migration.json"
    mutated.write_text(json.dumps(migration, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    result = subprocess.run(
        [
            sys.executable,
            str(CHECKER),
            "--inventory",
            str(INVENTORY_PATH),
            "--migration",
            str(mutated),
            "--scope",
            "all",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 1
    assert "FAIL:" in result.stderr


def test_later_target_marker_audit_binds_ids_to_their_own_decorated_function() -> None:
    checker = runpy.run_path(str(CHECKER))
    tree = ast.parse(
        textwrap.dedent(
            """
            import pytest

            @pytest.mark.alphalens_upstream_case("other-case")
            def test_other():
                assert True

            @pytest.mark.parametrize(
                "case_id",
                [
                    pytest.param(
                        "wanted-case",
                        id="wanted-case",
                        marks=pytest.mark.alphalens_upstream_case("wanted-case"),
                    ),
                ],
            )
            def test_target(case_id):
                assert case_id == "wanted-case"
            """
        )
    )
    definitions = checker["_target_definitions"](tree)
    assert definitions["test_other"][1] == {"other-case"}
    assert definitions["test_target"][1] == {"wanted-case"}


def test_no_accepted_upstream_case_is_silently_weakened_or_omitted() -> None:
    migration = _load(MIGRATION_PATH)
    cases = migration["cases"]
    assert isinstance(cases, dict)
    forbidden = {"skip", "xfail", "smoke_only", "raw_copy", "unmapped"}
    assert not {item["disposition"] for item in cases.values()} & forbidden


def test_inventory_generator_is_byte_idempotent_from_the_pinned_git_tree_when_available(tmp_path: Path) -> None:
    source = ROOT.parent / "alphalens"
    if not source.is_dir():
        pytest.skip("pinned Alphalens sibling checkout is optional for static fixture CI")
    result = subprocess.run(
        [
            sys.executable,
            str(GENERATOR),
            "--source",
            str(source),
            "--commit",
            PINNED_COMMIT,
            "--check",
            str(INVENTORY_PATH),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "inventory check OK" in result.stdout
    first = tmp_path / "first.json"
    second = tmp_path / "second.json"
    for output in (first, second):
        generated = subprocess.run(
            [
                sys.executable,
                str(GENERATOR),
                "--source",
                str(source),
                "--commit",
                PINNED_COMMIT,
                "--output",
                str(output),
            ],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        assert generated.returncode == 0, generated.stdout + generated.stderr
    assert first.read_bytes() == second.read_bytes() == INVENTORY_PATH.read_bytes()


@pytest.mark.parametrize(
    ("marker", "marker_arguments"),
    [
        ("skip", 'reason="not allowed"'),
        ("skipif", 'True, reason="not allowed"'),
        ("xfail", 'reason="not allowed"'),
        ("flaky", "reruns=1"),
        ("rerun", "1"),
        ("reruns", "1"),
    ],
)
def test_upstream_marker_rejects_collection_weakeners(
    pytester: pytest.Pytester, marker: str, marker_arguments: str
) -> None:
    conftest = (ROOT / "tests" / "conftest.py").read_text(encoding="utf-8")
    pytester.makeconftest(conftest)
    pytester.makepyfile(
        f"""
        import pytest

        @pytest.mark.alphalens_upstream_case("source-case")
        @pytest.mark.{marker}({marker_arguments})
        def test_collection_weakened():
            pass
        """
    )
    collection = pytester.runpytest("-q")
    assert collection.ret == pytest.ExitCode.USAGE_ERROR
    collection.stderr.fnmatch_lines([f"*upstream case cannot carry {marker}*"])


@pytest.mark.parametrize(
    ("case_id", "body", "expected_outcome"),
    [
        ("runtime-skip", 'pytest.skip("not allowed")', "skipped"),
        ("runtime-xfail", 'pytest.xfail("not allowed")', "xfailed"),
        ("runtime-failure", 'raise AssertionError("not allowed")', "failed"),
    ],
)
def test_upstream_marker_records_runtime_nonpassing_phase_as_a_failure(
    pytester: pytest.Pytester, case_id: str, body: str, expected_outcome: str
) -> None:
    conftest = (ROOT / "tests" / "conftest.py").read_text(encoding="utf-8")
    pytester.makeconftest(conftest)

    pytester.makepyfile(
        f"""
        import pytest

        @pytest.mark.alphalens_upstream_case("{case_id}")
        def test_runtime_nonpass():
            {body}
        """
    )
    result = pytester.runpytest("-q", "--alphalens-upstream-result-json", "build/upstream-results.json")
    assert result.ret == pytest.ExitCode.TESTS_FAILED
    document = json.loads((pytester.path / "build" / "upstream-results.json").read_text(encoding="utf-8"))
    assert document["schema_version"] == "alphalens-upstream-case-results-v2"
    assert document["xdist"] is False
    assert document["results"] == [
        {
            "case_id": case_id,
            "nodeid": "test_upstream_marker_records_runtime_nonpassing_phase_as_a_failure.py::test_runtime_nonpass",
            "outcomes": {"call": expected_outcome, "setup": "passed", "teardown": "passed"},
            "attempts": [
                {
                    "outcomes": {"call": expected_outcome, "setup": "passed", "teardown": "passed"},
                }
            ],
        }
    ]


def test_upstream_marker_rerun_is_never_accepted_after_an_earlier_failure(
    pytester: pytest.Pytester,
) -> None:
    """Exercise the installed rerun plugin in a real subprocess, not a hook mock."""
    pytest.importorskip("pytest_rerunfailures")
    pytester.makeconftest((ROOT / "tests" / "conftest.py").read_text(encoding="utf-8"))
    pytester.makepyfile(
        """
        from pathlib import Path

        import pytest

        @pytest.mark.alphalens_upstream_case("rerun-case")
        def test_fails_once_then_passes():
            attempt_path = Path("attempt-count.txt")
            attempt = int(attempt_path.read_text()) if attempt_path.exists() else 0
            attempt_path.write_text(str(attempt + 1))
            assert attempt == 1
        """
    )
    result = pytester.runpytest_subprocess(
        "-q",
        "--reruns",
        "1",
        "--alphalens-upstream-result-json",
        "build/upstream-results.json",
    )
    assert result.ret == pytest.ExitCode.USAGE_ERROR
    result.stderr.fnmatch_lines(["*upstream case cannot enable reruns via --reruns*"])


def test_upstream_marker_history_forces_failure_if_a_late_plugin_enables_reruns(
    pytester: pytest.Pytester,
) -> None:
    """Keep the session non-accepted if a future plugin enables retries after collection."""
    pytest.importorskip("pytest_rerunfailures")
    pytester.makeconftest((ROOT / "tests" / "conftest.py").read_text(encoding="utf-8"))
    pytester.makepyfile(
        late_rerun_plugin="""
        import pytest

        @pytest.hookimpl(trylast=True)
        def pytest_collection_modifyitems(config, items):
            config.option.reruns = 1
        """,
        test_rerun_history="""
        from pathlib import Path

        import pytest

        @pytest.mark.alphalens_upstream_case("rerun-history-case")
        def test_fails_once_then_passes():
            attempt_path = Path("attempt-count.txt")
            attempt = int(attempt_path.read_text()) if attempt_path.exists() else 0
            attempt_path.write_text(str(attempt + 1))
            assert attempt == 1
        """,
    )
    result = pytester.runpytest_subprocess(
        "-q",
        "-p",
        "late_rerun_plugin",
        "--alphalens-upstream-result-json",
        "build/upstream-results.json",
    )
    assert result.ret == pytest.ExitCode.TESTS_FAILED
    document = json.loads((pytester.path / "build" / "upstream-results.json").read_text(encoding="utf-8"))
    assert document["pytest_exitstatus"] == pytest.ExitCode.TESTS_FAILED
    assert document["results"][0]["outcomes"]["call"] == "failed"
    assert [attempt["outcomes"]["call"] for attempt in document["results"][0]["attempts"]] == [
        "failed",
        "passed",
    ]


def test_result_checker_rejects_a_nonpassing_rerun_attempt_even_if_terminal_outcomes_pass(
    tmp_path: Path,
) -> None:
    """A later plugin ordering must not erase a failed first attempt from proof JSON."""
    checker = runpy.run_path(str(CHECKER))
    proof = tmp_path / "upstream-results.json"
    proof.write_text(
        json.dumps(
            {
                "schema_version": "alphalens-upstream-case-results-v2",
                "xdist": False,
                "pytest_exitstatus": 0,
                "results": [
                    {
                        "case_id": "rerun-case",
                        "nodeid": "tests/compat/alphalens/test_target.py::test_target[rerun-case]",
                        "outcomes": {"setup": "passed", "call": "passed", "teardown": "passed"},
                        "attempts": [
                            {
                                "outcomes": {"setup": "passed", "call": "failed", "teardown": "passed"},
                            },
                            {
                                "outcomes": {"setup": "passed", "call": "passed", "teardown": "passed"},
                            },
                        ],
                    }
                ],
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    with pytest.raises(checker["MigrationAuditError"], match="attempt"):
        checker["_validate_results"](
            proof,
            {"rerun-case": "tests/compat/alphalens/test_target.py::test_target[rerun-case]"},
        )


def test_target_ast_rejects_dynamic_upstream_import_and_sibling_source_path(tmp_path: Path) -> None:
    checker, selected_inventory, selected_map, marker_id, target_path, function_name = _single_target_context(
        tmp_path, "tests/test_utils.py"
    )
    _write_marked_target(
        target_path,
        function_name,
        marker_id,
        "assert source_case_id == " + repr(marker_id),
        imports="""
        import importlib
        import sys
        import pytest

        sys.path.insert(0, "/Users/example/new_projects/alphalens/tests")
        upstream = importlib.import_module("alphalens")
        """,
    )
    with pytest.raises(checker["MigrationAuditError"], match="upstream|source path|sys.path"):
        checker["_validate_target_ast"](selected_inventory, selected_map)


def test_target_ast_rejects_from_tests_source_fixture_import(tmp_path: Path) -> None:
    checker, selected_inventory, selected_map, marker_id, target_path, function_name = _single_target_context(
        tmp_path, "tests/test_utils.py"
    )
    _write_marked_target(
        target_path,
        function_name,
        marker_id,
        "assert source_case_id == " + repr(marker_id),
        imports="""
        from tests import test_utils

        import pytest
        """,
    )
    with pytest.raises(checker["MigrationAuditError"], match="source-side test module"):
        checker["_validate_target_ast"](selected_inventory, selected_map)


def test_target_ast_rejects_builtin_dynamic_source_test_import(tmp_path: Path) -> None:
    checker, selected_inventory, selected_map, marker_id, target_path, function_name = _single_target_context(
        tmp_path, "tests/test_utils.py"
    )
    _write_marked_target(
        target_path,
        function_name,
        marker_id,
        "assert source_case_id == " + repr(marker_id),
        imports="""
        import pytest

        source_test = __import__("tests.test_utils")
        """,
    )
    with pytest.raises(checker["MigrationAuditError"], match="dynamically imports upstream source"):
        checker["_validate_target_ast"](selected_inventory, selected_map)


def test_target_ast_allows_safe_local_imports(tmp_path: Path) -> None:
    checker, selected_inventory, selected_map, marker_id, target_path, function_name = _single_target_context(
        tmp_path, "tests/test_utils.py"
    )
    _write_marked_target(
        target_path,
        function_name,
        marker_id,
        "assert source_case_id == " + repr(marker_id),
        imports="""
        from . import local_helpers
        from fincore import metrics

        import pytest
        """,
    )
    checker["_validate_target_ast"](selected_inventory, selected_map)


def test_c4_target_ast_rejects_bare_true_assertion(tmp_path: Path) -> None:
    checker, selected_inventory, selected_map, marker_id, target_path, function_name = _single_target_context(
        tmp_path, "tests/test_tears.py"
    )
    _write_marked_target(target_path, function_name, marker_id, "assert True")
    with pytest.raises(checker["MigrationAuditError"], match="C4"):
        checker["_validate_target_ast"](selected_inventory, selected_map)


def test_c4_target_ast_rejects_pure_numeric_assertion(tmp_path: Path) -> None:
    checker, selected_inventory, selected_map, marker_id, target_path, function_name = _single_target_context(
        tmp_path, "tests/test_tears.py"
    )
    _write_marked_target(target_path, function_name, marker_id, "assert 1 == 1")
    with pytest.raises(checker["MigrationAuditError"], match="C4"):
        checker["_validate_target_ast"](selected_inventory, selected_map)


def test_c4_target_ast_accepts_figure_show_close_and_owned_artifact_evidence(tmp_path: Path) -> None:
    checker, selected_inventory, selected_map, marker_id, target_path, function_name = _single_target_context(
        tmp_path, "tests/test_tears.py"
    )
    _write_marked_target(
        target_path,
        function_name,
        marker_id,
        """
        figure = build_figure()
        assert_figure_axes(figure)
        figure.show()
        close_figure(figure)
        assert_artifact_ownership({"primary": figure})
        """,
    )
    checker["_validate_target_ast"](selected_inventory, selected_map)
