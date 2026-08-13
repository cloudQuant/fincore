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


@pytest.mark.parametrize("marker", ("skip", "skipif", "xfail"))
def test_upstream_marker_rejects_collection_weakeners(pytester: pytest.Pytester, marker: str) -> None:
    conftest = (ROOT / "tests" / "conftest.py").read_text(encoding="utf-8")
    pytester.makeconftest(conftest)
    marker_arguments = "True, reason=\"not allowed\"" if marker == "skipif" else "reason=\"not allowed\""
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
    assert collection.ret != 0
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
    assert document["xdist"] is False
    assert document["results"] == [
        {
            "case_id": case_id,
            "nodeid": "test_upstream_marker_records_runtime_nonpassing_phase_as_a_failure.py::test_runtime_nonpass",
            "outcomes": {"call": expected_outcome, "setup": "passed", "teardown": "passed"},
        }
    ]
