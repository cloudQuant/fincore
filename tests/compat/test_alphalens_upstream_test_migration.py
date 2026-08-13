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
        '    "source_case_id",\n'
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
        item["disposition"] in {"rewritten_strict", "rewritten_invariant", "rebuilt_c4"}
        for item in mapped_cases.values()
    )
    assert all(item["target_selectors"] and item["assertion_grade"] for item in mapped_cases.values())

    tear_cases = [case for case in cases if case["source_path"] == "tests/test_tears.py"]
    assert len(tear_cases) == 24
    assert all(case["source_collection_state"] == "commented_out" for case in tear_cases)
    assert sum(len(case["invocation_ids"]) for case in tear_cases) == 96
    invocation_ids = {invocation_id for case in tear_cases for invocation_id in case["invocation_ids"]}
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

    shadowed = [case for case in cases if case["source_collection_state"] == "shadowed_by_generated_method_name"]
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
    ("fixture_name", "schema_version"),
    [
        ("inventory", "alphalens-upstream-test-inventory-v999"),
        ("migration", "alphalens-upstream-test-migration-v999"),
        ("inventory", 1),
        ("migration", None),
    ],
)
def test_static_checker_fails_closed_for_unknown_or_nonstring_schema_versions(
    tmp_path: Path, fixture_name: str, schema_version: object
) -> None:
    inventory = _load(INVENTORY_PATH)
    migration = _load(MIGRATION_PATH)
    document = inventory if fixture_name == "inventory" else migration
    document["schema_version"] = schema_version
    inventory_path = tmp_path / "inventory.json"
    migration_path = tmp_path / "migration.json"
    inventory_path.write_text(json.dumps(inventory, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    migration_path.write_text(json.dumps(migration, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    result = subprocess.run(
        [
            sys.executable,
            str(CHECKER),
            "--inventory",
            str(inventory_path),
            "--migration",
            str(migration_path),
            "--scope",
            "all",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 1
    assert "schema_version" in result.stderr


@pytest.mark.parametrize(
    "mutation",
    [
        "inventory-missing-extraction",
        "inventory-malformed-extraction",
        "inventory-wrong-profile",
        "inventory-malformed-cases",
        "inventory-extra-envelope-field",
        "migration-missing-review",
        "migration-malformed-review",
        "migration-wrong-profile",
        "migration-malformed-cases",
        "migration-extra-envelope-field",
    ],
)
def test_static_checker_fails_closed_for_missing_or_malformed_top_level_evidence(tmp_path: Path, mutation: str) -> None:
    inventory = _load(INVENTORY_PATH)
    migration = _load(MIGRATION_PATH)
    if mutation == "inventory-missing-extraction":
        inventory.pop("extraction")
    elif mutation == "inventory-malformed-extraction":
        inventory["extraction"] = []
    elif mutation == "inventory-wrong-profile":
        inventory["profile"] = "other-profile"
    elif mutation == "inventory-malformed-cases":
        inventory["cases"] = {}
    elif mutation == "inventory-extra-envelope-field":
        inventory["unrecognized_evidence"] = True
    elif mutation == "migration-missing-review":
        migration.pop("review")
    elif mutation == "migration-malformed-review":
        migration["review"] = []
    elif mutation == "migration-wrong-profile":
        migration["profile"] = "other-profile"
    elif mutation == "migration-malformed-cases":
        migration["cases"] = []
    elif mutation == "migration-extra-envelope-field":
        migration["unrecognized_evidence"] = True
    else:  # pragma: no cover - parametrization is the contract under test.
        raise AssertionError(f"unknown mutation: {mutation}")
    inventory_path = tmp_path / "inventory.json"
    migration_path = tmp_path / "migration.json"
    inventory_path.write_text(json.dumps(inventory, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    migration_path.write_text(json.dumps(migration, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    result = subprocess.run(
        [
            sys.executable,
            str(CHECKER),
            "--inventory",
            str(inventory_path),
            "--migration",
            str(migration_path),
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


def _collection_proof(
    checker: dict[str, object], *, exitstatus: int = 0, errors: list[str] | None = None
) -> dict[str, object]:
    target_paths = sorted(checker["GROUP_PATHS"]["utils"])
    return {
        "schema_version": "alphalens-upstream-collection-proof-v1",
        "command_identity": "fincore.alphalens-upstream-collect-v1",
        "scope": "utils",
        "command": [sys.executable, "-m", "pytest", "-o", "addopts=", "--collect-only", "-q", *target_paths],
        "target_paths": target_paths,
        "exitstatus": exitstatus,
        "nodeids": [
            "tests/compat/alphalens/test_forward_returns.py::test_forward_returns["
            "tests/test_utils.py::UtilsTestCase::test_compute_forward_returns#00]"
        ],
        "collection_errors": [] if errors is None else errors,
    }


def test_collection_proof_rejects_nonzero_or_error_even_with_a_correct_nodeid(tmp_path: Path) -> None:
    checker = runpy.run_path(str(CHECKER))
    for exitstatus, errors in ((1, []), (0, ["ERROR collecting target test"])):
        proof_path = tmp_path / f"proof-{exitstatus}-{bool(errors)}.json"
        proof_path.write_text(
            json.dumps(_collection_proof(checker, exitstatus=exitstatus, errors=errors), indent=2, sort_keys=True)
            + "\n",
            encoding="utf-8",
        )
        with pytest.raises(checker["MigrationAuditError"], match="collection proof"):
            checker["_read_collection_proof"](proof_path, "utils")


@pytest.mark.parametrize(("exitstatus", "errors"), [(1, []), (0, ["ERROR collecting target test"])])
def test_checker_cli_rejects_malicious_collection_proof_before_target_audit(
    tmp_path: Path, exitstatus: int, errors: list[str]
) -> None:
    checker = runpy.run_path(str(CHECKER))
    proof_path = tmp_path / "malicious-proof.json"
    proof_path.write_text(
        json.dumps(_collection_proof(checker, exitstatus=exitstatus, errors=errors), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    result_path = tmp_path / "unused-results.json"
    result_path.write_text("{}\n", encoding="utf-8")

    result = subprocess.run(
        [
            sys.executable,
            str(CHECKER),
            "--inventory",
            str(INVENTORY_PATH),
            "--migration",
            str(MIGRATION_PATH),
            "--scope",
            "utils",
            "--collection-proof",
            str(proof_path),
            "--results",
            str(result_path),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 1
    assert "collection proof" in result.stderr


@pytest.mark.parametrize(
    ("field", "value", "match"),
    [
        ("schema_version", "alphalens-upstream-collection-proof-v999", "schema_version"),
        ("command_identity", "manual-transcript", "controlled collector"),
        ("scope", "performance", "scope mismatch"),
        ("nodeids", [], "nodeids"),
    ],
)
def test_collection_proof_requires_the_complete_controlled_envelope(
    tmp_path: Path, field: str, value: object, match: str
) -> None:
    checker = runpy.run_path(str(CHECKER))
    proof = _collection_proof(checker)
    proof[field] = value
    proof_path = tmp_path / f"invalid-{field}.json"
    proof_path.write_text(json.dumps(proof, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    with pytest.raises(checker["MigrationAuditError"], match=match):
        checker["_read_collection_proof"](proof_path, "utils")


def test_collection_proof_accepts_a_passing_controlled_envelope(tmp_path: Path) -> None:
    checker = runpy.run_path(str(CHECKER))
    proof = _collection_proof(checker)
    proof_path = tmp_path / "passing-proof.json"
    proof_path.write_text(json.dumps(proof, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    assert checker["_read_collection_proof"](proof_path, "utils") == proof["nodeids"]


def test_controlled_collector_writes_the_verifiable_scope_bound_proof(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    checker = runpy.run_path(str(CHECKER))
    nodeid = (
        "tests/compat/alphalens/test_forward_returns.py::test_forward_returns["
        "tests/test_utils.py::UtilsTestCase::test_compute_forward_returns#00]"
    )
    calls: list[tuple[list[str], dict[str, object]]] = []

    class Completed:
        returncode = 0
        stdout = f"{nodeid}\n1 test collected\n"
        stderr = ""

    def fake_run(command: list[str], **kwargs: object) -> Completed:
        calls.append((command, kwargs))
        return Completed()

    monkeypatch.setattr(checker["_write_collection_proof"].__globals__["subprocess"], "run", fake_run)
    proof_path = tmp_path / "controlled-proof.json"
    checker["_write_collection_proof"](proof_path, "utils")

    proof = _load(proof_path)
    assert proof["command_identity"] == "fincore.alphalens-upstream-collect-v1"
    assert proof["scope"] == "utils"
    assert proof["target_paths"] == sorted(checker["GROUP_PATHS"]["utils"])
    assert proof["exitstatus"] == 0
    assert proof["nodeids"] == [nodeid]
    assert proof["collection_errors"] == []
    assert calls == [
        (
            proof["command"],
            {"cwd": ROOT, "capture_output": True, "text": True, "check": False},
        )
    ]


def test_checker_rejects_plain_nodeid_transcripts_and_accepts_structured_proof_cli() -> None:
    checker = runpy.run_path(str(CHECKER))
    with pytest.raises(SystemExit) as plain_nodeids:
        checker["parse_args"](
            [
                "--inventory",
                str(INVENTORY_PATH),
                "--migration",
                str(MIGRATION_PATH),
                "--nodeids",
                "legacy-nodeids.txt",
                "--results",
                "results.json",
            ]
        )
    assert plain_nodeids.value.code == 2

    arguments = checker["parse_args"](
        [
            "--inventory",
            str(INVENTORY_PATH),
            "--migration",
            str(MIGRATION_PATH),
            "--collection-proof",
            "collection-proof.json",
            "--results",
            "results.json",
        ]
    )
    assert arguments.collection_proof == Path("collection-proof.json")


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
        ("rerunfailures", "1"),
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


@pytest.mark.parametrize(
    "imports",
    [
        "from tests import test_utils",
        "from tests.test_utils import make_factor_data",
        "import tests.test_utils",
        "from test_utils import make_factor_data",
        "import test_utils",
        "from tests.test_performance import PerformanceTestCase",
        "import tests.test_tears",
        "from test_performance import PerformanceTestCase",
        "import test_tears",
    ],
)
def test_target_ast_rejects_exact_upstream_source_test_imports(tmp_path: Path, imports: str) -> None:
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
        """
        + imports,
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


@pytest.mark.parametrize(
    "expression",
    [
        "builtins.__import__('alphalens')",
        "builtins.__import__('tests.test_utils')",
        "__builtins__.__import__('alphalens')",
        "getattr(builtins, '__import__')('tests.test_utils')",
        "builtins.getattr(builtins, '__import__')('alphalens')",
        "load_attribute(builtins, '__import__')('alphalens')",
        "builtins.__dict__['__import__']('alphalens')",
        "getattr(builtins, '__' + 'import__')('tests.test_utils')",
        "load_from_dict('alphalens')",
    ],
)
def test_target_ast_rejects_builtins_and_getattr_dynamic_imports(tmp_path: Path, expression: str) -> None:
    checker, selected_inventory, selected_map, marker_id, target_path, function_name = _single_target_context(
        tmp_path, "tests/test_utils.py"
    )
    _write_marked_target(
        target_path,
        function_name,
        marker_id,
        "assert source_case_id == " + repr(marker_id),
        imports=f"""
        import builtins
        import pytest

        load_attribute = builtins.getattr
        load_from_dict = builtins.__dict__["__import__"]
        loaded_upstream = {expression}
        """,
    )
    with pytest.raises(checker["MigrationAuditError"], match="dynamically imports upstream source"):
        checker["_validate_target_ast"](selected_inventory, selected_map)


@pytest.mark.parametrize(
    ("imports", "message"),
    [
        (
            """
            import builtins
            import pytest

            b = builtins
            loaded_upstream = b.__import__('alphalens')
            """,
            "dynamically imports upstream source",
        ),
        (
            """
            import importlib
            import pytest

            il = importlib
            loaded_upstream = il.import_module('alphalens')
            """,
            "dynamically imports upstream source",
        ),
        (
            """
            import runpy
            from pathlib import Path

            import pytest

            r = runpy
            upstream_source = Path('/Users/example/new_projects') / 'alphalens/tests/test_utils.py'
            r.run_path(upstream_source)
            """,
            "upstream/source path",
        ),
    ],
)
def test_target_ast_rejects_assigned_dangerous_module_aliases(tmp_path: Path, imports: str, message: str) -> None:
    checker, selected_inventory, selected_map, marker_id, target_path, function_name = _single_target_context(
        tmp_path, "tests/test_utils.py"
    )
    _write_marked_target(
        target_path,
        function_name,
        marker_id,
        "assert source_case_id == " + repr(marker_id),
        imports=imports,
    )
    with pytest.raises(checker["MigrationAuditError"], match=message):
        checker["_validate_target_ast"](selected_inventory, selected_map)


def test_target_ast_rejects_imported_os_path_join_for_upstream_execution(tmp_path: Path) -> None:
    checker, selected_inventory, selected_map, marker_id, target_path, function_name = _single_target_context(
        tmp_path, "tests/test_utils.py"
    )
    _write_marked_target(
        target_path,
        function_name,
        marker_id,
        "assert source_case_id == " + repr(marker_id),
        imports="""
        from os.path import join
        import runpy

        import pytest

        upstream_source = join('/Users/example/new_projects', 'alphalens/tests/test_utils.py')
        runpy.run_path(upstream_source)
        """,
    )
    with pytest.raises(checker["MigrationAuditError"], match="upstream/source path"):
        checker["_validate_target_ast"](selected_inventory, selected_map)


def test_target_ast_allows_safe_module_alias_and_imported_join(tmp_path: Path) -> None:
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
        from os.path import join
        from pathlib import Path
        import runpy

        import pytest

        il = importlib
        r = runpy
        stdlib_module = il.import_module('json')
        local_path = Path(join('tests', 'local_target.py'))
        r.run_path(local_path)
        """,
    )
    checker["_validate_target_ast"](selected_inventory, selected_map)


@pytest.mark.parametrize(
    ("imports", "expression"),
    [
        (
            "import importlib\nimport pytest",
            "importlib.import_module(name='alphalens')",
        ),
        (
            "from importlib import import_module as load_module\nimport pytest",
            "load_module(name='alphalens')",
        ),
        (
            "import importlib\nimport pytest",
            "importlib.import_module(name='.test_utils', package='tests')",
        ),
        (
            "import importlib\nimport pytest",
            "importlib.import_module('.test_utils', 'tests')",
        ),
        (
            "import runpy\nimport pytest",
            "runpy.run_module(mod_name='alphalens')",
        ),
        (
            "import builtins\nimport pytest",
            "builtins.__import__(name='alphalens')",
        ),
    ],
)
def test_target_ast_rejects_keyworded_upstream_module_execution(tmp_path: Path, imports: str, expression: str) -> None:
    checker, selected_inventory, selected_map, marker_id, target_path, function_name = _single_target_context(
        tmp_path, "tests/test_utils.py"
    )
    _write_marked_target(
        target_path,
        function_name,
        marker_id,
        "assert source_case_id == " + repr(marker_id),
        imports=imports + "\n\n" + f"loaded_upstream = {expression}",
    )
    with pytest.raises(checker["MigrationAuditError"], match="dynamically imports upstream source"):
        checker["_validate_target_ast"](selected_inventory, selected_map)


def test_target_ast_allows_keyworded_stdlib_import(tmp_path: Path) -> None:
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
        import pytest

        stdlib_module = importlib.import_module(name="json")
        """,
    )
    checker["_validate_target_ast"](selected_inventory, selected_map)


@pytest.mark.parametrize(
    ("imports", "executor"),
    [
        (
            "import runpy\nfrom pathlib import Path\nimport pytest",
            "runpy.run_path(path_name=upstream_source)",
        ),
        (
            "from pathlib import Path\nfrom runpy import run_path as execute_path\nimport pytest",
            "execute_path(path_name=upstream_source)",
        ),
        (
            "import builtins\nfrom pathlib import Path\nimport pytest",
            "builtins.exec(source=upstream_source.read_text())",
        ),
    ],
)
def test_target_ast_rejects_keyworded_upstream_path_execution(tmp_path: Path, imports: str, executor: str) -> None:
    checker, selected_inventory, selected_map, marker_id, target_path, function_name = _single_target_context(
        tmp_path, "tests/test_utils.py"
    )
    _write_marked_target(
        target_path,
        function_name,
        marker_id,
        "assert source_case_id == " + repr(marker_id),
        imports=(
            imports
            + "\n\nupstream_source = Path('/Users/example/new_projects') / 'alphalens/tests/test_utils.py'"
            + f"\n{executor}"
        ),
    )
    with pytest.raises(checker["MigrationAuditError"], match="upstream/source path"):
        checker["_validate_target_ast"](selected_inventory, selected_map)


@pytest.mark.parametrize(
    "executor",
    [
        "runpy.run_path(upstream_source)",
        "getattr(runpy, 'run_path')(upstream_source)",
        "exec(upstream_source.read_text())",
        "getattr(builtins, 'exec')(upstream_source.read_text())",
        "builtins.getattr(builtins, 'exec')(upstream_source.read_text())",
    ],
)
def test_target_ast_rejects_static_upstream_path_assembly_when_executed(tmp_path: Path, executor: str) -> None:
    checker, selected_inventory, selected_map, marker_id, target_path, function_name = _single_target_context(
        tmp_path, "tests/test_utils.py"
    )
    _write_marked_target(
        target_path,
        function_name,
        marker_id,
        "assert source_case_id == " + repr(marker_id),
        imports=f"""
        import builtins
        import runpy
        from pathlib import Path

        import pytest

        upstream_source = Path('/Users/example/new_projects') / 'alphalens/tests/test_utils.py'
        {executor}
        """,
    )
    with pytest.raises(checker["MigrationAuditError"], match="upstream/source path"):
        checker["_validate_target_ast"](selected_inventory, selected_map)


def test_target_ast_rejects_static_upstream_string_join_when_executed(tmp_path: Path) -> None:
    checker, selected_inventory, selected_map, marker_id, target_path, function_name = _single_target_context(
        tmp_path, "tests/test_utils.py"
    )
    _write_marked_target(
        target_path,
        function_name,
        marker_id,
        "assert source_case_id == " + repr(marker_id),
        imports="""
        import runpy

        import pytest

        upstream_root = '/Users/example/new_projects'
        upstream_source = upstream_root + '/alphalens/tests/test_utils.py'
        runpy.run_path(str(upstream_source))
        """,
    )
    with pytest.raises(checker["MigrationAuditError"], match="upstream/source path"):
        checker["_validate_target_ast"](selected_inventory, selected_map)


@pytest.mark.parametrize(
    "path_expression",
    ["upstream_source.resolve()", "upstream_source.resolve(strict=False)", "upstream_source.absolute()"],
)
def test_target_ast_rejects_static_upstream_path_after_static_normalization(
    tmp_path: Path, path_expression: str
) -> None:
    checker, selected_inventory, selected_map, marker_id, target_path, function_name = _single_target_context(
        tmp_path, "tests/test_utils.py"
    )
    _write_marked_target(
        target_path,
        function_name,
        marker_id,
        "assert source_case_id == " + repr(marker_id),
        imports=f"""
        import runpy
        from pathlib import Path

        import pytest

        upstream_source = Path('/Users/example/new_projects') / 'alphalens/tests/test_utils.py'
        runpy.run_path({path_expression})
        """,
    )
    with pytest.raises(checker["MigrationAuditError"], match="upstream/source path"):
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
        from tests.test_factor_analysis import local_helpers as factor_helpers

        import pytest
        """,
    )
    checker["_validate_target_ast"](selected_inventory, selected_map)


@pytest.mark.parametrize(
    "body",
    [
        """
        if False:
            assert source_case_id == "tests/test_utils.py::UtilsTestCase::test_compute_forward_returns#00"
        pass
        """,
        """
        return
        assert source_case_id == "tests/test_utils.py::UtilsTestCase::test_compute_forward_returns#00"
        """,
        """
        def helper():
            assert source_case_id == "tests/test_utils.py::UtilsTestCase::test_compute_forward_returns#00"
        pass
        """,
        """
        class Helper:
            def check(self):
                assert source_case_id == "tests/test_utils.py::UtilsTestCase::test_compute_forward_returns#00"
        pass
        """,
        """
        check = lambda: np.testing.assert_array_equal([1], [1])
        pass
        """,
        """
        [np.testing.assert_array_equal([1], [1]) for _ in ()]
        pass
        """,
        """
        (np.testing.assert_array_equal([1], [1]) for _ in (1,))
        pass
        """,
    ],
)
def test_ordinary_target_ast_rejects_assertions_in_bounded_unreachable_regions(tmp_path: Path, body: str) -> None:
    checker, selected_inventory, selected_map, marker_id, target_path, function_name = _single_target_context(
        tmp_path, "tests/test_utils.py"
    )
    _write_marked_target(
        target_path,
        function_name,
        marker_id,
        body,
        imports="import numpy as np\nimport pytest",
    )
    with pytest.raises(checker["MigrationAuditError"], match="no reachable assertion"):
        checker["_validate_target_ast"](selected_inventory, selected_map)


@pytest.mark.parametrize(
    "body",
    [
        "assert source_case_id == 'tests/test_utils.py::UtilsTestCase::test_compute_forward_returns#00'",
        "pd.testing.assert_series_equal(actual, expected)",
        "np.testing.assert_array_equal(actual, expected)",
    ],
)
def test_ordinary_target_ast_accepts_reachable_assertion_shapes(tmp_path: Path, body: str) -> None:
    checker, selected_inventory, selected_map, marker_id, target_path, function_name = _single_target_context(
        tmp_path, "tests/test_utils.py"
    )
    _write_marked_target(
        target_path,
        function_name,
        marker_id,
        body,
        imports="import numpy as np\nimport pandas as pd\nimport pytest",
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


def test_c4_target_ast_rejects_evidence_hidden_in_a_statically_false_branch(tmp_path: Path) -> None:
    checker, selected_inventory, selected_map, marker_id, target_path, function_name = _single_target_context(
        tmp_path, "tests/test_tears.py"
    )
    _write_marked_target(
        target_path,
        function_name,
        marker_id,
        """
        if False:
            figure = build_figure()
            assert_figure_axes(figure)
            figure.show()
            close_figure(figure)
            assert_artifact_ownership({"primary": figure})
        assert True
        """,
    )
    with pytest.raises(checker["MigrationAuditError"], match="C4"):
        checker["_validate_target_ast"](selected_inventory, selected_map)


@pytest.mark.parametrize("short_circuit", ["False and", "True or"])
def test_c4_target_ast_rejects_evidence_hidden_in_short_circuited_boolop(tmp_path: Path, short_circuit: str) -> None:
    checker, selected_inventory, selected_map, marker_id, target_path, function_name = _single_target_context(
        tmp_path, "tests/test_tears.py"
    )
    _write_marked_target(
        target_path,
        function_name,
        marker_id,
        f"""
        {short_circuit} (
            assert_figure_axes(figure),
            figure.show(),
            close_figure(figure),
            assert_artifact_ownership({{"primary": figure}}),
        )
        assert True
        """,
    )
    with pytest.raises(checker["MigrationAuditError"], match="C4"):
        checker["_validate_target_ast"](selected_inventory, selected_map)


@pytest.mark.parametrize(
    "expression",
    [
        """
        [
            (
                assert_figure_axes(figure),
                figure.show(),
                close_figure(figure),
                assert_artifact_ownership({"primary": figure}),
            )
            for figure in ()
        ]
        """,
        """
        (
            (
                assert_figure_axes(figure),
                figure.show(),
                close_figure(figure),
                assert_artifact_ownership({"primary": figure}),
            )
            for figure in (1,)
        )
        """,
    ],
)
def test_c4_target_ast_rejects_evidence_in_comprehension_or_unconsumed_generator(
    tmp_path: Path, expression: str
) -> None:
    checker, selected_inventory, selected_map, marker_id, target_path, function_name = _single_target_context(
        tmp_path, "tests/test_tears.py"
    )
    _write_marked_target(
        target_path,
        function_name,
        marker_id,
        textwrap.dedent(expression).strip() + "\nassert True",
    )
    with pytest.raises(checker["MigrationAuditError"], match="C4"):
        checker["_validate_target_ast"](selected_inventory, selected_map)


@pytest.mark.parametrize("literal_range", ["range(0)", "range(1, 1)", "range(0, 4, -1)"])
def test_c4_target_ast_rejects_evidence_hidden_in_empty_literal_range(tmp_path: Path, literal_range: str) -> None:
    checker, selected_inventory, selected_map, marker_id, target_path, function_name = _single_target_context(
        tmp_path, "tests/test_tears.py"
    )
    _write_marked_target(
        target_path,
        function_name,
        marker_id,
        f"""
        for figure in {literal_range}:
            assert_figure_axes(figure)
            figure.show()
            close_figure(figure)
            assert_artifact_ownership({{"primary": figure}})
        assert True
        """,
    )
    with pytest.raises(checker["MigrationAuditError"], match="C4"):
        checker["_validate_target_ast"](selected_inventory, selected_map)


def test_c4_target_ast_rejects_evidence_hidden_in_an_empty_literal_for_loop(tmp_path: Path) -> None:
    checker, selected_inventory, selected_map, marker_id, target_path, function_name = _single_target_context(
        tmp_path, "tests/test_tears.py"
    )
    _write_marked_target(
        target_path,
        function_name,
        marker_id,
        """
        for _ in ():
            figure = build_figure()
            assert_figure_axes(figure)
            figure.show()
            close_figure(figure)
            assert_artifact_ownership({"primary": figure})
        assert True
        """,
    )
    with pytest.raises(checker["MigrationAuditError"], match="C4"):
        checker["_validate_target_ast"](selected_inventory, selected_map)


def test_c4_target_ast_rejects_evidence_after_a_nonempty_literal_for_return(tmp_path: Path) -> None:
    checker, selected_inventory, selected_map, marker_id, target_path, function_name = _single_target_context(
        tmp_path, "tests/test_tears.py"
    )
    _write_marked_target(
        target_path,
        function_name,
        marker_id,
        """
        for _ in (1,):
            return
        figure = build_figure()
        assert_figure_axes(figure)
        figure.show()
        close_figure(figure)
        assert_artifact_ownership({"primary": figure})
        assert True
        """,
    )
    with pytest.raises(checker["MigrationAuditError"], match="C4"):
        checker["_validate_target_ast"](selected_inventory, selected_map)


def test_c4_target_ast_rejects_evidence_after_a_literal_terminating_branch(tmp_path: Path) -> None:
    checker, selected_inventory, selected_map, marker_id, target_path, function_name = _single_target_context(
        tmp_path, "tests/test_tears.py"
    )
    _write_marked_target(
        target_path,
        function_name,
        marker_id,
        """
        if True:
            return
        figure = build_figure()
        assert_figure_axes(figure)
        figure.show()
        close_figure(figure)
        assert_artifact_ownership({"primary": figure})
        assert True
        """,
    )
    with pytest.raises(checker["MigrationAuditError"], match="C4"):
        checker["_validate_target_ast"](selected_inventory, selected_map)


def test_c4_target_ast_rejects_evidence_after_a_literal_terminating_loop(tmp_path: Path) -> None:
    checker, selected_inventory, selected_map, marker_id, target_path, function_name = _single_target_context(
        tmp_path, "tests/test_tears.py"
    )
    _write_marked_target(
        target_path,
        function_name,
        marker_id,
        """
        while True:
            return
        figure = build_figure()
        assert_figure_axes(figure)
        figure.show()
        close_figure(figure)
        assert_artifact_ownership({"primary": figure})
        assert True
        """,
    )
    with pytest.raises(checker["MigrationAuditError"], match="C4"):
        checker["_validate_target_ast"](selected_inventory, selected_map)


def test_c4_target_ast_rejects_reachable_signals_bound_to_different_figures(tmp_path: Path) -> None:
    checker, selected_inventory, selected_map, marker_id, target_path, function_name = _single_target_context(
        tmp_path, "tests/test_tears.py"
    )
    _write_marked_target(
        target_path,
        function_name,
        marker_id,
        """
        figure = build_figure()
        other_figure = build_figure()
        assert_figure_axes(figure)
        figure.show()
        close_figure(other_figure)
        assert_artifact_ownership({"primary": figure})
        """,
    )
    with pytest.raises(checker["MigrationAuditError"], match="not bound"):
        checker["_validate_target_ast"](selected_inventory, selected_map)


def test_c4_target_ast_binds_pyplot_close_argument_to_the_same_figure(tmp_path: Path) -> None:
    checker, selected_inventory, selected_map, marker_id, target_path, function_name = _single_target_context(
        tmp_path, "tests/test_tears.py"
    )
    _write_marked_target(
        target_path,
        function_name,
        marker_id,
        """
        figure = build_figure()
        other_figure = build_figure()
        assert_figure_axes(figure)
        figure.show()
        plt.close(other_figure)
        assert_artifact_ownership({"primary": figure})
        """,
    )
    with pytest.raises(checker["MigrationAuditError"], match="not bound"):
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
