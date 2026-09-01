"""Fail-closed contract tests for the 0042-R2 capability ledger."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from tests.support.complete_surface_inputs import write_minimal_complete_surface_inputs
from tests.support.frozen_capture_tooling import create_frozen_capture_tooling_root
from tests.support.repository_surface_inputs import write_minimal_repository_surface_inputs

SCRIPT = (
    Path(os.environ.get("FINCORE_0042R2_SOURCE_ROOT", Path(__file__).parents[2])).resolve()
    / "scripts"
    / "capture_capability_baseline.py"
)


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def _write_text_with_windows_newlines(
    path: Path,
    data: str,
    encoding: str | None = None,
    errors: str | None = None,
    newline: str | None = None,
) -> int:
    del newline
    return path.write_bytes(data.replace("\n", "\r\n").encode(encoding or "utf-8", errors or "strict"))


def _commit_source(source_root: Path) -> None:
    subprocess.run(["git", "init", "-q"], cwd=source_root, check=True)
    subprocess.run(["git", "add", "."], cwd=source_root, check=True)
    subprocess.run(
        [
            "git",
            "-c",
            "user.name=0042-R2 test",
            "-c",
            "user.email=0042-r2@example.invalid",
            "commit",
            "-qm",
            "baseline inputs",
        ],
        cwd=source_root,
        check=True,
    )


def _minimal_inputs(tmp_path: Path) -> tuple[Path, dict[str, Path]]:
    source_root = tmp_path / "source"
    source_root.mkdir()
    fixture_dir = source_root / "goldens"
    fixture_dir.mkdir()
    (fixture_dir / "annual-return.json").write_text('{"value": 0.1}\n', encoding="utf-8")

    paths = {
        "legacy_discovery": source_root / "legacy-discovery.json",
        "surface_union": source_root / "surface-union.json",
        "inventory": source_root / "inventory.json",
        "module_disposition": source_root / "module-disposition.json",
        "test_disposition": source_root / "test-disposition.json",
        "ledger": source_root / "ledger.json",
        "fixture_dir": fixture_dir,
        "tooling_root": create_frozen_capture_tooling_root(tmp_path / "frozen-tooling", SCRIPT.parent),
    }
    paths.update(write_minimal_repository_surface_inputs(source_root))
    write_minimal_complete_surface_inputs(paths)
    _write_json(paths["module_disposition"], {"entries": [{"module": "fincore.metrics", "disposition": "keep"}]})
    _write_json(
        paths["test_disposition"],
        {"entries": [{"nodeid": "tests/legacy/test_metrics.py::test_annual_return", "disposition": "migrate"}]},
    )
    _write_json(
        paths["ledger"],
        {
            "schema_version": 1,
            "decision_status": "complete",
            "entries": [
                {
                    "capability_id": "metrics.annual_return",
                    "owner": "metrics",
                    "disposition": "required",
                    "target_operation_id": "metrics.annual_return",
                    "source_nodeids": ["tests/legacy/test_metrics.py::test_annual_return"],
                    "wheel_nodeids": ["tests/parity/test_metrics.py::test_annual_return"],
                    "scenarios": [
                        {
                            "scenario_id": "ordinary_returns",
                            "authority": {
                                "kind": "pinned_upstream_oracle",
                                "source_project": "empyrical",
                                "reference": "empyrical.annual_return",
                                "version": "0.5.5",
                            },
                            "golden_path": "annual-return.json",
                        }
                    ],
                }
            ],
        },
    )
    _commit_source(source_root)
    return source_root, paths


def _capture(source_root: Path, paths: dict[str, Path], output: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable,
            "-I",
            str(paths["tooling_root"] / "scripts" / "capture_capability_baseline.py"),
            "--legacy-discovery",
            str(paths["legacy_discovery"]),
            "--surface-union",
            str(paths["surface_union"]),
            "--inventory",
            str(paths["inventory"]),
            "--module-disposition",
            str(paths["module_disposition"]),
            "--test-disposition",
            str(paths["test_disposition"]),
            "--ledger",
            str(paths["ledger"]),
            "--repository-surface-facts",
            str(paths["repository_surface_facts"]),
            "--repository-surface-disposition",
            str(paths["repository_surface_disposition"]),
            "--tooling-root",
            str(paths["tooling_root"]),
            "--fixture-dir",
            str(paths["fixture_dir"]),
            "--output",
            str(output),
            "--deny-network",
        ],
        cwd=source_root,
        capture_output=True,
        text=True,
        check=False,
    )


def test_complete_surface_inputs_bind_initial_git_blobs_after_windows_line_ending_normalization(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source_root = tmp_path / "source"
    source_root.mkdir()
    paths = {
        "legacy_discovery": source_root / "legacy-discovery.json",
        "surface_union": source_root / "surface-union.json",
        "inventory": source_root / "inventory.json",
    }

    monkeypatch.setattr(Path, "write_text", _write_text_with_windows_newlines)
    write_minimal_complete_surface_inputs(paths)

    subprocess.run(["git", "init", "-q"], cwd=source_root, check=True)
    subprocess.run(["git", "config", "core.autocrlf", "true"], cwd=source_root, check=True)
    subprocess.run(["git", "add", "."], cwd=source_root, check=True)
    subprocess.run(
        [
            "git",
            "-c",
            "user.name=0042-R2 test",
            "-c",
            "user.email=0042-r2@example.invalid",
            "commit",
            "-qm",
            "baseline inputs",
        ],
        cwd=source_root,
        check=True,
    )

    inventory = json.loads(paths["inventory"].read_text(encoding="utf-8"))
    source_artifacts = {artifact["source_id"]: artifact for artifact in inventory["source_artifacts"]}
    for source_id, path_key in (("legacy_surface_discovery", "legacy_discovery"), ("surface_union", "surface_union")):
        path = paths[path_key]
        result = subprocess.run(
            ["git", "cat-file", "blob", f"HEAD:{path.name}"],
            cwd=source_root,
            capture_output=True,
            check=True,
        )

        assert b"\r\n" not in result.stdout
        assert source_artifacts[source_id]["sha256"] == hashlib.sha256(result.stdout).hexdigest()


def test_repository_surface_inputs_bind_initial_git_blobs_after_windows_line_ending_normalization(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source_root = tmp_path / "source"
    source_root.mkdir()

    monkeypatch.setattr(Path, "write_text", _write_text_with_windows_newlines)
    paths = write_minimal_repository_surface_inputs(source_root)

    subprocess.run(["git", "init", "-q"], cwd=source_root, check=True)
    subprocess.run(["git", "config", "core.autocrlf", "true"], cwd=source_root, check=True)
    subprocess.run(["git", "add", "."], cwd=source_root, check=True)
    subprocess.run(
        [
            "git",
            "-c",
            "user.name=0042-R2 test",
            "-c",
            "user.email=0042-r2@example.invalid",
            "commit",
            "-qm",
            "baseline inputs",
        ],
        cwd=source_root,
        check=True,
    )

    def git_blob(relative_path: str) -> bytes:
        return subprocess.run(
            ["git", "cat-file", "blob", f"HEAD:{relative_path}"],
            cwd=source_root,
            capture_output=True,
            check=True,
        ).stdout

    facts_blob = git_blob(paths["repository_surface_facts"].name)
    disposition_blob = git_blob(paths["repository_surface_disposition"].name)
    source_blob = git_blob("scripts/check_quality.py")
    disposition = json.loads(disposition_blob)
    facts = json.loads(facts_blob)

    assert b"\r\n" not in facts_blob
    assert b"\r\n" not in disposition_blob
    assert b"\r\n" not in source_blob
    assert disposition["source_facts"]["sha256"] == hashlib.sha256(facts_blob).hexdigest()
    assert facts["records"][0]["blob_sha256"] == hashlib.sha256(source_blob).hexdigest()


def test_minimal_ledger_is_accepted_after_windows_git_line_ending_normalization(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(Path, "write_text", _write_text_with_windows_newlines)
    monkeypatch.setenv("GIT_CONFIG_COUNT", "1")
    monkeypatch.setenv("GIT_CONFIG_KEY_0", "core.autocrlf")
    monkeypatch.setenv("GIT_CONFIG_VALUE_0", "true")
    source_root, paths = _minimal_inputs(tmp_path)

    result = _capture(source_root, paths, tmp_path / "capture.json")

    assert result.returncode == 0, result.stderr


def test_valid_minimal_ledger_is_accepted(tmp_path: Path) -> None:
    source_root, paths = _minimal_inputs(tmp_path)

    result = _capture(source_root, paths, tmp_path / "capture.json")

    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize(
    ("mutate", "expected_error"),
    [
        (
            lambda entry: entry.update(capability_id="metrics.annual_return") or entry,
            "duplicate capability_id",
        ),
        (lambda entry: entry.pop("owner") or entry, "owner"),
        (lambda entry: entry.pop("disposition") or entry, "disposition"),
        (lambda entry: entry.update(disposition="REQUIRED") or entry, "ledger disposition"),
        (lambda entry: entry.update(disposition=" required ") or entry, "ledger disposition"),
        (lambda entry: entry.update(disposition="unknown") or entry, "ledger disposition"),
        (
            lambda entry: (
                entry.update(
                    scenarios=[
                        {
                            "scenario_id": "ordinary_returns",
                            "authority": {
                                "kind": "candidate_fincore_result",
                                "reference": "candidate value",
                                "version": "0.5.0",
                            },
                            "golden_path": "annual-return.json",
                        }
                    ]
                )
                or entry
            ),
            "independent authority kind",
        ),
        (
            lambda entry: (
                entry.update(
                    scenarios=[
                        {
                            "scenario_id": "ordinary_returns",
                            "authority": {
                                "kind": "unrecognized_oracle",
                                "reference": "unknown implementation",
                                "version": "1.0",
                            },
                            "golden_path": "annual-return.json",
                        }
                    ]
                )
                or entry
            ),
            "independent authority kind",
        ),
        (
            lambda entry: (
                entry.update(
                    scenarios=[
                        {
                            "scenario_id": "ordinary_returns",
                            "authority": {
                                "kind": "pinned_upstream_oracle",
                                "source_project": "empyrical",
                                "reference": "empyrical.annual_return",
                            },
                            "golden_path": "annual-return.json",
                        }
                    ]
                )
                or entry
            ),
            "version or artifact_digest",
        ),
    ],
    ids=[
        "duplicate",
        "missing-owner",
        "missing-disposition",
        "uppercase-disposition",
        "whitespace-disposition",
        "unknown-disposition",
        "candidate-oracle",
        "unknown-oracle",
        "untraceable-oracle",
    ],
)
def test_ledger_invalid_entries_are_rejected(
    tmp_path: Path,
    mutate: object,
    expected_error: str,
) -> None:
    source_root, paths = _minimal_inputs(tmp_path)
    ledger = json.loads(paths["ledger"].read_text(encoding="utf-8"))
    entry = ledger["entries"][0]
    assert isinstance(entry, dict)
    if expected_error == "duplicate capability_id":
        ledger["entries"].append(mutate(entry.copy()))
    else:
        mutate(entry)
    _write_json(paths["ledger"], ledger)
    _commit_source(source_root)

    result = _capture(source_root, paths, tmp_path / "capture.json")

    assert result.returncode != 0
    assert expected_error in result.stderr


@pytest.mark.parametrize(
    ("authority", "expected_error"),
    [
        (
            {
                "kind": "pinned_upstream_oracle",
                "reference": "empyrical.annual_return",
                "version": "0.5.5",
            },
            "source_project",
        ),
        (
            {
                "kind": "pinned_upstream_oracle",
                "source_project": "Fincore",
                "reference": "empyrical.annual_return",
                "version": "0.5.5",
            },
            "external project",
        ),
        (
            {
                "kind": "pinned_upstream_oracle",
                "source_project": "candidate",
                "reference": "empyrical.annual_return",
                "version": "0.5.5",
            },
            "external project",
        ),
        (
            {
                "kind": "pinned_upstream_oracle",
                "source_project": "CURRENT",
                "reference": "empyrical.annual_return",
                "version": "0.5.5",
            },
            "external project",
        ),
        (
            {
                "kind": "pinned_upstream_oracle",
                "source_project": "empyrical",
                "reference": "empyrical.annual_return",
            },
            "version or artifact_digest",
        ),
        (
            {
                "kind": "published_standard",
                "reference": "annualized return definition",
            },
            "publication, doi, version, or digest",
        ),
        (
            {
                "kind": "property_invariant",
                "reference": "annual returns scale with compounding",
                "digest": "sha256:fixture",
            },
            "invariant_id",
        ),
    ],
    ids=[
        "missing-project",
        "fincore-project",
        "candidate-project",
        "current-project",
        "missing-artifact-identity",
        "paper-metadata",
        "invariant-id",
    ],
)
def test_required_authorities_require_kind_specific_structured_provenance(
    tmp_path: Path,
    authority: dict[str, str],
    expected_error: str,
) -> None:
    source_root, paths = _minimal_inputs(tmp_path)
    ledger = json.loads(paths["ledger"].read_text(encoding="utf-8"))
    entry = ledger["entries"][0]
    assert isinstance(entry, dict)
    entry["scenarios"][0]["authority"] = authority
    _write_json(paths["ledger"], ledger)
    _commit_source(source_root)

    result = _capture(source_root, paths, tmp_path / "capture.json")

    assert result.returncode != 0
    assert expected_error in result.stderr


@pytest.mark.parametrize("document_name", ["inventory", "module_disposition", "test_disposition"])
def test_executable_disposition_records_must_not_be_blank(tmp_path: Path, document_name: str) -> None:
    source_root, paths = _minimal_inputs(tmp_path)
    _write_json(paths[document_name], {"entries": [{"item_id": "unmapped", "disposition": ""}]})
    _commit_source(source_root)

    result = _capture(source_root, paths, tmp_path / "capture.json")

    assert result.returncode != 0
    assert document_name.replace("_", " ") in result.stderr
    assert "disposition" in result.stderr


@pytest.mark.parametrize("document_name", ["inventory", "module_disposition", "test_disposition"])
def test_disposition_documents_must_not_be_empty(tmp_path: Path, document_name: str) -> None:
    source_root, paths = _minimal_inputs(tmp_path)
    _write_json(paths[document_name], {"entries": []})
    _commit_source(source_root)

    result = _capture(source_root, paths, tmp_path / "capture.json")

    assert result.returncode != 0
    assert "entries must be non-empty" in result.stderr


@pytest.mark.parametrize(
    ("marker", "value", "expected_error"),
    [
        ("decision_status", "scoped", "preparatory non-D0 artifact"),
        ("not_for_d0", True, "not_for_d0"),
    ],
    ids=["scoped-decision-status", "explicit-not-for-d0"],
)
def test_scoped_ledger_cannot_enter_baseline_capture(
    tmp_path: Path, marker: str, value: object, expected_error: str
) -> None:
    source_root, paths = _minimal_inputs(tmp_path)
    ledger = json.loads(paths["ledger"].read_text(encoding="utf-8"))
    ledger[marker] = value
    _write_json(paths["ledger"], ledger)
    _commit_source(source_root)

    result = _capture(source_root, paths, tmp_path / "capture.json")

    assert result.returncode != 0
    assert expected_error in result.stderr


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("target_operation_id", None),
        ("source_nodeids", []),
        ("wheel_nodeids", []),
        ("scenarios", []),
    ],
)
def test_required_ledger_entry_requires_target_nodes_and_scenarios(
    tmp_path: Path,
    field: str,
    value: object,
) -> None:
    source_root, paths = _minimal_inputs(tmp_path)
    ledger = json.loads(paths["ledger"].read_text(encoding="utf-8"))
    entry = ledger["entries"][0]
    assert isinstance(entry, dict)
    if value is None:
        entry.pop(field)
    else:
        entry[field] = value
    _write_json(paths["ledger"], ledger)
    _commit_source(source_root)

    result = _capture(source_root, paths, tmp_path / "capture.json")

    assert result.returncode != 0
    assert field in result.stderr
