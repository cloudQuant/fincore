"""Fail-closed contract tests for the 0042-R2 capability ledger."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).parents[2] / "scripts" / "capture_capability_baseline.py"


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


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
        "inventory": source_root / "inventory.json",
        "module_disposition": source_root / "module-disposition.json",
        "test_disposition": source_root / "test-disposition.json",
        "ledger": source_root / "ledger.json",
        "fixture_dir": fixture_dir,
    }
    _write_json(paths["inventory"], {"entries": [{"item_id": "metrics.annual_return", "disposition": "required"}]})
    _write_json(paths["module_disposition"], {"entries": [{"module": "fincore.metrics", "disposition": "keep"}]})
    _write_json(
        paths["test_disposition"],
        {"entries": [{"nodeid": "tests/legacy/test_metrics.py::test_annual_return", "disposition": "migrate"}]},
    )
    _write_json(
        paths["ledger"],
        {
            "schema_version": 1,
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
            str(SCRIPT),
            "--inventory",
            str(paths["inventory"]),
            "--module-disposition",
            str(paths["module_disposition"]),
            "--test-disposition",
            str(paths["test_disposition"]),
            "--ledger",
            str(paths["ledger"]),
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
