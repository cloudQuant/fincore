"""Integration contracts for the 0042-R2 baseline capture command."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from tests.support.frozen_capture_tooling import create_frozen_capture_tooling_root
from tests.support.repository_surface_inputs import write_minimal_repository_surface_inputs

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


def _canonical_sha256(value: object) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _source_provenance() -> dict[str, object]:
    return {"commit": "a" * 40, "tree": "b" * 40, "clean": True}


def _write_minimal_complete_surface_inputs(paths: dict[str, Path]) -> None:
    provenance = _source_provenance()
    legacy_entry = {
        "entry_id": "legacy:metrics.annual_return",
        "source_id": "legacy_manifest",
        "source_kind": "compat_manifest",
        "source_locator": {"artifact_path": "fixtures/legacy.json", "locator": "entries[0]"},
    }
    legacy = {
        "schema_version": 1,
        "artifact_type": "legacy_surface_discovery",
        "discovery_status": "partial",
        "not_for_d0": True,
        "source": provenance,
        "entries": [legacy_entry],
    }
    _write_json(paths["legacy_discovery"], legacy)
    kinds = (
        "public_definition",
        "registry",
        "manifest",
        "documentation",
        "example",
        "benchmark",
        "extra",
        "wheel_content",
    )
    union_entries = [
        {
            "entry_id": f"{kind}:fixtures/{kind}.json",
            "source_kind": kind,
            "source": {
                "artifact_path": f"fixtures/{kind}.json",
                "artifact_sha256": "d" * 64,
                "locator": f"{kind}:{index}",
            },
        }
        for index, kind in enumerate(kinds)
    ]
    union = {
        "schema_version": 1,
        "artifact_type": "surface_union_facts_discovery",
        "discovery_status": "complete",
        "not_for_d0": True,
        "does_not_assert": ["D-TECH", "D0", "installed_wheel_behavior", "legacy_zero"],
        "source_provenance": provenance,
        "wheel": {"filename": "fincore.whl", "sha256": "c" * 64, "member_count": 1},
        "entry_count": len(union_entries),
        "kind_counts": dict.fromkeys(kinds, 1),
        "entries": union_entries,
        "canonical_entries_sha256": _canonical_sha256(union_entries),
    }
    _write_json(paths["surface_union"], union)
    supporting_entries = [
        {
            "inventory_entry_id": f"surface_union:{raw_entry['entry_id']}",
            "source_id": "surface_union",
            "raw_entry_id": raw_entry["entry_id"],
            "raw_entry_sha256": _canonical_sha256(raw_entry),
            "owner": "platform",
            "disposition": "supporting",
            "capability_ids": ["platform.surface"],
            "target_operation_id": "platform.surface",
            "scenario_ids": ["platform.surface"],
            "source_nodeids": [],
            "wheel_nodeids": [],
            "evidence": {"oracle_reference": "reviewed supporting surface record"},
            "completion_gate": "D0",
            "rationale": "The raw union record is supporting evidence for the complete surface decision.",
            "rule_id": "supporting-surface",
        }
        for raw_entry in union_entries
    ]
    inventory = {
        "schema_version": 1,
        "artifact_type": "complete_surface_inventory",
        "scope": "complete_legacy_surface_union",
        "decision_status": "complete",
        "does_not_assert": ["D-TECH", "D0", "installed_wheel_behavior", "legacy_zero"],
        "owners": ["metrics", "platform"],
        "source_artifacts": [
            {
                "source_id": "legacy_surface_discovery",
                "path": paths["legacy_discovery"].name,
                "sha256": _sha256(paths["legacy_discovery"]),
                "entry_count": 1,
                "source_provenance": provenance,
            },
            {
                "source_id": "surface_union",
                "path": paths["surface_union"].name,
                "sha256": _sha256(paths["surface_union"]),
                "entry_count": len(union_entries),
                "source_provenance": provenance,
            },
        ],
        "entries": [
            {
                "inventory_entry_id": "legacy_surface_discovery:legacy:metrics.annual_return",
                "source_id": "legacy_surface_discovery",
                "raw_entry_id": legacy_entry["entry_id"],
                "raw_entry_sha256": _canonical_sha256(legacy_entry),
                "owner": "metrics",
                "disposition": "required",
                "capability_ids": ["metrics.annual_return"],
                "target_operation_id": "metrics.annual_return",
                "scenario_ids": ["ordinary_returns"],
                "source_nodeids": ["tests/legacy/test_metrics.py::test_annual_return"],
                "wheel_nodeids": ["tests/parity/test_metrics.py::test_annual_return"],
                "evidence": {"golden_path": "annual-return.json"},
                "completion_gate": "D-DOMAIN",
                "rationale": "The canonical metrics operation remains a required analytical capability.",
                "rule_id": "unified-operation",
            },
            *supporting_entries,
        ],
    }
    _write_json(paths["inventory"], inventory)


def _minimal_inputs(tmp_path: Path, *, include_candidate_checker_spoof: bool = False) -> tuple[Path, dict[str, Path]]:
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
    _write_minimal_complete_surface_inputs(paths)
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
    if include_candidate_checker_spoof:
        (source_root / "scripts" / "check_0042_r2_complete_surface_inventory.py").write_text(
            "raise RuntimeError('candidate checker must not be imported')\n",
            encoding="utf-8",
        )
        (source_root / "scripts" / "check_0042_r2_repository_surface_disposition.py").write_text(
            "raise RuntimeError('candidate checker must not be imported')\n",
            encoding="utf-8",
        )
    _commit_source(source_root)
    return source_root, paths


def _capture(
    source_root: Path,
    paths: dict[str, Path],
    output: Path,
    *,
    include_deny_network: bool = True,
    isolated: bool = True,
    declared_tooling_root: Path | None = None,
    runner_script: Path | None = None,
) -> subprocess.CompletedProcess[str]:
    command = [sys.executable]
    if isolated:
        command.append("-I")
    command.extend(
        [
            str(runner_script or paths["tooling_root"] / "scripts" / "capture_capability_baseline.py"),
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
            str(declared_tooling_root or paths["tooling_root"]),
            "--fixture-dir",
            str(paths["fixture_dir"]),
            "--output",
            str(output),
        ]
    )
    if include_deny_network:
        command.append("--deny-network")
    return subprocess.run(command, cwd=source_root, capture_output=True, text=True, check=False)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _capture_module(tooling_root: Path):
    spec = importlib.util.spec_from_file_location(
        "capture_capability_baseline",
        tooling_root / "scripts" / "capture_capability_baseline.py",
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    original_dont_write_bytecode = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    try:
        spec.loader.exec_module(module)
    finally:
        sys.dont_write_bytecode = original_dont_write_bytecode
    return module


def test_capture_records_clean_git_provenance_and_input_hashes(tmp_path: Path) -> None:
    source_root, paths = _minimal_inputs(tmp_path)
    output = tmp_path / "capture.json"

    result = _capture(source_root, paths, output)

    assert result.returncode == 0, result.stderr
    artifact = json.loads(output.read_text(encoding="utf-8"))
    expected_commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=source_root, capture_output=True, text=True, check=True
    ).stdout.strip()
    assert artifact["artifact_type"] == "capability_baseline_capture"
    assert artifact["does_not_assert"] == ["D0", "D-TECH"]
    assert artifact["source"]["commit"] == expected_commit
    assert artifact["source"]["clean"] is True
    assert len(artifact["source"]["tree"]) == 40
    assert artifact["tooling"]["source"]["clean"] is True
    assert artifact["tooling"]["capture"]["path"] == "scripts/capture_capability_baseline.py"
    assert artifact["tooling"]["repository_surface_disposition_checker"]["path"] == (
        "scripts/check_0042_r2_repository_surface_disposition.py"
    )
    assert artifact["tooling"]["complete_surface_inventory_checker"]["path"] == (
        "scripts/check_0042_r2_complete_surface_inventory.py"
    )
    assert artifact["tooling"]["capture"]["sha256"] == _sha256(
        paths["tooling_root"] / "scripts" / "capture_capability_baseline.py"
    )
    assert artifact["tooling"]["repository_surface_disposition_checker"]["sha256"] == _sha256(
        paths["tooling_root"] / "scripts" / "check_0042_r2_repository_surface_disposition.py"
    )
    assert artifact["tooling"]["complete_surface_inventory_checker"]["sha256"] == _sha256(
        paths["tooling_root"] / "scripts" / "check_0042_r2_complete_surface_inventory.py"
    )
    for key in (
        "legacy_discovery",
        "surface_union",
        "inventory",
        "module_disposition",
        "test_disposition",
        "ledger",
        "repository_surface_facts",
        "repository_surface_disposition",
    ):
        assert artifact["inputs"][key]["sha256"] == _sha256(paths[key])
    assert artifact["complete_surface_inventory"]["scope"] == "complete_legacy_surface_union"
    assert artifact["complete_surface_inventory"]["validation"]["artifact_type"] == (
        "complete_surface_inventory_validation"
    )
    assert artifact["repository_surface"]["scope"] == "classified_repository_surface_only"
    assert artifact["repository_surface"]["not_for_d0"] is True
    assert artifact["repository_surface"]["facts_sha256"] == _sha256(paths["repository_surface_facts"])
    assert artifact["repository_surface"]["disposition_sha256"] == _sha256(paths["repository_surface_disposition"])
    assert artifact["repository_surface"]["validation"]["not_for_d0"] is True
    assert artifact["fixtures"]["annual-return.json"]["sha256"] == _sha256(paths["fixture_dir"] / "annual-return.json")


def test_invalid_repository_surface_disposition_does_not_overwrite_capture_output(tmp_path: Path) -> None:
    source_root, paths = _minimal_inputs(tmp_path)
    disposition = json.loads(paths["repository_surface_disposition"].read_text(encoding="utf-8"))
    disposition["not_for_d0"] = False
    _write_json(paths["repository_surface_disposition"], disposition)
    _commit_source(source_root)
    output = tmp_path / "capture.json"
    output.write_text('{"previous": "success"}\n', encoding="utf-8")

    result = _capture(source_root, paths, output)

    assert result.returncode != 0
    assert "not_for_d0" in result.stderr
    assert output.read_text(encoding="utf-8") == '{"previous": "success"}\n'


def test_capture_rejects_a_scoped_raw_only_legacy_inventory(tmp_path: Path) -> None:
    source_root, paths = _minimal_inputs(tmp_path)
    _write_json(
        paths["inventory"],
        {
            "schema_version": 1,
            "artifact_type": "legacy_surface_inventory",
            "scope": "raw_legacy_surface_only",
            "decision_status": "scoped",
            "not_for_d0": True,
            "entries": [{"legacy_entry_id": "metric_registry:annual_return", "disposition": "required"}],
        },
    )
    _commit_source(source_root)
    output = tmp_path / "capture.json"
    output.write_text('{"previous": "success"}\n', encoding="utf-8")

    result = _capture(source_root, paths, output)

    assert result.returncode != 0
    assert "raw_legacy_surface_only" in result.stderr
    assert output.read_text(encoding="utf-8") == '{"previous": "success"}\n'


def test_capture_rejects_a_relabelled_legacy_surface_inventory(tmp_path: Path) -> None:
    source_root, paths = _minimal_inputs(tmp_path)
    _write_json(
        paths["inventory"],
        {
            "schema_version": 1,
            "artifact_type": "legacy_surface_inventory",
            "scope": "complete_legacy_surface",
            "decision_status": "complete",
            "not_for_d0": False,
            "entries": [{"legacy_entry_id": "metric_registry:annual_return", "disposition": "required"}],
        },
    )
    _commit_source(source_root)
    output = tmp_path / "capture.json"
    output.write_text('{"previous": "success"}\n', encoding="utf-8")

    result = _capture(source_root, paths, output)

    assert result.returncode != 0
    assert "legacy_surface_inventory" in result.stderr
    assert output.read_text(encoding="utf-8") == '{"previous": "success"}\n'


def test_capture_rejects_any_inventory_explicitly_marked_not_for_d0(tmp_path: Path) -> None:
    source_root, paths = _minimal_inputs(tmp_path)
    inventory = json.loads(paths["inventory"].read_text(encoding="utf-8"))
    inventory["not_for_d0"] = True
    _write_json(paths["inventory"], inventory)
    _commit_source(source_root)
    output = tmp_path / "capture.json"
    output.write_text('{"previous": "success"}\n', encoding="utf-8")

    result = _capture(source_root, paths, output)

    assert result.returncode != 0
    assert "not_for_d0" in result.stderr
    assert output.read_text(encoding="utf-8") == '{"previous": "success"}\n'


def test_capture_rejects_duplicate_json_policy_keys(tmp_path: Path) -> None:
    source_root, paths = _minimal_inputs(tmp_path)
    paths["inventory"].write_bytes(
        b'{"entries": [{"item_id": "metrics.annual_return", "disposition": "required"}], '
        b'"scope": "candidate", "scope": "candidate"}\n'
    )
    _commit_source(source_root)
    output = tmp_path / "capture.json"
    output.write_text('{"previous": "success"}\n', encoding="utf-8")

    result = _capture(source_root, paths, output)

    assert result.returncode != 0
    assert "duplicate JSON key" in result.stderr
    assert output.read_text(encoding="utf-8") == '{"previous": "success"}\n'


def test_capture_uses_static_repository_surface_checker_under_isolated_mode(tmp_path: Path) -> None:
    source_root, paths = _minimal_inputs(tmp_path, include_candidate_checker_spoof=True)
    output = tmp_path / "capture.json"

    result = _capture(source_root, paths, output, isolated=True)

    assert result.returncode == 0, result.stderr
    artifact = json.loads(output.read_text(encoding="utf-8"))
    assert artifact["repository_surface"]["validation"]["artifact_type"] == "repository_surface_disposition_validation"
    assert (
        artifact["complete_surface_inventory"]["validation"]["artifact_type"] == "complete_surface_inventory_validation"
    )


def test_capture_rejects_a_complete_inventory_that_no_longer_binds_its_raw_union(tmp_path: Path) -> None:
    source_root, paths = _minimal_inputs(tmp_path)
    union = json.loads(paths["surface_union"].read_text(encoding="utf-8"))
    union["entries"][0]["source"]["artifact_sha256"] = "e" * 64
    union["canonical_entries_sha256"] = _canonical_sha256(union["entries"])
    _write_json(paths["surface_union"], union)
    _commit_source(source_root)
    output = tmp_path / "capture.json"
    output.write_text('{"previous": "success"}\n', encoding="utf-8")

    result = _capture(source_root, paths, output)

    assert result.returncode != 0
    assert "complete-surface inventory validation failed" in result.stderr
    assert output.read_text(encoding="utf-8") == '{"previous": "success"}\n'


def test_capture_rejects_a_tooling_root_that_does_not_own_the_running_script(tmp_path: Path) -> None:
    source_root, paths = _minimal_inputs(tmp_path)
    output = tmp_path / "capture.json"

    result = _capture(source_root, paths, output, declared_tooling_root=tmp_path / "unrelated-tooling")

    assert result.returncode != 0
    assert "supplied frozen tooling root" in result.stderr
    assert not output.exists()


def test_capture_rejects_candidate_tooling_even_when_its_script_matches(tmp_path: Path) -> None:
    source_root, paths = _minimal_inputs(tmp_path)
    candidate_scripts = source_root / "scripts"
    candidate_capture = candidate_scripts / "capture_capability_baseline.py"
    shutil.copy2(paths["tooling_root"] / "scripts" / candidate_capture.name, candidate_capture)
    shutil.copy2(
        paths["tooling_root"] / "scripts" / "check_0042_r2_complete_surface_inventory.py",
        candidate_scripts / "check_0042_r2_complete_surface_inventory.py",
    )
    shutil.copy2(
        paths["tooling_root"] / "scripts" / "check_0042_r2_repository_surface_disposition.py",
        candidate_scripts / "check_0042_r2_repository_surface_disposition.py",
    )
    _commit_source(source_root)
    output = tmp_path / "capture.json"

    result = _capture(
        source_root,
        paths,
        output,
        declared_tooling_root=source_root,
        runner_script=candidate_capture,
    )

    assert result.returncode != 0
    assert "must be distinct from the source worktree" in result.stderr
    assert not output.exists()


def test_invalid_ledger_does_not_write_or_overwrite_capture_output(tmp_path: Path) -> None:
    source_root, paths = _minimal_inputs(tmp_path)
    ledger = json.loads(paths["ledger"].read_text(encoding="utf-8"))
    ledger["entries"][0]["scenarios"][0]["authority"] = {
        "kind": "current_fincore_output",
        "reference": "candidate result",
    }
    _write_json(paths["ledger"], ledger)
    _commit_source(source_root)
    output = tmp_path / "capture.json"
    output.write_text('{"previous": "success"}\n', encoding="utf-8")

    result = _capture(source_root, paths, output)

    assert result.returncode != 0
    assert "independent authority" in result.stderr
    assert output.read_text(encoding="utf-8") == '{"previous": "success"}\n'


def test_capture_rejects_self_referential_pinned_oracle_without_overwriting_output(tmp_path: Path) -> None:
    source_root, paths = _minimal_inputs(tmp_path)
    ledger = json.loads(paths["ledger"].read_text(encoding="utf-8"))
    ledger["entries"][0]["scenarios"][0]["authority"] = {
        "kind": "pinned_upstream_oracle",
        "source_project": "empyrical",
        "reference": "current Fincore output",
        "version": "0.5.0",
    }
    _write_json(paths["ledger"], ledger)
    _commit_source(source_root)
    output = tmp_path / "capture.json"
    output.write_text('{"previous": "success"}\n', encoding="utf-8")

    result = _capture(source_root, paths, output)

    assert result.returncode != 0
    assert "self-output" in result.stderr
    assert output.read_text(encoding="utf-8") == '{"previous": "success"}\n'


def test_capture_rejects_absolute_golden_path_without_overwriting_output(tmp_path: Path) -> None:
    source_root, paths = _minimal_inputs(tmp_path)
    ledger = json.loads(paths["ledger"].read_text(encoding="utf-8"))
    ledger["entries"][0]["scenarios"][0]["golden_path"] = str(paths["fixture_dir"] / "annual-return.json")
    _write_json(paths["ledger"], ledger)
    _commit_source(source_root)
    output = tmp_path / "capture.json"
    output.write_text('{"previous": "success"}\n', encoding="utf-8")

    result = _capture(source_root, paths, output)

    assert result.returncode != 0
    assert "portable fixture-relative" in result.stderr
    assert output.read_text(encoding="utf-8") == '{"previous": "success"}\n'


def test_capture_rejects_initial_head_symlink_fixture_without_writing_output(tmp_path: Path) -> None:
    source_root, paths = _minimal_inputs(tmp_path)
    fixture = paths["fixture_dir"] / "annual-return.json"
    outside_fixture = tmp_path / "outside-fixture.json"
    outside_fixture.write_text('{"value": 999.0}\n', encoding="utf-8")
    fixture.unlink()
    fixture.symlink_to(outside_fixture)
    _commit_source(source_root)
    output = tmp_path / "capture.json"

    result = _capture(source_root, paths, output)

    assert result.returncode != 0
    assert "regular-file blobs only" in result.stderr
    assert not output.exists()


def test_capture_rejects_missing_deny_network_flag(tmp_path: Path) -> None:
    source_root, paths = _minimal_inputs(tmp_path)

    result = _capture(source_root, paths, tmp_path / "capture.json", include_deny_network=False)

    assert result.returncode != 0
    assert "--deny-network is required" in result.stderr


def test_capture_rejects_dirty_source_tree_without_writing_output(tmp_path: Path) -> None:
    source_root, paths = _minimal_inputs(tmp_path)
    (source_root / "untracked.txt").write_text("dirty\n", encoding="utf-8")
    output = tmp_path / "capture.json"

    result = _capture(source_root, paths, output)

    assert result.returncode != 0
    assert "clean Git worktree" in result.stderr
    assert not output.exists()


@pytest.mark.parametrize("document_name", ["inventory", "module_disposition", "test_disposition"])
def test_empty_disposition_document_does_not_overwrite_capture_output(tmp_path: Path, document_name: str) -> None:
    source_root, paths = _minimal_inputs(tmp_path)
    _write_json(paths[document_name], {"entries": []})
    _commit_source(source_root)
    output = tmp_path / "capture.json"
    output.write_text('{"previous": "success"}\n', encoding="utf-8")

    result = _capture(source_root, paths, output)

    assert result.returncode != 0
    assert "entries must be non-empty" in result.stderr
    assert output.read_text(encoding="utf-8") == '{"previous": "success"}\n'


def test_capture_rejects_git_state_changed_during_input_hashing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source_root, paths = _minimal_inputs(tmp_path)
    module = _capture_module(paths["tooling_root"])
    output = tmp_path / "capture.json"
    output.write_text('{"previous": "success"}\n', encoding="utf-8")
    original_provenance = module._source_provenance
    source_provenance_calls = 0

    def mutate_before_second_provenance(root: Path):
        nonlocal source_provenance_calls
        if root.resolve() == source_root.resolve():
            source_provenance_calls += 1
        if source_provenance_calls == 2 and root.resolve() == source_root.resolve():
            paths["inventory"].write_text(paths["inventory"].read_text(encoding="utf-8") + "\n", encoding="utf-8")
            _commit_source(root)
        return original_provenance(root)

    monkeypatch.setattr(module, "_source_provenance", mutate_before_second_provenance)

    with pytest.raises(module.CaptureValidationError, match="source Git provenance changed during capture"):
        module.capture(
            source_root=source_root,
            legacy_discovery_path=paths["legacy_discovery"],
            surface_union_path=paths["surface_union"],
            inventory_path=paths["inventory"],
            module_disposition_path=paths["module_disposition"],
            test_disposition_path=paths["test_disposition"],
            ledger_path=paths["ledger"],
            repository_surface_facts_path=paths["repository_surface_facts"],
            repository_surface_disposition_path=paths["repository_surface_disposition"],
            tooling_root=paths["tooling_root"],
            fixture_dir=paths["fixture_dir"],
            output_path=output,
            deny_network=True,
        )

    assert source_provenance_calls == 2
    assert output.read_text(encoding="utf-8") == '{"previous": "success"}\n'


def test_capture_uses_initial_head_blobs_when_worktree_bytes_change_then_recover(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source_root, paths = _minimal_inputs(tmp_path)
    module = _capture_module(paths["tooling_root"])
    output = tmp_path / "capture.json"
    fixture = paths["fixture_dir"] / "annual-return.json"
    initial_ledger = paths["ledger"].read_bytes()
    initial_repository_surface_facts = paths["repository_surface_facts"].read_bytes()
    initial_repository_surface_disposition = paths["repository_surface_disposition"].read_bytes()
    initial_fixture = fixture.read_bytes()
    malicious_ledger = b"\n" + initial_ledger + b"\n"
    malicious_repository_surface_facts = b'{"invalid": "facts"}\n'
    malicious_repository_surface_disposition = b'{"invalid": "disposition"}\n'
    malicious_fixture = b'{"value": 999.0}\n'
    original_provenance = module._source_provenance
    source_provenance_calls = 0

    def mutate_then_recover_before_final_provenance(root: Path):
        nonlocal source_provenance_calls
        if root.resolve() == source_root.resolve():
            source_provenance_calls += 1
        if source_provenance_calls == 1 and root.resolve() == source_root.resolve():
            provenance = original_provenance(root)
            paths["ledger"].write_bytes(malicious_ledger)
            paths["repository_surface_facts"].write_bytes(malicious_repository_surface_facts)
            paths["repository_surface_disposition"].write_bytes(malicious_repository_surface_disposition)
            fixture.write_bytes(malicious_fixture)
            assert paths["ledger"].read_bytes() == malicious_ledger
            assert paths["repository_surface_facts"].read_bytes() == malicious_repository_surface_facts
            assert paths["repository_surface_disposition"].read_bytes() == malicious_repository_surface_disposition
            assert fixture.read_bytes() == malicious_fixture
            return provenance
        if source_provenance_calls == 2 and root.resolve() == source_root.resolve():
            paths["ledger"].write_bytes(initial_ledger)
            paths["repository_surface_facts"].write_bytes(initial_repository_surface_facts)
            paths["repository_surface_disposition"].write_bytes(initial_repository_surface_disposition)
            fixture.write_bytes(initial_fixture)
        return original_provenance(root)

    monkeypatch.setattr(module, "_source_provenance", mutate_then_recover_before_final_provenance)

    artifact = module.capture(
        source_root=source_root,
        legacy_discovery_path=paths["legacy_discovery"],
        surface_union_path=paths["surface_union"],
        inventory_path=paths["inventory"],
        module_disposition_path=paths["module_disposition"],
        test_disposition_path=paths["test_disposition"],
        ledger_path=paths["ledger"],
        repository_surface_facts_path=paths["repository_surface_facts"],
        repository_surface_disposition_path=paths["repository_surface_disposition"],
        tooling_root=paths["tooling_root"],
        fixture_dir=paths["fixture_dir"],
        output_path=output,
        deny_network=True,
    )

    assert source_provenance_calls == 2
    assert artifact["inputs"]["ledger"]["sha256"] == hashlib.sha256(initial_ledger).hexdigest()
    assert artifact["inputs"]["ledger"]["sha256"] != hashlib.sha256(malicious_ledger).hexdigest()
    assert (
        artifact["inputs"]["repository_surface_facts"]["sha256"]
        == hashlib.sha256(initial_repository_surface_facts).hexdigest()
    )
    assert (
        artifact["inputs"]["repository_surface_facts"]["sha256"]
        != hashlib.sha256(malicious_repository_surface_facts).hexdigest()
    )
    assert (
        artifact["inputs"]["repository_surface_disposition"]["sha256"]
        == hashlib.sha256(initial_repository_surface_disposition).hexdigest()
    )
    assert (
        artifact["inputs"]["repository_surface_disposition"]["sha256"]
        != hashlib.sha256(malicious_repository_surface_disposition).hexdigest()
    )
    assert artifact["fixtures"]["annual-return.json"]["sha256"] == hashlib.sha256(initial_fixture).hexdigest()
    assert artifact["fixtures"]["annual-return.json"]["sha256"] != hashlib.sha256(malicious_fixture).hexdigest()


def test_capture_executes_the_frozen_checker_blob_when_its_worktree_file_changes_then_recovers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source_root, paths = _minimal_inputs(tmp_path)
    module = _capture_module(paths["tooling_root"])
    checker_path = paths["tooling_root"] / "scripts" / "check_0042_r2_repository_surface_disposition.py"
    initial_checker = checker_path.read_bytes()
    original_validate = module._validate_repository_surface_inputs

    def mutate_then_validate(*args: object, **kwargs: object):
        checker_path.write_text("raise RuntimeError('mutable checker executed')\n", encoding="utf-8")
        try:
            return original_validate(*args, **kwargs)
        finally:
            checker_path.write_bytes(initial_checker)

    monkeypatch.setattr(module, "_validate_repository_surface_inputs", mutate_then_validate)

    artifact = module.capture(
        source_root=source_root,
        tooling_root=paths["tooling_root"],
        legacy_discovery_path=paths["legacy_discovery"],
        surface_union_path=paths["surface_union"],
        inventory_path=paths["inventory"],
        module_disposition_path=paths["module_disposition"],
        test_disposition_path=paths["test_disposition"],
        ledger_path=paths["ledger"],
        repository_surface_facts_path=paths["repository_surface_facts"],
        repository_surface_disposition_path=paths["repository_surface_disposition"],
        fixture_dir=paths["fixture_dir"],
        output_path=tmp_path / "capture.json",
        deny_network=True,
    )

    assert checker_path.read_bytes() == initial_checker
    assert artifact["repository_surface"]["validation"]["not_for_d0"] is True


def test_capture_executes_the_frozen_complete_inventory_checker_blob_when_mutated(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source_root, paths = _minimal_inputs(tmp_path)
    module = _capture_module(paths["tooling_root"])
    checker_path = paths["tooling_root"] / "scripts" / "check_0042_r2_complete_surface_inventory.py"
    initial_checker = checker_path.read_bytes()
    original_validate = module._validate_complete_inventory_inputs

    def mutate_then_validate(*args: object, **kwargs: object):
        checker_path.write_text("raise RuntimeError('mutable checker executed')\n", encoding="utf-8")
        try:
            return original_validate(*args, **kwargs)
        finally:
            checker_path.write_bytes(initial_checker)

    monkeypatch.setattr(module, "_validate_complete_inventory_inputs", mutate_then_validate)

    artifact = module.capture(
        source_root=source_root,
        tooling_root=paths["tooling_root"],
        legacy_discovery_path=paths["legacy_discovery"],
        surface_union_path=paths["surface_union"],
        inventory_path=paths["inventory"],
        module_disposition_path=paths["module_disposition"],
        test_disposition_path=paths["test_disposition"],
        ledger_path=paths["ledger"],
        repository_surface_facts_path=paths["repository_surface_facts"],
        repository_surface_disposition_path=paths["repository_surface_disposition"],
        fixture_dir=paths["fixture_dir"],
        output_path=tmp_path / "capture.json",
        deny_network=True,
    )

    assert checker_path.read_bytes() == initial_checker
    assert artifact["complete_surface_inventory"]["validation"]["not_a_d0_verdict"] is True


def test_capture_uses_initial_lexical_fixture_path_when_symlink_is_repointed_then_restored(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source_root, paths = _minimal_inputs(tmp_path)
    module = _capture_module(paths["tooling_root"])
    fixture_dir = paths["fixture_dir"]
    initial_fixture = fixture_dir / "annual-return.json"
    initial_fixture_bytes = initial_fixture.read_bytes()
    alternate_dir = source_root / "other-goldens"
    alternate_dir.mkdir()
    alternate_fixture = alternate_dir / "annual-return.json"
    alternate_fixture.write_text('{"value": 999.0}\n', encoding="utf-8")
    _commit_source(source_root)
    output = tmp_path / "capture.json"
    backup_dir = source_root / "goldens-backup"
    original_provenance = module._source_provenance
    source_provenance_calls = 0

    def repoint_then_restore_fixture_directory(root: Path):
        nonlocal source_provenance_calls
        if root.resolve() == source_root.resolve():
            source_provenance_calls += 1
        if source_provenance_calls == 1 and root.resolve() == source_root.resolve():
            provenance = original_provenance(root)
            fixture_dir.rename(backup_dir)
            fixture_dir.symlink_to(alternate_dir, target_is_directory=True)
            return provenance
        if source_provenance_calls == 2 and root.resolve() == source_root.resolve():
            fixture_dir.unlink()
            backup_dir.rename(fixture_dir)
        return original_provenance(root)

    monkeypatch.setattr(module, "_source_provenance", repoint_then_restore_fixture_directory)

    artifact = module.capture(
        source_root=source_root,
        legacy_discovery_path=paths["legacy_discovery"],
        surface_union_path=paths["surface_union"],
        inventory_path=paths["inventory"],
        module_disposition_path=paths["module_disposition"],
        test_disposition_path=paths["test_disposition"],
        ledger_path=paths["ledger"],
        repository_surface_facts_path=paths["repository_surface_facts"],
        repository_surface_disposition_path=paths["repository_surface_disposition"],
        tooling_root=paths["tooling_root"],
        fixture_dir=fixture_dir,
        output_path=output,
        deny_network=True,
    )

    assert source_provenance_calls == 2
    assert artifact["fixtures"]["annual-return.json"]["sha256"] == hashlib.sha256(initial_fixture_bytes).hexdigest()
    assert (
        artifact["fixtures"]["annual-return.json"]["sha256"]
        != hashlib.sha256(alternate_fixture.read_bytes()).hexdigest()
    )


def test_capture_module_documents_the_fail_closed_schema() -> None:
    spec = importlib.util.spec_from_file_location("capture_capability_baseline", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    assert "schema_version" in module.__doc__
    assert "independent authority" in module.__doc__
