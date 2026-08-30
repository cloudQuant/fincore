"""Integration contracts for the 0042-R2 baseline capture command."""

from __future__ import annotations

import hashlib
import importlib.util
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


def _capture(
    source_root: Path,
    paths: dict[str, Path],
    output: Path,
    *,
    include_deny_network: bool = True,
) -> subprocess.CompletedProcess[str]:
    command = [
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
    ]
    if include_deny_network:
        command.append("--deny-network")
    return subprocess.run(command, cwd=source_root, capture_output=True, text=True, check=False)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _capture_module():
    spec = importlib.util.spec_from_file_location("capture_capability_baseline", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
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
    for key in ("inventory", "module_disposition", "test_disposition", "ledger"):
        assert artifact["inputs"][key]["sha256"] == _sha256(paths[key])
    assert artifact["fixtures"]["annual-return.json"]["sha256"] == _sha256(paths["fixture_dir"] / "annual-return.json")


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
    module = _capture_module()
    source_root, paths = _minimal_inputs(tmp_path)
    output = tmp_path / "capture.json"
    output.write_text('{"previous": "success"}\n', encoding="utf-8")
    original_provenance = module._source_provenance
    provenance_calls = 0

    def mutate_before_second_provenance(root: Path):
        nonlocal provenance_calls
        provenance_calls += 1
        if provenance_calls == 2:
            paths["inventory"].write_text(paths["inventory"].read_text(encoding="utf-8") + "\n", encoding="utf-8")
            _commit_source(root)
        return original_provenance(root)

    monkeypatch.setattr(module, "_source_provenance", mutate_before_second_provenance)

    with pytest.raises(module.CaptureValidationError, match="source Git provenance changed during capture"):
        module.capture(
            source_root=source_root,
            inventory_path=paths["inventory"],
            module_disposition_path=paths["module_disposition"],
            test_disposition_path=paths["test_disposition"],
            ledger_path=paths["ledger"],
            fixture_dir=paths["fixture_dir"],
            output_path=output,
            deny_network=True,
        )

    assert provenance_calls == 2
    assert output.read_text(encoding="utf-8") == '{"previous": "success"}\n'


def test_capture_uses_initial_head_blobs_when_worktree_bytes_change_then_recover(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _capture_module()
    source_root, paths = _minimal_inputs(tmp_path)
    output = tmp_path / "capture.json"
    fixture = paths["fixture_dir"] / "annual-return.json"
    initial_ledger = paths["ledger"].read_bytes()
    initial_fixture = fixture.read_bytes()
    malicious_ledger = b"\n" + initial_ledger + b"\n"
    malicious_fixture = b'{"value": 999.0}\n'
    original_provenance = module._source_provenance
    provenance_calls = 0

    def mutate_then_recover_before_final_provenance(root: Path):
        nonlocal provenance_calls
        provenance_calls += 1
        if provenance_calls == 1:
            provenance = original_provenance(root)
            paths["ledger"].write_bytes(malicious_ledger)
            fixture.write_bytes(malicious_fixture)
            assert paths["ledger"].read_bytes() == malicious_ledger
            assert fixture.read_bytes() == malicious_fixture
            return provenance
        if provenance_calls == 2:
            paths["ledger"].write_bytes(initial_ledger)
            fixture.write_bytes(initial_fixture)
        return original_provenance(root)

    monkeypatch.setattr(module, "_source_provenance", mutate_then_recover_before_final_provenance)

    artifact = module.capture(
        source_root=source_root,
        inventory_path=paths["inventory"],
        module_disposition_path=paths["module_disposition"],
        test_disposition_path=paths["test_disposition"],
        ledger_path=paths["ledger"],
        fixture_dir=paths["fixture_dir"],
        output_path=output,
        deny_network=True,
    )

    assert provenance_calls == 2
    assert artifact["inputs"]["ledger"]["sha256"] == hashlib.sha256(initial_ledger).hexdigest()
    assert artifact["inputs"]["ledger"]["sha256"] != hashlib.sha256(malicious_ledger).hexdigest()
    assert artifact["fixtures"]["annual-return.json"]["sha256"] == hashlib.sha256(initial_fixture).hexdigest()
    assert artifact["fixtures"]["annual-return.json"]["sha256"] != hashlib.sha256(malicious_fixture).hexdigest()


def test_capture_uses_initial_lexical_fixture_path_when_symlink_is_repointed_then_restored(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _capture_module()
    source_root, paths = _minimal_inputs(tmp_path)
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
    provenance_calls = 0

    def repoint_then_restore_fixture_directory(root: Path):
        nonlocal provenance_calls
        provenance_calls += 1
        if provenance_calls == 1:
            provenance = original_provenance(root)
            fixture_dir.rename(backup_dir)
            fixture_dir.symlink_to(alternate_dir, target_is_directory=True)
            return provenance
        if provenance_calls == 2:
            fixture_dir.unlink()
            backup_dir.rename(fixture_dir)
        return original_provenance(root)

    monkeypatch.setattr(module, "_source_provenance", repoint_then_restore_fixture_directory)

    artifact = module.capture(
        source_root=source_root,
        inventory_path=paths["inventory"],
        module_disposition_path=paths["module_disposition"],
        test_disposition_path=paths["test_disposition"],
        ledger_path=paths["ledger"],
        fixture_dir=fixture_dir,
        output_path=output,
        deny_network=True,
    )

    assert provenance_calls == 2
    assert artifact["fixtures"]["annual-return.json"]["sha256"] == hashlib.sha256(initial_fixture_bytes).hexdigest()
    assert (
        artifact["fixtures"]["annual-return.json"]["sha256"]
        != hashlib.sha256(alternate_fixture.read_bytes()).hexdigest()
    )


def test_capture_module_documents_the_fail_closed_schema() -> None:
    module = _capture_module()

    assert "schema_version" in module.__doc__
    assert "independent authority" in module.__doc__
