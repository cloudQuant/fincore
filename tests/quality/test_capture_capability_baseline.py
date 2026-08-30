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


def test_capture_module_documents_the_fail_closed_schema() -> None:
    module = _capture_module()

    assert "schema_version" in module.__doc__
    assert "independent authority" in module.__doc__
