"""Contracts for frozen D0 capability-scenario evaluation.

The evaluator must execute source nodeids from an immutable ledger, bind every
``ok`` record to the input capture/ledger/source identities, and keep missing
evidence distinct from an executed failing scenario.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

from tests.support.frozen_capture_tooling import create_frozen_capture_tooling_root

REPOSITORY_ROOT = Path(os.environ.get("FINCORE_0042R2_SOURCE_ROOT", Path(__file__).parents[2])).resolve()
SCRIPT = REPOSITORY_ROOT / "scripts" / "evaluate_0042_r2_capability_scenarios.py"


def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _git_text(root: Path, *arguments: str) -> str:
    return subprocess.run(["git", *arguments], cwd=root, capture_output=True, text=True, check=True).stdout.strip()


def _commit(root: Path) -> None:
    subprocess.run(["git", "init", "-q"], cwd=root, check=True)
    subprocess.run(["git", "add", "."], cwd=root, check=True)
    subprocess.run(
        [
            "git",
            "-c",
            "user.name=0042-R2 evaluator test",
            "-c",
            "user.email=0042-r2-evaluator@example.invalid",
            "commit",
            "-qm",
            "frozen D0 source",
        ],
        cwd=root,
        check=True,
    )


def _source_identity(root: Path) -> dict[str, object]:
    return {
        "commit": _git_text(root, "rev-parse", "HEAD"),
        "tree": _git_text(root, "rev-parse", "HEAD^{tree}"),
        "clean": True,
    }


def _scenario(*, source_nodeids: list[str] | None = None, golden: bool = True) -> dict[str, object]:
    scenario: dict[str, object] = {
        "scenario_id": "ordinary_inputs",
        "authority": {
            "kind": "property_invariant",
            "reference": "metrics.demo.invariant.v1",
            "invariant_id": "metrics.demo.invariant.v1",
        },
    }
    if golden:
        scenario["golden_path"] = "metric.json"
    else:
        scenario["oracle_reference"] = "metric invariant"
    return scenario


def _write_source(
    tmp_path: Path,
    *,
    test_body: str = "assert 1 + 1 == 2",
    source_nodeids: list[str] | None = None,
    golden: bool = True,
) -> tuple[Path, Path]:
    source_root = tmp_path / "source"
    source_root.mkdir()
    _write(source_root / "tests" / "test_metric.py", f"def test_metric():\n    {test_body}\n")
    _write(source_root / "goldens" / "metric.json", '{"value": 1}\n')
    ledger = {
        "schema_version": 1,
        "artifact_type": "capability_ledger",
        "scope": "complete",
        "decision_status": "complete",
        "entries": [
            {
                "capability_id": "metrics.demo",
                "owner": "metrics",
                "disposition": "required",
                "target_operation_id": "metrics.demo",
                "source_nodeids": source_nodeids
                if source_nodeids is not None
                else ["tests/test_metric.py::test_metric"],
                "wheel_nodeids": ["tests/parity/test_metric.py::test_metric"],
                "scenarios": [_scenario(golden=golden)],
            }
        ],
    }
    ledger_path = source_root / "ledger.json"
    _write(ledger_path, json.dumps(ledger, indent=2) + "\n")
    _commit(source_root)
    return source_root, ledger_path


def _write_capture(tmp_path: Path, source_root: Path, ledger_path: Path, *, fixture_sha256: str | None = None) -> Path:
    fixture = source_root / "goldens" / "metric.json"
    capture = {
        "schema_version": 1,
        "artifact_type": "capability_baseline_capture",
        "capture_status": "captured",
        "source": _source_identity(source_root),
        "inputs": {
            "ledger": {
                "path": ledger_path.relative_to(source_root).as_posix(),
                "sha256": hashlib.sha256(ledger_path.read_bytes()).hexdigest(),
            }
        },
        "fixture_root": "goldens",
        "fixtures": {
            "metric.json": {
                "sha256": fixture_sha256 or hashlib.sha256(fixture.read_bytes()).hexdigest(),
            }
        },
    }
    path = tmp_path / "input-capture.json"
    _write(path, json.dumps(capture, indent=2) + "\n")
    return path


def _run(
    tmp_path: Path, source_root: Path, capture: Path, *, extra: list[str] | None = None
) -> subprocess.CompletedProcess[str]:
    tooling_root = create_frozen_capture_tooling_root(tmp_path / "tooling", SCRIPT.parent)
    output = tmp_path / "evaluated.json"
    result = subprocess.run(
        [
            sys.executable,
            "-I",
            str(tooling_root / "scripts" / SCRIPT.name),
            "--input-capture",
            str(capture),
            "--source-root",
            str(source_root),
            "--tooling-root",
            str(tooling_root),
            "--output",
            str(output),
            "--batch-size",
            "1",
            "--timeout-seconds",
            "30",
            "--deny-network",
            *(extra or []),
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
    )
    result.output_path = output  # type: ignore[attr-defined]
    return result


def test_evaluator_materializes_ok_with_immutable_execution_bindings(tmp_path: Path) -> None:
    source_root, ledger_path = _write_source(tmp_path)
    capture = _write_capture(tmp_path, source_root, ledger_path)

    result = _run(tmp_path, source_root, capture)

    assert result.returncode == 0, result.stderr
    artifact = json.loads(result.output_path.read_text(encoding="utf-8"))  # type: ignore[attr-defined]
    record = artifact["capabilities"]["metrics.demo"]
    assert artifact["evaluation_status"] == "evaluated_source"
    assert artifact["evaluation"]["verdict"] == "PASS"
    assert record["status"] == "ok"
    assert record["input_capture_sha256"] == hashlib.sha256(capture.read_bytes()).hexdigest()
    assert record["ledger_sha256"] == hashlib.sha256(ledger_path.read_bytes()).hexdigest()
    assert record["source_identity"] == _source_identity(source_root)
    scenario = record["scenarios"][0]
    assert scenario["status"] == "ok"
    assert (
        scenario["golden_sha256"] == hashlib.sha256((source_root / "goldens" / "metric.json").read_bytes()).hexdigest()
    )
    assert scenario["execution_ids"] == ["source-0001"]
    execution = artifact["executions"]["source-0001"]
    assert execution["exit_code"] == 0
    assert execution["argv"][-1] == "tests/test_metric.py::test_metric"
    assert len(execution["output_sha256"]) == 64


def test_evaluator_records_executed_source_failure_as_fail_not_pending(tmp_path: Path) -> None:
    source_root, ledger_path = _write_source(tmp_path, test_body="assert False")
    capture = _write_capture(tmp_path, source_root, ledger_path)

    result = _run(tmp_path, source_root, capture)

    assert result.returncode == 1
    artifact = json.loads(result.output_path.read_text(encoding="utf-8"))  # type: ignore[attr-defined]
    record = artifact["capabilities"]["metrics.demo"]
    assert artifact["evaluation"]["verdict"] == "FAIL"
    assert record["status"] == "failed"
    assert record["scenarios"][0]["failed_execution_ids"] == ["source-0001"]
    assert artifact["executions"]["source-0001"]["exit_code"] != 0


def test_evaluator_marks_missing_source_execution_prerequisite_pending(tmp_path: Path) -> None:
    source_root, ledger_path = _write_source(tmp_path, source_nodeids=[])
    capture = _write_capture(tmp_path, source_root, ledger_path)

    result = _run(tmp_path, source_root, capture)

    assert result.returncode == 3
    artifact = json.loads(result.output_path.read_text(encoding="utf-8"))  # type: ignore[attr-defined]
    record = artifact["capabilities"]["metrics.demo"]
    assert artifact["evaluation"]["verdict"] == "BLOCKED"
    assert record["status"] == "pending"
    assert "source_nodeids are missing" in record["pending_reasons"]


def test_evaluator_rejects_fixture_digest_that_does_not_match_captured_source(tmp_path: Path) -> None:
    source_root, ledger_path = _write_source(tmp_path)
    capture = _write_capture(tmp_path, source_root, ledger_path, fixture_sha256="0" * 64)

    result = _run(tmp_path, source_root, capture)

    assert result.returncode == 2
    assert "fixture blob SHA256" in result.stderr
    assert not result.output_path.exists()  # type: ignore[attr-defined]


def test_evaluator_rejects_a_source_identity_that_differs_from_the_input_capture(tmp_path: Path) -> None:
    source_root, ledger_path = _write_source(tmp_path)
    capture = _write_capture(tmp_path, source_root, ledger_path)
    payload = json.loads(capture.read_text(encoding="utf-8"))
    payload["source"]["commit"] = "0" * 40
    _write(capture, json.dumps(payload, indent=2) + "\n")

    result = _run(tmp_path, source_root, capture)

    assert result.returncode == 2
    assert "does not match" in result.stderr
