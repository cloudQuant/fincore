"""Fail-closed contracts for the development-time feature parity checker.

The checker must never sign parity from pending baselines, incomplete scoped
ledgers, or missing capability evidence.  Formal conclusions remain reserved
for the detached D0_TOOLING_SHA acceptance runner.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).parents[2]
SCRIPT = REPOSITORY_ROOT / "scripts" / "check_feature_parity.py"
COMMITTED_LEDGER = REPOSITORY_ROOT / "tests" / "parity" / "fixtures" / "capability-ledger-0042-r2.json"


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def _run(args: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-I", str(SCRIPT), *args],
        cwd=REPOSITORY_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )


def _complete_ledger(family: str = "metrics") -> dict:
    return {
        "schema_version": 1,
        "artifact_type": "capability_ledger",
        "scope": "complete",
        "decision_status": "complete",
        "does_not_assert": ["D-TECH", "D0", "installed_wheel_behavior", "legacy_zero"],
        "coverage_gaps": [],
        "entries": [
            {
                "capability_id": f"{family}.demo_capability",
                "owner": family,
                "disposition": "required",
                "target_operation_id": f"{family}.demo_capability",
                "source_nodeids": ["tests/parity/test_demo.py::test_demo"],
                "wheel_nodeids": ["tests/parity/test_demo.py::test_demo"],
                "scenarios": [
                    {
                        "scenario_id": "ordinary_inputs",
                        "oracle_reference": "demo oracle",
                        "authority": {
                            "kind": "property_invariant",
                            "reference": "demo invariant",
                            "invariant_id": f"{family}.demo_capability.invariant.v1",
                        },
                    }
                ],
            }
        ],
    }


def _captured_baseline(capabilities: dict) -> dict:
    return {
        "artifact_type": "capability_baseline_capture",
        "capture_status": "captured",
        "source": {"commit": "a" * 40, "tree": "b" * 40, "clean": True},
        "capabilities": capabilities,
    }


def test_missing_baseline_is_a_usage_error(tmp_path: Path) -> None:
    ledger = tmp_path / "ledger.json"
    _write_json(ledger, _complete_ledger())

    result = _run(["--baseline", str(tmp_path / "absent.json"), "--ledger", str(ledger), "--families", "metrics"])

    assert result.returncode == 2
    assert "--baseline" in result.stderr


def test_pending_baseline_is_blocked(tmp_path: Path) -> None:
    ledger = tmp_path / "ledger.json"
    baseline = tmp_path / "baseline.json"
    _write_json(ledger, _complete_ledger())
    _write_json(baseline, {"artifact_type": "capability_baseline_capture", "capture_status": "pending"})

    result = _run(["--baseline", str(baseline), "--ledger", str(ledger), "--families", "metrics"])

    assert result.returncode == 3
    assert "pending" in result.stderr


def test_baseline_without_capability_section_is_blocked(tmp_path: Path) -> None:
    ledger = tmp_path / "ledger.json"
    baseline = tmp_path / "baseline.json"
    _write_json(ledger, _complete_ledger())
    _write_json(baseline, _captured_baseline({}))
    payload = json.loads(baseline.read_text(encoding="utf-8"))
    payload.pop("capabilities")
    _write_json(baseline, payload)

    result = _run(["--baseline", str(baseline), "--ledger", str(ledger), "--families", "metrics"])

    assert result.returncode == 3
    assert "capability parity section" in result.stderr


def test_baseline_with_wrong_artifact_type_is_a_usage_error(tmp_path: Path) -> None:
    ledger = tmp_path / "ledger.json"
    baseline = tmp_path / "baseline.json"
    _write_json(ledger, _complete_ledger())
    _write_json(baseline, {"artifact_type": "something_else", "capture_status": "captured"})

    result = _run(["--baseline", str(baseline), "--ledger", str(ledger), "--families", "metrics"])

    assert result.returncode == 2
    assert "capability_baseline_capture" in result.stderr


def test_unknown_family_is_a_usage_error(tmp_path: Path) -> None:
    ledger = tmp_path / "ledger.json"
    baseline = tmp_path / "baseline.json"
    _write_json(ledger, _complete_ledger())
    _write_json(baseline, _captured_baseline({}))

    result = _run(["--baseline", str(baseline), "--ledger", str(ledger), "--families", "not_a_family"])

    assert result.returncode == 2
    assert "unknown families" in result.stderr


def test_committed_scoped_ledger_is_blocked_for_families_with_gaps(tmp_path: Path) -> None:
    baseline = tmp_path / "baseline.json"
    _write_json(baseline, _captured_baseline({}))

    result = _run(
        ["--baseline", str(baseline), "--ledger", str(COMMITTED_LEDGER), "--families", "metrics", "performance"]
    )

    assert result.returncode == 3
    assert "coverage gaps" in result.stderr


def test_scoped_ledger_without_gaps_still_cannot_sign_parity(tmp_path: Path) -> None:
    ledger = tmp_path / "ledger.json"
    baseline = tmp_path / "baseline.json"
    scoped = _complete_ledger()
    scoped["decision_status"] = "scoped"
    scoped["not_for_d0"] = True
    scoped["scope"] = "metrics_family_only"
    _write_json(ledger, scoped)
    _write_json(baseline, _captured_baseline({"metrics.demo_capability": {"status": "ok"}}))

    result = _run(["--baseline", str(baseline), "--ledger", str(ledger), "--families", "metrics"])

    assert result.returncode == 3
    assert "scoped" in result.stderr


def test_family_outside_ledger_scope_is_blocked(tmp_path: Path) -> None:
    ledger = tmp_path / "ledger.json"
    baseline = tmp_path / "baseline.json"
    scoped = _complete_ledger("metrics")
    scoped["scope"] = "metrics_family_only"
    _write_json(ledger, scoped)
    _write_json(baseline, _captured_baseline({}))

    result = _run(["--baseline", str(baseline), "--ledger", str(ledger), "--families", "factor"])

    assert result.returncode == 3
    assert "does not cover" in result.stderr


def test_complete_ledger_with_ok_baseline_passes_and_writes_evidence(tmp_path: Path) -> None:
    ledger = tmp_path / "ledger.json"
    baseline = tmp_path / "baseline.json"
    output = tmp_path / "evidence.json"
    _write_json(ledger, _complete_ledger())
    _write_json(baseline, _captured_baseline({"metrics.demo_capability": {"status": "ok"}}))

    result = _run(
        [
            "--baseline",
            str(baseline),
            "--ledger",
            str(ledger),
            "--families",
            "metrics",
            "--output",
            str(output),
        ]
    )

    assert result.returncode == 0, result.stderr
    evidence = json.loads(output.read_text(encoding="utf-8"))
    assert evidence["verdict"] == "PASS"
    assert evidence["scope"] == "development_diagnostic_only"
    assert evidence["results"][0]["verified_capabilities"] == 1
    assert evidence["results"][0]["missing_capabilities"] == []


def test_divergent_capability_fails_with_unresolved_difference(tmp_path: Path) -> None:
    ledger = tmp_path / "ledger.json"
    baseline = tmp_path / "baseline.json"
    output = tmp_path / "evidence.json"
    _write_json(ledger, _complete_ledger())
    _write_json(baseline, _captured_baseline({"metrics.demo_capability": {"status": "divergent"}}))

    result = _run(
        [
            "--baseline",
            str(baseline),
            "--ledger",
            str(ledger),
            "--families",
            "metrics",
            "--output",
            str(output),
        ]
    )

    assert result.returncode == 1
    evidence = json.loads(output.read_text(encoding="utf-8"))
    assert evidence["verdict"] == "FAIL"
    assert evidence["results"][0]["unresolved_differences"] == ["metrics.demo_capability"]


def test_capability_absent_from_baseline_fails_as_missing(tmp_path: Path) -> None:
    ledger = tmp_path / "ledger.json"
    baseline = tmp_path / "baseline.json"
    _write_json(ledger, _complete_ledger())
    _write_json(baseline, _captured_baseline({}))

    result = _run(["--baseline", str(baseline), "--ledger", str(ledger), "--families", "metrics"])

    assert result.returncode == 1


def test_family_without_ledger_capabilities_is_blocked(tmp_path: Path) -> None:
    ledger = tmp_path / "ledger.json"
    baseline = tmp_path / "baseline.json"
    _write_json(ledger, _complete_ledger("metrics"))
    _write_json(baseline, _captured_baseline({"metrics.demo_capability": {"status": "ok"}}))

    result = _run(["--baseline", str(baseline), "--ledger", str(ledger), "--families", "metrics", "factor"])

    assert result.returncode == 3
    assert "no capabilities for family" in result.stderr


def test_dist_directory_without_a_wheel_is_blocked(tmp_path: Path) -> None:
    ledger = tmp_path / "ledger.json"
    baseline = tmp_path / "baseline.json"
    dist = tmp_path / "dist"
    dist.mkdir()
    _write_json(ledger, _complete_ledger())
    _write_json(baseline, _captured_baseline({"metrics.demo_capability": {"status": "ok"}}))

    result = _run(
        ["--baseline", str(baseline), "--ledger", str(ledger), "--families", "metrics", "--dist", str(dist)]
    )

    assert result.returncode == 3
    assert "no wheel" in result.stderr


def test_missing_dist_directory_is_a_usage_error(tmp_path: Path) -> None:
    ledger = tmp_path / "ledger.json"
    baseline = tmp_path / "baseline.json"
    _write_json(ledger, _complete_ledger())
    _write_json(baseline, _captured_baseline({"metrics.demo_capability": {"status": "ok"}}))

    result = _run(
        [
            "--baseline",
            str(baseline),
            "--ledger",
            str(ledger),
            "--families",
            "metrics",
            "--dist",
            str(tmp_path / "absent-dist"),
        ]
    )

    assert result.returncode == 2
    assert "--dist" in result.stderr
