"""Task 11 gates for enhanced factor-analysis benchmark artifacts."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(os.environ.get("FINCORE_0042R2_SOURCE_ROOT", Path(__file__).resolve().parents[2])).resolve()
BENCHMARKS = ROOT / "benchmarks"
if str(BENCHMARKS) not in sys.path:
    sys.path.insert(0, str(BENCHMARKS))

from bench_factor_analysis import SCENARIOS, SEED

from scripts.compare_benchmarks import list_candidate_baselines, select_baseline

RUNNER = ROOT / "scripts" / "run_factor_benchmarks.py"
COMPARATOR = ROOT / "scripts" / "compare_benchmarks.py"
BASELINES_DIR = ROOT / "benchmarks" / "factor-analysis-baselines"
BASELINE = BASELINES_DIR / "darwin-arm64.json"
WORKFLOW = ROOT / ".github" / "workflows" / "ci.yml"
SHA256 = "a" * 64
REVIEWED_SHA256 = "b" * 64

EXPECTED_SCENARIOS = {
    "small-ci": {
        "input_shape": {"dates": 252, "assets": 100, "rows": 25200},
        "kernels": ("prepare", "quantize", "information-coefficient", "weights"),
    },
    "medium-artifact": {
        "input_shape": {"dates": 1260, "assets": 500, "rows": 630000},
        "kernels": ("prepare", "factor-returns", "full-model"),
    },
    "event": {
        "input_shape": {"dates": 756, "assets": 200, "rows": 151200},
        "kernels": ("common-start", "event-average"),
    },
}


def _expected_output_shape(scenario: str, kernel: str) -> list[int] | dict[str, object]:
    if scenario == "small-ci":
        return {
            "prepare": [25200, 5],
            "quantize": [25200],
            "information-coefficient": [252, 2],
            "weights": [25200],
        }[kernel]
    if scenario == "medium-artifact":
        return {
            "prepare": [630000, 5],
            "factor-returns": [1260, 2],
            "full-model": {"factor_data": [630000, 5], "forward_periods": ["1D", "5D"]},
        }[kernel]
    return {"common-start": [26, 36], "event-average": [10, 26]}[kernel]


def _digest(scenario: str, kernel: str) -> str:
    return hashlib.sha256(f"{scenario}/{kernel}".encode()).hexdigest()


def _payload(
    *,
    machine: str,
    scenarios: tuple[str, ...] = ("small-ci",),
    repeats: int = 2,
    warmups: int = 1,
    baseline: bool = False,
    approval: str = "approved",
    wall: float = 1.0,
    rss: int = 100,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "schema": "fincore-factor-analysis-benchmarks-v1",
        "kind": "factor_analysis",
        "rss_unit": "bytes",
        "provenance": {
            "commit": "0" * 40,
            "python": platform.python_version(),
            "numpy": "2",
            "pandas": "3",
            "scipy": "1",
            "statsmodels": "1",
            "os": sys.platform,
            "arch": machine,
            "platform_label": f"{sys.platform}-{machine}",
            "dirty": False,
        },
        "runner": {"warmups": warmups, "repeats": repeats},
        "cases": [],
    }
    for scenario in scenarios:
        contract = EXPECTED_SCENARIOS[scenario]
        for kernel in contract["kernels"]:
            for repeat in range(repeats):
                payload["cases"].append(
                    {
                        "scenario": scenario,
                        "kernel": kernel,
                        "input_shape": deepcopy(contract["input_shape"]),
                        "output_shape": _expected_output_shape(scenario, kernel),
                        "seed": SEED,
                        "wall_seconds": wall,
                        "peak_rss_bytes": rss,
                        "rss_before_bytes": 1,
                        "rss_delta_bytes": rss - 1,
                        "tracemalloc_peak_bytes": rss,
                        "rss_unit": "bytes",
                        "output_digest": _digest(scenario, kernel),
                        "warmup": warmups,
                        "repeat": repeat,
                        "repeats": repeats,
                    }
                )
    if baseline:
        if approval == "approved":
            payload.update(
                {
                    "baseline_status": "approved",
                    "approval": {
                        "status": "approved",
                        "approved_by": "kernel-owner",
                        "approved_at": "2026-08-15T00:00:00+00:00",
                        "reviewed_candidate_sha256": REVIEWED_SHA256,
                    },
                    "candidate_protocol": {
                        "reference_platform": f"{sys.platform}-{machine}",
                        "candidate_path": "build/factor-benchmark-candidate.json",
                        "captured_candidate_sha256": REVIEWED_SHA256,
                        "captured_candidate_review_status": "reviewed",
                        "required_reviewers": ["kernel owner", "Track E"],
                        "approval_steps": ["reviewed"],
                        "current_blockers": [],
                    },
                }
            )
        else:
            payload.update(
                {
                    "baseline_status": "candidate-only-not-release-approved",
                    "approval": {
                        "status": "pending",
                        "approved_by": None,
                        "approved_at": None,
                        "reviewed_candidate_sha256": None,
                    },
                    "candidate_protocol": {
                        "reference_platform": f"{sys.platform}-{machine}",
                        "candidate_path": "build/factor-benchmark-candidate.json",
                        "captured_candidate_sha256": REVIEWED_SHA256,
                        "captured_candidate_review_status": "unreviewed",
                        "required_reviewers": ["kernel owner", "Track E"],
                        "approval_steps": ["review"],
                        "current_blockers": ["approval pending"],
                    },
                }
            )
    return payload


def _run_compare(
    tmp_path: Path, baseline: dict[str, Any], candidate: dict[str, Any], *extra: str
) -> subprocess.CompletedProcess[str]:
    baseline_path = tmp_path / "baseline.json"
    candidate_path = tmp_path / "candidate.json"
    baseline_path.write_text(json.dumps(baseline), encoding="utf-8")
    candidate_path.write_text(json.dumps(candidate), encoding="utf-8")
    return subprocess.run(
        [
            sys.executable,
            str(COMPARATOR),
            "--baseline",
            str(baseline_path),
            "--candidate",
            str(candidate_path),
            *extra,
        ],
        capture_output=True,
        text=True,
        timeout=60,
    )


def _approved_pair(*, machine: str | None = None) -> tuple[dict[str, Any], dict[str, Any]]:
    selected_machine = platform.machine() if machine is None else machine
    return (
        _payload(machine=selected_machine, baseline=True),
        _payload(machine=selected_machine),
    )


def _case(payload: dict[str, Any], kernel: str, repeat: int = 0) -> dict[str, Any]:
    return next(case for case in payload["cases"] if case["kernel"] == kernel and case["repeat"] == repeat)


def test_scenario_registry_has_the_three_reviewed_exact_contracts() -> None:
    assert set(SCENARIOS) == set(EXPECTED_SCENARIOS)
    for name, expected in EXPECTED_SCENARIOS.items():
        assert SCENARIOS[name].input_shape == expected["input_shape"]
        assert SCENARIOS[name].kernels == expected["kernels"]


@pytest.mark.skipif(sys.platform == "win32", reason="runner uses resource.getrusage")
@pytest.mark.skipif(bool(os.environ.get("PYTEST_XDIST_WORKER")), reason="benchmarks do not run under xdist")
def test_small_ci_runner_emits_provenance_measurements_and_stable_outputs(tmp_path: Path) -> None:
    output = tmp_path / "factor-benchmark.json"
    subprocess.run(
        [
            sys.executable,
            str(RUNNER),
            "--scenarios",
            "small-ci",
            "--warmups",
            "0",
            "--repeats",
            "1",
            "--output",
            str(output),
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=600,
    )
    payload = json.loads(output.read_text(encoding="utf-8"))

    assert payload["schema"] == "fincore-factor-analysis-benchmarks-v1"
    assert payload["kind"] == "factor_analysis"
    assert payload["rss_unit"] == "bytes"
    assert payload["runner"] == {"warmups": 0, "repeats": 1}
    assert "approval" not in payload
    assert {key: payload["provenance"][key] for key in ("os", "arch")} == {
        "os": sys.platform,
        "arch": platform.machine(),
    }
    assert payload["provenance"]["platform_label"] == f"{sys.platform}-{platform.machine()}"
    assert isinstance(payload["provenance"]["dirty"], bool)

    assert {(case["scenario"], case["kernel"]) for case in payload["cases"]} == {
        ("small-ci", kernel) for kernel in EXPECTED_SCENARIOS["small-ci"]["kernels"]
    }
    for case in payload["cases"]:
        assert case["input_shape"] == EXPECTED_SCENARIOS["small-ci"]["input_shape"]
        assert case["output_shape"] == _expected_output_shape("small-ci", case["kernel"])
        assert case["seed"] == SEED
        assert case["wall_seconds"] > 0
        assert case["peak_rss_bytes"] > 0
        assert case["rss_delta_bytes"] >= 0
        assert len(case["output_digest"]) == 64
        int(case["output_digest"], 16)
        assert case["warmup"] == 0
        assert case["repeat"] == 0
        assert case["repeats"] == 1

    artifact_sha = hashlib.sha256(output.read_bytes()).hexdigest()
    assert len(artifact_sha) == 64


def test_checked_in_darwin_arm64_baseline_is_transparently_pending() -> None:
    baseline = json.loads(BASELINE.read_text(encoding="utf-8"))
    assert baseline["baseline_status"] == "candidate-only-not-release-approved"
    assert baseline["provenance"]["platform_label"] == "darwin-arm64"
    assert baseline["approval"] == {
        "status": "pending",
        "approved_by": None,
        "approved_at": None,
        "reviewed_candidate_sha256": None,
    }
    assert baseline["candidate_protocol"]["candidate_path"] == "build/factor-benchmark-candidate.json"
    assert "kernel owner" in baseline["candidate_protocol"]["required_reviewers"]
    assert "Track E" in baseline["candidate_protocol"]["required_reviewers"]


def test_digest_gate_accepts_an_approved_same_platform_complete_candidate(tmp_path: Path) -> None:
    baseline, candidate = _approved_pair()

    result = _run_compare(tmp_path, baseline, candidate, "--digest-gate", "sha256")

    assert result.returncode == 0
    assert "all factor-analysis digest, shape, time, and RSS gates passed" in result.stdout


def test_digest_gate_rejects_output_mismatch_before_performance(tmp_path: Path) -> None:
    baseline, candidate = _approved_pair()
    _case(candidate, "prepare")["output_digest"] = "c" * 64
    for case in candidate["cases"]:
        case["wall_seconds"] = 100.0
        case["peak_rss_bytes"] = 10_000
        case["rss_delta_bytes"] = 9_999

    result = _run_compare(tmp_path, baseline, candidate, "--digest-gate", "sha256")

    assert result.returncode == 1
    assert "output_digest mismatch" in result.stderr
    assert "performance comparison not run" in result.stderr
    assert "wall_seconds regressed" not in result.stdout + result.stderr
    assert "peak_rss_bytes regressed" not in result.stdout + result.stderr


def test_digest_gate_rejects_shape_mismatch_before_performance(tmp_path: Path) -> None:
    baseline, candidate = _approved_pair()
    _case(candidate, "prepare")["output_shape"] = [1, 1]
    for case in candidate["cases"]:
        case["wall_seconds"] = 100.0

    result = _run_compare(tmp_path, baseline, candidate, "--digest-gate", "sha256")

    assert result.returncode == 1
    assert "output_shape mismatch" in result.stderr
    assert "performance comparison not run" in result.stderr
    assert "wall_seconds regressed" not in result.stdout + result.stderr


def test_digest_gate_rejects_missing_baseline_case_before_approval(tmp_path: Path) -> None:
    baseline, candidate = _approved_pair()
    candidate["cases"] = [case for case in candidate["cases"] if case["kernel"] != "weights"]

    result = _run_compare(tmp_path, baseline, candidate, "--digest-gate", "sha256")

    assert result.returncode == 1
    assert "candidate: missing required cases" in result.stderr
    assert "performance comparison not run" in result.stderr


def test_digest_gate_rejects_complete_but_missing_candidate_scenario(tmp_path: Path) -> None:
    baseline = _payload(machine=platform.machine(), scenarios=("small-ci", "event"), baseline=True)
    candidate = _payload(machine=platform.machine())

    result = _run_compare(tmp_path, baseline, candidate, "--digest-gate", "sha256")

    assert result.returncode == 1
    assert "candidate missing required cases: [('event', 'common-start')" in result.stderr
    assert "performance comparison not run" in result.stderr


def test_digest_gate_rejects_unexpected_canonical_case_key(tmp_path: Path) -> None:
    baseline, candidate = _approved_pair()
    for payload in (baseline, candidate):
        for case in payload["cases"]:
            if case["kernel"] == "weights":
                case["kernel"] = "unexpected-kernel"

    result = _run_compare(tmp_path, baseline, candidate, "--digest-gate", "sha256")

    assert result.returncode == 1
    assert "unknown scenario/kernel" in result.stderr
    assert "performance comparison not run" in result.stderr


def test_digest_gate_rejects_complete_but_unexpected_candidate_scenario(tmp_path: Path) -> None:
    baseline, candidate = _approved_pair()
    extra = _payload(machine=platform.machine(), scenarios=("event",))
    candidate["cases"].extend(extra["cases"])

    result = _run_compare(tmp_path, baseline, candidate, "--digest-gate", "sha256")

    assert result.returncode == 1
    assert "candidate has unexpected cases: [('event', 'common-start')" in result.stderr
    assert "performance comparison not run" in result.stderr


def test_digest_gate_rejects_noncanonical_input_shape_and_seed(tmp_path: Path) -> None:
    baseline, candidate = _approved_pair()
    _case(candidate, "prepare")["input_shape"] = {"dates": 999, "assets": 100, "rows": 99900}
    _case(candidate, "quantize")["seed"] = SEED + 1

    result = _run_compare(tmp_path, baseline, candidate, "--digest-gate", "sha256")

    assert result.returncode == 1
    assert "noncanonical input_shape" in result.stderr
    assert "noncanonical seed" in result.stderr
    assert "performance comparison not run" in result.stderr


@pytest.mark.parametrize("mode", ("duplicate", "missing"))
def test_digest_gate_rejects_incomplete_or_duplicate_repeat_protocol(tmp_path: Path, mode: str) -> None:
    baseline, candidate = _approved_pair()
    if mode == "duplicate":
        _case(candidate, "prepare", 1)["repeat"] = 0
    else:
        candidate["cases"].remove(_case(candidate, "prepare", 1))

    result = _run_compare(tmp_path, baseline, candidate, "--digest-gate", "sha256")

    assert result.returncode == 1
    assert "repeat IDs" in result.stderr
    assert "performance comparison not run" in result.stderr


def test_digest_gate_rejects_malformed_candidate_schema_before_comparison(tmp_path: Path) -> None:
    baseline, candidate = _approved_pair()
    candidate["schema"] = "untrusted-v0"
    candidate["provenance"].pop("dirty")
    candidate["provenance"]["commit"] = "a" * 64
    candidate["provenance"]["platform_label"] = "wrong-label"
    candidate.pop("runner")
    first = candidate["cases"][0]
    first["output_digest"] = "z" * 64
    first["wall_seconds"] = float("nan")
    first["peak_rss_bytes"] = -1
    first["rss_unit"] = "KiB"

    result = _run_compare(tmp_path, baseline, candidate, "--digest-gate", "sha256")

    assert result.returncode == 1
    assert "candidate: schema" in result.stderr
    assert "candidate: provenance missing 'dirty'" in result.stderr
    assert "candidate: provenance commit must be a 40-hex Git SHA" in result.stderr
    assert "candidate: platform_label" in result.stderr
    assert "candidate: runner" in result.stderr
    assert "candidate: case 0 has invalid SHA256 digest" in result.stderr
    assert "candidate: case 0 has invalid wall_seconds" in result.stderr
    assert "candidate: case 0 has invalid peak_rss_bytes" in result.stderr
    assert "candidate: case 0 has invalid rss_unit" in result.stderr
    assert "performance comparison not run" in result.stderr


def test_digest_gate_requires_factor_kind_before_legacy_dispatch(tmp_path: Path) -> None:
    baseline, candidate = _approved_pair()
    candidate["kind"] = "rolling"

    result = _run_compare(tmp_path, baseline, candidate, "--digest-gate", "sha256")

    assert result.returncode == 1
    assert "candidate: kind must be 'factor_analysis'" in result.stderr
    assert "performance comparison not run" in result.stderr


def test_digest_gate_rejects_dirty_or_inconsistent_approved_baseline(tmp_path: Path) -> None:
    baseline, candidate = _approved_pair()
    baseline["provenance"]["dirty"] = True
    baseline["baseline_status"] = "candidate-only-not-release-approved"
    baseline["candidate_protocol"]["captured_candidate_review_status"] = "unreviewed"

    result = _run_compare(tmp_path, baseline, candidate, "--digest-gate", "sha256")

    assert result.returncode == 1
    assert "baseline: approved baseline must have dirty=false" in result.stderr
    assert "baseline: approved baseline_status" in result.stderr
    assert "baseline: approved candidate_protocol" in result.stderr
    assert "performance comparison not run" in result.stderr


def test_digest_gate_blocks_inconsistent_pending_baseline_before_performance(tmp_path: Path) -> None:
    baseline = _payload(machine=platform.machine(), baseline=True, approval="pending")
    candidate = _payload(machine=platform.machine())
    baseline["approval"]["approved_by"] = "fabricated-reviewer"

    result = _run_compare(tmp_path, baseline, candidate, "--digest-gate", "sha256")

    assert result.returncode == 1
    assert "baseline: pending approval metadata must be null" in result.stderr
    assert "performance comparison not run" in result.stderr


def test_digest_gate_rejects_malformed_required_reviewers_without_traceback(tmp_path: Path) -> None:
    baseline, candidate = _approved_pair()
    baseline["candidate_protocol"]["required_reviewers"] = [{"role": "kernel owner"}, "Track E"]

    result = _run_compare(tmp_path, baseline, candidate, "--digest-gate", "sha256")

    assert result.returncode == 1
    assert "candidate_protocol must name kernel owner and Track E" in result.stderr
    assert "performance comparison not run" in result.stderr
    assert "Traceback" not in result.stderr


def test_digest_gate_blocks_unapproved_same_platform_baseline(tmp_path: Path) -> None:
    baseline = _payload(machine=platform.machine(), baseline=True, approval="pending")
    candidate = _payload(machine=platform.machine())

    result = _run_compare(tmp_path, baseline, candidate, "--digest-gate", "sha256")

    assert result.returncode == 1
    assert "baseline approval is pending" in result.stderr
    assert "performance comparison not run" in result.stderr


def test_digest_gate_skips_cross_platform_hard_comparison(tmp_path: Path) -> None:
    other_machine = "x86_64" if platform.machine() != "x86_64" else "arm64"
    baseline, _candidate = _approved_pair(machine=other_machine)
    candidate = _payload(machine=platform.machine(), wall=100.0, rss=10_000)

    result = _run_compare(tmp_path, baseline, candidate, "--digest-gate", "sha256")

    assert result.returncode == 0
    assert "platform mismatch" in result.stdout
    assert "artifact only" in result.stdout


def test_digest_gate_rejects_wall_time_regression_above_25_percent(tmp_path: Path) -> None:
    baseline, candidate = _approved_pair()
    for case in candidate["cases"]:
        case["wall_seconds"] = 1.26

    result = _run_compare(tmp_path, baseline, candidate, "--digest-gate", "sha256")

    assert result.returncode == 1
    assert "wall_seconds regressed" in result.stderr
    assert "peak_rss_bytes regressed" not in result.stderr


def test_digest_gate_rejects_peak_rss_regression_above_25_percent(tmp_path: Path) -> None:
    baseline, candidate = _approved_pair()
    for case in candidate["cases"]:
        case["peak_rss_bytes"] = 126
        case["rss_delta_bytes"] = 125

    result = _run_compare(tmp_path, baseline, candidate, "--digest-gate", "sha256")

    assert result.returncode == 1
    assert "peak_rss_bytes regressed" in result.stderr
    assert "wall_seconds regressed" not in result.stderr


def test_digest_gate_accepts_time_and_rss_at_exactly_25_percent(tmp_path: Path) -> None:
    baseline, candidate = _approved_pair()
    for case in candidate["cases"]:
        case["wall_seconds"] = 1.25
        case["peak_rss_bytes"] = 125
        case["rss_delta_bytes"] = 124

    result = _run_compare(tmp_path, baseline, candidate, "--digest-gate", "sha256")

    assert result.returncode == 0
    assert "all factor-analysis digest, shape, time, and RSS gates passed" in result.stdout


def test_ci_workflow_schedules_non_xdist_medium_and_event_artifacts() -> None:
    workflow = WORKFLOW.read_text(encoding="utf-8")
    artifact_job = workflow.split("  factor-benchmark-artifacts:", 1)[1].split("\n  # Release build gate", 1)[0]

    assert "factor-benchmark-artifacts:" in workflow
    assert "--scenarios event" in workflow
    assert "--scenarios medium-artifact" in workflow
    assert "factor-event.json" in workflow
    assert "factor-medium.json" in workflow
    assert "tests/benchmarks/test_factor_analysis_performance.py" in workflow
    assert "-n" not in artifact_job


def test_factor_payload_without_digest_gate_keeps_legacy_unknown_kind_behavior(tmp_path: Path) -> None:
    baseline, candidate = _approved_pair()

    result = _run_compare(tmp_path, baseline, candidate)

    assert result.returncode == 1
    assert "unknown candidate kind 'factor_analysis'" in result.stderr


def test_selects_only_an_approved_matching_platform_baseline(tmp_path: Path) -> None:
    approved = _payload(machine="x86_64", baseline=True)
    approved["provenance"]["platform_label"] = "linux-x86_64"
    pending = _payload(machine="x86_64", baseline=True, approval="pending")
    pending["provenance"]["platform_label"] = "linux-x86_64"
    other = _payload(machine="arm64", baseline=True)
    other["provenance"]["platform_label"] = "darwin-arm64"

    (tmp_path / "linux-x86_64.json").write_text(json.dumps(approved), encoding="utf-8")
    (tmp_path / "linux-x86_64-pending.json").write_text(json.dumps(pending), encoding="utf-8")
    (tmp_path / "darwin-arm64.json").write_text(json.dumps(other), encoding="utf-8")

    baseline = select_baseline(tmp_path, platform_label="linux-x86_64")

    assert baseline is not None
    assert baseline["approval"]["status"] == "approved"
    assert baseline["provenance"]["platform_label"] == "linux-x86_64"


def test_select_baseline_returns_none_without_an_approved_match(tmp_path: Path) -> None:
    pending = _payload(machine="x86_64", baseline=True, approval="pending")
    pending["provenance"]["platform_label"] = "linux-x86_64"
    (tmp_path / "linux-x86_64.json").write_text(json.dumps(pending), encoding="utf-8")

    assert select_baseline(tmp_path, platform_label="linux-x86_64") is None
    assert select_baseline(tmp_path, platform_label="darwin-arm64") is None


def test_candidate_only_baseline_is_not_a_release_gate() -> None:
    candidates = list_candidate_baselines()
    approved = [c for c in candidates if c.get("approval", {}).get("status") == "approved"]
    unapproved = [c for c in candidates if c.get("approval", {}).get("status") != "approved"]

    assert candidates, "no checked-in factor baselines found"
    assert len(unapproved) > 0
    assert len(approved) == 0 or all(a["provenance"]["platform_label"] for a in approved)


def test_checked_in_baselines_are_platform_labelled() -> None:
    baselines = list_candidate_baselines()

    labels = {b["provenance"]["platform_label"] for b in baselines}

    assert "darwin-arm64" in labels
    assert "linux-x86_64" in labels
