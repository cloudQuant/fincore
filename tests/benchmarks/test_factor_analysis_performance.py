"""Task 11 gates for enhanced factor-analysis benchmark artifacts."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
RUNNER = ROOT / "scripts" / "run_factor_benchmarks.py"
COMPARATOR = ROOT / "scripts" / "compare_benchmarks.py"
BASELINE = ROOT / "benchmarks" / "factor-analysis-baseline.json"


def _payload(*, machine: str, digest: str = "a" * 64, wall: float = 1.0, rss: int = 100) -> dict:
    return {
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
        "approval": {
            "status": "approved",
            "approved_by": "kernel-owner",
            "approved_at": "2026-08-15T00:00:00Z",
            "reviewed_candidate_sha256": "b" * 64,
        },
        "cases": [
            {
                "scenario": "small-ci",
                "kernel": "prepare",
                "input_shape": {"dates": 252, "assets": 100, "rows": 25200},
                "output_shape": [25000, 5],
                "seed": 20260815,
                "wall_seconds": wall,
                "peak_rss_bytes": rss,
                "rss_before_bytes": 1,
                "rss_delta_bytes": rss - 1,
                "tracemalloc_peak_bytes": rss,
                "rss_unit": "bytes",
                "output_digest": digest,
                "warmup": 1,
                "repeat": 0,
                "repeats": 3,
            }
        ],
    }


def _run_compare(tmp_path: Path, baseline: dict, candidate: dict, *extra: str) -> subprocess.CompletedProcess[str]:
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
    assert payload["provenance"]["commit"]
    assert payload["provenance"]["python"]
    assert payload["provenance"]["numpy"]
    assert payload["provenance"]["pandas"]
    assert payload["provenance"]["scipy"]
    assert payload["provenance"]["statsmodels"]
    assert payload["provenance"]["os"]
    assert payload["provenance"]["arch"]
    assert isinstance(payload["provenance"]["dirty"], bool)

    assert {(case["scenario"], case["kernel"]) for case in payload["cases"]} == {
        ("small-ci", "prepare"),
        ("small-ci", "quantize"),
        ("small-ci", "information-coefficient"),
        ("small-ci", "weights"),
    }
    for case in payload["cases"]:
        assert case["input_shape"] == {"dates": 252, "assets": 100, "rows": 25200}
        assert case["seed"] == 20260815
        assert case["wall_seconds"] > 0
        assert case["peak_rss_bytes"] > 0
        assert case["rss_delta_bytes"] >= 0
        assert len(case["output_digest"]) == 64
        int(case["output_digest"], 16)
        assert case["output_shape"]
        assert case["warmup"] == 0
        assert case["repeat"] == 0
        assert case["repeats"] == 1

    artifact_sha = hashlib.sha256(output.read_bytes()).hexdigest()
    assert len(artifact_sha) == 64


def test_checked_in_darwin_arm64_baseline_is_transparently_pending() -> None:
    baseline = json.loads(BASELINE.read_text(encoding="utf-8"))
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


def test_digest_gate_rejects_output_mismatch_before_performance(tmp_path: Path) -> None:
    baseline = _payload(machine=platform.machine())
    candidate = _payload(machine=platform.machine(), digest="c" * 64, wall=100.0, rss=10_000)

    result = _run_compare(tmp_path, baseline, candidate, "--digest-gate", "sha256")

    assert result.returncode == 1
    assert "output_digest mismatch" in result.stderr
    assert "performance comparison not run" in result.stderr
    assert "wall_seconds regressed" not in result.stdout + result.stderr


def test_digest_gate_blocks_unapproved_same_platform_baseline(tmp_path: Path) -> None:
    baseline = _payload(machine=platform.machine())
    baseline["approval"] = {
        "status": "pending",
        "approved_by": None,
        "approved_at": None,
        "reviewed_candidate_sha256": None,
    }
    candidate = _payload(machine=platform.machine())

    result = _run_compare(tmp_path, baseline, candidate, "--digest-gate", "sha256")

    assert result.returncode == 1
    assert "baseline approval is pending" in result.stderr
    assert "performance comparison not run" in result.stderr


def test_digest_gate_skips_cross_platform_hard_comparison(tmp_path: Path) -> None:
    other_machine = "x86_64" if platform.machine() != "x86_64" else "arm64"
    baseline = _payload(machine=other_machine, wall=0.001, rss=2)
    candidate = _payload(machine=platform.machine(), wall=100.0, rss=10_000)

    result = _run_compare(tmp_path, baseline, candidate, "--digest-gate", "sha256")

    assert result.returncode == 0
    assert "platform mismatch" in result.stdout
    assert "artifact only" in result.stdout


def test_factor_payload_without_digest_gate_keeps_legacy_unknown_kind_behavior(tmp_path: Path) -> None:
    payload = _payload(machine=platform.machine())

    result = _run_compare(tmp_path, payload, payload)

    assert result.returncode == 1
    assert "unknown candidate kind 'factor_analysis'" in result.stderr
