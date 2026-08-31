"""Contract tests for the reproducible workload-profile orchestrator."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(os.environ.get("FINCORE_0042R2_SOURCE_ROOT", Path(__file__).resolve().parents[2])).resolve()


def _run_profile_workloads(output: Path, *arguments: str) -> subprocess.CompletedProcess[str]:
    """Run the public profiler CLI against one caller-owned artifact path."""

    return subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "profile_workloads.py"),
            *arguments,
            "--output",
            str(output),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=900,
    )


def test_profile_workloads_records_stable_semantic_digests_for_all_families(tmp_path: Path) -> None:
    output = tmp_path / "profiles.json"
    result = _run_profile_workloads(
        output,
        "--sizes",
        "small",
        "--kinds",
        "metrics",
        "rolling",
        "transactions",
        "factor",
        "risk",
        "report",
        "--warmups",
        "1",
        "--repeats",
        "2",
        "--require-output-digest",
    )

    assert result.returncode == 0, result.stderr
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["schema"] == "fincore-workload-profiles-v2"
    assert payload["measurement"] == {
        "warmups": 1,
        "repeats": 2,
        "require_output_digest": True,
        "timing_unit": "seconds",
        "percentile_method": "linear",
    }
    assert [case["kind"] for case in payload["cases"]] == [
        "metrics",
        "rolling",
        "transactions",
        "factor",
        "risk",
        "report",
    ]

    for case in payload["cases"]:
        assert len(case["workload"]["input_digest"]) == 64
        assert len(case["execution_input_digest"]) == 64
        assert len(case["output_digest"]) == 64
        assert case["warmup_output_digests"] == [case["output_digest"]]
        assert case["measured_output_digests"] == [case["output_digest"], case["output_digest"]]
        assert case["profiled_output_digest"] == case["output_digest"]
        assert len(case["timing_samples_seconds"]) == 2
        assert all(sample > 0.0 for sample in case["timing_samples_seconds"])
        assert case["timing"]["minimum_seconds"] > 0.0
        assert case["timing"]["minimum_seconds"] <= case["timing"]["median_seconds"]
        assert case["timing"]["median_seconds"] <= case["timing"]["p95_seconds"]
        assert case["timing"]["p95_seconds"] <= case["timing"]["maximum_seconds"]
        assert "wall_seconds" not in case
        assert case["peak_rss_bytes"] > 0
        assert case["hotspots"]


def test_profile_workloads_reproducibly_digests_the_same_financial_output(tmp_path: Path) -> None:
    first_output = tmp_path / "first.json"
    second_output = tmp_path / "second.json"
    arguments = (
        "--sizes",
        "small",
        "--kinds",
        "metrics",
        "--warmups",
        "0",
        "--repeats",
        "1",
        "--require-output-digest",
    )

    first = _run_profile_workloads(first_output, *arguments)
    second = _run_profile_workloads(second_output, *arguments)

    assert first.returncode == 0, first.stderr
    assert second.returncode == 0, second.stderr
    first_case = json.loads(first_output.read_text(encoding="utf-8"))["cases"][0]
    second_case = json.loads(second_output.read_text(encoding="utf-8"))["cases"][0]
    assert first_case["workload"]["input_digest"] == second_case["workload"]["input_digest"]
    assert first_case["execution_input_digest"] == second_case["execution_input_digest"]
    assert first_case["output_digest"] == second_case["output_digest"]


def test_profile_workloads_rejects_nonpositive_repeat_count(tmp_path: Path) -> None:
    output = tmp_path / "profiles.json"
    result = _run_profile_workloads(output, "--repeats", "0")

    assert result.returncode != 0
    assert "--repeats" in result.stderr
    assert not output.exists()
