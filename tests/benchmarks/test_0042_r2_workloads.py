"""Pre-registered 0042-R2 hotspot workload evidence tests.

The plan pre-registers six hotspot families (rolling metrics, portfolio
transactions/round trips, factor preparation/IC, risk forecasts, report
compute/render) and requires every benchmark to verify output digests before
timing with at least two warmups plus five measured repeats.  These tests run
the frozen orchestrator at exactly that contract and audit the resulting
evidence.  Formal regression and improvement gates remain with the detached
D0_TOOLING_SHA acceptance runner.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

REPOSITORY_ROOT = Path(os.environ.get("FINCORE_0042R2_SOURCE_ROOT", Path(__file__).parents[2])).resolve()
SCRIPT = REPOSITORY_ROOT / "scripts" / "profile_workloads.py"
R2_KINDS = ("metrics", "rolling", "transactions", "factor", "risk", "report")
GIT_OBJECT_ID = re.compile(r"^[0-9a-f]{40}$")
SHA256 = re.compile(r"^[0-9a-f]{64}$")


def _profile(output: Path, *extra: str) -> dict:
    result = subprocess.run(
        [
            sys.executable,
            "-I",
            str(SCRIPT),
            "--sizes",
            "small",
            "--kinds",
            *R2_KINDS,
            "--warmups",
            "2",
            "--repeats",
            "5",
            "--require-output-digest",
            "--output",
            str(output),
            *extra,
        ],
        cwd=REPOSITORY_ROOT,
        capture_output=True,
        text=True,
        check=False,
        timeout=900,
    )
    assert result.returncode == 0, result.stderr or result.stdout
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    return payload


@pytest.mark.slow
def test_preregistered_hotspots_run_with_the_plan_measurement_contract(tmp_path: Path) -> None:
    payload = _profile(tmp_path / "r2-hotspots.json")

    assert payload["schema"] == "fincore-workload-profiles-v2"
    assert payload["measurement"] == {
        "warmups": 2,
        "repeats": 5,
        "require_output_digest": True,
        "timing_unit": "seconds",
        "percentile_method": "linear",
    }
    cases = payload["cases"]
    assert sorted(case["kind"] for case in cases) == sorted(R2_KINDS)

    for case in cases:
        kind = case["kind"]
        assert case["measurement"]["warmups"] == 2, kind
        assert case["measurement"]["repeats"] == 5, kind
        assert case["measurement"]["require_output_digest"] is True, kind
        assert SHA256.fullmatch(case["output_digest"]), kind
        assert len(case["warmup_output_digests"]) == 2, kind
        assert len(case["measured_output_digests"]) == 5, kind
        assert all(digest == case["output_digest"] for digest in case["measured_output_digests"]), kind
        assert len(case["timing_samples_seconds"]) == 5, kind
        assert all(sample > 0 for sample in case["timing_samples_seconds"]), kind
        assert set(case["timing"]) == {
            "minimum_seconds",
            "median_seconds",
            "p95_seconds",
            "maximum_seconds",
        }, kind
        assert isinstance(case["peak_rss_bytes"], int) and case["peak_rss_bytes"] > 0, kind


@pytest.mark.slow
def test_hotspot_evidence_records_git_and_platform_provenance(tmp_path: Path) -> None:
    payload = _profile(tmp_path / "r2-provenance.json")

    for case in payload["cases"]:
        provenance = case["provenance"]
        kind = case["kind"]
        assert GIT_OBJECT_ID.fullmatch(provenance["commit"]), kind
        assert isinstance(provenance["dirty"], bool), kind
        assert provenance["python"].strip(), kind
        assert provenance["platform_label"].strip(), kind
        assert provenance["numpy"].strip(), kind
        assert provenance["pandas"].strip(), kind
        assert SHA256.fullmatch(case["workload"]["input_digest"]), kind


@pytest.mark.slow
def test_hotspot_output_digests_are_reproducible_across_runs(tmp_path: Path) -> None:
    first = _profile(tmp_path / "r2-first.json")
    second = _profile(tmp_path / "r2-second.json")

    first_digests = {case["kind"]: case["output_digest"] for case in first["cases"]}
    second_digests = {case["kind"]: case["output_digest"] for case in second["cases"]}
    assert first_digests == second_digests
