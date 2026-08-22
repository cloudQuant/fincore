"""Contract tests for the multi-scale workload-profile orchestrator."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def test_profile_workloads_writes_semantic_evidence(tmp_path: Path) -> None:
    output = tmp_path / "profiles.json"
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "profile_workloads.py"),
            "--sizes",
            "small",
            "--kinds",
            "metrics",
            "--output",
            str(output),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=900,
    )

    assert result.returncode == 0, result.stderr
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["schema"] == "fincore-workload-profiles-v1"
    assert len(payload["cases"]) == 1
    case = payload["cases"][0]
    assert case["workload"]["input_digest"]
    assert case["wall_seconds"] > 0.0
    assert case["peak_rss_bytes"] > 0
