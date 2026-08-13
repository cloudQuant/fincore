"""Runner schema tests for the fresh-subprocess rolling benchmark payload.

``scripts/run_rolling_benchmarks.py`` — not pytest-benchmark — produces
the JSON these tests consume.  The payload must carry environment
provenance (commit/python/numpy/pandas) plus per-case input_size,
window, wall time, and RSS figures normalised to bytes, so downstream
comparisons never guess units or provenance.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

SCRIPTS_DIR = Path(__file__).resolve().parents[2] / "scripts"

pytestmark = pytest.mark.skipif(sys.platform == "win32", reason="runner uses resource.getrusage")


@pytest.fixture(scope="module")
def benchmark_payload(tmp_path_factory: pytest.TempPathFactory) -> dict:
    output = tmp_path_factory.mktemp("rolling-bench") / "payload.json"
    subprocess.run(
        [
            sys.executable,
            str(SCRIPTS_DIR / "run_rolling_benchmarks.py"),
            "--sizes",
            "63",
            "--windows",
            "21",
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
    data = json.loads(output.read_text(encoding="utf-8"))
    first = data["cases"][0]
    return {**data["provenance"], **first}


def test_subprocess_benchmark_payload_has_provenance(benchmark_payload: dict) -> None:
    assert benchmark_payload["commit"]
    assert benchmark_payload["python"]
    assert benchmark_payload["numpy"]
    assert benchmark_payload["pandas"]
    assert benchmark_payload["input_size"]
    assert benchmark_payload["window"]
    assert benchmark_payload["wall_seconds"] > 0
    assert benchmark_payload["rss_before_bytes"] > 0
    assert benchmark_payload["peak_rss_bytes"] > 0
    assert benchmark_payload["rss_delta_bytes"] >= 0
    assert benchmark_payload["tracemalloc_peak_bytes"] > 0


def test_runner_covers_every_metric_case(tmp_path: Path) -> None:
    output = tmp_path / "payload.json"
    subprocess.run(
        [
            sys.executable,
            str(SCRIPTS_DIR / "run_rolling_benchmarks.py"),
            "--sizes",
            "63",
            "--windows",
            "21",
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
    data = json.loads(output.read_text(encoding="utf-8"))
    metrics = {case["metric"] for case in data["cases"]}
    assert metrics == {
        "sharpe",
        "volatility",
        "sortino",
        "max_drawdown",
        "beta",
        "mean_return",
        "roll_alpha",
        "roll_alpha_beta",
        "engine_all",
    }
    assert data["rss_unit"] == "bytes"
