"""Quantity-deque scaling gates for round-trip extraction.

The round-trip engine stores quantity blocks in a deque — one node per
transaction block, never one node per share — so with a fixed number of
transaction rows the produced PnL scales by the share amount (by
definition) while the queue node count, wall time, and RSS must NOT
scale with the share count.  These tests lock that property so a future
regression to upstream's expand-every-share behaviour is caught.
"""

from __future__ import annotations

import json
import os
import statistics
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from fincore.portfolio.round_trips import extract_round_trips

SCRIPTS_DIR = Path(os.environ.get("FINCORE_0042R2_SOURCE_ROOT", Path(__file__).resolve().parents[2])).resolve() / "scripts"
MIB = 1024 * 1024
AMOUNT_SCALE = 1_000_000


def _transactions(amount: float, rows: int, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    prices = rng.uniform(50.0, 150.0, rows)
    amounts = np.where(np.arange(rows) % 2 == 0, amount, -amount)
    index = pd.bdate_range("2024-01-02", periods=rows)
    return pd.DataFrame({"symbol": "AAA", "amount": amounts, "price": prices}, index=index)


def test_pnl_scales_by_share_amount_but_round_trip_count_does_not() -> None:
    small = extract_round_trips(_transactions(10.0, 100))
    large = extract_round_trips(_transactions(10.0 * AMOUNT_SCALE, 100))

    # The queue keeps one node per transaction block, so a 1e6x share
    # scale-up must not change the number of matched round trips.
    assert len(large) == len(small) > 0
    np.testing.assert_allclose(large["pnl"].to_numpy(), small["pnl"].to_numpy() * AMOUNT_SCALE, rtol=1e-12)
    np.testing.assert_allclose(large["rt_returns"].to_numpy(), small["rt_returns"].to_numpy(), rtol=1e-12)


@pytest.mark.skipif(sys.platform == "win32", reason="runner uses resource.getrusage")
def test_fresh_process_wall_time_and_rss_do_not_scale_with_share_amount(tmp_path: Path) -> None:
    output = tmp_path / "payload.json"
    subprocess.run(
        [
            sys.executable,
            str(SCRIPTS_DIR / "run_round_trip_benchmarks.py"),
            "--amounts",
            "10",
            str(10.0 * AMOUNT_SCALE),
            "--rows",
            "100",
            "--repeats",
            "5",
            "--output",
            str(output),
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=600,
    )
    data = json.loads(output.read_text(encoding="utf-8"))
    by_amount: dict[float, list[dict]] = {}
    for case in data["cases"]:
        by_amount.setdefault(float(case["amount"]), []).append(case)

    small = by_amount[10.0]
    large = by_amount[10.0 * AMOUNT_SCALE]
    small_wall = statistics.median(c["wall_seconds"] for c in small)
    large_wall = statistics.median(c["wall_seconds"] for c in large)
    small_rss = statistics.median(c["rss_delta_bytes"] for c in small)
    large_rss = statistics.median(c["rss_delta_bytes"] for c in large)

    assert {int(c["round_trip_count"]) for c in small} == {int(c["round_trip_count"]) for c in large}

    assert large_rss <= max(1.25 * small_rss, small_rss + 32 * MIB), (
        f"rss_delta scaled with share amount: {small_rss} -> {large_rss} bytes"
    )
    assert large_wall <= max(1.25 * small_wall, small_wall + 0.005), (
        f"wall time scaled with share amount: {small_wall:.4f}s -> {large_wall:.4f}s"
    )
