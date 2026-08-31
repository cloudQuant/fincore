#!/usr/bin/env python3
"""Fail-closed performance budget gate for the canonical runtime.

Runs fixed-overhead micro-benchmarks for catalog resolution, request planning,
and snapshot construction.  The script fails when a p95 measurement exceeds
the documented budget in ``docs/quality/performance-budget.md``.  This is the
local, deterministic half of the performance gate; full multi-scale
benchmarks run in CI.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


# A direct script invocation otherwise resolves ``fincore`` from an editable,
# site, or PYTHONPATH-shadowed installation.  Performance evidence must use
# this source candidate.
def _is_checkout_path(entry: str) -> bool:
    try:
        return Path(entry or ".").resolve() == ROOT
    except (OSError, RuntimeError, ValueError):
        return False


sys.path[:] = [entry for entry in sys.path if not _is_checkout_path(entry)]
sys.path.insert(0, str(ROOT))

from fincore.runtime.builtins import builtin_catalog
from fincore.runtime.data import AnalysisSnapshot
from fincore.runtime.engine import OperationRequest, plan

CATALOG_RESOLUTION_BUDGET_SECONDS = 500e-6
REQUEST_PLANNING_BUDGET_SECONDS = 1e-3
SNAPSHOT_BUDGET_SECONDS = 10e-3
_SHARPE_OPERATION = "metrics.ratios.sharpe_ratio"


def _p95(values: list[float]) -> float:
    return float(np.percentile(values, 95))


def _benchmark_catalog_resolution(n: int = 1000) -> float:
    catalog = builtin_catalog()
    timings: list[float] = []
    for _ in range(n):
        start = time.perf_counter()
        catalog.resolve(_SHARPE_OPERATION)
        timings.append(time.perf_counter() - start)
    return _p95(timings)


def _benchmark_request_planning(n: int = 500) -> float:
    catalog = builtin_catalog()
    returns = pd.Series([0.01, -0.005, 0.002, 0.004])
    timings: list[float] = []
    for _ in range(n):
        start = time.perf_counter()
        request = OperationRequest(_SHARPE_OPERATION, {"returns": returns})
        plan((request,), catalog=catalog)
        timings.append(time.perf_counter() - start)
    return _p95(timings)


def _benchmark_snapshot(n: int = 200) -> float:
    rng = np.random.default_rng(20260830)
    index = pd.date_range("2020-01-01", periods=1260, freq="B")
    returns = pd.Series(rng.normal(0.0005, 0.01, size=len(index)), index=index)
    benchmark = pd.Series(rng.normal(0.0004, 0.01, size=len(index)), index=index)
    timings: list[float] = []
    for _ in range(n):
        start = time.perf_counter()
        AnalysisSnapshot.from_inputs({"returns": returns, "benchmark": benchmark})
        timings.append(time.perf_counter() - start)
    return _p95(timings)


def main() -> int:
    violations: list[str] = []
    catalog_resolution_p95 = _benchmark_catalog_resolution()
    print(
        "catalog resolution p95: "
        f"{catalog_resolution_p95 * 1e6:.1f} µs "
        f"(budget {CATALOG_RESOLUTION_BUDGET_SECONDS * 1e6:.0f} µs)"
    )
    if catalog_resolution_p95 > CATALOG_RESOLUTION_BUDGET_SECONDS:
        violations.append(f"catalog resolution p95 {catalog_resolution_p95 * 1e6:.1f} µs exceeds budget")

    request_planning_p95 = _benchmark_request_planning()
    print(
        "request planning p95: "
        f"{request_planning_p95 * 1e6:.1f} µs "
        f"(budget {REQUEST_PLANNING_BUDGET_SECONDS * 1e6:.0f} µs)"
    )
    if request_planning_p95 > REQUEST_PLANNING_BUDGET_SECONDS:
        violations.append(f"request planning p95 {request_planning_p95 * 1e6:.1f} µs exceeds budget")

    snapshot_p95 = _benchmark_snapshot()
    print(f"snapshot construction p95: {snapshot_p95 * 1e3:.2f} ms (budget {SNAPSHOT_BUDGET_SECONDS * 1e3:.0f} ms)")
    if snapshot_p95 > SNAPSHOT_BUDGET_SECONDS:
        violations.append(f"snapshot construction p95 {snapshot_p95 * 1e3:.2f} ms exceeds budget")

    if violations:
        for violation in violations:
            print(f"FAIL: {violation}")
        return 1
    print("performance budget is within limits.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
