#!/usr/bin/env python3
"""Fail-closed performance budget gate.

Runs the fixed-overhead micro-benchmarks (catalog dispatch, DAG executor,
snapshot construction) and fails when their p95 overhead exceeds the
documented budget in ``docs/quality/performance-budget.md``.  This is the
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

from fincore.api import build_builtin_catalog
from fincore.api.invoke import invoke

DISPATCH_BUDGET_SECONDS = 500e-6
DAG_BUDGET_SECONDS = 1e-3
SNAPSHOT_BUDGET_SECONDS = 10e-3


def _p95(values: list[float]) -> float:
    return float(np.percentile(values, 95))


def _benchmark_dispatch(n: int = 1000) -> float:
    catalog = build_builtin_catalog()
    returns = pd.Series([0.01, -0.005, 0.002, 0.004])
    timings: list[float] = []
    for _ in range(n):
        start = time.perf_counter()
        invoke(catalog, "sharpe_ratio", "enhanced_v1", returns)
        timings.append(time.perf_counter() - start)
    return _p95(timings)


def _benchmark_dag(n: int = 500) -> float:
    from fincore.core.execution import DAGExecutor
    from fincore.core.planner import DAGNode, ExecutionPlan

    def identity(**kwargs: object) -> float:
        return 1.0

    plan = ExecutionPlan(
        (
            DAGNode("a", cache_key="k", compute=identity),
            DAGNode("b", dependencies=("a",), cache_key="k", compute=identity),
            DAGNode("c", dependencies=("b",), cache_key="k", compute=identity),
        )
    )
    timings: list[float] = []
    for _ in range(n):
        executor = DAGExecutor(plan)
        start = time.perf_counter()
        executor.execute("c")
        timings.append(time.perf_counter() - start)
    return _p95(timings)


def _benchmark_snapshot(n: int = 200) -> float:
    from fincore.core.snapshot import AnalysisSnapshot

    rng = np.random.default_rng(20260830)
    index = pd.date_range("2020-01-01", periods=1260, freq="B")
    returns = pd.Series(rng.normal(0.0005, 0.01, size=len(index)), index=index)
    benchmark = pd.Series(rng.normal(0.0004, 0.01, size=len(index)), index=index)
    timings: list[float] = []
    for _ in range(n):
        start = time.perf_counter()
        AnalysisSnapshot.from_data(returns, benchmark=benchmark)
        timings.append(time.perf_counter() - start)
    return _p95(timings)


def main() -> int:
    violations: list[str] = []
    dispatch_p95 = _benchmark_dispatch()
    print(f"catalog dispatch p95: {dispatch_p95 * 1e6:.1f} µs (budget {DISPATCH_BUDGET_SECONDS * 1e6:.0f} µs)")
    if dispatch_p95 > DISPATCH_BUDGET_SECONDS:
        violations.append(f"catalog dispatch p95 {dispatch_p95 * 1e6:.1f} µs exceeds budget")

    dag_p95 = _benchmark_dag()
    print(f"DAG executor p95: {dag_p95 * 1e6:.1f} µs (budget {DAG_BUDGET_SECONDS * 1e6:.0f} µs)")
    if dag_p95 > DAG_BUDGET_SECONDS:
        violations.append(f"DAG executor p95 {dag_p95 * 1e6:.1f} µs exceeds budget")

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
