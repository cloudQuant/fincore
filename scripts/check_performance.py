#!/usr/bin/env python3
"""Fail-closed performance budget gate.

Runs the fixed-overhead micro-benchmarks (catalog dispatch, DAG executor) and
fails when their p95 overhead exceeds the documented budget in
``docs/quality/performance-budget.md``.  This is the local, deterministic half
of the performance gate; full multi-scale benchmarks run in CI.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd

from fincore.api import build_builtin_catalog
from fincore.api.invoke import invoke

DISPATCH_BUDGET_SECONDS = 500e-6
DAG_BUDGET_SECONDS = 1e-3


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

    if violations:
        for violation in violations:
            print(f"FAIL: {violation}")
        return 1
    print("performance budget is within limits.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
