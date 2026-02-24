"""Performance benchmarks for fincore core metrics.

Run with:
    python benchmarks/bench_metrics.py

Or via pytest-benchmark:
    pytest tests/benchmarks/ --benchmark-only
"""

from __future__ import annotations

import time

import numpy as np
import pandas as pd


def _make_data(n: int = 2520) -> tuple[pd.Series, pd.Series]:
    np.random.seed(42)
    dates = pd.bdate_range("2020-01-01", periods=n)
    returns = pd.Series(np.random.normal(0.0005, 0.02, n), index=dates)
    benchmark = pd.Series(np.random.normal(0.0003, 0.015, n), index=dates)
    return returns, benchmark


def bench(name: str, fn, *, n_iter: int = 100) -> dict:
    """Benchmark a function and return timing stats."""
    fn()  # warmup
    times = []
    for _ in range(n_iter):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    arr = np.array(times) * 1000  # ms
    return {
        "name": name,
        "mean_ms": float(np.mean(arr)),
        "median_ms": float(np.median(arr)),
        "std_ms": float(np.std(arr)),
        "min_ms": float(np.min(arr)),
        "max_ms": float(np.max(arr)),
        "ops_per_sec": float(1000 / np.mean(arr)),
    }


def run_benchmarks() -> list[dict]:
    returns, benchmark = _make_data()

    from fincore.metrics import alpha_beta, drawdown, ratios, risk, rolling
    from fincore.metrics import returns as ret_mod

    cases = [
        ("sharpe_ratio", lambda: ratios.sharpe_ratio(returns)),
        ("sortino_ratio", lambda: ratios.sortino_ratio(returns)),
        ("calmar_ratio", lambda: ratios.calmar_ratio(returns)),
        ("max_drawdown", lambda: drawdown.max_drawdown(returns)),
        ("annual_volatility", lambda: risk.annual_volatility(returns)),
        ("cum_returns_final", lambda: ret_mod.cum_returns_final(returns)),
        ("alpha_beta", lambda: alpha_beta.alpha_beta(returns, benchmark)),
        ("roll_sharpe(w=252)", lambda: rolling.roll_sharpe_ratio(returns, window=252)),
        ("roll_max_drawdown(w=252)", lambda: rolling.roll_max_drawdown(returns, window=252)),
        ("roll_beta(w=252)", lambda: rolling.roll_beta(returns, benchmark, window=252)),
    ]

    results = []
    for name, fn in cases:
        r = bench(name, fn)
        results.append(r)

    return results


def main():
    results = run_benchmarks()

    print(f"{'Function':<30} {'Mean(ms)':>10} {'Median':>10} {'Std':>8} {'Ops/s':>10}")
    print("-" * 72)
    for r in results:
        print(
            f"{r['name']:<30} {r['mean_ms']:>10.3f} {r['median_ms']:>10.3f} "
            f"{r['std_ms']:>8.3f} {r['ops_per_sec']:>10.0f}"
        )

    # Write JSON for CI tracking
    import json
    from pathlib import Path

    out = Path(__file__).parent / "results.json"
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out}")


if __name__ == "__main__":
    main()
