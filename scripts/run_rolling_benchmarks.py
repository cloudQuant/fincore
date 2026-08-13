#!/usr/bin/env python
"""Fresh-subprocess rolling-metric benchmark runner.

Every ``metric x input_size x window`` case is measured in a NEW Python
interpreter so wall time and peak RSS are not polluted by earlier cases.
Each payload records provenance (commit/python/numpy/pandas), the RSS
baseline taken AFTER imports and input construction, the peak RSS read
from ``resource.getrusage`` and normalised to bytes (macOS reports bytes,
Linux reports KiB), and the ``tracemalloc`` peak over the workload.

This runner produces the JSON schema consumed by
``scripts/compare_benchmarks.py`` and by
``tests/benchmarks/test_rolling_regression.py`` — it does not depend on
pytest-benchmark.

Usage::

    python scripts/run_rolling_benchmarks.py \\
        --sizes 2520 25200 --windows 21 63 252 504 \\
        --repeats 7 --output /tmp/fincore-rolling.json
"""

from __future__ import annotations

import argparse
import json
import platform
import resource
import subprocess
import sys
import time
import tracemalloc
from pathlib import Path

_ROOT = str(Path(__file__).resolve().parent.parent)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

ENGINE_METRICS = ("sharpe", "volatility", "sortino", "max_drawdown", "beta", "mean_return")
BINARY_ROLL_METRICS = ("roll_alpha", "roll_alpha_beta")
BATCH_METRICS = ("engine_all",)
METRICS = ENGINE_METRICS + BINARY_ROLL_METRICS + BATCH_METRICS
RSS_UNIT = "bytes"

_KNOWN_RSS_UNITS = {"bytes"}


def rss_bytes() -> int:
    """Current process peak RSS in bytes, normalised across platforms.

    ``resource.getrusage`` reports bytes on macOS and KiB on Linux; the
    unit is normalised here and the payload always records
    ``rss_unit == "bytes"`` so downstream consumers never guess.
    """
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if sys.platform == "darwin":
        return int(peak)
    return int(peak) * 1024


def _build_inputs(size: int):
    import numpy as np
    import pandas as pd

    rng = np.random.default_rng(20240101 + size)
    index = pd.bdate_range("2000-01-03", periods=size)
    returns = pd.Series(rng.normal(0.0, 0.01, size), index=index)
    factor_returns = pd.Series(rng.normal(0.0, 0.008, size), index=index)
    return returns, factor_returns


def run_case(metric: str, size: int, window: int) -> dict:
    """Measure one workload in this interpreter; return the case payload.

    All imports (fincore included) happen before the RSS baseline so
    import cost never pollutes the workload measurement.
    """
    from fincore.core.engine import RollingEngine
    from fincore.metrics import rolling as rolling_module

    returns, factor_returns = _build_inputs(size)
    rss_before = rss_bytes()
    tracemalloc.start()
    start = time.perf_counter()
    if metric == "roll_alpha":
        rolling_module.roll_alpha(returns, factor_returns, window=window)
    elif metric == "roll_alpha_beta":
        rolling_module.roll_alpha_beta(returns, factor_returns, window=window)
    elif metric == "engine_all":
        RollingEngine(returns, factor_returns=factor_returns, window=window).compute("all")
    else:
        RollingEngine(returns, factor_returns=factor_returns, window=window).compute([metric])
    elapsed = time.perf_counter() - start
    _current, tracemalloc_peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    peak_rss = rss_bytes()
    return {
        "metric": metric,
        "input_size": size,
        "window": window,
        "wall_seconds": elapsed,
        "rss_before_bytes": rss_before,
        "peak_rss_bytes": peak_rss,
        "rss_delta_bytes": max(peak_rss - rss_before, 0),
        "tracemalloc_peak_bytes": tracemalloc_peak,
        "rss_unit": RSS_UNIT,
    }


def _provenance() -> dict:
    import numpy as np
    import pandas as pd

    commit = subprocess.run(
        ["git", "-C", _ROOT, "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    return {
        "commit": commit,
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "platform": sys.platform,
        "machine": platform.machine(),
    }


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sizes", nargs="+", type=int, default=[])
    parser.add_argument("--windows", nargs="+", type=int, default=[])
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--output", default="")
    parser.add_argument("--metrics", nargs="*", choices=METRICS, default=list(METRICS))
    # Internal: each --case spawns one fresh interpreter for one workload.
    parser.add_argument("--case", nargs=3, metavar=("METRIC", "SIZE", "WINDOW"), help=argparse.SUPPRESS)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.case is not None:
        metric, size, window = args.case
        print(json.dumps(run_case(metric, int(size), int(window))))
        return 0
    if not args.sizes or not args.windows or not args.output:
        _parse_args(["--help"])
        return 2

    payload = {
        "schema": "fincore-rolling-benchmarks-v1",
        "kind": "rolling",
        "rss_unit": RSS_UNIT,
        "known_rss_units": sorted(_KNOWN_RSS_UNITS),
        "provenance": _provenance(),
        "cases": [],
    }
    cases = payload["cases"]
    total = len(args.metrics) * len(args.sizes) * len(args.windows) * args.repeats
    done = 0
    for metric in args.metrics:
        for size in args.sizes:
            for window in args.windows:
                for repeat in range(args.repeats):
                    result = subprocess.run(
                        [sys.executable, str(Path(__file__).resolve()), "--case", metric, str(size), str(window)],
                        capture_output=True,
                        text=True,
                        check=True,
                        cwd=_ROOT,
                    )
                    case = json.loads(result.stdout)
                    case["repeat"] = repeat
                    cases.append(case)
                    done += 1
                    print(
                        f"[{done}/{total}] {metric} n={size} w={window} r={repeat}: "
                        f"{case['wall_seconds']:.4f}s rss_delta={case['rss_delta_bytes']}"
                    )
    with Path(args.output).open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
    print(f"wrote {len(cases)} cases to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
