#!/usr/bin/env python
"""Fresh-subprocess round-trip extraction benchmark runner.

Each ``amount x rows`` case runs ``extract_round_trips`` in a NEW Python
interpreter and records wall time, RSS baseline/peak/delta (normalised to
bytes), the tracemalloc peak, and the number of round trips produced.

The quantity-deque implementation stores one node per quantity block, so
neither wall time, RSS delta, nor the produced round-trip count may scale
with the per-row share amount — only with the transaction row count.

Usage::

    python scripts/run_round_trip_benchmarks.py \\
        --amounts 10 10000000 --rows 100 10000 \\
        --repeats 7 --output /tmp/fincore-round-trips.json
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

RSS_UNIT = "bytes"


def rss_bytes() -> int:
    """Current process peak RSS in bytes, normalised across platforms."""
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if sys.platform == "darwin":
        return int(peak)
    return int(peak) * 1024


def _build_transactions(amount: float, rows: int):
    import numpy as np
    import pandas as pd

    rng = np.random.default_rng(987654 + rows)
    prices = rng.uniform(50.0, 150.0, rows)
    amounts = np.where(np.arange(rows) % 2 == 0, amount, -amount)
    index = pd.bdate_range("2020-01-06", periods=rows)
    return pd.DataFrame({"symbol": "AAA", "amount": amounts, "price": prices}, index=index)


def run_case(amount: float, rows: int) -> dict:
    """Measure one extraction workload in this interpreter."""
    from fincore.portfolio.round_trips import extract_round_trips

    transactions = _build_transactions(amount, rows)
    rss_before = rss_bytes()
    tracemalloc.start()
    start = time.perf_counter()
    round_trips = extract_round_trips(transactions)
    elapsed = time.perf_counter() - start
    _current, tracemalloc_peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    peak_rss = rss_bytes()
    return {
        "amount": amount,
        "rows": rows,
        "round_trip_count": len(round_trips),
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
    parser.add_argument("--amounts", nargs="+", type=float, default=[])
    parser.add_argument("--rows", nargs="+", type=int, default=[])
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--output", default="")
    # Internal: each --case spawns one fresh interpreter for one workload.
    parser.add_argument("--case", nargs=2, metavar=("AMOUNT", "ROWS"), help=argparse.SUPPRESS)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.case is not None:
        amount, rows = args.case
        print(json.dumps(run_case(float(amount), int(rows))))
        return 0
    if not args.amounts or not args.rows or not args.output:
        _parse_args(["--help"])
        return 2

    payload = {
        "schema": "fincore-round-trip-benchmarks-v1",
        "kind": "round_trips",
        "rss_unit": RSS_UNIT,
        "provenance": _provenance(),
        "cases": [],
    }
    cases = payload["cases"]
    total = len(args.amounts) * len(args.rows) * args.repeats
    done = 0
    for amount in args.amounts:
        for rows in args.rows:
            for repeat in range(args.repeats):
                result = subprocess.run(
                    [sys.executable, str(Path(__file__).resolve()), "--case", repr(amount), str(rows)],
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
                    f"[{done}/{total}] amount={amount} rows={rows} r={repeat}: "
                    f"{case['wall_seconds']:.4f}s rss_delta={case['rss_delta_bytes']} "
                    f"round_trips={case['round_trip_count']}"
                )
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with Path(args.output).open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
    print(f"wrote {len(cases)} cases to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
