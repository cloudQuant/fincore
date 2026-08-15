#!/usr/bin/env python
"""Run reproducible enhanced factor-analysis benchmark scenarios.

The default is the bounded ``small-ci`` scenario.  ``medium-artifact`` and
``event`` are explicit artifact jobs and never silently expand the CI workload.
Each measured repeat runs in a fresh subprocess after its requested warmups.
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

ROOT = Path(__file__).resolve().parent.parent
BENCHMARKS = ROOT / "benchmarks"
for path in (str(ROOT), str(BENCHMARKS)):
    if path not in sys.path:
        sys.path.insert(0, path)

from bench_factor_analysis import SCENARIOS, SEED, build_workload, output_metadata

RSS_UNIT = "bytes"


def rss_bytes() -> int:
    """Return peak process RSS normalized to bytes on Darwin and Linux."""

    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(peak) if sys.platform == "darwin" else int(peak) * 1024


def _git(*args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(ROOT), *args],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def provenance() -> dict:
    """Capture interpreter, dependency, platform, commit, and dirty state."""

    import numpy as np
    import pandas as pd
    import scipy
    import statsmodels

    return {
        "commit": _git("rev-parse", "HEAD"),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scipy": scipy.__version__,
        "statsmodels": statsmodels.__version__,
        "os": sys.platform,
        "arch": platform.machine(),
        "platform_label": f"{sys.platform}-{platform.machine()}",
        "dirty": bool(_git("status", "--porcelain")),
    }


def run_case(scenario_name: str, kernel: str, warmups: int, repeat: int, repeats: int) -> dict:
    """Measure one kernel after input construction and deterministic warmups."""

    workload = build_workload(scenario_name, kernel)
    for _ in range(warmups):
        workload()

    rss_before = rss_bytes()
    tracemalloc.start()
    started = time.perf_counter()
    output = workload()
    wall_seconds = time.perf_counter() - started
    _current, tracemalloc_peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    peak_rss = rss_bytes()
    output_shape, output_digest = output_metadata(output)
    scenario = SCENARIOS[scenario_name]
    return {
        "scenario": scenario_name,
        "kernel": kernel,
        "input_shape": scenario.input_shape,
        "output_shape": output_shape,
        "seed": SEED,
        "wall_seconds": wall_seconds,
        "rss_before_bytes": rss_before,
        "peak_rss_bytes": peak_rss,
        "rss_delta_bytes": max(peak_rss - rss_before, 0),
        "tracemalloc_peak_bytes": tracemalloc_peak,
        "rss_unit": RSS_UNIT,
        "output_digest": output_digest,
        "warmup": warmups,
        "repeat": repeat,
        "repeats": repeats,
    }


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenarios", nargs="+", choices=tuple(SCENARIOS), default=["small-ci"])
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--case",
        nargs=5,
        metavar=("SCENARIO", "KERNEL", "WARMUPS", "REPEAT", "REPEATS"),
        help=argparse.SUPPRESS,
    )
    args = parser.parse_args(argv)
    if args.warmups < 0 or args.repeats < 1:
        parser.error("--warmups must be >= 0 and --repeats must be >= 1")
    return args


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.case is not None:
        scenario, kernel, warmups, repeat, repeats = args.case
        print(json.dumps(run_case(scenario, kernel, int(warmups), int(repeat), int(repeats)), sort_keys=True))
        return 0

    payload = {
        "schema": "fincore-factor-analysis-benchmarks-v1",
        "kind": "factor_analysis",
        "rss_unit": RSS_UNIT,
        "provenance": provenance(),
        "runner": {"warmups": args.warmups, "repeats": args.repeats},
        "cases": [],
    }
    total = sum(len(SCENARIOS[name].kernels) for name in args.scenarios) * args.repeats
    done = 0
    for scenario_name in args.scenarios:
        for kernel in SCENARIOS[scenario_name].kernels:
            for repeat in range(args.repeats):
                command = [
                    sys.executable,
                    str(Path(__file__).resolve()),
                    "--output",
                    args.output,
                    "--case",
                    scenario_name,
                    kernel,
                    str(args.warmups),
                    str(repeat),
                    str(args.repeats),
                ]
                result = subprocess.run(command, check=True, capture_output=True, text=True, cwd=ROOT)
                case = json.loads(result.stdout)
                payload["cases"].append(case)
                done += 1
                print(
                    f"[{done}/{total}] {scenario_name}/{kernel} r={repeat}: "
                    f"{case['wall_seconds']:.4f}s peak_rss={case['peak_rss_bytes']}"
                )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote {len(payload['cases'])} cases to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
