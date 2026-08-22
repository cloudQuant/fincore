#!/usr/bin/env python3
"""Profile deterministic workloads and emit machine-readable hotspots.

Runs a bounded workload under :mod:`cProfile` and writes a JSON report with the
top cumulative functions, plus a human Markdown summary beside it.  Cold-import
time and peak RSS are measured in a fresh subprocess so they are not polluted
by the profiler's own overhead.
"""

from __future__ import annotations

import argparse
import cProfile
import json
import platform
import pstats
import resource
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
BENCHMARKS = ROOT / "benchmarks"
for path in (str(ROOT), str(BENCHMARKS)):
    if path not in sys.path:
        sys.path.insert(0, path)

from workloads import describe_workload, factor_panel_workload, single_series_workload

TOP_FUNCTIONS = 15


def _rss_bytes() -> int:
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(peak) if sys.platform == "darwin" else int(peak) * 1024


def _cold_import_seconds() -> float:
    code = "import time; s=time.perf_counter(); import fincore; print(time.perf_counter()-s)"
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
        timeout=120,
    )
    return float(result.stdout.strip().splitlines()[-1])


def _profile_workload(workload) -> tuple[float, list[dict]]:
    def run():
        if workload.factor is not None:
            from fincore.factor_analysis.data import prepare_factor_data

            groups = pd.Series(
                {
                    asset: f"G{number % 10:02d}"
                    for number, asset in enumerate(workload.factor.index.unique(level="asset"))
                },
                name="group",
            )
            dates = workload.factor.index.get_level_values("date")
            assets = workload.factor.index.get_level_values("asset")
            rng = np.random.default_rng(workload.seed)
            innovations = rng.normal(0.0002, 0.01, (dates.nunique(), assets.nunique()))
            prices = pd.DataFrame(
                100.0 * np.exp(np.cumsum(innovations, axis=0)),
                index=pd.Index(dates.unique(), name="date"),
                columns=pd.Index(assets.unique(), name="asset"),
            )
            factor = workload.factor["factor"]
            prepare_factor_data(factor, prices, groupby=groups, periods=(1, 5), quantiles=5, max_loss=1)
        elif workload.returns is not None:
            from fincore.metrics.ratios import sharpe_ratio
            from fincore.metrics.risk import annual_volatility

            sharpe_ratio(workload.returns)
            annual_volatility(workload.returns)
        else:
            raise RuntimeError("unsupported workload")

    profiler = cProfile.Profile()
    started = time.perf_counter()
    profiler.enable()
    try:
        run()
    finally:
        profiler.disable()
    wall_seconds = time.perf_counter() - started
    stats = pstats.Stats(profiler)
    stats.sort_stats("cumulative")
    hot = []
    for name, func in stats.get_stats_profile().func_profiles.items():
        hot.append(
            {
                "function": f"{name} ({Path(func.file_name).name}:{func.line_number})",
                "cumtime_seconds": round(func.cumtime, 6),
                "calls": func.ncalls,
            }
        )
        if len(hot) >= TOP_FUNCTIONS:
            break
    return wall_seconds, hot


def _provenance() -> dict:
    import numpy
    import pandas

    return {
        "commit": subprocess.run(
            ["git", "-C", str(ROOT), "rev-parse", "HEAD"], check=True, capture_output=True, text=True
        ).stdout.strip(),
        "python": platform.python_version(),
        "numpy": numpy.__version__,
        "pandas": pandas.__version__,
        "platform_label": f"{sys.platform}-{platform.machine()}",
        "dirty": bool(
            subprocess.run(["git", "-C", str(ROOT), "status", "--porcelain"], capture_output=True, text=True).stdout
        ),
    }


def _render_markdown(data: dict) -> str:
    lines = [
        "# Hotspot Profile",
        "",
        f"- workload: `{data['workload']['name']}` size `{data['workload']['size']}`",
        f"- seed: `{data['workload']['seed']}` input_digest: `{data['workload']['input_digest']}`",
        f"- wall seconds: `{data['wall_seconds']:.3f}`",
        f"- cold import seconds: `{data['cold_import_seconds']:.3f}`",
        "",
        "| rank | function | cumtime (s) | calls |",
        "| --- | --- | ---: | ---: |",
    ]
    for rank, hot in enumerate(data["hotspots"], start=1):
        lines.append(f"| {rank} | `{hot['function']}` | {hot['cumtime_seconds']:.6f} | {hot['calls']} |")
    lines.append("")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario", choices=("small", "medium", "large"), default="medium")
    parser.add_argument("--kind", choices=("factor", "metrics"), default="factor")
    parser.add_argument("--output", required=True, help="JSON output path")
    parser.add_argument("--seed", type=int, default=20260817)
    args = parser.parse_args(argv)

    if args.kind == "factor":
        workload = factor_panel_workload(args.scenario, args.seed)
    else:
        workload = single_series_workload(args.scenario, args.seed)

    wall_seconds, hotspots = _profile_workload(workload)
    cold_import_seconds = _cold_import_seconds()
    data = {
        "schema": "fincore-hotspot-profile-v1",
        "provenance": _provenance(),
        "workload": describe_workload(workload),
        "wall_seconds": round(wall_seconds, 6),
        "cold_import_seconds": round(cold_import_seconds, 6),
        "peak_rss_bytes": _rss_bytes(),
        "hotspots": hotspots,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    markdown_path = output.with_suffix(".md")
    markdown_path.write_text(_render_markdown(data), encoding="utf-8")
    print(f"wrote {output} and {markdown_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
