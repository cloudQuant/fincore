#!/usr/bin/env python3
"""Profile deterministic metrics/factor workloads across named scales.

This orchestrator runs the bounded single-workload profiler in fresh
subprocesses and gathers the resulting semantic input digests, wall time,
cold-import time, RSS, and CPU hotspots into one JSON artifact.  It is a
profiling aid, not a release baseline writer: artifacts recorded from a dirty
checkout remain visibly dirty and cannot become performance evidence.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, cast

ROOT = Path(__file__).resolve().parent.parent
HOTSPOT_PROFILER = ROOT / "scripts" / "profile_hotspots.py"


def _profile_one(size: str, kind: str, output: Path) -> dict[str, Any]:
    """Run one isolated profile and return its machine-readable artifact."""
    _run = subprocess.run(
        [
            sys.executable,
            str(HOTSPOT_PROFILER),
            "--scenario",
            size,
            "--kind",
            kind,
            "--output",
            str(output),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=900,
    )
    if _run.returncode != 0:
        message = _run.stderr.strip() or _run.stdout.strip() or "no profiler output"
        raise RuntimeError(f"{kind}/{size} profile failed: {message}")
    payload = json.loads(output.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"{kind}/{size} profile emitted a non-object JSON payload")
    return cast("dict[str, Any]", payload)


def run_profiles(sizes: tuple[str, ...], kinds: tuple[str, ...]) -> dict[str, Any]:
    """Profile every requested kind/scale in separate temporary artifacts."""
    cases: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory(prefix="fincore-workload-profiles-") as temporary_directory:
        temporary_root = Path(temporary_directory)
        for kind in kinds:
            for size in sizes:
                artifact = temporary_root / f"{kind}-{size}.json"
                cases.append(_profile_one(size, kind, artifact))
    return {
        "schema": "fincore-workload-profiles-v1",
        "kinds": list(kinds),
        "sizes": list(sizes),
        "cases": cases,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sizes", nargs="+", choices=("small", "medium", "large"), default=["small"])
    parser.add_argument("--kinds", nargs="+", choices=("metrics", "factor"), default=["metrics", "factor"])
    parser.add_argument(
        "--output",
        default=str(ROOT / ".benchmarks" / "workload-profiles.json"),
        help="JSON output artifact (default: .benchmarks/workload-profiles.json)",
    )
    args = parser.parse_args(argv)

    payload = run_profiles(tuple(args.sizes), tuple(args.kinds))
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote {len(payload['cases'])} workload profiles to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
