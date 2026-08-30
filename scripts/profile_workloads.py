#!/usr/bin/env python3
"""Profile deterministic Fincore workloads across named scales.

The orchestrator gives every kind/scale pair an isolated profiler process and
collects its audited input identity, semantic output digest, repeat samples,
summary statistics, and hotspot evidence in one JSON artifact. It is a
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
WORKLOAD_KINDS = ("metrics", "rolling", "transactions", "factor", "risk", "report")
WORKLOAD_PROFILE_SCHEMA = "fincore-workload-profiles-v2"
HOTSPOT_PROFILE_SCHEMA = "fincore-hotspot-profile-v2"
SIZES = ("small", "medium", "large")


def _is_sha256(value: object) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(character in "0123456789abcdef" for character in value)


def _measurement_contract(warmups: int, repeats: int, require_output_digest: bool) -> dict[str, Any]:
    return {
        "warmups": warmups,
        "repeats": repeats,
        "require_output_digest": require_output_digest,
        "timing_unit": "seconds",
        "percentile_method": "linear",
    }


def _validate_case(
    payload: dict[str, Any],
    *,
    size: str,
    kind: str,
    warmups: int,
    repeats: int,
    require_output_digest: bool,
) -> None:
    """Reject a child result that cannot prove it measured the requested case."""

    if payload.get("schema") != HOTSPOT_PROFILE_SCHEMA:
        raise RuntimeError(f"{kind}/{size} profile emitted an unexpected schema")
    if payload.get("kind") != kind:
        raise RuntimeError(f"{kind}/{size} profile emitted a mismatched kind")
    if payload.get("measurement") != _measurement_contract(warmups, repeats, require_output_digest):
        raise RuntimeError(f"{kind}/{size} profile emitted a mismatched measurement contract")
    workload = payload.get("workload")
    if not isinstance(workload, dict) or workload.get("size") != size or not _is_sha256(workload.get("input_digest")):
        raise RuntimeError(f"{kind}/{size} profile omitted a valid workload input digest")
    if not _is_sha256(payload.get("execution_input_digest")):
        raise RuntimeError(f"{kind}/{size} profile omitted a valid execution input digest")
    output_digest = payload.get("output_digest")
    warmup_digests = payload.get("warmup_output_digests")
    measured_digests = payload.get("measured_output_digests")
    if not _is_sha256(output_digest):
        raise RuntimeError(f"{kind}/{size} profile omitted a valid semantic output digest")
    if not isinstance(warmup_digests, list) or not isinstance(measured_digests, list):
        raise RuntimeError(f"{kind}/{size} profile omitted output digest samples")
    if len(warmup_digests) != warmups or len(measured_digests) != repeats:
        raise RuntimeError(f"{kind}/{size} profile emitted the wrong number of output digest samples")
    all_digests = [*warmup_digests, *measured_digests, payload.get("profiled_output_digest")]
    if any(not _is_sha256(digest) for digest in all_digests) or any(digest != output_digest for digest in all_digests):
        raise RuntimeError(f"{kind}/{size} profile emitted an unstable semantic output digest")
    samples = payload.get("timing_samples_seconds")
    timing = payload.get("timing")
    if (
        not isinstance(samples, list)
        or len(samples) != repeats
        or not all(isinstance(sample, float) and sample > 0 for sample in samples)
    ):
        raise RuntimeError(f"{kind}/{size} profile omitted valid timing samples")
    if not isinstance(timing, dict) or set(timing) != {
        "minimum_seconds",
        "median_seconds",
        "p95_seconds",
        "maximum_seconds",
    }:
        raise RuntimeError(f"{kind}/{size} profile omitted an auditable timing summary")
    if not all(isinstance(value, float) and value > 0 for value in timing.values()):
        raise RuntimeError(f"{kind}/{size} profile emitted an invalid timing summary")


def _profile_one(
    size: str,
    kind: str,
    output: Path,
    *,
    warmups: int,
    repeats: int,
    require_output_digest: bool,
) -> dict[str, Any]:
    """Run one isolated profile and return its validated machine-readable artifact."""

    command = [
        sys.executable,
        str(HOTSPOT_PROFILER),
        "--scenario",
        size,
        "--kind",
        kind,
        "--warmups",
        str(warmups),
        "--repeats",
        str(repeats),
        "--output",
        str(output),
    ]
    if require_output_digest:
        command.append("--require-output-digest")
    result = subprocess.run(
        command,
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=900,
    )
    if result.returncode != 0:
        message = result.stderr.strip() or result.stdout.strip() or "no profiler output"
        raise RuntimeError(f"{kind}/{size} profile failed: {message}")
    try:
        payload = json.loads(output.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"{kind}/{size} profile did not write valid JSON") from error
    if not isinstance(payload, dict):
        raise RuntimeError(f"{kind}/{size} profile emitted a non-object JSON payload")
    profile = cast("dict[str, Any]", payload)
    _validate_case(
        profile,
        size=size,
        kind=kind,
        warmups=warmups,
        repeats=repeats,
        require_output_digest=require_output_digest,
    )
    return profile


def _validate_selection(sizes: tuple[str, ...], kinds: tuple[str, ...], warmups: int, repeats: int) -> None:
    if not sizes or any(size not in SIZES for size in sizes):
        raise ValueError(f"sizes must be a non-empty subset of {SIZES}")
    if len(set(sizes)) != len(sizes):
        raise ValueError("sizes must not contain duplicates")
    if not kinds or any(kind not in WORKLOAD_KINDS for kind in kinds):
        raise ValueError(f"kinds must be a non-empty subset of {WORKLOAD_KINDS}")
    if len(set(kinds)) != len(kinds):
        raise ValueError("kinds must not contain duplicates")
    if isinstance(warmups, bool) or not isinstance(warmups, int) or warmups < 0:
        raise ValueError("warmups must be a non-negative integer")
    if isinstance(repeats, bool) or not isinstance(repeats, int) or repeats < 1:
        raise ValueError("repeats must be a positive integer")


def run_profiles(
    sizes: tuple[str, ...],
    kinds: tuple[str, ...],
    *,
    warmups: int = 0,
    repeats: int = 1,
    require_output_digest: bool = False,
) -> dict[str, Any]:
    """Profile every requested kind/scale in separate temporary artifacts."""

    _validate_selection(sizes, kinds, warmups, repeats)
    cases: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory(prefix="fincore-workload-profiles-") as temporary_directory:
        temporary_root = Path(temporary_directory)
        for kind in kinds:
            for size in sizes:
                artifact = temporary_root / f"{kind}-{size}.json"
                cases.append(
                    _profile_one(
                        size,
                        kind,
                        artifact,
                        warmups=warmups,
                        repeats=repeats,
                        require_output_digest=require_output_digest,
                    )
                )
    return {
        "schema": WORKLOAD_PROFILE_SCHEMA,
        "kinds": list(kinds),
        "sizes": list(sizes),
        "measurement": _measurement_contract(warmups, repeats, require_output_digest),
        "cases": cases,
    }


def _nonnegative_int(raw: str) -> int:
    try:
        value = int(raw)
    except ValueError as error:
        raise argparse.ArgumentTypeError("must be an integer") from error
    if value < 0:
        raise argparse.ArgumentTypeError("must be greater than or equal to zero")
    return value


def _positive_int(raw: str) -> int:
    try:
        value = int(raw)
    except ValueError as error:
        raise argparse.ArgumentTypeError("must be an integer") from error
    if value < 1:
        raise argparse.ArgumentTypeError("must be greater than or equal to one")
    return value


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sizes", nargs="+", choices=SIZES, default=["small"])
    parser.add_argument("--kinds", nargs="+", choices=WORKLOAD_KINDS, default=["metrics", "factor"])
    parser.add_argument("--warmups", type=_nonnegative_int, default=0)
    parser.add_argument("--repeats", type=_positive_int, default=1)
    parser.add_argument("--require-output-digest", action="store_true")
    parser.add_argument(
        "--output",
        default=str(ROOT / ".benchmarks" / "workload-profiles.json"),
        help="JSON output artifact (default: .benchmarks/workload-profiles.json)",
    )
    args = parser.parse_args(argv)
    try:
        payload = run_profiles(
            tuple(args.sizes),
            tuple(args.kinds),
            warmups=args.warmups,
            repeats=args.repeats,
            require_output_digest=args.require_output_digest,
        )
    except ValueError as error:
        parser.error(str(error))
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote {len(payload['cases'])} workload profiles to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
