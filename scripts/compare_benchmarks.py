#!/usr/bin/env python
"""Gate benchmark regressions between a baseline and a candidate payload.

The comparison artifact reports per-axis log-log slopes, per-row bytes,
and every raw case, and fails the run (exit code 1) when:

* a case present in the candidate is missing from the baseline, or the
  payload records an unknown RSS unit (anything other than ``bytes``);
* any candidate median exceeds the baseline by more than
  ``--max-time-regression`` / ``--max-rss-regression`` (relative) AND
  the corresponding absolute slack (noise tolerance for tiny cases);
* the fixed-window time/RSS-delta slope versus input size is
  superlinear (slope > ``--max-slope``);
* at fixed input size, the RSS delta of the largest window exceeds
  ``max(--window-rss-ratio x smallest-window delta, delta + slack)`` —
  i.e. RSS must not grow roughly linearly with the window;
* ``roll_alpha`` / ``roll_alpha_beta`` medians exceed
  ``--upstream-time-ratio`` x the baseline median;
* for round trips, at fixed rows the share-amount scale-up moves time or
  RSS delta by more than ``--round-trip-time-ratio`` /
  ``--round-trip-rss-ratio`` (or the absolute slack, whichever is wider);
  fixed-amount row growth is reported but allowed to be near-linear.

When ``--baseline`` is omitted (or missing with
``--allow-missing-baseline``) only the structural gates that do not need
a reference run are enforced, so a cold CI cache still fails complexity
and memory-scaling regressions.

Usage::

    python scripts/compare_benchmarks.py \\
        --baseline /tmp/fincore-rolling-before.json \\
        --candidate /tmp/fincore-rolling-after.json \\
        --max-time-regression 0.25 --max-rss-regression 0.25
"""

from __future__ import annotations

import argparse
import json
import math
import re
import statistics
import sys
from datetime import datetime
from itertools import pairwise
from pathlib import Path

MIB = 1024 * 1024
KNOWN_RSS_UNITS = {"bytes"}
UPSTREAM_METRICS = {"roll_alpha", "roll_alpha_beta"}
FACTOR_BENCHMARK_SCHEMA = "fincore-factor-analysis-benchmarks-v1"
FACTOR_BENCHMARK_KIND = "factor_analysis"
FACTOR_BENCHMARK_SEED = 20260815
FACTOR_SCENARIOS: dict[str, tuple[dict[str, int], tuple[str, ...]]] = {
    "small-ci": (
        {"dates": 252, "assets": 100, "rows": 25200},
        ("prepare", "quantize", "information-coefficient", "weights"),
    ),
    "medium-artifact": (
        {"dates": 1260, "assets": 500, "rows": 630000},
        ("prepare", "factor-returns", "full-model"),
    ),
    "event": (
        {"dates": 756, "assets": 200, "rows": 151200},
        ("common-start", "event-average"),
    ),
}
SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
GIT_SHA_PATTERN = re.compile(r"[0-9a-f]{40}")


def _load(path: str) -> dict:
    with Path(path).open(encoding="utf-8") as fh:
        return json.load(fh)


def _median(values: list[float]) -> float:
    return float(statistics.median(values)) if values else float("nan")


def _rolling_key(case: dict) -> tuple:
    return (case["metric"], case["input_size"], case["window"])


def _round_trip_key(case: dict) -> tuple:
    return (case["amount"], case["rows"])


def _group_cases(cases: list[dict], key_fn) -> dict:
    groups: dict = {}
    for case in cases:
        groups.setdefault(key_fn(case), []).append(case)
    return groups


def _medians(group: list[dict]) -> dict:
    return {
        "wall_seconds": _median([c["wall_seconds"] for c in group]),
        "rss_delta_bytes": _median([c["rss_delta_bytes"] for c in group]),
        "tracemalloc_peak_bytes": _median([c["tracemalloc_peak_bytes"] for c in group]),
        "peak_rss_bytes": _median([c["peak_rss_bytes"] for c in group]),
        "rss_before_bytes": _median([c["rss_before_bytes"] for c in group]),
    }


def _factor_key(case: dict) -> tuple[str, str]:
    return (case["scenario"], case["kernel"])


def _is_nonnegative_integer(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def _is_nonnegative_finite_number(value: object) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
        and float(value) >= 0.0
    )


def _is_sha256(value: object) -> bool:
    return isinstance(value, str) and SHA256_PATTERN.fullmatch(value) is not None


def _is_git_sha(value: object) -> bool:
    return isinstance(value, str) and GIT_SHA_PATTERN.fullmatch(value) is not None


def _is_output_shape(value: object) -> bool:
    """Accept the JSON-safe array/model-shape forms emitted by the runner."""

    if isinstance(value, list):
        return bool(value) and all(_is_nonnegative_integer(item) for item in value)
    if not isinstance(value, dict) or set(value) != {"factor_data", "forward_periods"}:
        return False
    factor_data = value["factor_data"]
    forward_periods = value["forward_periods"]
    return (
        isinstance(factor_data, list)
        and bool(factor_data)
        and all(_is_nonnegative_integer(item) for item in factor_data)
        and isinstance(forward_periods, list)
        and bool(forward_periods)
        and all(isinstance(period, str) and period for period in forward_periods)
    )


def _is_timestamp(value: object) -> bool:
    if not isinstance(value, str) or not value:
        return False
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00")).tzinfo is not None
    except ValueError:
        return False


def _check_factor_baseline_metadata(data: dict, provenance: dict, label: str) -> list[str]:
    """Validate mutually exclusive approved and candidate-only baseline states."""

    violations: list[str] = []
    baseline_status = data.get("baseline_status")
    approval = data.get("approval")
    protocol = data.get("candidate_protocol")
    if not isinstance(approval, dict):
        return [f"{label}: approval metadata must be an object"]
    if not isinstance(protocol, dict):
        return [f"{label}: candidate_protocol metadata must be an object"]

    reference_platform = protocol.get("reference_platform")
    if reference_platform != provenance.get("platform_label"):
        violations.append(f"{label}: candidate_protocol reference_platform must match provenance platform_label")
    if not isinstance(protocol.get("candidate_path"), str) or not protocol["candidate_path"]:
        violations.append(f"{label}: candidate_protocol candidate_path must be a nonempty string")
    if not _is_sha256(protocol.get("captured_candidate_sha256")):
        violations.append(f"{label}: candidate_protocol captured_candidate_sha256 must be a 64-hex SHA256")
    required_reviewers = protocol.get("required_reviewers")
    if (
        not isinstance(required_reviewers, list)
        or not all(isinstance(reviewer, str) for reviewer in required_reviewers)
        or not {"kernel owner", "Track E"} <= set(required_reviewers)
    ):
        violations.append(f"{label}: candidate_protocol must name kernel owner and Track E")
    if not isinstance(protocol.get("approval_steps"), list) or not protocol["approval_steps"]:
        violations.append(f"{label}: candidate_protocol approval_steps must be nonempty")
    if not isinstance(protocol.get("current_blockers"), list):
        violations.append(f"{label}: candidate_protocol current_blockers must be a list")

    status = approval.get("status")
    approved_by = approval.get("approved_by")
    approved_at = approval.get("approved_at")
    reviewed_sha = approval.get("reviewed_candidate_sha256")
    if status == "pending":
        if baseline_status != "candidate-only-not-release-approved":
            violations.append(f"{label}: pending baseline_status must be candidate-only-not-release-approved")
        if any(value is not None for value in (approved_by, approved_at, reviewed_sha)):
            violations.append(f"{label}: pending approval metadata must be null")
        if protocol.get("captured_candidate_review_status") != "unreviewed":
            violations.append(f"{label}: pending candidate_protocol must be unreviewed")
        if not protocol.get("current_blockers"):
            violations.append(f"{label}: pending candidate_protocol must list current_blockers")
    elif status == "approved":
        if baseline_status != "approved":
            violations.append(f"{label}: approved baseline_status must be approved")
        if provenance.get("dirty") is not False:
            violations.append(f"{label}: approved baseline must have dirty=false")
        if not isinstance(approved_by, str) or not approved_by.strip():
            violations.append(f"{label}: approved baseline requires approved_by")
        if not _is_timestamp(approved_at):
            violations.append(f"{label}: approved baseline requires timezone-aware approved_at")
        if not _is_sha256(reviewed_sha):
            violations.append(f"{label}: approved baseline requires reviewed_candidate_sha256")
        if protocol.get("captured_candidate_review_status") != "reviewed":
            violations.append(f"{label}: approved candidate_protocol must be reviewed")
        if protocol.get("captured_candidate_sha256") != reviewed_sha:
            violations.append(f"{label}: approved candidate_protocol SHA must equal reviewed_candidate_sha256")
        if protocol.get("current_blockers"):
            violations.append(f"{label}: approved candidate_protocol current_blockers must be empty")
    else:
        violations.append(f"{label}: approval status must be 'pending' or 'approved'")
    return violations


def _check_factor_schema(data: object, label: str, *, baseline: bool) -> list[str]:
    """Fail closed on the complete opt-in factor-analysis artifact contract."""

    if not isinstance(data, dict):
        return [f"{label}: payload must be a JSON object"]

    violations: list[str] = []
    if data.get("schema") != FACTOR_BENCHMARK_SCHEMA:
        violations.append(f"{label}: schema must be {FACTOR_BENCHMARK_SCHEMA!r}")
    if data.get("kind") != FACTOR_BENCHMARK_KIND:
        violations.append(f"{label}: kind must be {FACTOR_BENCHMARK_KIND!r}")
    if data.get("rss_unit") != "bytes":
        violations.append(f"{label}: rss_unit must be 'bytes'")

    provenance = data.get("provenance")
    if not isinstance(provenance, dict):
        violations.append(f"{label}: provenance must be an object")
        provenance = {}
    violations.extend(
        f"{label}: provenance missing {field!r}"
        for field in ("python", "numpy", "pandas", "scipy", "statsmodels", "os", "arch")
        if not isinstance(provenance.get(field), str) or not provenance[field]
    )
    if not _is_git_sha(provenance.get("commit")):
        violations.append(f"{label}: provenance commit must be a 40-hex Git SHA")
    if not isinstance(provenance.get("dirty"), bool):
        violations.append(f"{label}: provenance missing 'dirty'")
    expected_platform = (
        f"{provenance.get('os')}-{provenance.get('arch')}"
        if isinstance(provenance.get("os"), str) and isinstance(provenance.get("arch"), str)
        else None
    )
    if provenance.get("platform_label") != expected_platform:
        violations.append(f"{label}: platform_label must match provenance os and arch")

    runner = data.get("runner")
    if not isinstance(runner, dict):
        violations.append(f"{label}: runner must be an object")
        runner = {}
    warmups = runner.get("warmups")
    repeats = runner.get("repeats")
    if not _is_nonnegative_integer(warmups):
        violations.append(f"{label}: runner warmups must be a nonnegative integer")
    if not _is_nonnegative_integer(repeats) or repeats < 1:
        violations.append(f"{label}: runner repeats must be a positive integer")

    cases = data.get("cases")
    if not isinstance(cases, list) or not cases:
        violations.append(f"{label}: payload has no cases")
        cases = []

    valid_groups: dict[tuple[str, str], list[dict]] = {}
    observed_scenarios: set[str] = set()
    for number, case in enumerate(cases):
        prefix = f"{label}: case {number}"
        if not isinstance(case, dict):
            violations.append(f"{prefix} must be an object")
            continue
        scenario = case.get("scenario")
        kernel = case.get("kernel")
        scenario_is_valid = isinstance(scenario, str) and scenario in FACTOR_SCENARIOS
        kernel_is_valid = scenario_is_valid and isinstance(kernel, str) and kernel in FACTOR_SCENARIOS[scenario][1]
        if not kernel_is_valid:
            violations.append(f"{prefix} has unknown scenario/kernel {scenario!r}/{kernel!r}")
        else:
            observed_scenarios.add(scenario)
            valid_groups.setdefault((scenario, kernel), []).append(case)
            if case.get("input_shape") != FACTOR_SCENARIOS[scenario][0]:
                violations.append(f"{prefix} has noncanonical input_shape")
        if case.get("seed") != FACTOR_BENCHMARK_SEED:
            violations.append(f"{prefix} has noncanonical seed")
        if not _is_output_shape(case.get("output_shape")):
            violations.append(f"{prefix} has invalid output_shape")
        if not _is_sha256(case.get("output_digest")):
            violations.append(f"{prefix} has invalid SHA256 digest")
        if case.get("rss_unit") != "bytes":
            violations.append(f"{prefix} has invalid rss_unit")
        if not _is_nonnegative_finite_number(case.get("wall_seconds")):
            violations.append(f"{prefix} has invalid wall_seconds")
        violations.extend(
            f"{prefix} has invalid {field}"
            for field in ("rss_before_bytes", "peak_rss_bytes", "rss_delta_bytes", "tracemalloc_peak_bytes")
            if not _is_nonnegative_integer(case.get(field))
        )
        before = case.get("rss_before_bytes")
        peak = case.get("peak_rss_bytes")
        delta = case.get("rss_delta_bytes")
        if all(_is_nonnegative_integer(value) for value in (before, peak, delta)):
            if peak < before:
                violations.append(f"{prefix} peak_rss_bytes must be >= rss_before_bytes")
            if delta != max(peak - before, 0):
                violations.append(f"{prefix} rss_delta_bytes must match peak minus before")
        if not _is_nonnegative_integer(case.get("warmup")):
            violations.append(f"{prefix} warmup must be a nonnegative integer")
        if not _is_nonnegative_integer(case.get("repeat")):
            violations.append(f"{prefix} repeat must be a nonnegative integer")
        if not _is_nonnegative_integer(case.get("repeats")) or case.get("repeats", 0) < 1:
            violations.append(f"{prefix} repeats must be a positive integer")

    if (
        isinstance(warmups, int)
        and not isinstance(warmups, bool)
        and isinstance(repeats, int)
        and not isinstance(repeats, bool)
        and repeats >= 1
    ):
        expected_keys = {
            (scenario, kernel) for scenario in observed_scenarios for kernel in FACTOR_SCENARIOS[scenario][1]
        }
        actual_keys = set(valid_groups)
        missing = expected_keys - actual_keys
        unexpected = actual_keys - expected_keys
        if missing:
            violations.append(f"{label}: missing required cases: {sorted(missing)}")
        if unexpected:
            violations.append(f"{label}: unexpected cases: {sorted(unexpected)}")
        expected_repeat_ids = list(range(repeats))
        for key, group in sorted(valid_groups.items()):
            repeat_ids = sorted(case["repeat"] for case in group if _is_nonnegative_integer(case.get("repeat")))
            if repeat_ids != expected_repeat_ids:
                violations.append(
                    f"{label}: {key[0]}/{key[1]} repeat IDs must be {expected_repeat_ids}, got {repeat_ids}"
                )
            if any(case.get("warmup") != warmups or case.get("repeats") != repeats for case in group):
                violations.append(f"{label}: {key[0]}/{key[1]} repeat protocol must match runner")

    if baseline:
        violations.extend(_check_factor_baseline_metadata(data, provenance, label))
    else:
        violations.extend(
            f"{label}: candidate must not contain {field}"
            for field in ("baseline_status", "approval", "candidate_protocol")
            if field in data
        )
    return violations


def _factor_digest_violations(baseline: dict, candidate: dict) -> list[str]:
    """Compare deterministic output digests and shapes before timing data."""

    violations = _check_factor_schema(baseline, "baseline", baseline=True) + _check_factor_schema(
        candidate, "candidate", baseline=False
    )
    if violations:
        return violations
    base_groups = _group_cases(baseline["cases"], _factor_key)
    cand_groups = _group_cases(candidate["cases"], _factor_key)
    missing_from_candidate = set(base_groups) - set(cand_groups)
    unexpected_in_candidate = set(cand_groups) - set(base_groups)
    if missing_from_candidate:
        violations.append(f"candidate missing required cases: {sorted(missing_from_candidate)}")
    if unexpected_in_candidate:
        violations.append(f"candidate has unexpected cases: {sorted(unexpected_in_candidate)}")
    for key in sorted(set(cand_groups) & set(base_groups)):
        base_digests = {case["output_digest"] for case in base_groups[key]}
        cand_digests = {case["output_digest"] for case in cand_groups[key]}
        base_shapes = {json.dumps(case["output_shape"], sort_keys=True) for case in base_groups[key]}
        cand_shapes = {json.dumps(case["output_shape"], sort_keys=True) for case in cand_groups[key]}
        base_inputs = {json.dumps(case["input_shape"], sort_keys=True) for case in base_groups[key]}
        cand_inputs = {json.dumps(case["input_shape"], sort_keys=True) for case in cand_groups[key]}
        base_seeds = {case["seed"] for case in base_groups[key]}
        cand_seeds = {case["seed"] for case in cand_groups[key]}
        if len(base_digests) != 1 or len(cand_digests) != 1 or base_digests != cand_digests:
            violations.append(
                f"{key[0]}/{key[1]}: output_digest mismatch "
                f"baseline={sorted(base_digests)} candidate={sorted(cand_digests)}"
            )
        if len(base_shapes) != 1 or len(cand_shapes) != 1 or base_shapes != cand_shapes:
            violations.append(
                f"{key[0]}/{key[1]}: output_shape mismatch "
                f"baseline={sorted(base_shapes)} candidate={sorted(cand_shapes)}"
            )
        if len(base_inputs) != 1 or len(cand_inputs) != 1 or base_inputs != cand_inputs:
            violations.append(
                f"{key[0]}/{key[1]}: input_shape mismatch "
                f"baseline={sorted(base_inputs)} candidate={sorted(cand_inputs)}"
            )
        if len(base_seeds) != 1 or len(cand_seeds) != 1 or base_seeds != cand_seeds:
            violations.append(
                f"{key[0]}/{key[1]}: seed mismatch baseline={sorted(base_seeds)} candidate={sorted(cand_seeds)}"
            )
    return violations


def _compare_factor_analysis(baseline: dict | None, candidate: dict, args) -> int:
    """Run the opt-in digest-first, same-platform factor benchmark gate."""

    if baseline is None:
        violations = _check_factor_schema(candidate, "candidate", baseline=False)
        if violations:
            for violation in violations:
                print(f"FAIL: {violation}", file=sys.stderr)
            return 1
        print("factor-analysis baseline absent; artifact only, performance comparison not run")
        return 0

    digest_violations = _factor_digest_violations(baseline, candidate)
    if digest_violations:
        for violation in digest_violations:
            print(f"FAIL: {violation}", file=sys.stderr)
        print("performance comparison not run because digest/shape gate failed", file=sys.stderr)
        return 1

    baseline_platform = baseline["provenance"]["platform_label"]
    candidate_platform = candidate["provenance"]["platform_label"]
    if baseline_platform != candidate_platform:
        print(
            f"platform mismatch ({baseline_platform} != {candidate_platform}); "
            "artifact only, performance comparison not run"
        )
        return 0

    approval = baseline["approval"]
    if approval["status"] != "approved":
        print("baseline approval is pending; performance comparison not run", file=sys.stderr)
        return 1

    base_groups = _group_cases(baseline["cases"], _factor_key)
    cand_groups = _group_cases(candidate["cases"], _factor_key)
    violations: list[str] = []
    for key in sorted(cand_groups):
        base_time = _median([case["wall_seconds"] for case in base_groups[key]])
        cand_time = _median([case["wall_seconds"] for case in cand_groups[key]])
        base_rss = _median([case["peak_rss_bytes"] for case in base_groups[key]])
        cand_rss = _median([case["peak_rss_bytes"] for case in cand_groups[key]])
        if cand_time > base_time * (1.0 + args.max_time_regression):
            violations.append(f"{key[0]}/{key[1]}: wall_seconds regressed {cand_time:.6g} vs {base_time:.6g}")
        if cand_rss > base_rss * (1.0 + args.max_rss_regression):
            violations.append(f"{key[0]}/{key[1]}: peak_rss_bytes regressed {cand_rss:.6g} vs {base_rss:.6g}")
    for violation in violations:
        print(f"FAIL: {violation}", file=sys.stderr)
    if violations:
        return 1
    print("all factor-analysis digest, shape, time, and RSS gates passed")
    return 0


def _slope(x1: float, x2: float, y1: float, y2: float) -> float | None:
    """Log-log slope between two (x, y) points; None when undefined."""
    if min(x1, x2) <= 0 or min(y1, y2) <= 0 or x1 == x2:
        return None
    return math.log(y2 / y1) / math.log(x2 / x1)


def _check_schema(data: dict, kind: str) -> list[str]:
    violations = []
    if data.get("rss_unit") != "bytes":
        violations.append(f"unknown RSS unit {data.get('rss_unit')!r}; known units: {sorted(KNOWN_RSS_UNITS)}")
    provenance = data.get("provenance") or {}
    violations.extend(
        f"provenance missing {field!r}"
        for field in ("commit", "python", "numpy", "pandas")
        if not provenance.get(field)
    )
    cases = data.get("cases") or []
    if not cases:
        violations.append("payload has no cases")
    common = (
        "wall_seconds",
        "rss_before_bytes",
        "peak_rss_bytes",
        "rss_delta_bytes",
        "tracemalloc_peak_bytes",
        "rss_unit",
    )
    kind_fields = ("metric", "input_size", "window") if kind == "rolling" else ("amount", "rows")
    for case in cases:
        for field in common + kind_fields:
            if field not in case:
                violations.append(f"case missing {field!r}: {case}")
                break
    return violations


def _generic_regressions(base_group: dict, cand_group: dict, args, label: str) -> list[str]:
    violations = []
    for axis, reg, slack in (
        ("wall_seconds", args.max_time_regression, args.time_slack),
        ("rss_delta_bytes", args.max_rss_regression, args.rss_slack_mib * MIB),
    ):
        base = base_group[axis]
        cand = cand_group[axis]
        rel_ok = cand <= base * (1.0 + reg)
        abs_ok = cand <= base + slack
        if not rel_ok and not abs_ok:
            violations.append(
                f"{label}: {axis} regressed {cand - base:.4g} "
                f"(candidate {cand:.4g} vs baseline {base:.4g}, "
                f"limit {base * (1.0 + reg):.4g} or {base + slack:.4g})"
            )
    return violations


def compare_rolling(baseline: dict | None, candidate: dict, args, report: dict) -> list[str]:
    violations = _check_schema(candidate, "rolling")
    cand_cases = candidate["cases"]
    cand_groups = _group_cases(cand_cases, _rolling_key)
    cand_medians = {key: _medians(group) for key, group in cand_groups.items()}
    report["rolling"] = {"cases": {}}
    report["per_row_bytes"] = {}
    report["slopes"] = {"time_vs_n": [], "rss_vs_n": [], "rss_vs_window": [], "time_vs_window": []}

    base_medians: dict = {}
    if baseline is not None:
        violations += _check_schema(baseline, "rolling")
        base_groups = _group_cases(baseline["cases"], _rolling_key)
        base_medians = {key: _medians(group) for key, group in base_groups.items()}
        missing = set(cand_groups) - set(base_groups)
        if missing:
            violations.append(f"candidate cases missing from baseline: {sorted(missing)}")

    for key in sorted(cand_groups):
        med = cand_medians[key]
        metric, size, window = key
        report["rolling"]["cases"][f"{metric}|n={size}|w={window}"] = med
        report["per_row_bytes"][f"{metric}|n={size}|w={window}"] = med["rss_delta_bytes"] / size if size else None
        if key in base_medians:
            violations += _generic_regressions(base_medians[key], med, args, f"{metric} n={size} w={window}")
            if metric in UPSTREAM_METRICS:
                limit = base_medians[key]["wall_seconds"] * args.upstream_time_ratio
                if med["wall_seconds"] > limit + args.upstream_time_slack:
                    violations.append(
                        f"{metric} n={size} w={window}: candidate median {med['wall_seconds']:.4g}s "
                        f"exceeds {args.upstream_time_ratio}x upstream ({limit:.4g}s)"
                    )

    # Fixed-window slope versus input size (superlinearity gate).
    for metric in sorted({k[0] for k in cand_groups}):
        for window in sorted({k[2] for k in cand_groups if k[0] == metric}):
            sizes = sorted({k[1] for k in cand_groups if k[0] == metric and k[2] == window})
            for s1, s2 in pairwise(sizes):
                m1 = cand_medians[(metric, s1, window)]
                m2 = cand_medians[(metric, s2, window)]
                for axis, bucket in (("wall_seconds", "time_vs_n"), ("rss_delta_bytes", "rss_vs_n")):
                    slope = _slope(s1, s2, m1[axis], m2[axis])
                    report["slopes"][bucket].append(
                        {
                            "metric": metric,
                            "window": window,
                            "sizes": [s1, s2],
                            "slope": slope,
                        }
                    )
                    if slope is not None and slope > args.max_slope:
                        violations.append(
                            f"{metric} w={window}: {axis} slope vs n ({s1}->{s2}) is {slope:.3f} "
                            f"(superlinear; limit {args.max_slope})"
                        )

    # Fixed-size window scaling gate: RSS delta of the largest window must
    # not grow roughly linearly with the window.
    for metric in sorted({k[0] for k in cand_groups}):
        for size in sorted({k[1] for k in cand_groups if k[0] == metric}):
            windows = sorted({k[2] for k in cand_groups if k[0] == metric and k[1] == size})
            if len(windows) < 2:
                continue
            small = cand_medians[(metric, size, windows[0])]
            large = cand_medians[(metric, size, windows[-1])]
            rss_small = small["rss_delta_bytes"]
            rss_large = large["rss_delta_bytes"]
            limit = max(args.window_rss_ratio * rss_small, rss_small + args.window_rss_slack_mib * MIB)
            slope = _slope(windows[0], windows[-1], rss_small, rss_large)
            report["slopes"]["rss_vs_window"].append(
                {
                    "metric": metric,
                    "size": size,
                    "windows": [windows[0], windows[-1]],
                    "slope": slope,
                }
            )
            report["slopes"]["time_vs_window"].append(
                {
                    "metric": metric,
                    "size": size,
                    "windows": [windows[0], windows[-1]],
                    "slope": _slope(windows[0], windows[-1], small["wall_seconds"], large["wall_seconds"]),
                }
            )
            if rss_large > limit:
                violations.append(
                    f"{metric} n={size}: rss_delta grows from {rss_small / MIB:.2f} MiB "
                    f"(w={windows[0]}) to {rss_large / MIB:.2f} MiB (w={windows[-1]}); "
                    f"limit {limit / MIB:.2f} MiB"
                )
    return violations


def compare_round_trips(baseline: dict | None, candidate: dict, args, report: dict) -> list[str]:
    violations = _check_schema(candidate, "round_trips")
    cand_cases = candidate["cases"]
    cand_groups = _group_cases(cand_cases, _round_trip_key)
    cand_medians = {key: _medians(group) for key, group in cand_groups.items()}
    report["round_trips"] = {"cases": {}}
    report["per_row_bytes"] = {}
    report["slopes"]["time_vs_rows"] = []
    report["slopes"]["rss_vs_rows"] = []

    base_medians: dict = {}
    if baseline is not None:
        violations += _check_schema(baseline, "round_trips")
        base_groups = _group_cases(baseline["cases"], _round_trip_key)
        base_medians = {key: _medians(group) for key, group in base_groups.items()}
        missing = set(cand_groups) - set(base_groups)
        if missing:
            violations.append(f"candidate cases missing from baseline: {sorted(missing)}")

    for key in sorted(cand_groups):
        med = cand_medians[key]
        amount, rows = key
        report["round_trips"]["cases"][f"amount={amount}|rows={rows}"] = med
        report["per_row_bytes"][f"amount={amount}|rows={rows}"] = med["rss_delta_bytes"] / rows if rows else None
        if key in base_medians:
            violations += _generic_regressions(base_medians[key], med, args, f"round_trips amount={amount} rows={rows}")

    # Fixed rows: share-amount scale-up must not move time or RSS delta
    # beyond the ratio gate or the absolute slack (whichever is wider).
    for rows in sorted({k[1] for k in cand_groups}):
        amounts = sorted({k[0] for k in cand_groups if k[1] == rows})
        for a1, a2 in pairwise(amounts):
            small = cand_medians[(a1, rows)]
            large = cand_medians[(a2, rows)]
            for axis, ratio, slack, unit in (
                ("wall_seconds", args.round_trip_time_ratio, args.round_trip_time_slack, "s"),
                ("rss_delta_bytes", args.round_trip_rss_ratio, args.round_trip_rss_slack_mib * MIB, "bytes"),
            ):
                limit = max(ratio * small[axis], small[axis] + slack)
                if large[axis] > limit:
                    violations.append(
                        f"round_trips rows={rows}: amount {a1}->{a2} scales {axis} from "
                        f"{small[axis]:.4g} to {large[axis]:.4g} {unit} (limit {limit:.4g})"
                    )

    # Fixed amount: row growth is allowed to be near-linear; report slopes.
    for amount in sorted({k[0] for k in cand_groups}):
        rows_list = sorted({k[1] for k in cand_groups if k[0] == amount})
        for r1, r2 in pairwise(rows_list):
            m1 = cand_medians[(amount, r1)]
            m2 = cand_medians[(amount, r2)]
            for axis, bucket in (("wall_seconds", "time_vs_rows"), ("rss_delta_bytes", "rss_vs_rows")):
                report["slopes"][bucket].append(
                    {
                        "amount": amount,
                        "rows": [r1, r2],
                        "slope": _slope(r1, r2, m1[axis], m2[axis]),
                    }
                )
    return violations


def _print_report(report: dict, violations: list[str], args) -> None:
    print("== per-axis log-log slopes ==")
    for axis, entries in report["slopes"].items():
        if not entries:
            continue
        print(f"-- {axis} --")
        for entry in entries:
            slope = entry["slope"]
            rendered = f"{slope:.3f}" if slope is not None else "undefined"
            print(f"   {entry} -> slope={rendered}")
    print("== per-row bytes (rss_delta / rows) ==")
    for key, value in sorted(report["per_row_bytes"].items()):
        rendered = f"{value:.1f}" if value else "undefined"
        print(f"   {key}: {rendered}")
    print("== raw cases (median over repeats) ==")
    for kind, section in (("rolling", report.get("rolling")), ("round_trips", report.get("round_trips"))):
        if not section:
            continue
        for key, med in sorted(section["cases"].items()):
            print(
                f"   [{kind}] {key}: wall={med['wall_seconds']:.4g}s "
                f"rss_delta={med['rss_delta_bytes'] / MIB:.3f} MiB "
                f"tracemalloc_peak={med['tracemalloc_peak_bytes'] / MIB:.3f} MiB"
            )
    print("== verdicts ==")
    for violation in violations:
        print(f"FAIL: {violation}")
    if args.report:
        with Path(args.report).open("w", encoding="utf-8") as fh:
            json.dump({"violations": violations, "report": report}, fh, indent=2)
        print(f"comparison artifact written to {args.report}")


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--baseline", help="baseline JSON produced by a benchmark runner")
    parser.add_argument("--candidate", required=True, help="candidate JSON produced by a benchmark runner")
    parser.add_argument("--max-time-regression", type=float, default=0.25)
    parser.add_argument("--max-rss-regression", type=float, default=0.25)
    parser.add_argument("--time-slack", type=float, default=0.005, help="absolute wall-time noise slack (s)")
    parser.add_argument("--rss-slack-mib", type=float, default=8.0, help="absolute RSS noise slack (MiB)")
    parser.add_argument("--max-slope", type=float, default=1.25, help="superlinear log-log slope limit")
    parser.add_argument("--window-rss-ratio", type=float, default=1.5)
    parser.add_argument("--window-rss-slack-mib", type=float, default=64.0)
    parser.add_argument(
        "--upstream-time-ratio", type=float, default=1.25, help="roll_alpha/roll_alpha_beta median limit vs baseline"
    )
    parser.add_argument("--upstream-time-slack", type=float, default=0.005)
    parser.add_argument("--round-trip-time-ratio", type=float, default=1.25)
    parser.add_argument("--round-trip-time-slack", type=float, default=0.005)
    parser.add_argument("--round-trip-rss-ratio", type=float, default=1.25)
    parser.add_argument("--round-trip-rss-slack-mib", type=float, default=32.0)
    parser.add_argument(
        "--allow-missing-baseline",
        action="store_true",
        help="run structural gates only when the baseline file is absent",
    )
    parser.add_argument("--digest-gate", choices=("sha256",), help="compare output digest/shape before performance")
    parser.add_argument("--report", help="write the full comparison artifact as JSON here")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    candidate = _load(args.candidate)
    baseline = None
    if args.baseline:
        try:
            baseline = _load(args.baseline)
        except OSError:
            if not args.allow_missing_baseline:
                print(
                    f"baseline {args.baseline!r} not found (pass --allow-missing-baseline "
                    f"to run structural gates only)",
                    file=sys.stderr,
                )
                return 1
            print(f"baseline {args.baseline!r} not found; running structural gates only")
    elif not args.baseline:
        print("no --baseline given; running structural gates only")

    if args.digest_gate == "sha256":
        return _compare_factor_analysis(baseline, candidate, args)

    kind = candidate.get("kind")
    report: dict = {"slopes": {}, "per_row_bytes": {}}
    if kind == "rolling":
        violations = compare_rolling(baseline, candidate, args, report)
    elif kind == "round_trips":
        violations = compare_round_trips(baseline, candidate, args, report)
    else:
        print(f"unknown candidate kind {kind!r} (expected 'rolling' or 'round_trips')", file=sys.stderr)
        return 1

    _print_report(report, violations, args)
    if violations:
        print(f"{len(violations)} gate violation(s)", file=sys.stderr)
        return 1
    print("all benchmark gates passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
