#!/usr/bin/env python3
"""Profile deterministic Fincore workloads with reproducible semantic output checks.

Each profile case is built from a deterministic workload factory, warmed up,
measured repeatedly, and then profiled once for hotspot attribution. The timing
samples deliberately exclude input construction and digesting; they measure the
Fincore computation itself. Input and output digests are retained in the JSON
artifact so a comparison never mistakes a different financial calculation for a
performance change.
"""

from __future__ import annotations

import argparse
import cProfile
import hashlib
import json
import math
import platform
import pstats
import resource
import subprocess
import sys
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, fields, is_dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
BENCHMARKS = ROOT / "benchmarks"
for path in (str(ROOT), str(BENCHMARKS)):
    if path not in sys.path:
        sys.path.insert(0, path)

from workloads import (
    Workload,
    describe_workload,
    factor_panel_workload,
    report_workload,
    rolling_returns_workload,
    single_series_workload,
    transactions_workload,
)

TOP_FUNCTIONS = 15
WORKLOAD_KINDS = ("metrics", "rolling", "transactions", "factor", "risk", "report")
HOTSPOT_PROFILE_SCHEMA = "fincore-hotspot-profile-v2"
SEMANTIC_DIGEST_SCHEMA = "fincore-semantic-digest-v1"
ROLLING_WINDOW = 63


@dataclass(frozen=True)
class _Invocation:
    """A fully materialized financial calculation ready to execute."""

    workload: dict[str, Any]
    execution_input_digest: str
    run: Callable[[], Any]


@dataclass(frozen=True)
class _Execution:
    """One invocation's identity, semantic result, and optional profile data."""

    workload: dict[str, Any]
    execution_input_digest: str
    output_digest: str
    wall_seconds: float
    hotspots: list[dict[str, Any]]


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


def _write_digest_bytes(hasher: hashlib._Hash, payload: bytes) -> None:
    """Write a length-delimited token so unlike structures cannot collide."""

    hasher.update(len(payload).to_bytes(8, "big"))
    hasher.update(payload)


def _write_digest_tag(hasher: hashlib._Hash, tag: str) -> None:
    _write_digest_bytes(hasher, tag.encode("utf-8"))


def _float_token(value: float) -> str:
    if math.isnan(value):
        return "nan"
    if math.isinf(value):
        return "positive-infinity" if value > 0 else "negative-infinity"
    return value.hex()


def _atom(value: Any) -> Any:
    """Return a JSON-safe, type-preserving representation of a scalar label."""

    if isinstance(value, np.generic):
        value = value.item()
    if value is None:
        return {"type": "none"}
    if isinstance(value, bool):
        return {"type": "bool", "value": value}
    if isinstance(value, int):
        return {"type": "int", "value": str(value)}
    if isinstance(value, float):
        return {"type": "float", "value": _float_token(value)}
    if isinstance(value, str):
        return {"type": "str", "value": value}
    if isinstance(value, bytes):
        return {"type": "bytes", "value": value.hex()}
    if value is pd.NaT:
        return {"type": "pandas.NaT"}
    if isinstance(value, pd.Timestamp):
        return {"type": "pandas.Timestamp", "value": value.isoformat()}
    if isinstance(value, pd.Timedelta):
        return {"type": "pandas.Timedelta", "value": value.isoformat()}
    if isinstance(value, np.datetime64):
        return {"type": "numpy.datetime64", "value": str(value), "dtype": str(value.dtype)}
    if isinstance(value, np.timedelta64):
        return {"type": "numpy.timedelta64", "value": str(value), "dtype": str(value.dtype)}
    if isinstance(value, datetime):
        return {"type": "datetime", "value": value.isoformat()}
    if isinstance(value, date):
        return {"type": "date", "value": value.isoformat()}
    if isinstance(value, tuple):
        return {"type": "tuple", "values": [_atom(item) for item in value]}
    raise TypeError(f"unsupported semantic label type: {type(value).__module__}.{type(value).__qualname__}")


def _index_header(index: pd.Index) -> dict[str, Any]:
    """Record metadata not carried by pandas' row hash alone."""

    header: dict[str, Any] = {
        "type": f"{type(index).__module__}.{type(index).__qualname__}",
        "names": [_atom(name) for name in index.names],
        "length": len(index),
    }
    if isinstance(index, pd.MultiIndex):
        header["level_dtypes"] = [str(level.dtype) for level in index.levels]
    else:
        header["dtype"] = str(index.dtype)
    frequency = getattr(index, "freqstr", None)
    if frequency is not None:
        header["frequency"] = frequency
    timezone = getattr(index, "tz", None)
    if timezone is not None:
        header["timezone"] = str(timezone)
    return header


def _write_json_header(hasher: hashlib._Hash, header: Mapping[str, Any]) -> None:
    _write_digest_bytes(
        hasher,
        json.dumps(header, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8"),
    )


def _update_pandas_digest(hasher: hashlib._Hash, value: pd.Series | pd.DataFrame) -> None:
    if isinstance(value, pd.Series):
        _write_digest_tag(hasher, "pandas-series")
        header: dict[str, Any] = {
            "name": _atom(value.name),
            "dtype": str(value.dtype),
            "shape": list(value.shape),
            "index": _index_header(value.index),
        }
    else:
        _write_digest_tag(hasher, "pandas-dataframe")
        header = {
            "columns": [_atom(column) for column in value.columns],
            "columns_metadata": _index_header(value.columns),
            "dtypes": [str(dtype) for dtype in value.dtypes],
            "shape": list(value.shape),
            "index": _index_header(value.index),
        }
    _write_json_header(hasher, header)
    try:
        hashes = pd.util.hash_pandas_object(value, index=True, categorize=True).to_numpy(
            dtype=np.uint64,
            copy=False,
        )
    except (TypeError, ValueError) as error:
        raise TypeError(f"cannot hash pandas semantic output: {error}") from error
    _write_digest_bytes(hasher, np.ascontiguousarray(hashes).tobytes())


def _update_semantic_digest(hasher: hashlib._Hash, value: Any) -> None:
    """Add a deterministic, type-aware representation of a result to ``hasher``."""

    if isinstance(value, pd.DataFrame | pd.Series):
        _update_pandas_digest(hasher, value)
        return
    if isinstance(value, np.generic):
        _update_semantic_digest(hasher, value.item())
        return
    if isinstance(value, np.ndarray):
        _write_digest_tag(hasher, "numpy-array")
        _write_json_header(hasher, {"dtype": str(value.dtype), "shape": list(value.shape)})
        if value.dtype.hasobject:
            _update_semantic_digest(hasher, value.tolist())
        else:
            _write_digest_bytes(hasher, np.ascontiguousarray(value).tobytes())
        return
    if isinstance(value, Mapping):
        _write_digest_tag(hasher, "mapping")
        keys = list(value)
        if not all(isinstance(key, str) for key in keys):
            raise TypeError("semantic mappings must use string keys")
        for key in sorted(keys):
            _write_digest_tag(hasher, "mapping-key")
            _write_digest_bytes(hasher, key.encode("utf-8"))
            _write_digest_tag(hasher, "mapping-value")
            _update_semantic_digest(hasher, value[key])
        return
    if isinstance(value, list):
        _write_digest_tag(hasher, "list")
        for item in value:
            _update_semantic_digest(hasher, item)
        return
    if isinstance(value, tuple):
        _write_digest_tag(hasher, "tuple")
        for item in value:
            _update_semantic_digest(hasher, item)
        return
    if is_dataclass(value) and not isinstance(value, type):
        _write_digest_tag(hasher, f"dataclass:{type(value).__module__}.{type(value).__qualname__}")
        for field in fields(value):
            _write_digest_bytes(hasher, field.name.encode("utf-8"))
            _update_semantic_digest(hasher, getattr(value, field.name))
        return
    _write_digest_tag(hasher, "atom")
    _write_json_header(hasher, {"value": _atom(value)})


def semantic_output_digest(value: Any) -> str:
    """Return a stable digest of the semantic financial output, or fail closed."""

    hasher = hashlib.sha256()
    _write_digest_tag(hasher, SEMANTIC_DIGEST_SCHEMA)
    _update_semantic_digest(hasher, value)
    return hasher.hexdigest()


def _factor_inputs(workload: Workload) -> tuple[pd.Series, pd.DataFrame, pd.Series]:
    if workload.factor is None:
        raise RuntimeError("factor workload did not provide factor data")
    factor = workload.factor["factor"]
    dates = factor.index.get_level_values("date").unique()
    assets = factor.index.get_level_values("asset").unique()
    groups = pd.Series(
        [f"G{number % 10:02d}" for number in range(len(assets))],
        index=pd.Index(assets, name="asset"),
        name="group",
    )
    rng = np.random.default_rng(workload.seed)
    innovations = rng.normal(0.0002, 0.01, (len(dates), len(assets)))
    prices = pd.DataFrame(
        100.0 * np.exp(np.cumsum(innovations, axis=0)),
        index=pd.Index(dates, name="date"),
        columns=pd.Index(assets, name="asset"),
    )
    return factor, prices, groups


def _calendar_descriptor(calendar: Any) -> dict[str, Any]:
    """Capture the deterministic calendar semantics returned by factor preparation."""

    holidays = getattr(calendar, "holidays", ())
    return {
        "type": f"{type(calendar).__module__}.{type(calendar).__qualname__}",
        "frequency": str(calendar),
        "n": int(getattr(calendar, "n", 1)),
        "weekmask": getattr(calendar, "weekmask", None),
        "holidays": [pd.Timestamp(value).isoformat() for value in holidays],
    }


def _workload_for(kind: str, scenario: str, seed: int) -> Workload:
    factories: dict[str, Callable[[str, int], Workload]] = {
        "metrics": single_series_workload,
        "rolling": rolling_returns_workload,
        "transactions": transactions_workload,
        "factor": factor_panel_workload,
        "risk": single_series_workload,
        "report": report_workload,
    }
    try:
        return factories[kind](scenario, seed)
    except KeyError as error:
        raise ValueError(f"kind must be one of {WORKLOAD_KINDS}") from error


def _invocation_for(kind: str, scenario: str, seed: int) -> _Invocation:
    """Build the exact inputs and direct Fincore callable for one measurement."""

    workload = _workload_for(kind, scenario, seed)
    workload_description = describe_workload(workload)

    if kind == "metrics":
        if workload.returns is None:
            raise RuntimeError("metrics workload did not provide returns")
        returns = workload.returns
        execution_inputs = {"kind": kind, "returns": returns, "period": "daily"}

        def run_metrics() -> Any:
            from fincore.metrics.perf_stats import perf_stats

            return perf_stats(returns, period="daily")

        run: Callable[[], Any] = run_metrics

    elif kind == "rolling":
        if workload.returns is None:
            raise RuntimeError("rolling workload did not provide returns")
        returns = workload.returns
        execution_inputs = {"kind": kind, "returns": returns, "window": ROLLING_WINDOW, "period": "daily"}

        def run_rolling() -> Any:
            from fincore.metrics.rolling import rolling_sharpe, rolling_volatility

            return {
                "rolling_sharpe": rolling_sharpe(returns, ROLLING_WINDOW, period="daily"),
                "rolling_volatility": rolling_volatility(returns, ROLLING_WINDOW, period="daily"),
            }

        run = run_rolling

    elif kind == "transactions":
        if workload.transactions is None:
            raise RuntimeError("transactions workload did not provide transactions")
        transactions = workload.transactions
        execution_inputs = {"kind": kind, "transactions": transactions, "matching": "fifo"}

        def run_transactions() -> Any:
            from fincore.portfolio.round_trips import extract_round_trips

            return extract_round_trips(transactions)

        run = run_transactions

    elif kind == "factor":
        factor, prices, groups = _factor_inputs(workload)
        execution_inputs = {
            "kind": kind,
            "factor": factor,
            "prices": prices,
            "groups": groups,
            "periods": [1, 5],
            "quantiles": 5,
            "max_loss": 1.0,
        }

        def run_factor() -> Any:
            from fincore.factor_analysis.data import prepare_factor_data

            prepared = prepare_factor_data(
                factor,
                prices,
                groupby=groups,
                periods=(1, 5),
                quantiles=5,
                max_loss=1.0,
            )
            return {
                "prepared_data": prepared.data,
                "loss_report": prepared.loss_report,
                "calendar": _calendar_descriptor(prepared.calendar),
            }

        run = run_factor

    elif kind == "risk":
        if workload.returns is None:
            raise RuntimeError("risk workload did not provide returns")
        returns = workload.returns
        execution_inputs = {"kind": kind, "returns": returns, "period": "daily", "cutoff": 0.05}

        def run_risk() -> Any:
            from fincore.metrics.risk import (
                annual_volatility,
                conditional_value_at_risk,
                downside_risk,
                tail_ratio,
                value_at_risk,
            )

            return {
                "annual_volatility": float(cast("float", annual_volatility(returns, period="daily"))),
                "conditional_value_at_risk": conditional_value_at_risk(returns, cutoff=0.05),
                "downside_risk": float(cast("float", downside_risk(returns, period="daily"))),
                "tail_ratio": tail_ratio(returns),
                "value_at_risk": value_at_risk(returns, cutoff=0.05),
            }

        run = run_risk

    elif kind == "report":
        if workload.returns is None:
            raise RuntimeError("report workload did not provide returns")
        returns = workload.returns
        execution_inputs = {
            "kind": kind,
            "returns": returns,
            "rolling_window": ROLLING_WINDOW,
            "period": "daily",
        }

        def run_report() -> Any:
            from fincore.report.compute import compute_sections

            return cast(
                "Mapping[str, Any]", compute_sections(returns, None, None, None, None, ROLLING_WINDOW, period="daily")
            )

        run = run_report

    else:  # pragma: no cover - _workload_for validates all public kinds first
        raise ValueError(f"kind must be one of {WORKLOAD_KINDS}")

    return _Invocation(
        workload=workload_description,
        execution_input_digest=semantic_output_digest(execution_inputs),
        run=run,
    )


def _profile_workload(workload: Callable[[], Any]) -> tuple[float, list[dict[str, Any]], Any]:
    """Run one callable under cProfile and return its semantic result separately."""

    result: Any = None

    def run() -> None:
        nonlocal result
        result = workload()

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
    hot: list[dict[str, Any]] = []
    for name, func in stats.get_stats_profile().func_profiles.items():
        hot.append(
            {
                "function": f"{name} ({Path(func.file_name).name}:{func.line_number})",
                "cumtime_seconds": round(func.cumtime, 9),
                "calls": func.ncalls,
            }
        )
        if len(hot) >= TOP_FUNCTIONS:
            break
    return wall_seconds, hot, result


def _execute(kind: str, scenario: str, seed: int, *, profile: bool) -> _Execution:
    invocation = _invocation_for(kind, scenario, seed)
    if profile:
        wall_seconds, hotspots, result = _profile_workload(invocation.run)
    else:
        started = time.perf_counter()
        result = invocation.run()
        wall_seconds = time.perf_counter() - started
        hotspots = []
    return _Execution(
        workload=invocation.workload,
        execution_input_digest=invocation.execution_input_digest,
        output_digest=semantic_output_digest(result),
        wall_seconds=wall_seconds,
        hotspots=hotspots,
    )


def _assert_case_identity(reference: _Execution | None, current: _Execution, *, kind: str, scenario: str) -> _Execution:
    if reference is None:
        return current
    if current.workload != reference.workload:
        raise RuntimeError(f"{kind}/{scenario} rebuilt a different workload identity")
    if current.execution_input_digest != reference.execution_input_digest:
        raise RuntimeError(f"{kind}/{scenario} rebuilt different execution inputs")
    return reference


def _linear_percentile(samples: list[float], percentile: float) -> float:
    if not samples:
        raise ValueError("cannot calculate a percentile without samples")
    ordered = sorted(samples)
    position = (len(ordered) - 1) * percentile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * fraction


def _timing_summary(samples: list[float]) -> dict[str, float]:
    if not samples:
        raise ValueError("repeats must provide at least one timing sample")
    return {
        "minimum_seconds": round(min(samples), 9),
        "median_seconds": round(_linear_percentile(samples, 0.5), 9),
        "p95_seconds": round(_linear_percentile(samples, 0.95), 9),
        "maximum_seconds": round(max(samples), 9),
    }


def _validate_measurement(warmups: int, repeats: int) -> None:
    if isinstance(warmups, bool) or not isinstance(warmups, int) or warmups < 0:
        raise ValueError("warmups must be a non-negative integer")
    if isinstance(repeats, bool) or not isinstance(repeats, int) or repeats < 1:
        raise ValueError("repeats must be a positive integer")


def profile_case(
    scenario: str,
    kind: str,
    *,
    seed: int = 20260817,
    warmups: int = 0,
    repeats: int = 1,
    require_output_digest: bool = False,
) -> dict[str, Any]:
    """Profile one workload with stable input/output identity checks."""

    if scenario not in ("small", "medium", "large"):
        raise ValueError("scenario must be one of ('small', 'medium', 'large')")
    if kind not in WORKLOAD_KINDS:
        raise ValueError(f"kind must be one of {WORKLOAD_KINDS}")
    _validate_measurement(warmups, repeats)

    reference: _Execution | None = None
    warmup_digests: list[str] = []
    measured_digests: list[str] = []
    timing_samples: list[float] = []
    for _ in range(warmups):
        execution = _execute(kind, scenario, seed, profile=False)
        reference = _assert_case_identity(reference, execution, kind=kind, scenario=scenario)
        warmup_digests.append(execution.output_digest)
    for _ in range(repeats):
        execution = _execute(kind, scenario, seed, profile=False)
        reference = _assert_case_identity(reference, execution, kind=kind, scenario=scenario)
        measured_digests.append(execution.output_digest)
        timing_samples.append(execution.wall_seconds)
    profiled = _execute(kind, scenario, seed, profile=True)
    reference = _assert_case_identity(reference, profiled, kind=kind, scenario=scenario)

    all_output_digests = [*warmup_digests, *measured_digests, profiled.output_digest]
    if not all_output_digests or any(len(digest) != 64 for digest in all_output_digests):
        raise RuntimeError(f"{kind}/{scenario} did not emit a SHA256 semantic output digest")
    if len(set(all_output_digests)) != 1:
        raise RuntimeError(f"{kind}/{scenario} emitted unstable semantic output digests: {all_output_digests!r}")
    if require_output_digest and not all_output_digests[0]:  # pragma: no cover - defensive CLI clarity
        raise RuntimeError(f"{kind}/{scenario} requires a semantic output digest")
    assert reference is not None
    timing = _timing_summary(timing_samples)
    return {
        "schema": HOTSPOT_PROFILE_SCHEMA,
        "kind": kind,
        "measurement": {
            "warmups": warmups,
            "repeats": repeats,
            "require_output_digest": require_output_digest,
            "timing_unit": "seconds",
            "percentile_method": "linear",
        },
        "provenance": _provenance(),
        "workload": reference.workload,
        "execution_input_digest": reference.execution_input_digest,
        "output_digest": all_output_digests[0],
        "warmup_output_digests": warmup_digests,
        "measured_output_digests": measured_digests,
        "profiled_output_digest": profiled.output_digest,
        "timing_samples_seconds": [round(sample, 9) for sample in timing_samples],
        "timing": timing,
        "profiled_wall_seconds": round(profiled.wall_seconds, 9),
        "cold_import_seconds": round(_cold_import_seconds(), 9),
        "peak_rss_bytes": _rss_bytes(),
        "hotspots": profiled.hotspots,
    }


def _provenance() -> dict[str, Any]:
    import numpy
    import pandas

    return {
        "commit": subprocess.run(
            ["git", "-C", str(ROOT), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip(),
        "python": platform.python_version(),
        "numpy": numpy.__version__,
        "pandas": pandas.__version__,
        "platform_label": f"{sys.platform}-{platform.machine()}",
        "dirty": bool(
            subprocess.run(["git", "-C", str(ROOT), "status", "--porcelain"], capture_output=True, text=True).stdout
        ),
    }


def _render_markdown(data: Mapping[str, Any]) -> str:
    lines = [
        "# Hotspot Profile",
        "",
        f"- kind: `{data['kind']}` workload: `{data['workload']['name']}` size `{data['workload']['size']}`",
        f"- seed: `{data['workload']['seed']}` input_digest: `{data['workload']['input_digest']}`",
        f"- execution_input_digest: `{data['execution_input_digest']}`",
        f"- output_digest: `{data['output_digest']}`",
        f"- repeats: `{data['measurement']['repeats']}` median seconds: `{data['timing']['median_seconds']:.9f}`",
        f"- cold import seconds: `{data['cold_import_seconds']:.9f}`",
        "",
        "| rank | function | cumtime (s) | calls |",
        "| --- | --- | ---: | ---: |",
    ]
    for rank, hot in enumerate(data["hotspots"], start=1):
        lines.append(f"| {rank} | `{hot['function']}` | {hot['cumtime_seconds']:.9f} | {hot['calls']} |")
    lines.append("")
    return "\n".join(lines)


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
    parser.add_argument("--scenario", choices=("small", "medium", "large"), default="medium")
    parser.add_argument("--kind", choices=WORKLOAD_KINDS, default="factor")
    parser.add_argument("--output", required=True, help="JSON output path")
    parser.add_argument("--seed", type=int, default=20260817)
    parser.add_argument("--warmups", type=_nonnegative_int, default=0)
    parser.add_argument("--repeats", type=_positive_int, default=1)
    parser.add_argument("--require-output-digest", action="store_true")
    args = parser.parse_args(argv)

    data = profile_case(
        args.scenario,
        args.kind,
        seed=args.seed,
        warmups=args.warmups,
        repeats=args.repeats,
        require_output_digest=args.require_output_digest,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    markdown_path = output.with_suffix(".md")
    markdown_path.write_text(_render_markdown(data), encoding="utf-8")
    print(f"wrote {output} and {markdown_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
