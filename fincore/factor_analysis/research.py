"""Thin research-evaluation entry point for externally collected factor panels.

The Backtrader research adapter collects factor panels as plain pandas objects
and delegates evaluation here.  This module deliberately does not re-implement
factor statistics: labels come from the same forward-return convention as
:mod:`fincore.factor_analysis.data`, binning/cleaning goes through
:func:`prepare_factor_data_from_forward_returns`, and the information
coefficient, quantile-return, spread and stability kernels are imported from
:mod:`fincore.factor_analysis.performance`.

The public surface is a single normalization helper, a capability registry and
:func:`evaluate_factor_research`.  Results are plain transport tables so a
caller can persist them without importing any renderer.
"""

from __future__ import annotations

import json
import math
import warnings
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from fincore.exceptions import FincoreError
from fincore.factor_analysis.calendar import infer_trading_calendar
from fincore.factor_analysis.data import (
    FactorLossExceededError,
    _period_label,
    prepare_factor_data_from_forward_returns,
)
from fincore.factor_analysis.models import fingerprint_value
from fincore.factor_analysis.performance import (
    compute_mean_returns_spread,
    factor_information_coefficient,
    factor_rank_autocorrelation,
    mean_return_by_quantile,
    quantile_turnover,
)

if TYPE_CHECKING:
    from collections.abc import Iterable

API_VERSION = "1.0"
RESULT_SCHEMA_VERSION = 1
EVALUATION_CONFIG_SCHEMA_VERSION = 1

LABELS = ("close_to_close", "next_open_to_open", "precomputed")
PREPROCESS_METHODS = ("cross_section_rank", "cross_section_zscore", "none")
METRICS = (
    "pearson_ic",
    "spearman_ic",
    "icir",
    "quantile_returns",
    "spread",
    "rank_autocorrelation",
    "quantile_turnover",
    "factor_correlation",
    "paired_comparison",
)
RESULT_TABLES = (
    "signals",
    "labels",
    "ic",
    "ic_summary",
    "quantile_returns",
    "turnover",
    "spreads",
    "stability",
    "correlation",
    "comparisons",
    "coverage",
    "diagnostics",
)
EXPORT_FORMATS = ("json", "csv")
RESULT_STATUSES = ("OK", "PARTIAL", "UNEVALUABLE")

# ``precomputed`` is supported only when ``precomputed_returns`` is supplied;
# the entry rejects it explicitly otherwise.  It is not a derived price label.
# definition: it selects a caller-supplied forward-return frame and therefore
# is accepted by the normalizer without being advertised as a computable label.
PRECOMPUTED_LABEL = "precomputed"

_SPEC_FIELDS = (
    "targets",
    "start",
    "end",
    "snapshot_id",
    "price_basis",
    "calendar_id",
    "frequency",
    "source_timezone",
    "feed_time_kind",
    "periods",
    "label",
    "precomputed_entry",
    "quantiles",
    "min_assets",
    "min_quantile_assets",
    "preprocess",
    "max_loss",
    "stability_lags",
    "direction_overrides",
    "combinations",
    "comparisons",
    "segments",
)


class _ResearchError(FincoreError):
    """Base class carrying a machine-readable code and a diagnostic report."""

    default_code = "research_error"

    def __init__(
        self,
        message: str = "",
        *,
        code: str | None = None,
        report: Mapping[str, object] | None = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.code = code or self.default_code
        self.report = dict(report) if report is not None else None

    def __str__(self) -> str:
        if self.report:
            return f"{self.message} [{self.code}] report={self.report!r}"
        return f"{self.message} [{self.code}]"


class ConfigError(_ResearchError):
    """The evaluation specification is invalid or contradictory."""

    default_code = "invalid_config"


class DatasetError(_ResearchError):
    """A supplied panel, grid, price, universe or group input is malformed."""

    default_code = "invalid_dataset"


class DependencyError(_ResearchError):
    """An optional dependency required by the requested operation is absent."""

    default_code = "missing_dependency"


class DataQualityError(_ResearchError):
    """The dataset is structurally valid but violates an integrity contract."""

    default_code = "data_quality"


class FactorComputationError(_ResearchError):
    """A reused fincore kernel failed unexpectedly for one factor slice."""

    default_code = "factor_computation_failed"


@dataclass(frozen=True)
class ResearchEvaluationSpec:
    """Immutable, validated description of one research evaluation request."""

    targets: tuple[str, ...]
    start: pd.Timestamp
    end: pd.Timestamp
    snapshot_id: str = "unspecified"
    price_basis: str = "unspecified"
    calendar_id: str = "unspecified"
    frequency: str = "unspecified"
    source_timezone: str = "UTC"
    feed_time_kind: str = "close_time"
    periods: tuple[int, ...] = (1, 5, 20)
    label: str = "next_open_to_open"
    precomputed_entry: str = "next_open"
    quantiles: int = 5
    min_assets: int = 10
    min_quantile_assets: int = 1
    preprocess: str = "cross_section_rank"
    max_loss: float = 1.0
    stability_lags: tuple[int, ...] = (1, 5, 20)
    direction_overrides: Mapping[str, int] = field(default_factory=dict)
    combinations: Mapping[str, Mapping[str, float]] = field(default_factory=dict)
    comparisons: Mapping[str, Sequence[str]] = field(default_factory=dict)
    segments: Mapping[str, tuple] = field(default_factory=dict)

    def __post_init__(self) -> None:
        normalized = _normalize_spec_values(_spec_values(self))
        for name, value in normalized.items():
            object.__setattr__(self, name, value)


@dataclass(frozen=True)
class ResearchEvaluationResult:
    """Transport-friendly evaluation tables plus provenance manifest."""

    signals: pd.DataFrame
    labels: pd.DataFrame
    ic: pd.DataFrame
    ic_summary: pd.DataFrame
    quantile_returns: pd.DataFrame
    turnover: pd.DataFrame
    spreads: pd.DataFrame
    stability: pd.DataFrame
    correlation: pd.DataFrame
    comparisons: pd.DataFrame
    coverage: pd.DataFrame
    diagnostics: pd.DataFrame
    status: str
    manifest: Mapping[str, Any]

    def table(self, name: str) -> pd.DataFrame:
        """Return one named result table (all names are public attributes)."""

        if name not in RESULT_TABLES:
            raise KeyError(f"unknown result table {name!r}")
        value = getattr(self, name)
        if not isinstance(value, pd.DataFrame):  # pragma: no cover - internal invariant
            raise TypeError(f"result table {name!r} is not a DataFrame")
        return value

    def to_serializable(self) -> dict[str, Any]:
        """Return a JSON-compatible mapping of every table and the manifest."""

        return {
            "status": self.status,
            "manifest": _json_ready(dict(self.manifest)),
            "tables": {name: _frame_records(self.table(name)) for name in RESULT_TABLES},
            "columns": {name: [str(column) for column in self.table(name).columns] for name in RESULT_TABLES},
        }

    def export(self, directory: str | Path, *, charts: bool = True) -> Path:
        """Write manifest, CSV tables and optional PNG charts into a new folder.

        The destination is ``<directory>/<evaluation_id>``.  An existing
        ``manifest.json`` there is never overwritten; the caller must choose a
        different root or evaluation.
        """

        if charts and not _charts_available():
            raise DependencyError(
                "matplotlib is required to export charts",
                code="charts_unavailable",
                report={"dependency": "matplotlib", "hint": "pass charts=False to export tables only"},
            )
        evaluation_id = str(self.manifest["evaluation_id"])
        target = Path(directory).expanduser() / evaluation_id
        target.mkdir(parents=True, exist_ok=True)
        manifest_path = target / "manifest.json"
        if manifest_path.exists():
            raise DatasetError(
                "evaluation export already exists and will not be overwritten",
                code="export_target_exists",
                report={"path": str(manifest_path)},
            )
        manifest = dict(self.manifest)
        manifest["exports"] = [f"{name}.csv" for name in RESULT_TABLES]
        if charts:
            manifest["exports"].extend(["ic_series.png", "quantile_returns.png"])
        payload = _json_ready(manifest)
        manifest_path.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False), encoding="utf-8")
        for name in RESULT_TABLES:
            self.table(name).to_csv(target / f"{name}.csv", index=False, encoding="utf-8")
        if charts:
            _write_charts(self, target)
        return target


def _spec_values(spec: ResearchEvaluationSpec) -> dict[str, Any]:
    return {name: getattr(spec, name) for name in _SPEC_FIELDS}


def _string(value: object, field_name: str, *, allow_empty: bool = False) -> str:
    if not isinstance(value, str):
        raise ConfigError(f"{field_name} must be a string", code="spec_field_type", report={"field": field_name})
    if not allow_empty and not value:
        raise ConfigError(f"{field_name} must not be empty", code="spec_field_empty", report={"field": field_name})
    return value


def _string_tuple(value: object, field_name: str, *, allow_empty: bool) -> tuple[str, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ConfigError(
            f"{field_name} must be a sequence of strings",
            code="spec_field_type",
            report={"field": field_name},
        )
    items = tuple(value)
    if not items and not allow_empty:
        raise ConfigError(f"{field_name} must not be empty", code="spec_field_empty", report={"field": field_name})
    if any(not isinstance(item, str) or not item for item in items):
        raise ConfigError(
            f"{field_name} must contain non-empty strings",
            code="spec_field_value",
            report={"field": field_name},
        )
    if len(set(items)) != len(items):
        raise ConfigError(
            f"{field_name} must not contain duplicates",
            code="spec_field_duplicate",
            report={"field": field_name},
        )
    return items


def _positive_int(value: object, field_name: str, *, minimum: int = 1) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise ConfigError(f"{field_name} must be an integer", code="spec_field_type", report={"field": field_name})
    number = int(value)
    if number < minimum:
        raise ConfigError(
            f"{field_name} must be at least {minimum}",
            code="spec_field_value",
            report={"field": field_name, "value": number},
        )
    return number


def _int_tuple(value: object, field_name: str, *, minimum: int, allow_empty: bool) -> tuple[int, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ConfigError(
            f"{field_name} must be a sequence of integers",
            code="spec_field_type",
            report={"field": field_name},
        )
    items = tuple(_positive_int(item, field_name, minimum=minimum) for item in value)
    if not items and not allow_empty:
        raise ConfigError(f"{field_name} must not be empty", code="spec_field_empty", report={"field": field_name})
    if len(set(items)) != len(items):
        raise ConfigError(
            f"{field_name} must not contain duplicates",
            code="spec_field_duplicate",
            report={"field": field_name},
        )
    return items


def _spec_timestamp(value: object, field_name: str) -> pd.Timestamp:
    try:
        timestamp = pd.Timestamp(value)
    except (TypeError, ValueError) as error:
        raise ConfigError(
            f"{field_name} must be a timestamp",
            code="spec_field_type",
            report={"field": field_name},
        ) from error
    if timestamp is pd.NaT or pd.isna(timestamp):
        raise ConfigError(f"{field_name} must not be NaT", code="spec_field_value", report={"field": field_name})
    if timestamp.tzinfo is None:
        timestamp = timestamp.tz_localize("UTC")
    elif not _is_utc(timestamp.tzinfo):
        raise ConfigError(
            f"{field_name} must be UTC",
            code="spec_timezone",
            report={"field": field_name, "timezone": str(timestamp.tzinfo)},
        )
    return timestamp


def _is_utc(timezone: object) -> bool:
    return timezone is not None and str(timezone).upper() in {"UTC", "UTC+00:00"}


def _mapping(value: object, field_name: str) -> Mapping[object, object]:
    if not isinstance(value, Mapping):
        raise ConfigError(f"{field_name} must be a mapping", code="spec_field_type", report={"field": field_name})
    return value


def _normalize_spec_values(values: Mapping[str, Any]) -> dict[str, Any]:
    targets = _string_tuple(values["targets"], "targets", allow_empty=False)
    start = _spec_timestamp(values["start"], "start")
    end = _spec_timestamp(values["end"], "end")
    if start > end:
        raise ConfigError(
            "spec.start must not be after spec.end",
            code="spec_window",
            report={"start": start.isoformat(), "end": end.isoformat()},
        )

    label = _string(values["label"], "label")
    if label not in (*LABELS, PRECOMPUTED_LABEL):
        raise ConfigError(
            f"unsupported label {label!r}",
            code="unsupported_label",
            report={"label": label, "supported": [*LABELS, PRECOMPUTED_LABEL]},
        )
    preprocess = _string(values["preprocess"], "preprocess")
    if preprocess not in PREPROCESS_METHODS:
        raise ConfigError(
            f"unsupported preprocess {preprocess!r}",
            code="unsupported_preprocess",
            report={"preprocess": preprocess, "supported": list(PREPROCESS_METHODS)},
        )

    periods = _int_tuple(values["periods"], "periods", minimum=1, allow_empty=False)
    stability_lags = _int_tuple(values["stability_lags"], "stability_lags", minimum=1, allow_empty=True)
    quantiles = _positive_int(values["quantiles"], "quantiles", minimum=1)
    min_assets = _positive_int(values["min_assets"], "min_assets", minimum=1)
    min_quantile_assets = _positive_int(values["min_quantile_assets"], "min_quantile_assets", minimum=1)
    max_loss_value = values["max_loss"]
    if isinstance(max_loss_value, bool) or not isinstance(max_loss_value, (int, float, np.number)):
        raise ConfigError("max_loss must be a number", code="spec_field_type", report={"field": "max_loss"})
    max_loss = float(max_loss_value)
    if not math.isfinite(max_loss) or not 0.0 <= max_loss <= 1.0:
        raise ConfigError(
            "max_loss must be between 0 and 1",
            code="spec_field_value",
            report={"field": "max_loss", "value": max_loss},
        )

    overrides: dict[str, int] = {}
    for name, direction in _mapping(values["direction_overrides"], "direction_overrides").items():
        if not isinstance(name, str) or not name:
            raise ConfigError(
                "direction_overrides keys must be non-empty strings",
                code="spec_field_value",
                report={"field": "direction_overrides"},
            )
        if isinstance(direction, bool) or not isinstance(direction, (int, np.integer)) or int(direction) not in (-1, 1):
            raise ConfigError(
                "direction_overrides values must be +1 or -1",
                code="spec_field_value",
                report={"field": "direction_overrides", "factor": name},
            )
        overrides[name] = int(direction)

    combinations: dict[str, dict[str, float]] = {}
    for name, weights_value in _mapping(values["combinations"], "combinations").items():
        if not isinstance(name, str) or not name:
            raise ConfigError(
                "combination names must be non-empty strings",
                code="spec_field_value",
                report={"field": "combinations"},
            )
        weights: dict[str, float] = {}
        for factor_name, weight_value in _mapping(weights_value, f"combinations[{name}]").items():
            if not isinstance(factor_name, str) or not factor_name:
                raise ConfigError(
                    "combination factor names must be non-empty strings",
                    code="spec_field_value",
                    report={"field": f"combinations[{name}]"},
                )
            if isinstance(weight_value, bool) or not isinstance(weight_value, (int, float, np.number)):
                raise ConfigError(
                    "combination weights must be numbers",
                    code="spec_field_type",
                    report={"combination": name, "factor": factor_name},
                )
            weight = float(weight_value)
            if not math.isfinite(weight):
                raise ConfigError(
                    "combination weights must be finite",
                    code="spec_field_value",
                    report={"combination": name, "factor": factor_name},
                )
            weights[factor_name] = weight
        if not weights or not any(weight != 0.0 for weight in weights.values()):
            raise ConfigError(
                "combinations require at least one non-zero weight",
                code="spec_field_value",
                report={"combination": name},
            )
        combinations[name] = weights

    comparisons: dict[str, tuple[str, str]] = {}
    for name, pair_value in _mapping(values["comparisons"], "comparisons").items():
        if not isinstance(name, str) or not name:
            raise ConfigError(
                "comparison names must be non-empty strings",
                code="spec_field_value",
                report={"field": "comparisons"},
            )
        pair = _string_tuple(pair_value, f"comparisons[{name}]", allow_empty=False)
        if len(pair) != 2:
            raise ConfigError(
                "comparisons must name exactly two factors",
                code="spec_field_value",
                report={"comparison": name, "length": len(pair)},
            )
        if pair[0] == pair[1]:
            raise ConfigError(
                "comparison sides must be distinct",
                code="spec_field_value",
                report={"comparison": name},
            )
        comparisons[name] = (pair[0], pair[1])

    segments: dict[str, tuple[object, object]] = {}
    for name, segment_value in _mapping(values["segments"], "segments").items():
        if not isinstance(name, str) or not name:
            raise ConfigError(
                "segment names must be non-empty strings",
                code="spec_field_value",
                report={"field": "segments"},
            )
        if isinstance(segment_value, (str, bytes)) or not isinstance(segment_value, Sequence):
            raise ConfigError(
                "segments must be two-element sequences",
                code="spec_field_type",
                report={"segment": name},
            )
        segment = tuple(segment_value)
        if len(segment) != 2:
            raise ConfigError(
                "segments must have exactly two bounds",
                code="spec_field_value",
                report={"segment": name, "length": len(segment)},
            )
        segments[name] = (segment[0], segment[1])

    return {
        "targets": targets,
        "start": start,
        "end": end,
        "snapshot_id": _string(values["snapshot_id"], "snapshot_id", allow_empty=True),
        "price_basis": _string(values["price_basis"], "price_basis", allow_empty=True),
        "calendar_id": _string(values["calendar_id"], "calendar_id", allow_empty=True),
        "frequency": _string(values["frequency"], "frequency", allow_empty=True),
        "source_timezone": _string(values["source_timezone"], "source_timezone", allow_empty=True),
        "feed_time_kind": _string(values["feed_time_kind"], "feed_time_kind", allow_empty=True),
        "periods": periods,
        "label": label,
        "precomputed_entry": _string(values["precomputed_entry"], "precomputed_entry", allow_empty=True),
        "quantiles": quantiles,
        "min_assets": min_assets,
        "min_quantile_assets": min_quantile_assets,
        "preprocess": preprocess,
        "max_loss": max_loss,
        "stability_lags": stability_lags,
        "direction_overrides": MappingProxyType(overrides),
        "combinations": MappingProxyType(
            {name: MappingProxyType(dict(weights)) for name, weights in combinations.items()}
        ),
        "comparisons": MappingProxyType(comparisons),
        "segments": MappingProxyType(segments),
    }


def _spec_defaults() -> dict[str, Any]:
    return {
        "targets": (),
        "start": None,
        "end": None,
        "snapshot_id": "unspecified",
        "price_basis": "unspecified",
        "calendar_id": "unspecified",
        "frequency": "unspecified",
        "source_timezone": "UTC",
        "feed_time_kind": "close_time",
        "periods": (1, 5, 20),
        "label": "next_open_to_open",
        "precomputed_entry": "next_open",
        "quantiles": 5,
        "min_assets": 10,
        "min_quantile_assets": 1,
        "preprocess": "cross_section_rank",
        "max_loss": 1.0,
        "stability_lags": (1, 5, 20),
        "direction_overrides": {},
        "combinations": {},
        "comparisons": {},
        "segments": {},
    }


def normalize_research_evaluation_spec(values: Mapping[str, Any]) -> ResearchEvaluationSpec:
    """Validate caller values and fill every documented default."""

    if not isinstance(values, Mapping):
        raise ConfigError(
            "research evaluation spec must be a mapping",
            code="spec_type",
            report={"type": type(values).__name__},
        )
    unknown = sorted(str(key) for key in values if key not in _SPEC_FIELDS)
    if unknown:
        raise ConfigError(
            "unknown research evaluation spec fields",
            code="unknown_spec_field",
            report={"fields": unknown},
        )
    merged = _spec_defaults()
    merged.update(values)
    return ResearchEvaluationSpec(**_normalize_spec_values(merged))


def research_capabilities() -> dict[str, Any]:
    """Describe this module's public contract to an external adapter."""

    return {
        "api_version": API_VERSION,
        "result_schema_version": RESULT_SCHEMA_VERSION,
        "evaluation_config_schema_version": EVALUATION_CONFIG_SCHEMA_VERSION,
        "python_min": "3.11",
        "entrypoint": "fincore.factor_analysis.research.evaluate_factor_research",
        "spec": "fincore.factor_analysis.research.ResearchEvaluationSpec",
        "normalizer": "fincore.factor_analysis.research.normalize_research_evaluation_spec",
        "labels": LABELS,
        "preprocess": PREPROCESS_METHODS,
        "metrics": METRICS,
        "result_tables": RESULT_TABLES,
        "export_formats": EXPORT_FORMATS,
        "charts_supported": _charts_available(),
        "result_statuses": RESULT_STATUSES,
        "precomputed_label": PRECOMPUTED_LABEL,
    }


def _charts_available() -> bool:
    try:
        import matplotlib  # presence probe only
    except ImportError:
        return False
    return True


def _coerce_spec(spec: ResearchEvaluationSpec | Mapping[str, Any]) -> ResearchEvaluationSpec:
    if isinstance(spec, ResearchEvaluationSpec):
        return normalize_research_evaluation_spec(_spec_values(spec))
    if isinstance(spec, Mapping):
        return normalize_research_evaluation_spec(spec)
    raise ConfigError(
        "spec must be a ResearchEvaluationSpec or a mapping",
        code="spec_type",
        report={"type": type(spec).__name__},
    )


def _spec_payload(spec: ResearchEvaluationSpec) -> dict[str, Any]:
    return {
        "targets": list(spec.targets),
        "start": spec.start.isoformat(),
        "end": spec.end.isoformat(),
        "snapshot_id": spec.snapshot_id,
        "price_basis": spec.price_basis,
        "calendar_id": spec.calendar_id,
        "frequency": spec.frequency,
        "source_timezone": spec.source_timezone,
        "feed_time_kind": spec.feed_time_kind,
        "periods": list(spec.periods),
        "label": spec.label,
        "precomputed_entry": spec.precomputed_entry,
        "quantiles": spec.quantiles,
        "min_assets": spec.min_assets,
        "min_quantile_assets": spec.min_quantile_assets,
        "preprocess": spec.preprocess,
        "max_loss": spec.max_loss,
        "stability_lags": list(spec.stability_lags),
        "direction_overrides": {name: int(direction) for name, direction in sorted(spec.direction_overrides.items())},
        "combinations": {
            name: {factor: float(weight) for factor, weight in sorted(weights.items())}
            for name, weights in sorted(spec.combinations.items())
        },
        "comparisons": {name: list(pair) for name, pair in sorted(spec.comparisons.items())},
        "segments": {name: [str(bound) for bound in bounds] for name, bounds in sorted(spec.segments.items())},
    }


def _frame_records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    return [{str(key): _json_ready(value) for key, value in row.items()} for row in frame.to_dict(orient="records")]


def _json_ready(value: object) -> Any:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        number = float(value)
        return number if math.isfinite(number) else None
    if isinstance(value, np.datetime64):
        return pd.Timestamp(value).isoformat()
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, pd.Timedelta):
        return str(value)
    if isinstance(value, np.generic):
        return _json_ready(value.item())
    if isinstance(value, Mapping):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_json_ready(item) for item in value]
    return str(value)


def _empty_frame(columns: Sequence[str]) -> pd.DataFrame:
    return pd.DataFrame({column: pd.Series(dtype=object) for column in columns})


def _require_utc_index(
    index: object,
    *,
    field: str,
    code: str,
    require_unique: bool = True,
) -> pd.DatetimeIndex:
    try:
        converted = pd.DatetimeIndex(index)
    except (TypeError, ValueError) as error:
        raise DataQualityError(
            f"{field} index must contain datetimes",
            code=f"{code}_index_type",
            report={"field": field},
        ) from error
    if converted.tz is None or not _is_utc(converted.tz):
        raise DataQualityError(
            f"{field} index must be timezone-aware UTC",
            code=f"{code}_timezone",
            report={"field": field, "timezone": None if converted.tz is None else str(converted.tz)},
        )
    if require_unique and converted.has_duplicates:
        raise DataQualityError(
            f"{field} index must be unique",
            code=f"{code}_duplicate_index",
            report={"field": field},
        )
    return converted


def _require_wide_frame(frame: object, *, field: str, code: str) -> pd.DataFrame:
    if not isinstance(frame, pd.DataFrame):
        raise DatasetError(
            f"{field} must be a pandas DataFrame",
            code=f"{code}_type",
            report={"field": field},
        )
    if frame.columns.has_duplicates:
        raise DataQualityError(
            f"{field} columns must be unique",
            code=f"{code}_duplicate_columns",
            report={"field": field},
        )
    return frame


def _numeric_frame(frame: pd.DataFrame, *, field: str, code: str) -> pd.DataFrame:
    try:
        return frame.astype(float)
    except (TypeError, ValueError) as error:
        raise DataQualityError(
            f"{field} values must be numeric",
            code=f"{code}_non_numeric",
            report={"field": field},
        ) from error


def _validate_factors(factors: object, spec: ResearchEvaluationSpec) -> pd.DataFrame:
    if not isinstance(factors, pd.DataFrame):
        raise DatasetError("factors must be a pandas DataFrame", code="factors_type")
    index = factors.index
    if not isinstance(index, pd.MultiIndex) or index.nlevels != 2:
        raise DatasetError("factors must use a two-level (time, asset) MultiIndex", code="factors_index")
    times = _require_utc_index(index.get_level_values(0), field="factors", code="factors", require_unique=False)
    assets = pd.Index(index.get_level_values(1))
    if index.has_duplicates:
        raise DataQualityError("factor index keys must be unique", code="factors_duplicate_keys")
    if factors.columns.has_duplicates:
        raise DataQualityError("factor columns must be unique", code="factors_duplicate_columns")
    if factors.shape[1] == 0:
        raise DatasetError("factors must contain at least one factor column", code="factors_empty")
    names = tuple(factors.columns)
    if any(not isinstance(name, str) or not name for name in names):
        raise DatasetError("factor column names must be non-empty strings", code="factors_column_name")
    missing_targets = [target for target in spec.targets if target not in assets]
    if missing_targets:
        raise DatasetError(
            "factor panel does not contain every requested target asset",
            code="factors_missing_targets",
            report={"missing": missing_targets},
        )
    panel = factors.copy(deep=True)
    panel.index = pd.MultiIndex.from_arrays((times, assets), names=("time", "asset"))
    return _numeric_frame(panel, field="factors", code="factors")


def _study_grid(grid: object, spec: ResearchEvaluationSpec) -> tuple[pd.DatetimeIndex | None, pd.DatetimeIndex]:
    if grid is None:
        return None, pd.DatetimeIndex([])
    frame = _require_wide_frame(grid, field="grid", code="grid")
    index = _require_utc_index(frame.index, field="grid", code="grid")
    if not index.is_monotonic_increasing:
        raise DataQualityError("grid index must be sorted increasing", code="grid_order")
    if spec.label == "next_open_to_open" and "open_time" not in frame.columns:
        raise DatasetError(
            "grid must contain an 'open_time' column for next_open_to_open labels",
            code="grid_missing_open_time",
        )
    study = index[(index >= spec.start) & (index <= spec.end)]
    if study.empty:
        raise DatasetError("grid contains no dates inside the requested window", code="grid_empty_window")
    return index, study


def _validate_prices(
    prices: object,
    spec: ResearchEvaluationSpec,
    grid_index: pd.DatetimeIndex | None,
) -> Mapping[str, pd.DataFrame]:
    if grid_index is None:
        return MappingProxyType({})
    if not isinstance(prices, Mapping):
        raise DatasetError("prices must be a mapping of price basis to wide frames", code="prices_type")
    validated: dict[str, pd.DataFrame] = {}
    for key in ("open", "close"):
        if key not in prices:
            continue
        frame = _require_wide_frame(prices[key], field=f"prices[{key}]", code="prices")
        frame_index = _require_utc_index(frame.index, field=f"prices[{key}]", code="prices")
        if not frame_index.equals(grid_index):
            raise DatasetError(
                f"prices[{key}] must be indexed exactly by the grid index",
                code="prices_index_mismatch",
                report={"basis": key},
            )
        missing = [target for target in spec.targets if target not in frame.columns]
        if missing:
            raise DatasetError(
                f"prices[{key}] is missing requested targets",
                code="prices_missing_targets",
                report={"basis": key, "missing": missing},
            )
        validated[key] = _numeric_frame(
            frame.reindex(columns=list(spec.targets)), field=f"prices[{key}]", code="prices"
        )
    return MappingProxyType(validated)


def _price_basis(spec: ResearchEvaluationSpec, prices: Mapping[str, pd.DataFrame]) -> str:
    basis = "close" if spec.label == "close_to_close" else "open"
    if basis not in prices:
        raise ConfigError(
            f"label {spec.label!r} requires prices[{basis!r}]",
            code="missing_price_basis",
            report={"label": spec.label, "basis": basis, "available": sorted(prices)},
        )
    return basis


def _canonical_period_labels(
    periods: Sequence[int],
    grid_index: pd.DatetimeIndex | None,
    study_dates: pd.DatetimeIndex,
) -> tuple[str, ...]:
    if grid_index is None or len(study_dates) == 0:
        labels = tuple(f"{period}D" for period in periods)
    else:
        calendar = infer_trading_calendar(study_dates, grid_index)
        labels = tuple(_period_label(grid_index, study_dates, period, calendar) for period in periods)
    if len(set(labels)) != len(labels):
        raise DataQualityError(
            "requested periods map to duplicate forward-return labels",
            code="labels_duplicate",
            report={"labels": list(labels)},
        )
    return labels


def _period_column(columns: pd.Index, period: int, label: str) -> object:
    for candidate in (period, label, str(period), f"{period}D"):
        if candidate in columns:
            return candidate
    raise DatasetError(
        "precomputed returns do not contain a requested period",
        code="precomputed_period_missing",
        report={"period": period, "label": label, "columns": [str(column) for column in columns]},
    )


def _from_wide_matrix(frame: pd.DataFrame) -> pd.Series:
    values = frame.to_numpy(dtype=float)
    index = pd.MultiIndex.from_product((frame.index, frame.columns), names=("time", "asset"))
    return pd.Series(values.reshape(-1), index=index)


def _precomputed_label_frame(
    returns: object,
    *,
    periods: Sequence[int],
    labels: Sequence[str],
    targets: Sequence[str],
) -> pd.DataFrame:
    if not isinstance(returns, pd.DataFrame):
        raise DatasetError("precomputed_returns must be a pandas DataFrame", code="precomputed_type")
    frame = returns
    if frame.columns.has_duplicates:
        raise DataQualityError(
            "precomputed_returns columns must be unique",
            code="precomputed_duplicate_columns",
        )
    extracted: dict[str, pd.Series] = {}
    if isinstance(frame.index, pd.MultiIndex) and frame.index.nlevels == 2:
        times = _require_utc_index(
            frame.index.get_level_values(0),
            field="precomputed_returns",
            code="precomputed",
            require_unique=False,
        )
        if frame.index.has_duplicates:
            raise DataQualityError("precomputed_returns keys must be unique", code="precomputed_duplicate_keys")
        assets = pd.Index(frame.index.get_level_values(1))
        for period, label in zip(periods, labels, strict=True):
            column = _period_column(frame.columns, period, label)
            series = frame[column].copy(deep=True)
            series.index = pd.MultiIndex.from_arrays((times, assets), names=("time", "asset"))
            extracted[label] = _numeric_series(series, field=f"precomputed_returns[{column}]")
    elif isinstance(frame.index, pd.DatetimeIndex):
        times = _require_utc_index(frame.index, field="precomputed_returns", code="precomputed")
        if isinstance(frame.columns, pd.MultiIndex) and frame.columns.nlevels == 2:
            for period, label in zip(periods, labels, strict=True):
                column = _period_column(frame.columns.get_level_values(0), period, label)
                wide = frame.xs(column, axis=1, level=0, drop_level=True)
                wide.index = times
                wide = _numeric_frame(wide, field=f"precomputed_returns[{column}]", code="precomputed")
                extracted[label] = _from_wide_matrix(wide)
        elif all(isinstance(column, str) for column in frame.columns):
            if len(periods) != 1:
                raise DatasetError(
                    "a single-period asset-column precomputed frame needs exactly one period",
                    code="precomputed_shape",
                    report={"periods": list(periods)},
                )
            wide = _numeric_frame(
                frame.reindex(columns=list(targets)),
                field="precomputed_returns",
                code="precomputed",
            )
            wide.index = times
            extracted[labels[0]] = _from_wide_matrix(wide)
        else:
            raise DatasetError(
                "unsupported precomputed_returns column layout",
                code="precomputed_shape",
                report={"columns": [str(column) for column in frame.columns]},
            )
    elif {"time", "asset", "period"}.issubset({str(column) for column in frame.columns}):
        value_column = "value" if "value" in frame.columns else "return" if "return" in frame.columns else None
        if value_column is None:
            raise DatasetError(
                "long precomputed_returns frames need a 'value' or 'return' column",
                code="precomputed_shape",
            )
        period_column_name = next(column for column in frame.columns if str(column) == "period")
        time_column = next(column for column in frame.columns if str(column) == "time")
        asset_column = next(column for column in frame.columns if str(column) == "asset")
        for period, label in zip(periods, labels, strict=True):
            subset = frame[frame[period_column_name].isin((period, label, str(period)))]
            if subset.empty:
                raise DatasetError(
                    "precomputed_returns contains no rows for a requested period",
                    code="precomputed_period_missing",
                    report={"period": period},
                )
            times = _require_utc_index(subset[time_column], field="precomputed_returns", code="precomputed")
            index = pd.MultiIndex.from_arrays((times, pd.Index(subset[asset_column])), names=("time", "asset"))
            if index.has_duplicates:
                raise DataQualityError(
                    "precomputed_returns keys must be unique",
                    code="precomputed_duplicate_keys",
                    report={"period": period},
                )
            extracted[label] = _numeric_series(
                pd.Series(subset[value_column].to_numpy(), index=index),
                field=f"precomputed_returns[{label}]",
            )
    else:
        raise DatasetError(
            "precomputed_returns must be long or wide",
            code="precomputed_shape",
            report={"index": type(frame.index).__name__},
        )
    return pd.DataFrame(extracted)


def _numeric_series(series: pd.Series, *, field: str) -> pd.Series:
    try:
        values = pd.to_numeric(series, errors="raise").astype(float)
    except (TypeError, ValueError) as error:
        raise DataQualityError(f"{field} values must be numeric", code="series_non_numeric") from error
    values.name = series.name
    return values


def _validate_universe(
    universe: object,
    *,
    spec: ResearchEvaluationSpec,
    study_dates: pd.DatetimeIndex,
) -> tuple[np.ndarray, int]:
    if universe is None:
        return np.ones((len(study_dates), len(spec.targets)), dtype=bool), 0
    if isinstance(universe, pd.Series) and isinstance(universe.index, pd.MultiIndex):
        # Accept the long (time, asset) boolean form as well as a wide frame.
        if tuple(universe.index.names) != ("time", "asset"):
            raise DatasetError(
                "long universe index must be named (time, asset)",
                code="universe_index",
                report={"names": list(universe.index.names)},
            )
        universe = universe.unstack("asset")
    frame = _require_wide_frame(universe, field="universe", code="universe")
    _require_utc_index(frame.index, field="universe", code="universe")
    missing = [target for target in spec.targets if target not in frame.columns]
    if missing:
        raise DatasetError(
            "universe is missing requested targets",
            code="universe_missing_targets",
            report={"missing": missing},
        )
    aligned = frame.reindex(index=study_dates, columns=list(spec.targets))
    absent = int(np.count_nonzero(aligned.isna().to_numpy()))
    # A missing or null universe cell is treated as out of universe, but the
    # caller is told explicitly rather than silently receiving zeros.
    return aligned.fillna(False).to_numpy(dtype=bool), absent


def _resolve_groups(groups: object, assets: pd.Index) -> dict[object, object]:
    if groups is None:
        return dict.fromkeys(assets, "default")
    if not isinstance(groups, Mapping):
        raise DatasetError("groups must be an asset to group mapping", code="groups_type")
    missing = [asset for asset in assets if asset not in groups]
    if missing:
        raise DatasetError(
            "groups do not cover every evaluated asset",
            code="groups_missing_assets",
            report={"missing": [str(asset) for asset in missing]},
        )
    return {asset: groups[asset] for asset in assets}


def _direction_map(factors: pd.DataFrame, spec: ResearchEvaluationSpec) -> dict[str, int]:
    raw = factors.attrs.get("factor_directions", {})
    if raw is None:
        raw = {}
    if not isinstance(raw, Mapping):
        raise DatasetError("factors.attrs['factor_directions'] must be a mapping", code="directions_type")
    directions: dict[str, int] = {}
    for name in factors.columns:
        value = raw.get(name, 1)
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or int(value) not in (-1, 1):
            raise DatasetError(
                "factor directions must be +1 or -1",
                code="directions_value",
                report={"factor": name},
            )
        directions[str(name)] = int(value)
    extra = sorted(str(name) for name in raw if name not in factors.columns)
    if extra:
        raise DatasetError(
            "factor_directions names unknown factors",
            code="directions_unknown_factor",
            report={"factors": extra},
        )
    for name, direction in spec.direction_overrides.items():
        if name not in factors.columns:
            raise ConfigError(
                "direction_overrides names an unknown factor",
                code="direction_unknown_factor",
                report={"factor": name},
            )
        directions[name] = int(direction)
    return directions


def _diagnose(
    rows: list[dict[str, object]],
    *,
    factor: str | None,
    stage: str,
    severity: str,
    code: str,
    message: str,
) -> None:
    rows.append(
        {
            "factor": factor,
            "stage": stage,
            "severity": severity,
            "code": code,
            "message": message,
        }
    )


def _preprocess_matrix(matrix: np.ndarray, method: str) -> tuple[np.ndarray, int]:
    """Apply a date-wise cross-sectional transform; return values and losses."""

    if method == "none":
        return matrix.copy(), 0
    frame = pd.DataFrame(matrix)
    if method == "cross_section_rank":
        processed = frame.rank(axis=1, method="average", pct=True).to_numpy(dtype=float)
    else:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            mean = frame.mean(axis=1, skipna=True)
            deviation = frame.std(axis=1, ddof=1, skipna=True)
            processed = frame.sub(mean, axis=0).div(deviation.mask(deviation == 0.0), axis=0).to_numpy(dtype=float)
    before = np.isfinite(matrix).any(axis=1)
    after = np.isfinite(processed).any(axis=1)
    collapsed = int(np.count_nonzero(before & ~after))
    return processed, collapsed


def _summary(values: Sequence[float]) -> tuple[float, float, float, int]:
    array = np.asarray([value for value in values if value is not None and np.isfinite(value)], dtype=float)
    count = int(array.size)
    if count == 0:
        return math.nan, math.nan, math.nan, 0
    mean = float(array.mean())
    if count < 2:
        return mean, math.nan, math.nan, count
    deviation = float(array.std(ddof=1))
    icir = mean / deviation if deviation > 0.0 and math.isfinite(deviation) else math.nan
    return mean, deviation, icir, count


def _safe_corr(left: pd.Series, right: pd.Series) -> float:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        value = left.corr(right, method="pearson")
    return float(value) if value is not None and np.isfinite(value) else math.nan


def _evaluate_period(
    *,
    factor: str,
    label: str,
    values: np.ndarray,
    label_values: np.ndarray,
    panel_index: pd.MultiIndex,
    group_map: Mapping[object, object],
    spec: ResearchEvaluationSpec,
    diagnostics: list[dict[str, object]],
) -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]]]:
    ic_rows: list[dict[str, object]] = []
    quantile_rows: list[dict[str, object]] = []
    spread_rows: list[dict[str, object]] = []
    turnover_rows: list[dict[str, object]] = []

    finite_factor = np.isfinite(values)
    finite_label = np.isfinite(label_values)
    if not finite_factor.any():
        _diagnose(
            diagnostics,
            factor=factor,
            stage="prepare",
            severity="error",
            code="no_finite_factor",
            message=f"factor {factor!r} has no finite values in the study window",
        )
        return ic_rows, quantile_rows, spread_rows, turnover_rows
    if not finite_label.any():
        _diagnose(
            diagnostics,
            factor=factor,
            stage="prepare",
            severity="error",
            code="no_labels",
            message=f"forward returns {label!r} are unavailable for every {factor!r} cell",
        )
        return ic_rows, quantile_rows, spread_rows, turnover_rows

    factor_series = pd.Series(values, index=panel_index, name="factor")
    forward = pd.DataFrame({label: label_values}, index=panel_index)
    try:
        prepared = prepare_factor_data_from_forward_returns(
            factor_series,
            forward,
            groupby=dict(group_map),
            quantiles=spec.quantiles,
            max_loss=spec.max_loss,
        )
    except FactorLossExceededError as error:
        _diagnose(
            diagnostics,
            factor=factor,
            stage="prepare",
            severity="error",
            code="max_loss_exceeded",
            message=f"{error} (period {label}, factor {factor!r})",
        )
        return ic_rows, quantile_rows, spread_rows, turnover_rows
    except (KeyError, ValueError) as error:
        _diagnose(
            diagnostics,
            factor=factor,
            stage="prepare",
            severity="warning",
            code="preparation_failed",
            message=f"period {label}, factor {factor!r}: {error}",
        )
        return ic_rows, quantile_rows, spread_rows, turnover_rows

    data = prepared.data
    if data.empty or label not in data.columns:
        _diagnose(
            diagnostics,
            factor=factor,
            stage="prepare",
            severity="warning",
            code="no_evaluable_cells",
            message=f"period {label}, factor {factor!r} has no rows after cleaning and binning",
        )
        return ic_rows, quantile_rows, spread_rows, turnover_rows

    grouped = data.groupby(level="date", observed=False, sort=True)
    counts = grouped["factor"].size()
    pearson: dict[object, float] = {}
    skipped_dates = 0
    for date, group in grouped:
        if len(group) < spec.min_assets:
            skipped_dates += 1
            continue
        pearson[date] = _safe_corr(group["factor"], group[label])
    if skipped_dates:
        _diagnose(
            diagnostics,
            factor=factor,
            stage="ic",
            severity="warning",
            code="insufficient_assets",
            message=(
                f"period {label}, factor {factor!r}: {skipped_dates} dates had fewer than "
                f"min_assets={spec.min_assets} evaluable assets"
            ),
        )

    try:
        with warnings.catch_warnings():
            # A degenerate cross-section (one finite asset) makes scipy return
            # ``ConstantInputWarning``; the resulting NaN is handled below.
            warnings.simplefilter("ignore", RuntimeWarning)
            spearman_frame = factor_information_coefficient(data)
        spearman = spearman_frame[label] if label in spearman_frame.columns else pd.Series(dtype=float)
    except (KeyError, ValueError) as error:
        raise FactorComputationError(
            "rank information coefficient failed",
            code="rank_ic_failed",
            report={"factor": factor, "period": label, "error": str(error)},
        ) from error

    for date in sorted(pearson):
        rank_value = spearman.get(date, math.nan)
        rank_value = float(rank_value) if np.isfinite(rank_value) else math.nan
        ic_rows.append(
            {
                "time": date,
                "factor": factor,
                "period": label,
                "ic": pearson[date],
                "rank_ic": rank_value,
                "n_assets": int(counts.loc[date]),
            }
        )

    try:
        mean_returns, _ = mean_return_by_quantile(data, by_date=True, demeaned=False)
    except (KeyError, ValueError) as error:
        raise FactorComputationError(
            "quantile return aggregation failed",
            code="quantile_return_failed",
            report={"factor": factor, "period": label, "error": str(error)},
        ) from error
    # ``factor_quantile`` is int when every bucket exists and float when pandas
    # widened the column; normalize the level once so ``xs`` lookups and the
    # integer quantile labels in the public table agree.
    mean_returns.index = pd.MultiIndex.from_arrays(
        (
            pd.Index(mean_returns.index.get_level_values("factor_quantile")).astype(int),
            pd.DatetimeIndex(mean_returns.index.get_level_values("date")),
        ),
        names=("factor_quantile", "date"),
    )

    count_frame = data.groupby(
        [data["factor_quantile"], data.index.get_level_values("date")],
        observed=False,
        sort=True,
    )[label].count()
    count_lookup = {
        (int(quantile), date): int(count) for (quantile, date), count in count_frame.items() if pd.notna(quantile)
    }
    filtered_quantiles = 0
    for (quantile, date), row in mean_returns.iterrows():
        if pd.isna(quantile):
            continue
        quantile_number = int(quantile)
        count = count_lookup.get((quantile_number, date), 0)
        if count < spec.min_quantile_assets:
            filtered_quantiles += 1
            continue
        quantile_rows.append(
            {
                "time": date,
                "factor": factor,
                "period": label,
                "quantile": quantile_number,
                "mean_return": float(row[label]),
                "n_assets": count,
            }
        )
    if filtered_quantiles:
        _diagnose(
            diagnostics,
            factor=factor,
            stage="quantile",
            severity="warning",
            code="min_quantile_assets",
            message=(
                f"period {label}, factor {factor!r}: dropped {filtered_quantiles} quantile observations "
                f"below min_quantile_assets={spec.min_quantile_assets}"
            ),
        )

    available = sorted({int(value) for value in data["factor_quantile"].dropna().unique()})
    if len(available) >= 2:
        try:
            spread, _ = compute_mean_returns_spread(mean_returns, available[-1], available[0])
        except (KeyError, ValueError) as error:
            raise FactorComputationError(
                "quantile spread aggregation failed",
                code="spread_failed",
                report={"factor": factor, "period": label, "error": str(error)},
            ) from error
        for date, value in spread[label].items():
            if pd.isna(value) or not np.isfinite(value):
                continue
            spread_rows.append(
                {
                    "time": date,
                    "factor": factor,
                    "period": label,
                    "spread": float(value),
                    "top_quantile": int(available[-1]),
                    "bottom_quantile": int(available[0]),
                }
            )
    else:
        _diagnose(
            diagnostics,
            factor=factor,
            stage="quantile",
            severity="warning",
            code="quantile_spread_unavailable",
            message=f"period {label}, factor {factor!r} has fewer than two populated quantiles",
        )

    # Quantile turnover reuses the same cleaned quantile assignments; the lag
    # is the label's bar count (5D -> 5), matching the HTML report contract.
    try:
        lag = int(str(label).rstrip("D"))
    except ValueError:
        lag = 1
    for quantile_number in available:
        try:
            series = quantile_turnover(data["factor_quantile"], quantile_number, period=lag)
        except (KeyError, ValueError, TypeError):
            continue
        for date, value in series.items():
            if pd.isna(value) or not np.isfinite(float(value)):
                continue
            turnover_rows.append(
                {
                    "time": date,
                    "factor": factor,
                    "period": label,
                    "quantile": int(quantile_number),
                    "turnover": float(value),
                }
            )
    return ic_rows, quantile_rows, spread_rows, turnover_rows


def _stability_rows(
    factor: str,
    values: np.ndarray,
    panel_index: pd.MultiIndex,
    spec: ResearchEvaluationSpec,
    diagnostics: list[dict[str, object]],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    frame = pd.DataFrame({"factor": values}, index=panel_index)
    for lag in spec.stability_lags:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            series = factor_rank_autocorrelation(frame, period=lag)
        mean, deviation, _, count = _summary(list(series.to_numpy(dtype=float)))
        rows.append(
            {
                "factor": factor,
                "lag": int(lag),
                "mean_rank_autocorrelation": mean,
                "std_rank_autocorrelation": deviation,
                "n_dates": count,
            }
        )
        if count == 0:
            _diagnose(
                diagnostics,
                factor=factor,
                stage="stability",
                severity="warning",
                code="stability_lag_unavailable",
                message=f"factor {factor!r} has no rank autocorrelation at lag {lag}",
            )
    return rows


def _correlation_rows(
    names: Sequence[str],
    *,
    matrix_lookup: Mapping[str, np.ndarray],
    spec: ResearchEvaluationSpec,
    diagnostics: list[dict[str, object]],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for left_position, left in enumerate(names):
        for right in names[left_position + 1 :]:
            correlations: list[float] = []
            skipped = 0
            left_matrix = matrix_lookup[left]
            right_matrix = matrix_lookup[right]
            for row in range(left_matrix.shape[0]):
                valid = np.isfinite(left_matrix[row]) & np.isfinite(right_matrix[row])
                if int(valid.sum()) < spec.min_assets:
                    skipped += 1
                    continue
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", RuntimeWarning)
                    value = float(np.corrcoef(left_matrix[row][valid], right_matrix[row][valid])[0, 1])
                if np.isfinite(value):
                    correlations.append(value)
            mean, deviation, _, count = _summary(correlations)
            rows.append(
                {
                    "factor_left": left,
                    "factor_right": right,
                    "mean_correlation": mean,
                    "std_correlation": deviation,
                    "n_dates": count,
                }
            )
            if count == 0:
                _diagnose(
                    diagnostics,
                    factor=left,
                    stage="correlation",
                    severity="warning",
                    code="correlation_insufficient_assets",
                    message=(
                        f"factors {left!r}/{right!r} have no date with at least "
                        f"min_assets={spec.min_assets} common assets"
                    ),
                )
            elif skipped:
                _diagnose(
                    diagnostics,
                    factor=left,
                    stage="correlation",
                    severity="info",
                    code="correlation_dates_skipped",
                    message=f"factors {left!r}/{right!r}: {skipped} dates were below min_assets",
                )
    return rows


def _combination_values(
    name: str,
    weights: Mapping[str, float],
    processed: Mapping[str, np.ndarray],
) -> np.ndarray:
    unknown = sorted(factor for factor in weights if factor not in processed)
    if unknown:
        raise ConfigError(
            "combination references unknown factors",
            code="combination_unknown_factor",
            report={"combination": name, "factors": unknown},
        )
    size = len(next(iter(processed.values())))
    blended = np.zeros(size, dtype=float)
    used = 0
    for factor, weight in weights.items():
        if weight == 0.0:
            continue
        blended = blended + float(weight) * processed[factor]
        used += 1
    if used == 0:
        raise ConfigError(
            "combination has no non-zero weights",
            code="combination_empty",
            report={"combination": name},
        )
    return blended


def evaluate_factor_research(
    factors: pd.DataFrame,
    *,
    grid: pd.DataFrame | None,
    prices: Mapping[str, pd.DataFrame] | None,
    universe: pd.DataFrame | None,
    groups: Mapping[object, object] | None,
    precomputed_returns: pd.DataFrame | None,
    spec: ResearchEvaluationSpec | Mapping[str, Any],
) -> ResearchEvaluationResult:
    """Evaluate factor panels against forward returns using fincore kernels.

    ``factors`` carries raw factor panels on a ``(time, asset)`` MultiIndex;
    ``factors.attrs['factor_directions']`` (plus any spec override) supplies the
    +1/-1 direction applied before preprocessing.  ``grid`` describes the
    trading dates and ``prices`` supplies the open/close panels used to derive
    labels.  When ``spec.label`` is ``"precomputed"`` (or when ``prices`` is
    ``None``), ``precomputed_returns`` is used instead and must be supplied; an
    unused precomputed frame is rejected rather than silently ignored.
    """

    normalized = _coerce_spec(spec)
    panel_frame = _validate_factors(factors, normalized)
    if normalized.label == PRECOMPUTED_LABEL:
        if precomputed_returns is None:
            raise ConfigError(
                "label 'precomputed' requires precomputed_returns",
                code="missing_precomputed_returns",
            )
        label_mode = PRECOMPUTED_LABEL
    elif prices is None:
        if precomputed_returns is None:
            raise ConfigError(
                "prices are required unless precomputed_returns is supplied",
                code="missing_prices",
            )
        label_mode = PRECOMPUTED_LABEL
    else:
        if precomputed_returns is not None:
            raise ConfigError(
                "precomputed_returns was supplied but the label is price-derived",
                code="precomputed_returns_unused",
                report={"label": normalized.label},
            )
        label_mode = normalized.label

    if label_mode != PRECOMPUTED_LABEL and grid is None:
        raise ConfigError(
            "grid is required for price-derived labels",
            code="missing_grid",
            report={"label": normalized.label},
        )

    grid_index, study_dates = _study_grid(grid, normalized)
    target_index = pd.Index(list(normalized.targets))

    times = pd.DatetimeIndex(panel_frame.index.get_level_values(0))
    in_window = (times >= normalized.start) & (times <= normalized.end)
    panel_frame = panel_frame.loc[in_window]
    if panel_frame.empty:
        raise DatasetError("factors contain no cells inside the requested window", code="factors_empty_window")
    panel_frame = panel_frame.copy(deep=True)
    panel_frame.index = pd.MultiIndex.from_arrays(
        (
            pd.DatetimeIndex(panel_frame.index.get_level_values(0)),
            pd.Index(panel_frame.index.get_level_values(1)),
        ),
        names=("time", "asset"),
    )
    panel_frame = panel_frame.loc[panel_frame.index.get_level_values("asset").isin(target_index)]
    present_targets = pd.Index(panel_frame.index.get_level_values("asset").unique())
    missing_targets = [target for target in normalized.targets if target not in present_targets]
    if missing_targets:
        raise DatasetError(
            "factors contain no cells for every requested target inside the window",
            code="factors_missing_targets",
            report={"missing": missing_targets},
        )
    panel_frame = panel_frame.sort_index()

    panel_times = pd.DatetimeIndex(panel_frame.index.get_level_values("time").unique()).sort_values()
    if grid_index is not None:
        off_grid = panel_times.difference(study_dates)
        if len(off_grid):
            raise DatasetError(
                "factor timestamps are not present inside the study grid",
                code="factors_off_grid",
                report={"count": len(off_grid)},
            )
        study_dates = grid_index[(grid_index >= normalized.start) & (grid_index <= normalized.end)]
    else:
        study_dates = panel_times
        grid_index = None

    universe_values, universe_absent = _validate_universe(universe, spec=normalized, study_dates=study_dates)
    time_lookup = pd.Index(study_dates)
    row_times = pd.DatetimeIndex(panel_frame.index.get_level_values("time"))
    row_assets = pd.Index(panel_frame.index.get_level_values("asset"))
    time_positions = time_lookup.get_indexer(row_times)
    asset_positions = target_index.get_indexer(row_assets)
    if (time_positions < 0).any() or (asset_positions < 0).any():  # pragma: no cover - guarded above
        raise DatasetError("factor panel cannot be aligned to the study grid", code="factors_alignment")
    in_universe = universe_values[time_positions, asset_positions]
    dropped_universe = int(np.count_nonzero(~in_universe))
    panel_frame = panel_frame.loc[in_universe]
    if panel_frame.empty:
        raise DatasetError("universe excludes every factor cell in the window", code="universe_empty")
    row_times = pd.DatetimeIndex(panel_frame.index.get_level_values("time"))
    row_assets = pd.Index(panel_frame.index.get_level_values("asset"))
    time_positions = time_lookup.get_indexer(row_times)
    asset_positions = target_index.get_indexer(row_assets)
    panel_index = panel_frame.index
    panel_assets = pd.Index(sorted(panel_index.get_level_values("asset").unique(), key=str))
    panel_cell_count = len(panel_frame)

    diagnostics: list[dict[str, object]] = []
    if universe_absent:
        _diagnose(
            diagnostics,
            factor=None,
            stage="universe",
            severity="warning",
            code="universe_missing_cells",
            message=f"{universe_absent} universe cells were absent and treated as out of universe",
        )
    if dropped_universe:
        _diagnose(
            diagnostics,
            factor=None,
            stage="universe",
            severity="info",
            code="universe_cells_excluded",
            message=f"{dropped_universe} factor cells were outside the requested universe",
        )

    labels = _canonical_period_labels(normalized.periods, grid_index, study_dates)
    if label_mode == PRECOMPUTED_LABEL:
        if precomputed_returns is None:  # pragma: no cover - guarded above
            raise ConfigError("precomputed_returns is required", code="missing_precomputed_returns")
        if prices is not None:
            _diagnose(
                diagnostics,
                factor=None,
                stage="labels",
                severity="info",
                code="prices_ignored",
                message="prices were provided but precomputed returns were selected",
            )
        precomputed_frame = _precomputed_label_frame(
            precomputed_returns,
            periods=normalized.periods,
            labels=labels,
            targets=normalized.targets,
        )
        label_values: dict[str, np.ndarray] = {}
        for label in labels:
            series = precomputed_frame[label].reindex(panel_index)
            label_values[label] = series.to_numpy(dtype=float)
    else:
        price_frames = _validate_prices(prices, normalized, grid_index)
        basis = _price_basis(normalized, price_frames)
        full_price_frame = price_frames[basis]
        label_values = {}
        for period, label in zip(normalized.periods, labels, strict=True):
            # The shift must use the complete grid so the label tail beyond
            # ``spec.end`` remains available; only afterwards is the study
            # window selected.
            if normalized.label == "next_open_to_open":
                wide = full_price_frame.shift(-(period + 1)) / full_price_frame.shift(-1) - 1.0
            else:
                wide = full_price_frame.shift(-period) / full_price_frame - 1.0
            wide = wide.replace([np.inf, -np.inf], np.nan).reindex(
                index=study_dates,
                columns=list(normalized.targets),
            )
            matrix = wide.to_numpy(dtype=float)
            label_values[label] = matrix[time_positions, asset_positions]

    label_table_parts: list[pd.DataFrame] = []
    for period, label in zip(normalized.periods, labels, strict=True):
        label_table_parts.append(
            pd.DataFrame(
                {
                    "time": row_times,
                    "asset": row_assets,
                    "period": label,
                    "value": label_values[label],
                }
            )
        )
    label_table = (
        pd.concat(label_table_parts, ignore_index=True)
        if label_table_parts
        else _empty_frame(["time", "asset", "period", "value"])
    )
    label_any = np.zeros(panel_cell_count, dtype=bool)
    for label in labels:
        label_any |= np.isfinite(label_values[label])

    group_map = _resolve_groups(groups, panel_assets)
    directions = _direction_map(factors, normalized)
    factor_names = [str(name) for name in panel_frame.columns]
    reserved_names = {"factor", "group", "factor_quantile", *labels}
    collisions = sorted(name for name in factor_names if name in reserved_names)
    if collisions:
        raise ConfigError(
            "factor names collide with reserved evaluation columns",
            code="factor_name_reserved",
            report={"names": collisions, "reserved": sorted(reserved_names)},
        )

    processed_matrices: dict[str, np.ndarray] = {}
    row_values: dict[str, np.ndarray] = {}
    signal_parts: list[pd.DataFrame] = []
    collapse_reports: list[tuple[str, int]] = []
    for name in factor_names:
        raw = panel_frame[name].to_numpy(dtype=float)
        direction = directions[name]
        adjusted = raw * float(direction)
        matrix = np.full((len(study_dates), len(target_index)), np.nan, dtype=float)
        matrix[time_positions, asset_positions] = adjusted
        processed_matrix, collapsed = _preprocess_matrix(matrix, normalized.preprocess)
        if collapsed:
            collapse_reports.append((name, collapsed))
        processed_matrices[name] = processed_matrix
        row_values[name] = processed_matrix[time_positions, asset_positions]
        group_values = np.array([group_map[asset] for asset in row_assets], dtype=object)
        signal_parts.append(
            pd.DataFrame(
                {
                    "time": row_times,
                    "asset": row_assets,
                    "factor": name,
                    "direction": direction,
                    "value": adjusted,
                    "processed_value": row_values[name],
                    "preprocess": normalized.preprocess,
                    "group": group_values,
                }
            )
        )
    for name, collapsed in collapse_reports:
        _diagnose(
            diagnostics,
            factor=name,
            stage="preprocess",
            severity="warning",
            code="preprocess_collapsed_cross_section",
            message=(
                f"{collapsed} cross-sections for factor {name!r} became fully undefined under "
                f"preprocess={normalized.preprocess!r}"
            ),
        )

    ic_rows: list[dict[str, object]] = []
    quantile_rows: list[dict[str, object]] = []
    spread_rows: list[dict[str, object]] = []
    turnover_rows: list[dict[str, object]] = []
    for name in factor_names:
        for period, label in zip(normalized.periods, labels, strict=True):
            period_ic, period_quantile, period_spread, period_turnover = _evaluate_period(
                factor=name,
                label=label,
                values=row_values[name],
                label_values=label_values[label],
                panel_index=panel_index,
                group_map=group_map,
                spec=normalized,
                diagnostics=diagnostics,
            )
            ic_rows.extend(period_ic)
            quantile_rows.extend(period_quantile)
            spread_rows.extend(period_spread)
            turnover_rows.extend(period_turnover)

    processed: dict[str, np.ndarray] = {
        name: processed_matrices[name][time_positions, asset_positions] for name in factor_names
    }
    stability_rows: list[dict[str, object]] = []
    for name in factor_names:
        stability_rows.extend(_stability_rows(name, processed[name], panel_index, normalized, diagnostics))

    evaluated_row_values: dict[str, np.ndarray] = {name: row_values[name] for name in factor_names}
    if normalized.combinations:
        conflicts = sorted(name for name in normalized.combinations if name in processed_matrices)
        if conflicts:
            raise ConfigError(
                "combination names collide with factor names",
                code="combination_name_conflict",
                report={"names": conflicts},
            )
        for name, weights in normalized.combinations.items():
            blended = _combination_values(name, weights, processed)
            matrix = np.full((len(study_dates), len(target_index)), np.nan, dtype=float)
            matrix[time_positions, asset_positions] = blended
            processed_matrices[name] = matrix
            evaluated_row_values[name] = blended
            _diagnose(
                diagnostics,
                factor=name,
                stage="combination",
                severity="info",
                code="combination_built",
                message=f"combination {name!r} blends {sorted(weights)}",
            )
            for label in labels:
                period_ic, period_quantile, period_spread, period_turnover = _evaluate_period(
                    factor=name,
                    label=label,
                    values=blended,
                    label_values=label_values[label],
                    panel_index=panel_index,
                    group_map=group_map,
                    spec=normalized,
                    diagnostics=diagnostics,
                )
                ic_rows.extend(period_ic)
                quantile_rows.extend(period_quantile)
                spread_rows.extend(period_spread)
            stability_rows.extend(_stability_rows(name, blended, panel_index, normalized, diagnostics))

    evaluated_names = [*factor_names, *normalized.combinations]
    summary_rows: list[dict[str, object]] = []
    summary_lookup: dict[tuple[str, str], dict[str, object]] = {}
    for name in evaluated_names:
        for label in labels:
            period_ic = [row["ic"] for row in ic_rows if row["factor"] == name and row["period"] == label]
            period_rank = [row["rank_ic"] for row in ic_rows if row["factor"] == name and row["period"] == label]
            mean_ic, std_ic, icir, n_dates = _summary(period_ic)
            mean_rank, std_rank, rank_icir, _ = _summary(period_rank)
            row: dict[str, object] = {
                "factor": name,
                "period": label,
                "mean_ic": mean_ic,
                "std_ic": std_ic,
                "icir": icir,
                "mean_rank_ic": mean_rank,
                "std_rank_ic": std_rank,
                "rank_icir": rank_icir,
                "n_dates": n_dates,
            }
            summary_rows.append(row)
            summary_lookup[(name, label)] = row

    correlation_rows = _correlation_rows(
        evaluated_names,
        matrix_lookup={name: processed_matrices[name] for name in evaluated_names},
        spec=normalized,
        diagnostics=diagnostics,
    )

    comparison_rows = _comparison_rows(normalized, evaluated_names, summary_lookup, labels)
    coverage_rows = _coverage_rows(
        factor_names=factor_names,
        combination_names=list(normalized.combinations),
        panel_frame=panel_frame,
        label_any=label_any,
        values_by_name={name: evaluated_row_values[name] for name in evaluated_names},
        directions=directions,
        summary_lookup=summary_lookup,
        labels=labels,
    )
    status = _overall_status(factor_names, summary_lookup, labels)

    signals = (
        pd.concat(signal_parts, ignore_index=True)
        if signal_parts
        else _empty_frame(["time", "asset", "factor", "direction", "value", "processed_value", "preprocess", "group"])
    )
    ic_table = pd.DataFrame(
        ic_rows,
        columns=["time", "factor", "period", "ic", "rank_ic", "n_assets"],
    )
    ic_summary = pd.DataFrame(
        summary_rows,
        columns=[
            "factor",
            "period",
            "mean_ic",
            "std_ic",
            "icir",
            "mean_rank_ic",
            "std_rank_ic",
            "rank_icir",
            "n_dates",
        ],
    )
    quantile_table = pd.DataFrame(
        quantile_rows,
        columns=["time", "factor", "period", "quantile", "mean_return", "n_assets"],
    )
    turnover_table = pd.DataFrame(
        turnover_rows,
        columns=["time", "factor", "period", "quantile", "turnover"],
    )
    spread_table = pd.DataFrame(
        spread_rows,
        columns=["time", "factor", "period", "spread", "top_quantile", "bottom_quantile"],
    )
    stability_table = pd.DataFrame(
        stability_rows,
        columns=["factor", "lag", "mean_rank_autocorrelation", "std_rank_autocorrelation", "n_dates"],
    )
    correlation_table = pd.DataFrame(
        correlation_rows,
        columns=["factor_left", "factor_right", "mean_correlation", "std_correlation", "n_dates"],
    )
    comparison_table = pd.DataFrame(
        comparison_rows,
        columns=[
            "comparison",
            "left",
            "right",
            "period",
            "mean_ic",
            "icir",
            "mean_ic_left",
            "mean_ic_right",
            "icir_left",
            "icir_right",
            "n_dates_left",
            "n_dates_right",
        ],
    )
    coverage_table = pd.DataFrame(
        coverage_rows,
        columns=[
            "factor",
            "kind",
            "direction",
            "input_cells",
            "finite_cells",
            "label_cells",
            "evaluable_cells",
            "status",
        ],
    )
    diagnostics_table = pd.DataFrame(
        diagnostics,
        columns=["factor", "stage", "severity", "code", "message"],
    )

    tables = {
        "signals": signals,
        "labels": label_table,
        "ic": ic_table,
        "ic_summary": ic_summary,
        "quantile_returns": quantile_table,
        "turnover": turnover_table,
        "spreads": spread_table,
        "stability": stability_table,
        "correlation": correlation_table,
        "comparisons": comparison_table,
        "coverage": coverage_table,
        "diagnostics": diagnostics_table,
    }
    input_fingerprints = {
        "factors": fingerprint_value(factors),
        "grid": "none" if grid is None else fingerprint_value(grid),
        "prices": "none"
        if prices is None
        else {str(key): fingerprint_value(value) for key, value in sorted(prices.items())},
        "universe": "none" if universe is None else fingerprint_value(universe),
        "groups": "none"
        if groups is None
        else {str(key): str(value) for key, value in sorted(groups.items(), key=lambda item: str(item[0]))},
        "precomputed_returns": "none" if precomputed_returns is None else fingerprint_value(precomputed_returns),
    }
    spec_payload = _spec_payload(normalized)
    evaluation_id = fingerprint_value(
        {
            "api_version": API_VERSION,
            "config_schema_version": EVALUATION_CONFIG_SCHEMA_VERSION,
            "spec": spec_payload,
            "inputs": input_fingerprints,
        }
    )
    result_fingerprint = fingerprint_value(
        {
            "result_schema_version": RESULT_SCHEMA_VERSION,
            "status": status,
            "tables": tables,
        }
    )
    counts = {
        "factors": len(factor_names),
        "combinations": len(normalized.combinations),
        "targets": len(normalized.targets),
        "study_dates": len(study_dates),
        "panel_cells": int(panel_cell_count),
        "label_rows": len(label_table),
        "signal_rows": len(signals),
        "ic_rows": len(ic_table),
        "ic_observations": int(np.isfinite(ic_table["ic"].to_numpy(dtype=float)).sum()) if len(ic_table) else 0,
        "evaluable_factor_periods": int(sum(1 for row in summary_rows if int(row["n_dates"]) > 0)),
        "diagnostics": len(diagnostics_table),
    }
    manifest = {
        "api_version": API_VERSION,
        "result_schema_version": RESULT_SCHEMA_VERSION,
        "evaluation_config_schema_version": EVALUATION_CONFIG_SCHEMA_VERSION,
        "evaluation_id": evaluation_id,
        "result_fingerprint": result_fingerprint,
        "status": status,
        "label_mode": label_mode,
        "labels": list(labels),
        "spec": spec_payload,
        "inputs": input_fingerprints,
        "counts": counts,
        "tables": {
            name: {"rows": len(table), "columns": [str(column) for column in table.columns]}
            for name, table in tables.items()
        },
        "capabilities": {
            "charts_available": _charts_available(),
            "export_formats": list(EXPORT_FORMATS),
        },
    }
    return ResearchEvaluationResult(
        signals=signals,
        labels=label_table,
        ic=ic_table,
        ic_summary=ic_summary,
        quantile_returns=quantile_table,
        turnover=turnover_table,
        spreads=spread_table,
        stability=stability_table,
        correlation=correlation_table,
        comparisons=comparison_table,
        coverage=coverage_table,
        diagnostics=diagnostics_table,
        status=status,
        manifest=manifest,
    )


def _comparison_rows(
    spec: ResearchEvaluationSpec,
    evaluated_names: Sequence[str],
    summary_lookup: Mapping[tuple[str, str], Mapping[str, object]],
    labels: Sequence[str],
) -> list[dict[str, object]]:
    pairs: list[tuple[str, str, str]] = []
    if spec.comparisons:
        for name, (left, right) in spec.comparisons.items():
            for side in (left, right):
                if side not in evaluated_names:
                    raise ConfigError(
                        "comparison references an unknown factor",
                        code="comparison_unknown_factor",
                        report={"comparison": name, "factor": side},
                    )
            pairs.append((name, left, right))
    else:
        for name, weights in spec.combinations.items():
            for factor in weights:
                if weights[factor] == 0.0:
                    continue
                pairs.append((f"{name}_vs_{factor}", name, factor))
    rows: list[dict[str, object]] = []
    for name, left, right in pairs:
        for label in labels:
            left_row = summary_lookup.get((left, label), {})
            right_row = summary_lookup.get((right, label), {})
            left_mean = float(left_row.get("mean_ic", math.nan))
            right_mean = float(right_row.get("mean_ic", math.nan))
            left_icir = float(left_row.get("icir", math.nan))
            right_icir = float(right_row.get("icir", math.nan))
            rows.append(
                {
                    "comparison": name,
                    "left": left,
                    "right": right,
                    "period": label,
                    "mean_ic": left_mean - right_mean
                    if np.isfinite(left_mean) and np.isfinite(right_mean)
                    else math.nan,
                    "icir": left_icir - right_icir if np.isfinite(left_icir) and np.isfinite(right_icir) else math.nan,
                    "mean_ic_left": left_mean,
                    "mean_ic_right": right_mean,
                    "icir_left": left_icir,
                    "icir_right": right_icir,
                    "n_dates_left": int(left_row.get("n_dates", 0)),
                    "n_dates_right": int(right_row.get("n_dates", 0)),
                }
            )
    return rows


def _coverage_rows(
    *,
    factor_names: Sequence[str],
    combination_names: Sequence[str],
    panel_frame: pd.DataFrame,
    label_any: np.ndarray,
    values_by_name: Mapping[str, np.ndarray],
    directions: Mapping[str, int],
    summary_lookup: Mapping[tuple[str, str], Mapping[str, object]],
    labels: Sequence[str],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    input_cells = len(panel_frame)
    label_cells = int(np.count_nonzero(label_any))
    ordered: list[tuple[str, str]] = [("factor", name) for name in factor_names]
    ordered.extend(("combination", name) for name in combination_names)
    for kind, name in ordered:
        values = np.asarray(values_by_name[name], dtype=float)
        finite = int(np.count_nonzero(np.isfinite(values)))
        evaluable = int(np.count_nonzero(np.isfinite(values) & label_any))
        direction: object = int(directions[name]) if kind == "factor" else None
        observed = [
            int(summary_lookup[(name, label)]["n_dates"]) for label in labels if (name, label) in summary_lookup
        ]
        if not observed or all(count == 0 for count in observed):
            status = "UNEVALUABLE"
        elif all(count > 0 for count in observed):
            status = "OK"
        else:
            status = "PARTIAL"
        rows.append(
            {
                "factor": name,
                "kind": kind,
                "direction": direction,
                "input_cells": int(input_cells),
                "finite_cells": finite,
                "label_cells": label_cells,
                "evaluable_cells": evaluable,
                "status": status,
            }
        )
    return rows


def _overall_status(
    factor_names: Sequence[str],
    summary_lookup: Mapping[tuple[str, str], Mapping[str, object]],
    labels: Sequence[str],
) -> str:
    evaluated = [any(int(summary_lookup[(name, label)]["n_dates"]) > 0 for label in labels) for name in factor_names]
    if evaluated and all(evaluated):
        return "OK"
    if any(evaluated):
        return "PARTIAL"
    return "UNEVALUABLE"


def _write_charts(result: ResearchEvaluationResult, target: Path) -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        # Use the Agg canvas directly instead of pyplot so exporting cannot
        # change a host application's global matplotlib backend.
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        from matplotlib.figure import Figure

        ic = result.ic
        figure = Figure(figsize=(8.0, 4.0))
        FigureCanvasAgg(figure)
        axis = figure.add_subplot(1, 1, 1)
        if len(ic):
            for period in ic["period"].drop_duplicates():
                subset = ic[ic["period"] == period].groupby("time", observed=False, sort=True)["ic"].mean()
                axis.plot(list(subset.index), subset.to_numpy(dtype=float), label=str(period))
            axis.legend(title="period")
        axis.set_title("Mean cross-sectional IC")
        axis.set_ylabel("IC")
        axis.grid(True, alpha=0.3)
        figure.tight_layout()
        figure.savefig(target / "ic_series.png", dpi=120)

        quantiles = result.quantile_returns
        figure = Figure(figsize=(8.0, 4.0))
        FigureCanvasAgg(figure)
        axis = figure.add_subplot(1, 1, 1)
        if len(quantiles):
            first_period = str(quantiles["period"].iloc[0])
            subset = quantiles[quantiles["period"] == first_period]
            profile = subset.groupby("quantile", observed=False, sort=True)["mean_return"].mean()
            axis.bar([str(int(index)) for index in profile.index], profile.to_numpy(dtype=float))
            axis.set_title(f"Mean quantile returns ({first_period})")
            axis.set_xlabel("quantile")
            axis.set_ylabel("mean forward return")
        else:
            axis.set_title("Mean quantile returns (no observations)")
        axis.grid(True, axis="y", alpha=0.3)
        figure.tight_layout()
        figure.savefig(target / "quantile_returns.png", dpi=120)


__all__ = [
    "API_VERSION",
    "EVALUATION_CONFIG_SCHEMA_VERSION",
    "RESULT_SCHEMA_VERSION",
    "ConfigError",
    "DataQualityError",
    "DatasetError",
    "DependencyError",
    "FactorComputationError",
    "ResearchEvaluationResult",
    "ResearchEvaluationSpec",
    "evaluate_factor_research",
    "normalize_research_evaluation_spec",
    "research_capabilities",
]
