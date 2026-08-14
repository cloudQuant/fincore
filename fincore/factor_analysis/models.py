"""Frozen, renderer-ready models for enhanced factor analysis.

These dataclasses deliberately contain only analytical data.  Matplotlib,
seaborn, IPython, and renderer callables belong to later adapter layers and
are intentionally absent from this module.
"""

from __future__ import annotations

import json
import math
import pickle
from collections.abc import Hashable, Mapping
from dataclasses import dataclass, field, fields, is_dataclass
from hashlib import sha256
from types import MappingProxyType
from typing import Any, Literal, TypeVar, cast

import numpy as np
import pandas as pd

from fincore.factor_analysis.portfolio import PyfolioFactorInputs

_MappingKey = TypeVar("_MappingKey", bound=Hashable)
_MappingValue = TypeVar("_MappingValue")


def _json_scalar(value: object) -> object:
    """Convert one scalar to a deterministic, JSON-safe representation."""

    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, np.generic):
        return _json_scalar(value.item())
    if isinstance(value, float):
        if math.isnan(value):
            return {"kind": "nonfinite", "value": "nan"}
        if math.isinf(value):
            return {"kind": "nonfinite", "value": "positive_infinity" if value > 0 else "negative_infinity"}
        return value
    if value is pd.NA or value is pd.NaT:
        return None
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, pd.Timedelta):
        return str(value)
    return repr(value)


_WIRE_TYPE = "__fincore_factor_analysis_type__"
_WIRE_SCHEMA = "fincore-factor-analysis-json-v1"


def _wire_scalar(value: object) -> object:
    """Encode one scalar without losing numeric bits or pandas sentinels."""

    if isinstance(value, np.generic):
        return _wire_scalar(value.item())
    if value is pd.NA:
        return {_WIRE_TYPE: "pandas-na"}
    if value is pd.NaT:
        return {_WIRE_TYPE: "pandas-nat"}
    if value is None or isinstance(value, (str, bool)):
        return value
    if isinstance(value, int):
        # A JSON number may be rounded by non-Python consumers. Keep every
        # integer as decimal text behind an explicit type tag instead.
        return {_WIRE_TYPE: "int", "value": str(value)}
    if isinstance(value, float):
        return {_WIRE_TYPE: "float", "hex": value.hex()}
    if isinstance(value, pd.Timestamp):
        return {
            _WIRE_TYPE: "timestamp",
            "nanoseconds": str(value.value),
            "timezone": None if value.tz is None else str(value.tz),
        }
    if isinstance(value, pd.Timedelta):
        return {_WIRE_TYPE: "timedelta", "nanoseconds": str(value.value)}
    raise TypeError(f"factor-analysis JSON handoff does not support scalar {value!r} ({type(value).__name__})")


def _unwire_scalar(value: object) -> object:
    """Decode one scalar produced by :func:`_wire_scalar`."""

    if not isinstance(value, dict) or _WIRE_TYPE not in value:
        return value
    wire_type = value[_WIRE_TYPE]
    if wire_type == "pandas-na":
        return pd.NA
    if wire_type == "pandas-nat":
        return pd.NaT
    if wire_type == "int":
        return int(cast("str", value["value"]))
    if wire_type == "float":
        return float.fromhex(cast("str", value["hex"]))
    if wire_type == "timestamp":
        nanoseconds = int(cast("str", value["nanoseconds"]))
        timezone = value.get("timezone")
        if timezone is None:
            return pd.Timestamp(nanoseconds, unit="ns")
        return pd.Timestamp(nanoseconds, unit="ns", tz="UTC").tz_convert(cast("str", timezone))
    if wire_type == "timedelta":
        return pd.Timedelta(int(cast("str", value["nanoseconds"])), unit="ns")
    raise ValueError(f"unknown factor-analysis scalar wire type {wire_type!r}")


def _index_payload(index: pd.Index) -> dict[str, object]:
    """Encode an index with its names, levels, categories, timezone, and freq."""

    if isinstance(index, pd.MultiIndex):
        return {
            _WIRE_TYPE: "multiindex",
            "names": [_wire_scalar(name) for name in index.names],
            "levels": [_index_payload(level) for level in index.levels],
            "codes": [codes.tolist() for codes in index.codes],
        }
    if isinstance(index, pd.CategoricalIndex):
        return {
            _WIRE_TYPE: "categorical-index",
            "name": _wire_scalar(index.name),
            "categories": _index_payload(index.categories),
            "codes": index.codes.tolist(),
            "ordered": cast("pd.CategoricalDtype", index.dtype).ordered,
        }
    if isinstance(index, pd.RangeIndex):
        return {
            _WIRE_TYPE: "range-index",
            "name": _wire_scalar(index.name),
            "start": str(index.start),
            "stop": str(index.stop),
            "step": str(index.step),
        }
    if isinstance(index, pd.DatetimeIndex):
        return {
            _WIRE_TYPE: "datetime-index",
            "name": _wire_scalar(index.name),
            "values": [_wire_scalar(value) for value in index],
            "freq": index.freqstr,
            "unit": index.unit,
        }
    if isinstance(index, pd.TimedeltaIndex):
        return {
            _WIRE_TYPE: "timedelta-index",
            "name": _wire_scalar(index.name),
            "values": [_wire_scalar(value) for value in index],
            "freq": index.freqstr,
            "unit": index.unit,
        }
    return {
        _WIRE_TYPE: "index",
        "name": _wire_scalar(index.name),
        "dtype": str(index.dtype),
        "values": [_wire_scalar(value) for value in index],
    }


def _restore_frequency(
    index: pd.DatetimeIndex | pd.TimedeltaIndex, frequency: object
) -> pd.DatetimeIndex | pd.TimedeltaIndex:
    """Restore a valid explicit frequency without rejecting irregular indexes."""

    if frequency is None:
        return index
    try:
        return type(index)(index, freq=cast("str", frequency), name=index.name)  # type: ignore[arg-type]
    except ValueError:
        return index


def _restore_datetime_unit(
    index: pd.DatetimeIndex | pd.TimedeltaIndex, unit: object
) -> pd.DatetimeIndex | pd.TimedeltaIndex:
    """Restore pandas' datetime resolution independently from its frequency."""

    if unit is None:
        return index
    try:
        return index.as_unit(cast("Literal['s', 'ms', 'us', 'ns']", unit))
    except (TypeError, ValueError):
        return index


def _index_from_payload(payload: Mapping[str, object]) -> pd.Index:
    """Restore one typed index payload from the JSON handoff schema."""

    wire_type = payload.get(_WIRE_TYPE)
    if wire_type == "multiindex":
        levels = [
            _index_from_payload(cast("Mapping[str, object]", item)) for item in cast("list[object]", payload["levels"])
        ]
        names = [_unwire_scalar(item) for item in cast("list[object]", payload["names"])]
        codes = cast("list[list[int]]", payload["codes"])
        return pd.MultiIndex(levels=cast("Any", levels), codes=cast("Any", codes), names=names)
    if wire_type == "categorical-index":
        categories = _index_from_payload(cast("Mapping[str, object]", payload["categories"]))
        categorical = pd.Categorical.from_codes(
            cast("list[int]", payload["codes"]),
            categories=categories,
            ordered=bool(payload["ordered"]),
        )
        return pd.CategoricalIndex(categorical, name=_unwire_scalar(payload["name"]))
    if wire_type == "range-index":
        return pd.RangeIndex(
            start=int(cast("str", payload["start"])),
            stop=int(cast("str", payload["stop"])),
            step=int(cast("str", payload["step"])),
            name=_unwire_scalar(payload["name"]),
        )
    if wire_type == "datetime-index":
        values = [_unwire_scalar(item) for item in cast("list[object]", payload["values"])]
        datetime_result = pd.DatetimeIndex(values, name=_unwire_scalar(payload["name"]))
        datetime_result = cast("pd.DatetimeIndex", _restore_datetime_unit(datetime_result, payload.get("unit")))
        return cast("pd.Index", _restore_frequency(datetime_result, payload.get("freq")))
    if wire_type == "timedelta-index":
        values = [_unwire_scalar(item) for item in cast("list[object]", payload["values"])]
        timedelta_result = pd.TimedeltaIndex(values, name=cast("str | None", _unwire_scalar(payload["name"])))
        timedelta_result = cast("pd.TimedeltaIndex", _restore_datetime_unit(timedelta_result, payload.get("unit")))
        return cast("pd.Index", _restore_frequency(timedelta_result, payload.get("freq")))
    if wire_type == "index":
        generic_result = pd.Index(
            [_unwire_scalar(item) for item in cast("list[object]", payload["values"])],
            name=_unwire_scalar(payload["name"]),
        )
        dtype = cast("str", payload["dtype"])
        try:
            return cast("pd.Index", generic_result.astype(cast("Any", dtype)))
        except (TypeError, ValueError):
            return generic_result
    raise ValueError(f"unknown factor-analysis index wire type {wire_type!r}")


def _dtype_payload(dtype: object) -> dict[str, object]:
    """Encode ordinary and categorical pandas dtypes without implicit inference."""

    if isinstance(dtype, pd.CategoricalDtype):
        return {
            _WIRE_TYPE: "categorical-dtype",
            "categories": _index_payload(dtype.categories),
            "ordered": dtype.ordered,
        }
    return {_WIRE_TYPE: "dtype", "name": str(dtype)}


def _restore_series_dtype(series: pd.Series, payload: Mapping[str, object]) -> pd.Series:
    """Restore one dtype after exact scalar and index reconstruction."""

    wire_type = payload.get(_WIRE_TYPE)
    if wire_type == "categorical-dtype":
        categories = _index_from_payload(cast("Mapping[str, object]", payload["categories"]))
        return pd.Series(
            pd.Categorical(series, categories=categories, ordered=bool(payload["ordered"])),
            index=series.index,
            name=series.name,
        )
    if wire_type == "dtype":
        return cast("pd.Series", series.astype(cast("Any", payload["name"])))
    raise ValueError(f"unknown factor-analysis dtype wire type {wire_type!r}")


def _pandas_payload(value: pd.Series | pd.DataFrame) -> dict[str, object]:
    """Return a standards-compliant, lossless table envelope for JSON handoff."""

    if isinstance(value, pd.Series):
        return {
            _WIRE_TYPE: "series",
            "schema": _WIRE_SCHEMA,
            "name": _wire_scalar(value.name),
            "index": _index_payload(value.index),
            "dtype": _dtype_payload(value.dtype),
            "data": [_wire_scalar(item) for item in value.astype(object).tolist()],
        }
    rows = [list(row) for row in value.astype(object).itertuples(index=False, name=None)]
    return {
        _WIRE_TYPE: "dataframe",
        "schema": _WIRE_SCHEMA,
        "index": _index_payload(value.index),
        "columns": _index_payload(value.columns),
        "dtypes": [_dtype_payload(dtype) for dtype in value.dtypes],
        "data": [[_wire_scalar(item) for item in row] for row in rows],
    }


def _pandas_from_payload(payload: Mapping[str, object]) -> pd.Series | pd.DataFrame:
    """Restore a Series or DataFrame encoded by :func:`_pandas_payload`."""

    if payload.get("schema") != _WIRE_SCHEMA:
        raise ValueError("unsupported factor-analysis pandas handoff schema")
    wire_type = payload.get(_WIRE_TYPE)
    if wire_type == "series":
        series_result = pd.Series(
            [_unwire_scalar(item) for item in cast("list[object]", payload["data"])],
            index=_index_from_payload(cast("Mapping[str, object]", payload["index"])),
            name=_unwire_scalar(payload["name"]),
        )
        return _restore_series_dtype(series_result, cast("Mapping[str, object]", payload["dtype"]))
    if wire_type == "dataframe":
        frame_result = pd.DataFrame(
            [
                [_unwire_scalar(item) for item in cast("list[object]", row)]
                for row in cast("list[object]", payload["data"])
            ],
            index=_index_from_payload(cast("Mapping[str, object]", payload["index"])),
            columns=_index_from_payload(cast("Mapping[str, object]", payload["columns"])),
        )
        for position, dtype in enumerate(cast("list[object]", payload["dtypes"])):
            restored = _restore_series_dtype(frame_result.iloc[:, position], cast("Mapping[str, object]", dtype))
            frame_result.isetitem(position, restored.array)
        return frame_result
    raise ValueError(f"unknown factor-analysis pandas wire type {wire_type!r}")


def serializable_value(value: object) -> object:
    """Convert analytical data to a lossless, standards-compliant JSON value."""

    if isinstance(value, (pd.Series, pd.DataFrame)):
        return _pandas_payload(value)
    if is_dataclass(value) and not isinstance(value, type):
        return {item.name: serializable_value(getattr(value, item.name)) for item in fields(value)}
    if isinstance(value, Mapping):
        return {
            _WIRE_TYPE: "mapping",
            "schema": _WIRE_SCHEMA,
            "entries": [[serializable_value(key), serializable_value(item)] for key, item in value.items()],
        }
    if isinstance(value, tuple):
        return {_WIRE_TYPE: "tuple", "items": [serializable_value(item) for item in value]}
    return _wire_scalar(value)


def deserialize_serializable_value(value: object) -> object:
    """Restore supported data from :func:`serializable_value` trusted payloads."""

    if isinstance(value, list):
        return [deserialize_serializable_value(item) for item in value]
    if not isinstance(value, dict):
        return value
    wire_type = value.get(_WIRE_TYPE)
    if wire_type in {"series", "dataframe"}:
        return _pandas_from_payload(cast("Mapping[str, object]", value))
    if wire_type == "mapping":
        if value.get("schema") != _WIRE_SCHEMA:
            raise ValueError("unsupported factor-analysis mapping handoff schema")
        entries = cast("list[list[object]]", value["entries"])
        return {deserialize_serializable_value(key): deserialize_serializable_value(item) for key, item in entries}
    if wire_type == "tuple":
        return tuple(deserialize_serializable_value(item) for item in cast("list[object]", value["items"]))
    if wire_type is not None:
        return _unwire_scalar(cast("dict[str, object]", value))
    return {key: deserialize_serializable_value(item) for key, item in value.items()}


def _fingerprint_bytes(value: object) -> bytes:
    """Encode model data without lossy JSON floating-point formatting.

    The handoff serializer above intentionally favors portable JSON. Pandas'
    JSON writer rounds floating-point values, however, so it cannot be the
    source of a provenance fingerprint: two adjacent IEEE-754 values could
    otherwise receive the same digest. Pandas' own protocol-5 pickle keeps
    table values, dtypes, indexes, and metadata bit-for-bit; the surrounding
    recursive envelope keeps mappings deterministic without ever unpickling
    user data.
    """

    if isinstance(value, (pd.Series, pd.DataFrame)):
        return b"pandas\0" + pickle.dumps(value, protocol=5)
    if is_dataclass(value) and not isinstance(value, type):
        parts = [
            b"dataclass\0",
            type(value).__module__.encode("utf-8"),
            b"\0",
            type(value).__qualname__.encode("utf-8"),
        ]
        for item in fields(value):
            parts.extend((b"\0", item.name.encode("utf-8"), b"\0", _fingerprint_bytes(getattr(value, item.name))))
        return b"".join(parts)
    if isinstance(value, Mapping):
        items = sorted(
            ((_fingerprint_bytes(key), _fingerprint_bytes(item)) for key, item in value.items()),
            key=lambda pair: pair[0],
        )
        return b"mapping\0" + b"".join(key + b"\0" + item + b"\0" for key, item in items)
    if isinstance(value, tuple):
        return b"tuple\0" + b"".join(_fingerprint_bytes(item) + b"\0" for item in value)
    scalar = json.dumps(_json_scalar(value), ensure_ascii=True, sort_keys=True, separators=(",", ":"))
    return b"scalar\0" + scalar.encode("utf-8")


def fingerprint_value(value: object) -> str:
    """Return a deterministic, lossless SHA-256 fingerprint for model data."""

    return sha256(_fingerprint_bytes(value)).hexdigest()


def _snapshot_value(value: object) -> object:
    """Deep-copy analytical data before storage or public model exposure."""

    if isinstance(value, (pd.Series, pd.DataFrame)):
        return value.copy(deep=True)
    if isinstance(value, Mapping):
        return frozen_mapping(value)
    if isinstance(value, tuple):
        return tuple(_snapshot_value(item) for item in value)
    if isinstance(value, PyfolioFactorInputs):
        return PyfolioFactorInputs(
            returns=value.returns.copy(deep=True),
            positions=value.positions.copy(deep=True),
            benchmark_rets=None if value.benchmark_rets is None else value.benchmark_rets.copy(deep=True),
        )
    if isinstance(value, FactorGroupAnalysis):
        return FactorGroupAnalysis(
            group=value.group,
            quantile_statistics=cast("pd.DataFrame", _snapshot_value(value.quantile_statistics)),
            factor_weights=cast("pd.DataFrame", _snapshot_value(value.factor_weights)),
            factor_returns=cast("pd.DataFrame", _snapshot_value(value.factor_returns)),
            mean_returns_by_quantile=cast("pd.DataFrame", _snapshot_value(value.mean_returns_by_quantile)),
            std_error_by_quantile=cast("pd.DataFrame", _snapshot_value(value.std_error_by_quantile)),
            information_coefficient=cast("pd.DataFrame", _snapshot_value(value.information_coefficient)),
            mean_information_coefficient=cast(
                "pd.Series | pd.DataFrame", _snapshot_value(value.mean_information_coefficient)
            ),
            quantile_turnover=cast("Mapping[int, pd.DataFrame]", _snapshot_value(value.quantile_turnover)),
            rank_autocorrelation=cast("pd.DataFrame", _snapshot_value(value.rank_autocorrelation)),
        )
    if isinstance(value, EventAnalysisModel):
        return EventAnalysisModel(
            event_windows=cast("pd.DataFrame", _snapshot_value(value.event_windows)),
            mean_returns=cast("pd.Series", _snapshot_value(value.mean_returns)),
            return_distribution=cast("pd.Series", _snapshot_value(value.return_distribution)),
            quantile_average_returns=cast("pd.DataFrame", _snapshot_value(value.quantile_average_returns)),
        )
    return value


def frozen_mapping(mapping: Mapping[_MappingKey, _MappingValue]) -> Mapping[_MappingKey, _MappingValue]:
    """Own mapping values and expose a fresh read-only cache boundary."""

    return MappingProxyType({key: cast("_MappingValue", _snapshot_value(value)) for key, value in mapping.items()})


@dataclass(frozen=True)
class FactorAnalysisConfig:
    """Every typed option that can alter an :class:`FactorAnalysisModel`."""

    long_short: bool = True
    group_neutral: bool = False
    equal_weight: bool = False
    by_group: bool = False
    periods: tuple[str, ...] = ()
    event_before: int | None = None
    event_after: int | None = None
    turnover_periods: tuple[int, ...] = (1,)
    time_aggregation: tuple[str, ...] = ("M",)
    include_pyfolio: bool = True
    pyfolio_capital: int | float | None = None
    pyfolio_benchmark_period: str = "1D"
    fingerprint: str = field(init=False)

    def __post_init__(self) -> None:
        """Normalize public options before computing the frozen config digest."""

        periods = tuple(self.periods)
        turnover_periods = tuple(self.turnover_periods)
        time_aggregation = tuple(self.time_aggregation)
        if any(not isinstance(period, str) for period in periods):
            raise TypeError("periods must contain strings")
        if any(not isinstance(period, int) or isinstance(period, bool) for period in turnover_periods):
            raise TypeError("turnover_periods must contain integers")
        if any(not isinstance(frequency, str) for frequency in time_aggregation):
            raise TypeError("time_aggregation must contain strings")
        capital = self.pyfolio_capital
        if capital is not None:
            if not isinstance(capital, (int, float)) or isinstance(capital, bool):
                raise TypeError("pyfolio_capital must be an int, float, or None")
            if isinstance(capital, int) and abs(capital) > 2**53:
                raise ValueError("pyfolio_capital integer must be representable exactly by float arithmetic")

        object.__setattr__(self, "periods", periods)
        object.__setattr__(self, "turnover_periods", turnover_periods)
        object.__setattr__(self, "time_aggregation", time_aggregation)

        payload = {
            "long_short": self.long_short,
            "group_neutral": self.group_neutral,
            "equal_weight": self.equal_weight,
            "by_group": self.by_group,
            "periods": periods,
            "event_before": self.event_before,
            "event_after": self.event_after,
            "turnover_periods": turnover_periods,
            "time_aggregation": time_aggregation,
            "include_pyfolio": self.include_pyfolio,
            "pyfolio_capital": self.pyfolio_capital,
            "pyfolio_benchmark_period": self.pyfolio_benchmark_period,
        }
        object.__setattr__(self, "fingerprint", fingerprint_value(payload))


@dataclass(frozen=True)
class FactorGroupAnalysis:
    """Typed, per-group analytical data used by grouped tables and charts."""

    group: Hashable
    quantile_statistics: pd.DataFrame
    factor_weights: pd.DataFrame
    factor_returns: pd.DataFrame
    mean_returns_by_quantile: pd.DataFrame
    std_error_by_quantile: pd.DataFrame
    information_coefficient: pd.DataFrame
    mean_information_coefficient: pd.Series | pd.DataFrame
    quantile_turnover: Mapping[int, pd.DataFrame]
    rank_autocorrelation: pd.DataFrame

    _DATA_FIELDS = frozenset(
        {
            "quantile_statistics",
            "factor_weights",
            "factor_returns",
            "mean_returns_by_quantile",
            "std_error_by_quantile",
            "information_coefficient",
            "mean_information_coefficient",
            "quantile_turnover",
            "rank_autocorrelation",
        }
    )

    def __post_init__(self) -> None:
        """Own independent group data rather than caller-visible work buffers."""

        for name in self._DATA_FIELDS:
            object.__setattr__(self, name, _snapshot_value(object.__getattribute__(self, name)))

    def __getattribute__(self, name: str) -> object:
        value = object.__getattribute__(self, name)
        if name in object.__getattribute__(self, "_DATA_FIELDS"):
            return _snapshot_value(value)
        return value


@dataclass(frozen=True)
class EventAnalysisModel:
    """Frozen event-window values, averages, and distribution inputs."""

    event_windows: pd.DataFrame
    mean_returns: pd.Series
    return_distribution: pd.Series
    quantile_average_returns: pd.DataFrame

    _DATA_FIELDS = frozenset({"event_windows", "mean_returns", "return_distribution", "quantile_average_returns"})

    def __post_init__(self) -> None:
        """Own independent event data rather than caller-visible work buffers."""

        for name in self._DATA_FIELDS:
            object.__setattr__(self, name, _snapshot_value(object.__getattribute__(self, name)))

    def __getattribute__(self, name: str) -> object:
        value = object.__getattribute__(self, name)
        if name in object.__getattribute__(self, "_DATA_FIELDS"):
            return _snapshot_value(value)
        return value


@dataclass(frozen=True)
class FactorAnalysisModel:
    """One compute-once analytical snapshot consumed by later renderers."""

    config: FactorAnalysisConfig
    factor_data: pd.DataFrame
    forward_periods: tuple[str, ...]
    quantile_statistics: pd.DataFrame
    factor_weights: pd.DataFrame
    factor_returns: pd.DataFrame
    factor_cumulative_returns: Mapping[str, pd.Series]
    factor_positions: Mapping[str, pd.DataFrame]
    alpha_beta: pd.DataFrame
    mean_returns_by_quantile: pd.DataFrame
    std_error_by_quantile: pd.DataFrame
    mean_returns_by_date: pd.DataFrame
    mean_return_spread: pd.DataFrame
    mean_return_spread_std: pd.DataFrame | None
    information_coefficient: pd.DataFrame
    mean_information_coefficient: pd.Series | pd.DataFrame
    quantile_turnover: Mapping[int, pd.DataFrame]
    rank_autocorrelation: pd.DataFrame
    grouped_results: Mapping[Hashable, FactorGroupAnalysis]
    time_aggregated_results: Mapping[str, pd.Series | pd.DataFrame]
    pyfolio_inputs: PyfolioFactorInputs | None
    event_returns: EventAnalysisModel | None = None
    result_fingerprint: str = ""

    _DATA_FIELDS = frozenset(
        {
            "factor_data",
            "quantile_statistics",
            "factor_weights",
            "factor_returns",
            "factor_cumulative_returns",
            "factor_positions",
            "alpha_beta",
            "mean_returns_by_quantile",
            "std_error_by_quantile",
            "mean_returns_by_date",
            "mean_return_spread",
            "mean_return_spread_std",
            "information_coefficient",
            "mean_information_coefficient",
            "quantile_turnover",
            "rank_autocorrelation",
            "grouped_results",
            "time_aggregated_results",
            "pyfolio_inputs",
            "event_returns",
        }
    )

    def __post_init__(self) -> None:
        """Store private canonical snapshots for every renderer-facing field."""

        for name in self._DATA_FIELDS:
            object.__setattr__(self, name, _snapshot_value(object.__getattribute__(self, name)))

    def __getattribute__(self, name: str) -> object:
        value = object.__getattribute__(self, name)
        if name in object.__getattribute__(self, "_DATA_FIELDS"):
            return _snapshot_value(value)
        return value

    def to_serializable(self) -> dict[str, object]:
        """Return a JSON-compatible, renderer-free representation of this model."""

        payload = serializable_value(self)
        if not isinstance(payload, dict):  # pragma: no cover - dataclass invariant
            raise TypeError("FactorAnalysisModel serialization must be a mapping")
        return payload


__all__ = [
    "EventAnalysisModel",
    "FactorAnalysisConfig",
    "FactorAnalysisModel",
    "FactorGroupAnalysis",
    "deserialize_serializable_value",
    "fingerprint_value",
    "frozen_mapping",
    "serializable_value",
]
