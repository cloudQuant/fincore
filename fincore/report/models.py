"""Canonical compute-once report models shared by every renderer.

The model is deliberately limited to financial values that were computed by a
domain workflow.  Renderers receive this document and may format or plot it,
but they do not receive raw inputs from which to recompute metrics.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import date, datetime
from types import MappingProxyType
from typing import Any

import numpy as np
import pandas as pd

__all__ = ["ReportDocument", "ReportSection"]


def _mapping(value: Mapping[str, Any], *, field_name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{field_name} must be a mapping")
    if any(not isinstance(key, str) or not key for key in value):
        raise TypeError(f"{field_name} keys must be non-empty strings")
    return MappingProxyType(dict(value))


def _copy_series(value: pd.Series, *, field_name: str) -> pd.Series:
    if not isinstance(value, pd.Series):
        raise TypeError(f"{field_name} values must be pandas Series")
    return value.copy(deep=True)


def _copy_table(value: pd.DataFrame, *, field_name: str) -> pd.DataFrame:
    if not isinstance(value, pd.DataFrame):
        raise TypeError(f"{field_name} values must be pandas DataFrame objects")
    return value.copy(deep=True)


def _freeze_series(values: Mapping[str, pd.Series]) -> Mapping[str, pd.Series]:
    return MappingProxyType({name: _copy_series(value, field_name="series") for name, value in values.items()})


def _freeze_tables(values: Mapping[str, pd.DataFrame]) -> Mapping[str, pd.DataFrame]:
    return MappingProxyType({name: _copy_table(value, field_name="tables") for name, value in values.items()})


def _semantic_value(value: Any) -> Any:
    """Convert a report value to deterministic JSON primitives without formatting it."""

    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if math.isnan(value):
            return {"type": "float", "value": "nan"}
        if math.isinf(value):
            return {"type": "float", "value": "inf" if value > 0 else "-inf"}
        return value
    if isinstance(value, np.generic):
        return _semantic_value(value.item())
    if value is pd.NA:
        return {"type": "pandas_na"}
    if value is pd.NaT:
        return {"type": "pandas_nat"}
    if isinstance(value, (pd.Timestamp, pd.Timedelta, datetime, date)):
        return {"type": type(value).__name__, "value": value.isoformat()}
    if isinstance(value, Mapping):
        if any(not isinstance(key, str) for key in value):
            raise TypeError("report semantic mappings must use string keys")
        return {key: _semantic_value(value[key]) for key in sorted(value)}
    if isinstance(value, tuple):
        return {"type": "tuple", "values": [_semantic_value(item) for item in value]}
    if isinstance(value, list):
        return [_semantic_value(item) for item in value]
    if isinstance(value, pd.Index):
        return {
            "type": "index",
            "name": _semantic_value(value.name),
            "values": [_semantic_value(item) for item in value.tolist()],
        }
    raise TypeError(f"report semantic payload does not support {type(value).__module__}.{type(value).__qualname__}")


def _series_payload(value: pd.Series) -> dict[str, Any]:
    return {
        "name": _semantic_value(value.name),
        "dtype": str(value.dtype),
        "index": _semantic_value(value.index),
        "values": [_semantic_value(item) for item in value.tolist()],
    }


def _table_payload(value: pd.DataFrame) -> dict[str, Any]:
    return {
        "columns": _semantic_value(value.columns),
        "dtypes": [str(dtype) for dtype in value.dtypes],
        "index": _semantic_value(value.index),
        "values": [[_semantic_value(item) for item in row] for row in value.itertuples(index=False, name=None)],
    }


@dataclass(frozen=True, slots=True)
class ReportSection:
    """One typed, renderer-ready section of a report document."""

    key: str
    title: str
    metrics: Mapping[str, Any] = field(default_factory=dict)
    tables: Mapping[str, pd.DataFrame] = field(default_factory=dict)
    series: Mapping[str, pd.Series] = field(default_factory=dict)
    units: Mapping[str, str] = field(default_factory=dict)
    legends: Mapping[str, str] = field(default_factory=dict)
    notes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.key, str) or not self.key:
            raise ValueError("report section key must be a non-empty string")
        if not isinstance(self.title, str) or not self.title:
            raise ValueError("report section title must be a non-empty string")
        object.__setattr__(self, "metrics", _mapping(self.metrics, field_name="metrics"))
        object.__setattr__(self, "tables", _freeze_tables(self.tables))
        object.__setattr__(self, "series", _freeze_series(self.series))
        units = _mapping(self.units, field_name="units")
        legends = _mapping(self.legends, field_name="legends")
        if any(not isinstance(value, str) or not value for value in units.values()):
            raise TypeError("report section units must be non-empty strings")
        if any(not isinstance(value, str) or not value for value in legends.values()):
            raise TypeError("report section legends must be non-empty strings")
        known_values = set(self.metrics) | set(self.tables) | set(self.series)
        if unknown := sorted((set(units) | set(legends)) - known_values):
            raise ValueError(f"report section metadata references unknown values: {unknown!r}")
        if any(not isinstance(note, str) or not note for note in self.notes):
            raise TypeError("report section notes must contain non-empty strings")
        object.__setattr__(self, "units", units)
        object.__setattr__(self, "legends", legends)

    def semantic_payload(self) -> dict[str, Any]:
        """Return a renderer-independent representation for golden comparison."""

        return {
            "key": self.key,
            "title": self.title,
            "metrics": _semantic_value(self.metrics),
            "tables": {key: _table_payload(self.tables[key]) for key in sorted(self.tables)},
            "series": {key: _series_payload(self.series[key]) for key in sorted(self.series)},
            "units": _semantic_value(self.units),
            "legends": _semantic_value(self.legends),
            "notes": list(self.notes),
        }


@dataclass(frozen=True, slots=True)
class ReportDocument:
    """The sole compute-model contract passed to report renderers."""

    domain: str
    title: str
    sections: tuple[ReportSection, ...]
    metadata: Mapping[str, Any] = field(default_factory=dict)
    offline_assets: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.domain, str) or not self.domain:
            raise ValueError("report domain must be a non-empty string")
        if not isinstance(self.title, str) or not self.title:
            raise ValueError("report title must be a non-empty string")
        if not isinstance(self.sections, tuple) or not all(isinstance(item, ReportSection) for item in self.sections):
            raise TypeError("report sections must be a tuple of ReportSection values")
        keys = tuple(section.key for section in self.sections)
        if len(set(keys)) != len(keys):
            raise ValueError("report section keys must be unique")
        assets = _mapping(self.offline_assets, field_name="offline_assets")
        if any(not isinstance(value, str) for value in assets.values()):
            raise TypeError("offline asset content must be text")
        object.__setattr__(self, "metadata", _mapping(self.metadata, field_name="metadata"))
        object.__setattr__(self, "offline_assets", assets)

    def section(self, key: str) -> ReportSection:
        """Resolve one uniquely named report section."""

        for section in self.sections:
            if section.key == key:
                return section
        raise KeyError(key)

    def semantic_payload(self) -> dict[str, Any]:
        """Return normalized data for report semantic golden tests."""

        return {
            "domain": self.domain,
            "title": self.title,
            "sections": [section.semantic_payload() for section in self.sections],
            "metadata": _semantic_value(self.metadata),
            "offline_assets": {
                key: hashlib.sha256(value.encode("utf-8")).hexdigest()
                for key, value in sorted(self.offline_assets.items())
            },
        }

    @property
    def semantic_digest(self) -> str:
        """Return a stable digest of financial/report semantics, not renderer bytes."""

        payload = json.dumps(self.semantic_payload(), sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode(
            "utf-8"
        )
        return hashlib.sha256(payload).hexdigest()
