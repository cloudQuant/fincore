"""Shared Alphalens compatibility fixtures and pinned-manifest helpers."""

from __future__ import annotations

import json
from collections.abc import Mapping
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

_REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
_MANIFEST_PATH = _REPOSITORY_ROOT / "tests/compat/fixtures/alphalens-0.4.0-cloudquant-api.json"
_ASSETS = tuple(f"asset_{ordinal:02d}" for ordinal in range(10))
_FIXTURE_TABLE_SCHEMA_VERSION = "fincore-factor-fixture-table-v1"


@lru_cache(maxsize=1)
def load_pinned_manifest() -> dict[str, Any]:
    """Load the development-only pinned API manifest for compatibility assertions."""

    return json.loads(_MANIFEST_PATH.read_text(encoding="utf-8"))


def manifest_entries() -> tuple[dict[str, Any], ...]:
    """Return the pinned C0 definition entries in stable manifest order."""

    return tuple(load_pinned_manifest()["entries"])


def callable_entries_with_signature() -> tuple[dict[str, Any], ...]:
    """Return entries whose manifest freezes a source or introspection signature."""

    return tuple(
        entry for entry in manifest_entries() if entry["kind"] == "function" or entry["symbol"] == "GridFigure"
    )


def accepted_call_cases() -> tuple[tuple[dict[str, Any], dict[str, Any]], ...]:
    """Return every manifest-declared accepted call grammar row."""

    return tuple((entry, case) for entry in manifest_entries() for case in entry["accepted_call_cases"])


def _is_missing(value: object) -> bool:
    """Return whether one scalar needs the explicit portable missing-value mask."""

    result = pd.isna(value)
    return isinstance(result, (bool, np.bool_)) and bool(result)


def _json_scalar(value: object) -> Any:
    """Convert supported fixture scalars to standards-compliant JSON values."""

    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, pd.Timedelta):
        return value.isoformat()
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    raise TypeError(f"fixture table serialization does not support scalar {value!r} ({type(value).__name__})")


def _serialize_vector(values: list[object]) -> dict[str, list[Any]]:
    """Encode scalar values and a parallel missing-value mask without JSON NaN."""

    nan_mask = [_is_missing(value) for value in values]
    return {
        "values": [
            None if is_missing else _json_scalar(value) for value, is_missing in zip(values, nan_mask, strict=True)
        ],
        "nan_mask": nan_mask,
    }


def _serialize_index(index: pd.Index) -> dict[str, Any]:
    """Capture all index metadata required for exact fixture reconstruction."""

    if isinstance(index, pd.MultiIndex):
        return {
            "kind": "multiindex",
            "names": [_json_scalar(name) for name in index.names],
            "levels": [_serialize_index(level) for level in index.levels],
            "codes": [code.tolist() for code in index.codes],
        }
    if isinstance(index, pd.DatetimeIndex):
        return {
            "kind": "datetimeindex",
            "name": _json_scalar(index.name),
            "dtype": str(index.dtype),
            "timezone": str(index.tz) if index.tz is not None else None,
            "freq": index.freqstr,
            **_serialize_vector(list(index)),
        }
    if isinstance(index, pd.TimedeltaIndex):
        return {
            "kind": "timedeltaindex",
            "name": _json_scalar(index.name),
            "dtype": str(index.dtype),
            "freq": index.freqstr,
            **_serialize_vector(list(index)),
        }
    return {
        "kind": "index",
        "name": _json_scalar(index.name),
        "dtype": str(index.dtype),
        **_serialize_vector(list(index)),
    }


def serialize_factor_fixture_table(value: pd.Series | pd.DataFrame) -> dict[str, Any]:
    """Return a portable v1 JSON-table envelope for a shared factor fixture.

    The envelope contains a row/column table, an explicit ``nan_mask`` (never
    a non-standard JSON ``NaN`` token), and enough index/column metadata to
    restore MultiIndex levels, dtype, timezone, frequency, names, and Series
    names exactly.  It intentionally avoids pandas ``to_json``/``read_json``
    because those default pathways discard MultiIndex structure.
    """

    if isinstance(value, pd.Series):
        return {
            "schema_version": _FIXTURE_TABLE_SCHEMA_VERSION,
            "kind": "series",
            "name": _json_scalar(value.name),
            "dtype": str(value.dtype),
            "index": _serialize_index(value.index),
            "data": _serialize_vector(value.tolist()),
        }
    if isinstance(value, pd.DataFrame):
        rows = [list(row) for row in value.astype(object).itertuples(index=False, name=None)]
        return {
            "schema_version": _FIXTURE_TABLE_SCHEMA_VERSION,
            "kind": "dataframe",
            "index": _serialize_index(value.index),
            "columns": _serialize_index(value.columns),
            "dtypes": [str(dtype) for dtype in value.dtypes],
            "data": {
                "values": [[None if _is_missing(cell) else _json_scalar(cell) for cell in row] for row in rows],
                "nan_mask": [[_is_missing(cell) for cell in row] for row in rows],
            },
        }
    raise TypeError(f"expected pandas Series or DataFrame, got {type(value).__name__}")


def _require_mapping(value: object, context: str) -> Mapping[str, Any]:
    """Fail clearly when a JSON payload is not the supported table shape."""

    if not isinstance(value, Mapping):
        raise ValueError(f"{context} must be a JSON object")
    return value


def _decode_vector(payload: Mapping[str, Any]) -> tuple[list[Any], list[bool]]:
    """Validate and decode one JSON vector plus its explicit missing-value mask."""

    values = payload.get("values")
    nan_mask = payload.get("nan_mask")
    if not isinstance(values, list) or not isinstance(nan_mask, list) or len(values) != len(nan_mask):
        raise ValueError("fixture table vector must contain equally sized values and nan_mask lists")
    if not all(isinstance(is_missing, bool) for is_missing in nan_mask):
        raise ValueError("fixture table nan_mask entries must be booleans")
    return values, nan_mask


def _decode_matrix(payload: Mapping[str, Any]) -> tuple[list[list[Any]], list[list[bool]]]:
    """Validate a two-dimensional table plus its row/column missing-value mask."""

    values = payload.get("values")
    nan_mask = payload.get("nan_mask")
    if not isinstance(values, list) or not isinstance(nan_mask, list) or len(values) != len(nan_mask):
        raise ValueError("fixture table matrix must contain equally sized values and nan_mask row lists")
    if any(not isinstance(row, list) for row in values):
        raise ValueError("fixture table value rows must be lists")
    if any(
        not isinstance(row, list) or not all(is_missing is True or is_missing is False for is_missing in row)
        for row in nan_mask
    ):
        raise ValueError("fixture table nan_mask rows must contain booleans")
    return values, nan_mask


def _deserialize_index(payload: object) -> pd.Index:
    """Rebuild a simple, datetime, timedelta, or multi-level fixture index."""

    data = _require_mapping(payload, "fixture table index")
    kind = data.get("kind")
    if kind == "multiindex":
        levels = data.get("levels")
        codes = data.get("codes")
        names = data.get("names")
        if not isinstance(levels, list) or not isinstance(codes, list) or not isinstance(names, list):
            raise ValueError("multiindex metadata requires levels, codes, and names lists")
        return pd.MultiIndex(
            levels=[_deserialize_index(level) for level in levels],
            codes=codes,
            names=names,
            verify_integrity=True,
        )

    values, nan_mask = _decode_vector(data)
    name = data.get("name")
    if kind == "datetimeindex":
        restored_values = [
            pd.NaT if is_missing else pd.Timestamp(value) for value, is_missing in zip(values, nan_mask, strict=True)
        ]
        index = pd.DatetimeIndex(restored_values, name=name)
        timezone = data.get("timezone")
        if timezone is not None:
            if not isinstance(timezone, str):
                raise ValueError("datetimeindex timezone must be a string or null")
            index = index.tz_localize(timezone) if index.tz is None else index.tz_convert(timezone)
        elif index.tz is not None:
            index = index.tz_localize(None)
        frequency = data.get("freq")
        if frequency is not None:
            if not isinstance(frequency, str):
                raise ValueError("datetimeindex frequency must be a string or null")
            index = pd.DatetimeIndex(index, name=name, freq=frequency)
        return index
    if kind == "timedeltaindex":
        restored_values = [
            pd.NaT if is_missing else pd.Timedelta(value) for value, is_missing in zip(values, nan_mask, strict=True)
        ]
        index = pd.TimedeltaIndex(restored_values, name=name)
        frequency = data.get("freq")
        if frequency is not None:
            if not isinstance(frequency, str):
                raise ValueError("timedeltaindex frequency must be a string or null")
            index = pd.TimedeltaIndex(index, name=name, freq=frequency)
        return index
    if kind == "index":
        dtype = data.get("dtype")
        if not isinstance(dtype, str):
            raise ValueError("index dtype must be a string")
        restored_values = [np.nan if is_missing else value for value, is_missing in zip(values, nan_mask, strict=True)]
        return pd.Index(restored_values, dtype=dtype, name=name)
    raise ValueError(f"unsupported fixture table index kind {kind!r}")


def _restore_array(values: list[Any], nan_mask: list[bool], dtype: str) -> Any:
    """Rebuild one typed data vector while retaining portable missing-value semantics."""

    restored_values = [np.nan if is_missing else value for value, is_missing in zip(values, nan_mask, strict=True)]
    try:
        return pd.array(restored_values, dtype=dtype)
    except (TypeError, ValueError):
        return pd.array(
            [pd.NA if is_missing else value for value, is_missing in zip(values, nan_mask, strict=True)],
            dtype=dtype,
        )


def deserialize_factor_fixture_table(payload: object) -> pd.Series | pd.DataFrame:
    """Rehydrate a :func:`serialize_factor_fixture_table` v1 JSON-table envelope."""

    data = _require_mapping(payload, "fixture table")
    if data.get("schema_version") != _FIXTURE_TABLE_SCHEMA_VERSION:
        raise ValueError(f"unsupported fixture table schema {data.get('schema_version')!r}")
    kind = data.get("kind")
    index = _deserialize_index(data.get("index"))
    table_data = _require_mapping(data.get("data"), "fixture table data")
    if kind == "series":
        dtype = data.get("dtype")
        if not isinstance(dtype, str):
            raise ValueError("series dtype must be a string")
        values, nan_mask = _decode_vector(table_data)
        if len(values) != len(index):
            raise ValueError("series data length does not match index length")
        return pd.Series(_restore_array(values, nan_mask, dtype), index=index, name=data.get("name"))
    if kind == "dataframe":
        columns = _deserialize_index(data.get("columns"))
        dtypes = data.get("dtypes")
        values, nan_mask = _decode_matrix(table_data)
        if not isinstance(dtypes, list) or not all(isinstance(dtype, str) for dtype in dtypes):
            raise ValueError("dataframe dtypes must be a list of strings")
        if len(columns) != len(dtypes):
            raise ValueError("dataframe dtype count does not match columns")
        if len(values) != len(index) or len(nan_mask) != len(index):
            raise ValueError("dataframe row count does not match index length")
        if any(not isinstance(row, list) or len(row) != len(columns) for row in values):
            raise ValueError("dataframe value rows must match the column count")
        if any(not isinstance(row, list) or len(row) != len(columns) for row in nan_mask):
            raise ValueError("dataframe nan_mask rows must match the column count")
        restored_columns = {
            ordinal: _restore_array(
                [row[ordinal] for row in values],
                [row[ordinal] for row in nan_mask],
                dtype,
            )
            for ordinal, dtype in enumerate(dtypes)
        }
        frame = pd.DataFrame(restored_columns, index=index)
        frame.columns = columns
        return frame
    raise ValueError(f"unsupported fixture table kind {kind!r}")


@lru_cache(maxsize=1)
def _shared_inputs() -> tuple[pd.Series, pd.DataFrame, pd.DataFrame, pd.Series]:
    """Build the immutable-source synthetic data contract shared by later tasks."""

    rng = np.random.default_rng(7)
    dates = pd.bdate_range("2024-01-02", periods=120)
    factor_index = pd.MultiIndex.from_product((dates, _ASSETS), names=("date", "asset"))
    raw_factor = pd.Series(rng.normal(0, 1, size=len(factor_index)), index=factor_index, name="factor")

    price_assets = (*_ASSETS, "asset_10", "asset_11")
    price_changes = rng.normal(0, 0.01, size=(len(dates), len(price_assets))).cumsum(axis=0)
    prices = pd.DataFrame(100 + price_changes, index=dates, columns=price_assets)
    prices.index.name = "date"
    tz_aware_prices = prices.copy()
    tz_aware_prices.index = tz_aware_prices.index.tz_localize("UTC")
    groups = pd.Series(
        ["sector_a" if ordinal % 2 == 0 else "sector_b" for ordinal in range(len(_ASSETS))],
        index=pd.Index(_ASSETS, name="asset"),
        name="group",
    )
    return raw_factor, prices, tz_aware_prices, groups


@pytest.fixture
def raw_factor() -> pd.Series:
    """A fresh factor series; callers may safely make local mutations."""

    return _shared_inputs()[0].copy()


@pytest.fixture
def prices() -> pd.DataFrame:
    """A fresh naive price frame with two non-factor assets."""

    return _shared_inputs()[1].copy()


@pytest.fixture
def tz_aware_prices() -> pd.DataFrame:
    """A fresh UTC version of :func:`prices`."""

    return _shared_inputs()[2].copy()


@pytest.fixture
def groups() -> pd.Series:
    """A fresh alternating-sector mapping for the ten factor assets."""

    return _shared_inputs()[3].copy()


@pytest.fixture(scope="session")
def clean_factor_data() -> pd.DataFrame:
    """Reserve the real cleaned-data fixture for Task 3 rather than fabricate it."""

    raise RuntimeError(
        "clean_factor_data is deferred until Task 3 provides prepare_factor_data; "
        "Task 2 deliberately does not fabricate cleaned factor output."
    )
