"""Shared Alphalens compatibility fixtures and pinned-manifest helpers."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from functools import lru_cache
from numbers import Real
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

_REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
_MANIFEST_PATH = _REPOSITORY_ROOT / "tests/compat/fixtures/alphalens-0.4.0-cloudquant-api.json"
_ASSETS = tuple(f"asset_{ordinal:02d}" for ordinal in range(10))
_FIXTURE_TABLE_SCHEMA_VERSION = "fincore-factor-fixture-table-v1"
_NONFINITE_TAGS = frozenset(("positive_infinity", "negative_infinity"))


@lru_cache(maxsize=1)
def _load_pinned_manifest_text() -> str:
    """Cache immutable manifest text while callers receive fresh decoded objects."""

    return _MANIFEST_PATH.read_text(encoding="utf-8")


def load_pinned_manifest() -> dict[str, Any]:
    """Load an independent development-only manifest object for each caller."""

    return json.loads(_load_pinned_manifest_text())


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
        return _json_scalar(value.item())
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, pd.Timedelta):
        return value.isoformat()
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("nonfinite fixture scalars require an explicit nonfinite tag")
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    raise TypeError(f"fixture table serialization does not support scalar {value!r} ({type(value).__name__})")


def _nonfinite_tag(value: object) -> str | None:
    """Return the portable JSON tag for one signed infinite scalar, if present."""

    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and math.isinf(value):
        return "positive_infinity" if value > 0 else "negative_infinity"
    return None


def _serialize_vector(values: list[object]) -> dict[str, list[Any]]:
    """Encode scalar values with separate missing and signed-infinity metadata."""

    nan_mask = [_is_missing(value) for value in values]
    nonfinite = [
        None if is_missing else _nonfinite_tag(value) for value, is_missing in zip(values, nan_mask, strict=True)
    ]
    return {
        "values": [
            None if is_missing or tag is not None else _json_scalar(value)
            for value, is_missing, tag in zip(values, nan_mask, nonfinite, strict=True)
        ],
        "nan_mask": nan_mask,
        "nonfinite": nonfinite,
    }


def _serialize_matrix(rows: list[list[object]]) -> dict[str, list[list[Any]]]:
    """Encode a rectangular table with the same scalar protocol as vectors."""

    encoded_rows = [_serialize_vector(row) for row in rows]
    return {
        "values": [encoded["values"] for encoded in encoded_rows],
        "nan_mask": [encoded["nan_mask"] for encoded in encoded_rows],
        "nonfinite": [encoded["nonfinite"] for encoded in encoded_rows],
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

    The envelope contains a row/column table, an explicit ``nan_mask`` for
    NaN/NaT, and a separate ``nonfinite`` tag vector for signed infinities;
    it therefore never emits a non-standard JSON ``NaN``/``Infinity`` token.
    It also includes enough index/column metadata to restore MultiIndex
    levels, dtype, timezone, frequency, names, and Series names exactly.
    It intentionally avoids pandas ``to_json``/``read_json`` because those
    default pathways discard MultiIndex structure.
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
            "data": _serialize_matrix(rows),
        }
    raise TypeError(f"expected pandas Series or DataFrame, got {type(value).__name__}")


def _require_mapping(value: object, context: str) -> Mapping[str, Any]:
    """Fail clearly when a JSON payload is not the supported table shape."""

    if not isinstance(value, Mapping):
        raise ValueError(f"{context} must be a JSON object")
    return value


def _validate_encoded_scalar(value: Any, is_missing: bool, nonfinite: object) -> str | None:
    """Fail closed unless a scalar's missing/nonfinite representation is unambiguous."""

    if nonfinite is not None and (not isinstance(nonfinite, str) or nonfinite not in _NONFINITE_TAGS):
        raise ValueError("fixture table nonfinite entries must be null or a supported signed-infinity tag")
    if is_missing and nonfinite is not None:
        raise ValueError("fixture table nonfinite entries cannot accompany a nan_mask value")
    if (is_missing or nonfinite is not None) and value is not None:
        raise ValueError("fixture table nonfinite or nan_mask values must use a null scalar slot")
    if not is_missing and nonfinite is None and value is None:
        raise ValueError("fixture table nonfinite metadata must disambiguate an otherwise null scalar")
    if isinstance(value, (float, np.floating)) and not math.isfinite(float(value)):
        raise ValueError("fixture table nonfinite values must use an explicit nonfinite tag")
    return nonfinite if isinstance(nonfinite, str) else None


def _decode_vector(payload: Mapping[str, Any]) -> tuple[list[Any], list[bool], list[str | None]]:
    """Validate and decode one JSON vector with missing and nonfinite metadata."""

    values = payload.get("values")
    nan_mask = payload.get("nan_mask")
    nonfinite = payload.get("nonfinite")
    if (
        not isinstance(values, list)
        or not isinstance(nan_mask, list)
        or not isinstance(nonfinite, list)
        or len(values) != len(nan_mask)
        or len(values) != len(nonfinite)
    ):
        raise ValueError("fixture table vector must contain equally sized values, nan_mask, and nonfinite lists")

    decoded_mask: list[bool] = []
    decoded_nonfinite: list[str | None] = []
    for value, is_missing, tag in zip(values, nan_mask, nonfinite, strict=True):
        if not isinstance(is_missing, bool):
            raise ValueError("fixture table nan_mask entries must be booleans")
        decoded_mask.append(is_missing)
        decoded_nonfinite.append(_validate_encoded_scalar(value, is_missing, tag))
    return values, decoded_mask, decoded_nonfinite


def _decode_matrix(payload: Mapping[str, Any]) -> tuple[list[list[Any]], list[list[bool]], list[list[str | None]]]:
    """Validate a two-dimensional table with missing and nonfinite metadata."""

    values = payload.get("values")
    nan_mask = payload.get("nan_mask")
    nonfinite = payload.get("nonfinite")
    if (
        not isinstance(values, list)
        or not isinstance(nan_mask, list)
        or not isinstance(nonfinite, list)
        or len(values) != len(nan_mask)
        or len(values) != len(nonfinite)
    ):
        raise ValueError("fixture table matrix must contain equally sized values, nan_mask, and nonfinite row lists")
    if any(not isinstance(row, list) for row in values):
        raise ValueError("fixture table value rows must be lists")
    if any(not isinstance(row, list) for row in nan_mask):
        raise ValueError("fixture table nan_mask rows must be lists")
    if any(not isinstance(row, list) for row in nonfinite):
        raise ValueError("fixture table nonfinite rows must be lists")

    decoded_mask: list[list[bool]] = []
    decoded_nonfinite: list[list[str | None]] = []
    for value_row, mask_row, tag_row in zip(values, nan_mask, nonfinite, strict=True):
        if len(value_row) != len(mask_row) or len(value_row) != len(tag_row):
            raise ValueError("fixture table nonfinite rows must align with values and nan_mask")
        decoded_mask_row: list[bool] = []
        decoded_tag_row: list[str | None] = []
        for value, is_missing, tag in zip(value_row, mask_row, tag_row, strict=True):
            if not isinstance(is_missing, bool):
                raise ValueError("fixture table nan_mask rows must contain booleans")
            decoded_mask_row.append(is_missing)
            decoded_tag_row.append(_validate_encoded_scalar(value, is_missing, tag))
        decoded_mask.append(decoded_mask_row)
        decoded_nonfinite.append(decoded_tag_row)
    return values, decoded_mask, decoded_nonfinite


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

    values, nan_mask, nonfinite = _decode_vector(data)
    name = data.get("name")
    if kind == "datetimeindex":
        if any(tag is not None for tag in nonfinite):
            raise ValueError("datetimeindex does not support nonfinite metadata")
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
        if any(tag is not None for tag in nonfinite):
            raise ValueError("timedeltaindex does not support nonfinite metadata")
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
        restored_values = [
            _restore_scalar(value, is_missing, tag)
            for value, is_missing, tag in zip(values, nan_mask, nonfinite, strict=True)
        ]
        index = pd.Index(restored_values, dtype=dtype, name=name)
        _validate_restored_nonfinite_slots(index, nonfinite, f"fixture table index dtype {dtype!r}")
        return index
    raise ValueError(f"unsupported fixture table index kind {kind!r}")


def _restore_scalar(value: Any, is_missing: bool, nonfinite: str | None) -> Any:
    """Restore one explicitly encoded missing or signed-infinite scalar."""

    if is_missing:
        return np.nan
    if nonfinite == "positive_infinity":
        return float("inf")
    if nonfinite == "negative_infinity":
        return float("-inf")
    return value


def _validate_restored_nonfinite_slots(restored_values: Any, nonfinite: list[str | None], context: str) -> None:
    """Reject a tag whose target dtype rewrote it to a non-numeric value."""

    if len(restored_values) != len(nonfinite):
        raise ValueError(f"fixture table nonfinite metadata does not align with {context}")
    for ordinal, tag in enumerate(nonfinite):
        if tag is None:
            continue
        restored = restored_values[ordinal]
        if isinstance(restored, np.generic):
            restored = restored.item()
        if isinstance(restored, (bool, np.bool_)) or not isinstance(restored, Real):
            raise ValueError(f"fixture table nonfinite tag at {context}[{ordinal}] must restore to a numeric infinity")
        restored_float = float(restored)
        has_expected_sign = (tag == "positive_infinity" and restored_float > 0) or (
            tag == "negative_infinity" and restored_float < 0
        )
        if not math.isinf(restored_float) or not has_expected_sign:
            raise ValueError(
                f"fixture table nonfinite tag at {context}[{ordinal}] must restore to matching signed numeric infinity"
            )


def _restore_array(values: list[Any], nan_mask: list[bool], nonfinite: list[str | None], dtype: str) -> Any:
    """Rebuild one typed data vector while retaining portable missing-value semantics."""

    restored_values = [
        _restore_scalar(value, is_missing, tag)
        for value, is_missing, tag in zip(values, nan_mask, nonfinite, strict=True)
    ]
    try:
        restored_array = pd.array(restored_values, dtype=dtype)
    except (TypeError, ValueError):
        restored_array = pd.array(
            [
                pd.NA if is_missing else _restore_scalar(value, False, tag)
                for value, is_missing, tag in zip(values, nan_mask, nonfinite, strict=True)
            ],
            dtype=dtype,
        )
    _validate_restored_nonfinite_slots(restored_array, nonfinite, f"fixture table dtype {dtype!r}")
    return restored_array


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
        values, nan_mask, nonfinite = _decode_vector(table_data)
        if len(values) != len(index):
            raise ValueError("series data length does not match index length")
        return pd.Series(_restore_array(values, nan_mask, nonfinite, dtype), index=index, name=data.get("name"))
    if kind == "dataframe":
        columns = _deserialize_index(data.get("columns"))
        dtypes = data.get("dtypes")
        values, nan_mask, nonfinite = _decode_matrix(table_data)
        if not isinstance(dtypes, list) or not all(isinstance(dtype, str) for dtype in dtypes):
            raise ValueError("dataframe dtypes must be a list of strings")
        if len(columns) != len(dtypes):
            raise ValueError("dataframe dtype count does not match columns")
        if len(values) != len(index) or len(nan_mask) != len(index) or len(nonfinite) != len(index):
            raise ValueError("dataframe row count does not match index length")
        if any(not isinstance(row, list) or len(row) != len(columns) for row in values):
            raise ValueError("dataframe value rows must match the column count")
        if any(not isinstance(row, list) or len(row) != len(columns) for row in nan_mask):
            raise ValueError("dataframe nan_mask rows must match the column count")
        if any(not isinstance(row, list) or len(row) != len(columns) for row in nonfinite):
            raise ValueError("dataframe nonfinite rows must match the column count")
        restored_columns = {
            ordinal: _restore_array(
                [row[ordinal] for row in values],
                [row[ordinal] for row in nan_mask],
                [row[ordinal] for row in nonfinite],
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


@lru_cache(maxsize=1)
def _shared_clean_factor_data() -> pd.DataFrame:
    """Compute the real Task 3 cleaned table once from immutable synthetic inputs."""

    from fincore.factor_analysis.data import prepare_factor_data

    factor, price_frame, _, group_series = _shared_inputs()
    prepared = prepare_factor_data(
        factor,
        price_frame,
        groupby=group_series,
        periods=(1, 5, 10),
        max_loss=1,
    )
    return prepared.data.copy(deep=True)


@pytest.fixture
def clean_factor_data() -> pd.DataFrame:
    """Return a fresh copy of a session-cached real cleaned factor-data table."""

    return _shared_clean_factor_data().copy(deep=True)
