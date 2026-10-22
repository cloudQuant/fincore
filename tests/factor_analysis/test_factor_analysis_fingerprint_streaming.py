"""Exact-byte and memory-bound contracts for model fingerprint streaming."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from decimal import Decimal
from hashlib import sha256
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from fincore.factor_analysis import models

if TYPE_CHECKING:
    import pytest


@dataclass
class _DataclassBackedMapping(Mapping[str, object]):
    frame: pd.DataFrame

    def __getitem__(self, key: str) -> object:
        if key != "mapping-view":
            raise KeyError(key)
        return pd.DataFrame({"hidden": [10, 20, 30, 40]})

    def __iter__(self) -> Iterator[str]:
        return iter(("mapping-view",))

    def __len__(self) -> int:
        return 1


@dataclass
class _NestedPandasRecord:
    frame: pd.DataFrame
    series: pd.Series


def _complex_fingerprint_payload() -> dict[str, object]:
    timestamps = pd.DatetimeIndex(
        ["2025-01-01", "2025-01-01", "2025-01-02", "2025-01-02"],
        tz="Asia/Shanghai",
    )
    assets = pd.Categorical(
        ["A", "A", "B", "B"],
        categories=["A", "B", "unused"],
        ordered=True,
    )
    index = pd.MultiIndex.from_arrays([timestamps, assets], names=["timestamp", "asset"])
    nullable = pd.Series(pd.array([1, None, 3, 4], dtype="Int64"), index=index)
    categorical = pd.Series(
        pd.Categorical(
            ["low", "high", None, "low"],
            categories=["low", "high", "unused"],
            ordered=True,
        ),
        index=index,
    )
    signed_float = pd.Series([-0.0, 0.0, np.nan, 1.25], index=index)
    object_values = pd.Series([None, pd.NA, float("nan"), -0.0], index=index, dtype=object)
    frame = pd.concat([nullable, categorical, signed_float, object_values], axis=1)
    frame.columns = pd.Index(["duplicate", "duplicate", "float", "object"], name="field")
    frame.attrs = {
        "nested": {"decimal": Decimal("1.2300"), "labels": ("a", None)},
        "note": "fingerprint metadata",
    }
    timezone_series = pd.Series(
        pd.date_range("2025-02-01", periods=4, freq="h", tz="America/New_York"),
        index=index,
        name="timezone-values",
    )
    timezone_series.attrs = {"source": "tz-aware"}
    return {"nested": {"tables": [frame], "series": timezone_series}}


def _assert_streamed_bytes_equal_legacy(payload: object) -> None:
    legacy = models._fingerprint_bytes(payload)
    plan = models._fingerprint_stream_plan(payload)
    digest = sha256()
    chunks = []
    for chunk in models._fingerprint_stream_chunks(plan):
        chunks.append(chunk)
        digest.update(chunk)

    assert b"".join(chunks) == legacy
    assert digest.hexdigest() == sha256(legacy).hexdigest()


def test_streaming_preserves_legacy_bytes_for_nested_pandas_values(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = _complex_fingerprint_payload()
    before = {
        "frame": payload["nested"]["tables"][0].copy(deep=True),  # type: ignore[index]
        "series": payload["nested"]["series"].copy(deep=True),  # type: ignore[index]
    }
    legacy = models._fingerprint_bytes(payload)

    monkeypatch.setattr(models, "_STREAMING_FINGERPRINT_MIN_PANDAS_SIZE", 0)
    _assert_streamed_bytes_equal_legacy(payload)
    assert models.fingerprint_value(payload) == sha256(legacy).hexdigest()

    frame = payload["nested"]["tables"][0]  # type: ignore[index]
    series = payload["nested"]["series"]  # type: ignore[index]
    pd.testing.assert_frame_equal(frame, before["frame"])
    pd.testing.assert_series_equal(series, before["series"])
    assert np.signbit(frame.iloc[0, 2])
    assert pd.isna(frame.iloc[2, 3])


def test_small_pandas_payload_keeps_legacy_fingerprint_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    frame = pd.DataFrame({"value": [1.0, -0.0, np.nan]})
    payload = {"table": frame}
    expected = sha256(models._fingerprint_bytes(payload)).hexdigest()

    def unexpected_stream_plan(value: object) -> models._FingerprintStreamPlan:
        raise AssertionError(f"small pandas values should not stream: {type(value).__name__}")

    monkeypatch.setattr(models, "_fingerprint_stream_plan", unexpected_stream_plan)
    assert models.fingerprint_value(payload) == expected


def test_mapping_key_encoding_collision_uses_legacy_mapping_bytes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(models, "_STREAMING_FINGERPRINT_MIN_PANDAS_SIZE", 0)
    first_nan = float("nan")
    second_nan = float("nan")
    payload = {
        first_nan: pd.DataFrame({"first": [1]}),
        second_nan: pd.DataFrame({"second": [2]}),
    }

    plan = models._fingerprint_stream_plan(payload)
    legacy = models._fingerprint_bytes(payload)

    assert plan.payload == legacy
    assert plan.producer is None
    assert models.fingerprint_value(payload) == sha256(legacy).hexdigest()


def test_dataclass_dispatch_precedes_mapping_and_nested_pandas_is_streamed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(models, "_STREAMING_FINGERPRINT_MIN_PANDAS_SIZE", 2)
    dataclass_mapping = _DataclassBackedMapping(frame=pd.DataFrame({"field": [1, 2]}))
    nested_record = _NestedPandasRecord(
        frame=pd.DataFrame({"field": [3, 4, 5]}),
        series=pd.Series([6, 7], name="nested-series"),
    )
    payload = {"dataclass-mapping": dataclass_mapping, "record": nested_record}
    legacy = models._fingerprint_bytes(payload)
    plan = models._fingerprint_stream_plan(payload)

    assert b"".join(models._fingerprint_stream_chunks(plan)) == legacy
    assert models.fingerprint_value(payload) == sha256(legacy).hexdigest()


def test_stream_mapping_sorts_distinct_encoded_keys_by_length_then_bytes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(models, "_STREAMING_FINGERPRINT_MIN_PANDAS_SIZE", 0)
    payload = {
        "long-key": pd.DataFrame({"value": [1, 2]}),
        10: pd.Series([3, 4]),
        b"x": pd.DataFrame({"value": [5, 6]}),
        "a": pd.Series([7, 8]),
    }
    legacy = models._fingerprint_bytes(payload)
    plan = models._fingerprint_stream_plan(payload)
    expected_keys = sorted(
        (models._fingerprint_bytes(key) for key in payload),
        key=lambda encoded: (len(encoded), encoded),
    )
    planned_keys = [entry.parts[0].payload for entry in plan.parts]

    assert len({len(key) for key in expected_keys}) > 1
    assert planned_keys == expected_keys
    assert b"".join(models._fingerprint_stream_chunks(plan)) == legacy


def test_streaming_threshold_crosses_at_inclusive_cell_count(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(models, "_STREAMING_FINGERPRINT_MIN_PANDAS_SIZE", 6)
    original_plan = models._fingerprint_stream_plan
    planned_values: list[object] = []

    def observe_plan(value: object) -> models._FingerprintStreamPlan:
        planned_values.append(value)
        return original_plan(value)

    monkeypatch.setattr(models, "_fingerprint_stream_plan", observe_plan)
    pandas_cases = (
        (
            {"value": pd.DataFrame(np.arange(5).reshape(1, 5))},
            {"value": pd.DataFrame(np.arange(6).reshape(2, 3))},
        ),
        ({"value": pd.Series(range(5))}, {"value": pd.Series(range(6))}),
    )

    for below, boundary in pandas_cases:
        below_expected = sha256(models._fingerprint_bytes(below)).hexdigest()
        boundary_expected = sha256(models._fingerprint_bytes(boundary)).hexdigest()
        planned_values.clear()

        assert models.fingerprint_value(below) == below_expected
        assert planned_values == []
        assert models.fingerprint_value(boundary) == boundary_expected
        assert planned_values


def test_stream_chunks_are_bounded_by_one_dataframe_column_payload(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(models, "_STREAMING_FINGERPRINT_MIN_PANDAS_SIZE", 0)
    frame = pd.DataFrame(
        {
            "first": np.arange(512, dtype=np.int64),
            "second": np.arange(512, dtype=np.int64) * -3,
            "third": np.arange(512, dtype=np.int64) + 11,
        }
    )
    payload = {"result": frame}
    plan = models._fingerprint_stream_plan(payload)
    chunks = list(models._fingerprint_stream_chunks(plan))
    largest_column = max(
        len(models._fingerprint_bytes(tuple(frame.iloc[:, position].astype(object).tolist())))
        for position in range(frame.shape[1])
    )

    assert max(map(len, chunks)) <= largest_column
    assert max(map(len, chunks)) < len(models._fingerprint_bytes(payload))
