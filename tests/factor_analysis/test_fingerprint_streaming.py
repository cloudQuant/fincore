from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from decimal import Decimal
from hashlib import sha256
from types import MappingProxyType
from typing import Any
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import pytest

from fincore.factor_analysis import models


class _CaptureWriter:
    def __init__(self) -> None:
        self.parts: list[bytes] = []

    def write(self, value: bytes | bytearray | memoryview) -> None:
        self.parts.append(bytes(value))

    def getvalue(self) -> bytes:
        return b"".join(self.parts)


class _CustomCell:
    def __init__(self, label: str) -> None:
        self.label = label


class _HashEqualityTrap:
    def __init__(self, label: str) -> None:
        self.label = label

    def __hash__(self) -> int:
        raise AssertionError("custom hash must not be called")

    def __eq__(self, other: object) -> bool:
        raise AssertionError("custom equality must not be called")


class _StringSubclassWithHashEqualityTrap(str):
    def __hash__(self) -> int:
        raise AssertionError("string-subclass hash must not be called")

    def __eq__(self, other: object) -> bool:
        raise AssertionError("string-subclass equality must not be called")


class _ChangingDescriptor:
    def __init__(self) -> None:
        self.reads = 0

    def __get__(self, instance: object | None, owner: type[object]) -> str:
        if instance is None:
            return "initial"
        self.reads += 1
        return f"read-{self.reads}"

    def __set__(self, instance: object, value: str) -> None:
        vars(instance)["_stored_value"] = value


_CHANGING_DESCRIPTOR = _ChangingDescriptor()


@dataclass
class _DescriptorRecord:
    value: str = _CHANGING_DESCRIPTOR


def _stream_bytes(value: object) -> bytes:
    writer = _CaptureWriter()
    length = models._stream_fingerprint(value, writer)
    encoded = writer.getvalue()
    assert len(encoded) == length
    return encoded


def _assert_stream_matches_legacy(value: object) -> None:
    expected = models._fingerprint_bytes(value)
    actual = _stream_bytes(value)
    assert actual == expected
    assert models.fingerprint_value(value) == sha256(expected).hexdigest()


@pytest.mark.parametrize(
    "values",
    [
        np.array([0.0, -0.0, np.inf, -np.inf, np.nan], dtype=np.float64),
        np.array([0.0, -0.0, np.inf, -np.inf, np.nan], dtype=np.float32),
        np.array([3.5, -4.25, 1.0, 8.125], dtype=">f8"),
        np.array([3.5, -4.25, 1.0, 8.125], dtype="<f4")[::2],
        np.arange(32, dtype=np.float64)[::3],
    ],
)
def test_float_series_stream_is_byte_identical(values: np.ndarray[Any, Any]) -> None:
    series = pd.Series(
        values, index=pd.RangeIndex(len(values), name="row"), name="signal"
    )
    series.attrs = {"source": "fixture", "revision": 3}
    _assert_stream_matches_legacy(series)


@pytest.mark.parametrize("dtype", [np.float64, np.float32])
def test_float_stream_preserves_nan_payload_and_sign_bits(dtype: Any) -> None:
    if dtype is np.float64:
        bits = np.array(
            [
                0x7FF8000000000042,
                0xFFF8000000000123,
                0x7FF0000000000042,
                0xFFF0000000000123,
                0x8000000000000000,
            ],
            dtype=np.uint64,
        )
    else:
        bits = np.array(
            [0x7FC00042, 0xFFC00123, 0x7F800042, 0xFF800123, 0x80000000],
            dtype=np.uint32,
        )
    values = bits.view(dtype)
    with np.errstate(invalid="ignore"):
        _assert_stream_matches_legacy(pd.Series(values))
        _assert_stream_matches_legacy(pd.DataFrame({"signal": values}))


@pytest.mark.parametrize("dtype", [">f8", ">f4"])
def test_big_endian_nan_payloads_match_legacy_object_conversion(dtype: str) -> None:
    if dtype == ">f8":
        bits = np.array(
            [0x7FF0000000000042, 0xFFF8000000000123], dtype=">u8"
        )
    else:
        bits = np.array([0x7F800042, 0xFFC00123], dtype=">u4")
    values = bits.view(dtype)
    series = pd.Series(values, name="big-endian-nan")
    with np.errstate(invalid="ignore"):
        _assert_stream_matches_legacy(series)


def test_equal_object_scalars_with_distinct_encodings_do_not_share_cached_bytes() -> None:
    utc_timestamp = pd.Timestamp("2025-01-01 00:00:00", tz="UTC")
    eastern_timestamp = pd.Timestamp("2024-12-31 19:00:00", tz="America/New_York")
    seconds_timestamp = pd.Timestamp(1_700_000_000, unit="s", tz="UTC")
    milliseconds_timestamp = pd.Timestamp(1_700_000_000_000, unit="ms", tz="UTC")
    zone = ZoneInfo("America/New_York")
    first_fold = dt.datetime(2025, 11, 2, 1, 30, tzinfo=zone, fold=0)
    second_fold = dt.datetime(2025, 11, 2, 1, 30, tzinfo=zone, fold=1)
    assert utc_timestamp == eastern_timestamp
    assert seconds_timestamp == milliseconds_timestamp
    assert Decimal("1.0") == Decimal("1.00")
    assert first_fold == second_fold

    values = np.empty(10, dtype=object)
    values[:] = [
        0.0,
        -0.0,
        Decimal("1.0"),
        Decimal("1.00"),
        utc_timestamp,
        eastern_timestamp,
        seconds_timestamp,
        milliseconds_timestamp,
        first_fold,
        second_fold,
    ]
    frame = pd.DataFrame({"value": pd.Series(values, dtype=object)})
    frame.attrs = {
        "decimal": (Decimal("1.0"), Decimal("1.00")),
        "timestamp": (utc_timestamp, eastern_timestamp),
        "resolution": (seconds_timestamp, milliseconds_timestamp),
        "fold": (first_fold, second_fold),
        "signed_zero": (0.0, -0.0),
    }
    _assert_stream_matches_legacy(frame)
    _assert_stream_matches_legacy(tuple(values))


@pytest.mark.parametrize(
    "items",
    [
        ["资产/α", "资产/α", "βeta", "βeta", "e\u0301", "é"],
        [-1, -1, -(2**130), -(2**130), 0, 2**128, 2**128],
        [True, 1, True, 1, False, 0, False, 0],
        [float("nan"), float("-nan"), 0.0, -0.0, float("inf"), -float("inf")],
    ],
)
def test_object_scalar_runs_match_legacy_bytes(items: list[object]) -> None:
    values = np.empty(len(items), dtype=object)
    values[:] = items
    series = pd.Series(values, name="object-scalars")
    _assert_stream_matches_legacy(series)
    _assert_stream_matches_legacy(tuple(items))


def test_repeated_object_scalar_runs_use_bounded_frames_and_skip_generic_length(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    values = np.empty(50_000, dtype=object)
    values[:18_000] = "资产-α/long-repeated-label"
    values[18_000:38_000] = -(2**130)
    values[38_000:] = "tail"
    expected = models._fingerprint_bytes(tuple(values))

    class _FrameDigestSpy:
        def __init__(self) -> None:
            self.digest = sha256()
            self.sizes: list[int] = []
            self.total = 0

        def write(self, data: bytes | bytearray | memoryview) -> None:
            self.sizes.append(len(data))
            self.total += len(data)
            self.digest.update(data)

    scalar_calls = 0
    generic_length_calls = 0
    original_scalar_bytes = models._FingerprintStreamContext.scalar_bytes
    original_stream_length = models._stream_length

    def count_scalar_bytes(
        context: models._FingerprintStreamContext, value: object
    ) -> bytes:
        nonlocal scalar_calls
        scalar_calls += 1
        return original_scalar_bytes(context, value)

    def count_generic_length(value: object, context: models._FingerprintStreamContext) -> int:
        nonlocal generic_length_calls
        generic_length_calls += 1
        return original_stream_length(value, context)

    monkeypatch.setattr(
        models._FingerprintStreamContext, "scalar_bytes", count_scalar_bytes
    )
    monkeypatch.setattr(models, "_stream_length", count_generic_length)

    context = models._FingerprintStreamContext()
    encoded_length = models._stream_tuple_array_length(values, values.dtype, context)
    writer = _FrameDigestSpy()
    models._stream_tuple_array(values, values.dtype, writer, context)

    assert writer.total == encoded_length == len(expected)
    assert writer.digest.hexdigest() == sha256(expected).hexdigest()
    assert max(writer.sizes) <= models._FINGERPRINT_LEAF_BATCH_BYTES
    assert generic_length_calls == 0
    # Three adjacent runs are encoded once during each of the length and write
    # passes, instead of dispatching once per object cell.
    assert scalar_calls == 6
    cache_entries = (
        len(context._strings)
        + len(context._integers)
        + len(context._other_scalars)
        + len(context._timestamps)
        + len(context._framed_parts)
    )
    assert cache_entries <= models._FINGERPRINT_LEAF_CACHE_ENTRIES
    assert context._cache_bytes <= models._FINGERPRINT_LEAF_CACHE_BYTES


def test_interleaved_builtin_string_chunks_match_v1_with_bounded_batches(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    chunk_rows = models._FINGERPRINT_OBJECT_STRING_CHUNK_ROWS
    count = chunk_rows * 2 + 137
    labels = ("", "资产/α", "e\u0301", "é", "asset\x00x")
    values = np.empty(count, dtype=object)
    values[:] = [labels[position % len(labels)] for position in range(count)]
    series = pd.Series(values, name="asset")
    expected = models._fingerprint_bytes(series)
    original_factorize = models.pd.factorize
    factorized_sizes: list[int] = []

    def factorize(values: Any, **kwargs: Any) -> tuple[Any, Any]:
        assert kwargs == {"sort": False}
        chunk = np.asarray(values, dtype=object)
        assert len(chunk) <= chunk_rows
        assert all(type(item) is str for item in chunk)
        factorized_sizes.append(len(chunk))
        return original_factorize(values, **kwargs)

    monkeypatch.setattr(models.pd, "factorize", factorize)
    writer = _CaptureWriter()
    encoded_length = models._stream_fingerprint(series, writer)

    assert encoded_length == len(expected)
    assert writer.getvalue() == expected
    assert factorized_sizes == [
        min(chunk_rows, count - start)
        for start in range(0, count, chunk_rows)
    ] * 2
    assert max(map(len, writer.parts)) <= models._FINGERPRINT_LEAF_BATCH_BYTES


def test_low_run_string_column_keeps_the_run_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    values = np.empty(20_000, dtype=object)
    values[:8_000] = "first"
    values[8_000:17_000] = "second"
    values[17_000:] = "third"
    series = pd.Series(values, name="group")
    expected = models._fingerprint_bytes(series)

    def unexpected_factorize(values: Any, **kwargs: Any) -> tuple[Any, Any]:
        raise AssertionError("low-run strings should use the adjacent-run path")

    monkeypatch.setattr(models.pd, "factorize", unexpected_factorize)
    _assert_stream_matches_legacy(series)
    assert _stream_bytes(series) == expected


def test_mixed_builtin_type_chunk_keeps_the_run_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    values = np.empty(16_384, dtype=object)
    values[:] = ["A", 7] * (len(values) // 2)
    series = pd.Series(values, name="mixed")
    expected = models._fingerprint_bytes(series)

    def unexpected_factorize(values: Any, **kwargs: Any) -> tuple[Any, Any]:
        raise AssertionError("mixed scalar types should use the adjacent-run path")

    monkeypatch.setattr(models.pd, "factorize", unexpected_factorize)
    _assert_stream_matches_legacy(series)
    assert _stream_bytes(series) == expected


def test_unsampled_non_string_values_fallback_without_custom_hash_or_equality(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    values = np.empty(16_384, dtype=object)
    values[:] = ["A", "B"] * (len(values) // 2)
    # The sample checks indexes 0/1, 16/17, ...; the first actual chunk scan
    # must still reject these unsampled values before pandas hashes the chunk.
    values[2] = _StringSubclassWithHashEqualityTrap("A")
    values[3] = _HashEqualityTrap("custom")
    series = pd.Series(values, name="mixed")
    expected = models._fingerprint_bytes(series)

    def unexpected_factorize(values: Any, **kwargs: Any) -> tuple[Any, Any]:
        raise AssertionError("a non-exact string chunk must not be factorized")

    monkeypatch.setattr(models.pd, "factorize", unexpected_factorize)
    with pytest.raises(models._FingerprintNotStreamable):
        models._stream_fingerprint(series, _CaptureWriter())
    assert models.fingerprint_value(series) == sha256(expected).hexdigest()


def _assert_string_array_writer_matches_legacy(
    values: np.ndarray[Any, Any],
) -> _CaptureWriter:
    expected = models._fingerprint_bytes(tuple(values.tolist()))
    context = models._FingerprintStreamContext()
    encoded_length = models._stream_object_string_array_length(values, context)
    writer = _CaptureWriter()
    models._stream_write_object_string_array(values, writer, context)

    assert encoded_length == len(expected)
    assert writer.getvalue() == expected
    assert all(
        len(part) <= models._FINGERPRINT_LEAF_BATCH_BYTES for part in writer.parts
    )
    return writer


def test_string_writer_joins_exactly_one_mib() -> None:
    rows = models._FINGERPRINT_OBJECT_STRING_CHUNK_ROWS
    leaf_bytes = len(models._FingerprintStreamContext().scalar_bytes("x" * 105)) + 8
    assert leaf_bytes * rows == models._FINGERPRINT_LEAF_BATCH_BYTES
    values = np.empty(rows, dtype=object)
    values[::2] = "x" * 105
    values[1::2] = "y" * 105
    assert models._stream_object_string_array_batch_worthwhile(values)
    writer = _CaptureWriter()

    expected = models._fingerprint_bytes(tuple(values.tolist()))
    context = models._FingerprintStreamContext()
    encoded_length = models._stream_object_string_array_length(values, context)
    models._stream_write_object_string_array(values, writer, context)

    assert encoded_length == len(expected)
    assert writer.getvalue() == expected
    assert models._FINGERPRINT_LEAF_BATCH_BYTES in map(len, writer.parts)
    assert all(
        len(part) <= models._FINGERPRINT_LEAF_BATCH_BYTES for part in writer.parts
    )


def test_interleaved_string_frame_falls_back_above_one_mib(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rows = models._FINGERPRINT_OBJECT_STRING_CHUNK_ROWS
    leaf_bytes = len(models._FingerprintStreamContext().scalar_bytes("x" * 106)) + 8
    assert leaf_bytes * rows > models._FINGERPRINT_LEAF_BATCH_BYTES
    values = np.empty(rows, dtype=object)
    values[::2] = "x" * 106
    values[1::2] = "y" * 106
    frame = pd.DataFrame({"asset": pd.Series(values, dtype=object)})
    expected = models._fingerprint_bytes(frame)
    assert models._stream_object_string_array_batch_worthwhile(values)
    original_factorize = models.pd.factorize
    factorized_sizes: list[int] = []
    original_write_runs = models._stream_write_object_scalar_runs
    fallback_sizes: list[int] = []

    def factorize(items: Any, **kwargs: Any) -> tuple[Any, Any]:
        assert len(items) <= rows
        assert all(type(item) is str for item in items)
        factorized_sizes.append(len(items))
        return original_factorize(items, **kwargs)

    def record_fallback(
        items: np.ndarray[Any, Any],
        writer: Any,
        context: models._FingerprintStreamContext,
    ) -> None:
        fallback_sizes.append(len(items))
        original_write_runs(items, writer, context)

    monkeypatch.setattr(models.pd, "factorize", factorize)
    monkeypatch.setattr(models, "_stream_write_object_scalar_runs", record_fallback)
    writer = _CaptureWriter()
    encoded_length = models._stream_fingerprint(frame, writer)

    assert encoded_length == len(expected)
    assert writer.getvalue() == expected
    assert factorized_sizes == [rows, rows]
    # The one-element object-typed columns Index uses the same legacy writer;
    # the oversized asset chunk is the second call.
    assert sorted(fallback_sizes) == [len(frame.columns), rows]
    assert max(map(len, writer.parts)) <= models._FINGERPRINT_LEAF_BATCH_BYTES


def test_high_cardinality_string_column_uses_actual_bounded_batch_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rows = models._FINGERPRINT_OBJECT_STRING_CHUNK_ROWS
    values = np.array(
        [f"asset-{position:04d}" for position in range(rows)], dtype=object
    )
    frame = pd.DataFrame({"asset": values}, index=pd.RangeIndex(rows, name="row"))
    expected = models._fingerprint_bytes(frame)
    original_factorize = models.pd.factorize
    factorized_sizes: list[int] = []
    batch_writes: list[tuple[int, ...]] = []
    fallback_sizes: list[int] = []
    original_write_chunk = models._stream_write_object_string_chunk
    original_write_runs = models._stream_write_object_scalar_runs

    def factorize(items: Any, **kwargs: Any) -> tuple[Any, Any]:
        assert kwargs == {"sort": False}
        assert len(items) <= rows
        assert all(type(item) is str for item in items)
        factorized_sizes.append(len(items))
        return original_factorize(items, **kwargs)

    def record_batch(
        items: np.ndarray[Any, Any],
        writer: Any,
        context: models._FingerprintStreamContext,
    ) -> None:
        before = len(writer.parts)
        original_write_chunk(items, writer, context)
        batch_writes.append(tuple(len(part) for part in writer.parts[before:]))

    def record_fallback(
        items: np.ndarray[Any, Any],
        writer: Any,
        context: models._FingerprintStreamContext,
    ) -> None:
        fallback_sizes.append(len(items))
        original_write_runs(items, writer, context)

    monkeypatch.setattr(models.pd, "factorize", factorize)
    monkeypatch.setattr(models, "_stream_write_object_string_chunk", record_batch)
    monkeypatch.setattr(models, "_stream_write_object_scalar_runs", record_fallback)
    writer = _CaptureWriter()
    encoded_length = models._stream_fingerprint(frame, writer)

    assert encoded_length == len(expected)
    assert writer.getvalue() == expected
    assert factorized_sizes == [rows, rows]
    assert len(batch_writes) == 1 and len(batch_writes[0]) == 1
    assert 0 < batch_writes[0][0] <= models._FINGERPRINT_LEAF_BATCH_BYTES
    assert fallback_sizes == [len(frame.columns)]
    assert max(map(len, writer.parts)) <= models._FINGERPRINT_LEAF_BATCH_BYTES

@pytest.mark.parametrize(("unique_count", "repeats"), ((500, 18), (5471, 4)))
def test_market_cardinality_string_columns_use_stream_batches(
    unique_count: int,
    repeats: int,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rows = unique_count * repeats
    labels = np.array(
        [f"asset-{position:04d}" for position in range(unique_count)], dtype=object
    )
    values = np.resize(labels, rows)
    frame = pd.DataFrame(
        {"asset": values, "signal": np.arange(rows, dtype=np.float64)},
        index=pd.RangeIndex(rows, name="row"),
    )
    expected = models._fingerprint_bytes(frame)

    sample_count = min(models._FINGERPRINT_OBJECT_STRING_SAMPLE_ROWS, len(values) - 1)
    sample_positions = [
        sample * (len(values) - 2) // (sample_count - 1)
        for sample in range(sample_count)
    ]
    sample_unique_ratio = (
        len({values[position] for position in sample_positions}) / sample_count
    )
    if unique_count == 5471:
        # The old 0.75 uniqueness gate rejected this interleaved full-market shape.
        assert sample_unique_ratio > 0.75
    else:
        assert sample_unique_ratio <= 0.75

    original_factorize = models.pd.factorize
    original_write_chunk = models._stream_write_object_string_chunk
    original_write_runs = models._stream_write_object_scalar_runs
    original_context_init = models._FingerprintStreamContext.__init__
    original_scalar_bytes = models._FingerprintStreamContext.scalar_bytes
    factorized_sizes: list[int] = []
    string_batch_writes: list[tuple[int, tuple[int, ...]]] = []
    contexts: list[models._FingerprintStreamContext] = []
    asset_labels = set(labels.tolist())
    string_cache_hits = 0
    fallback_sizes: list[int] = []

    def factorize(items: Any, **kwargs: Any) -> tuple[Any, Any]:
        assert kwargs == {"sort": False}
        assert len(items) <= models._FINGERPRINT_OBJECT_STRING_CHUNK_ROWS
        assert all(type(item) is str for item in items)
        factorized_sizes.append(len(items))
        return original_factorize(items, **kwargs)

    def record_batch(
        items: np.ndarray[Any, Any],
        writer: Any,
        context: models._FingerprintStreamContext,
    ) -> None:
        before = len(writer.parts)
        original_write_chunk(items, writer, context)
        string_batch_writes.append(
            (len(items), tuple(len(part) for part in writer.parts[before:]))
        )

    def record_fallback(
        items: np.ndarray[Any, Any],
        writer: Any,
        context: models._FingerprintStreamContext,
    ) -> None:
        fallback_sizes.append(len(items))
        original_write_runs(items, writer, context)

    def record_context_init(context: models._FingerprintStreamContext) -> None:
        original_context_init(context)
        contexts.append(context)

    def record_scalar_bytes(
        context: models._FingerprintStreamContext, value: object
    ) -> bytes:
        nonlocal string_cache_hits
        if (
            type(value) is str
            and value in asset_labels
            and value in context._strings
        ):
            string_cache_hits += 1
        return original_scalar_bytes(context, value)

    monkeypatch.setattr(models.pd, "factorize", factorize)
    monkeypatch.setattr(models, "_stream_write_object_string_chunk", record_batch)
    monkeypatch.setattr(models, "_stream_write_object_scalar_runs", record_fallback)
    monkeypatch.setattr(models._FingerprintStreamContext, "__init__", record_context_init)
    monkeypatch.setattr(models._FingerprintStreamContext, "scalar_bytes", record_scalar_bytes)
    writer = _CaptureWriter()
    encoded_length = models._stream_fingerprint(frame, writer)

    expected_chunks = [
        min(models._FINGERPRINT_OBJECT_STRING_CHUNK_ROWS, rows - start)
        for start in range(0, rows, models._FINGERPRINT_OBJECT_STRING_CHUNK_ROWS)
    ]
    assert encoded_length == len(expected)
    assert writer.getvalue() == expected
    assert factorized_sizes == expected_chunks * 2
    assert [item[0] for item in string_batch_writes] == expected_chunks
    assert all(len(parts) == 1 for _, parts in string_batch_writes)
    assert all(
        8 * chunk_rows < parts[0] <= models._FINGERPRINT_LEAF_BATCH_BYTES
        for chunk_rows, parts in string_batch_writes
    )
    assert fallback_sizes == [len(frame.columns)]
    assert len(contexts) == 1
    context = contexts[0]
    assert string_cache_hits > 0
    assert len(context._strings) >= unique_count
    cache_entries = (
        len(context._strings)
        + len(context._integers)
        + len(context._other_scalars)
        + len(context._timestamps)
        + len(context._framed_parts)
    )
    assert cache_entries <= models._FINGERPRINT_LEAF_CACHE_ENTRIES
    assert context._cache_bytes <= models._FINGERPRINT_LEAF_CACHE_BYTES


def test_string_chunk_rejects_overlength_values_before_factorize(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rows = models._FINGERPRINT_OBJECT_STRING_CHUNK_ROWS
    values = np.resize(np.array(["asset-a", "asset-b"], dtype=object), rows)
    # This position is outside the 1024-row sample; the bounded chunk scan must
    # still reject it before pandas factorize sees any values.
    values[2] = "x" * (models._FINGERPRINT_OBJECT_STRING_MAX_CHARS + 1)
    frame = pd.DataFrame({"asset": values}, index=pd.RangeIndex(rows))
    expected = models._fingerprint_bytes(frame)

    def unexpected_factorize(items: Any, **kwargs: Any) -> tuple[Any, Any]:
        raise AssertionError("overlength exact strings must not be factorized")

    monkeypatch.setattr(models.pd, "factorize", unexpected_factorize)
    writer = _CaptureWriter()
    encoded_length = models._stream_fingerprint(frame, writer)

    assert encoded_length == len(expected)
    assert writer.getvalue() == expected
    assert max(map(len, writer.parts)) <= models._FINGERPRINT_LEAF_BATCH_BYTES

def test_empty_object_string_array_matches_empty_tuple_oracle() -> None:
    values = np.empty(0, dtype=object)
    writer = _assert_string_array_writer_matches_legacy(values)

    assert writer.getvalue() == models._fingerprint_bytes(())
    assert writer.parts == [len(b"tuple").to_bytes(4, "big"), b"tuple"]


def test_custom_object_column_uses_legacy_fallback() -> None:
    series = pd.Series(
        np.array([_CustomCell("custom"), _CustomCell("cell")], dtype=object),
        name="extension-values",
    )
    expected = sha256(models._fingerprint_bytes(series)).hexdigest()

    with pytest.raises(models._FingerprintNotStreamable):
        models._stream_fingerprint(series, _CaptureWriter())
    assert models.fingerprint_value(series) == expected


def test_stream_handles_standard_dataframe_index_metadata_and_duplicate_columns() -> (
    None
):
    index = pd.MultiIndex.from_arrays(
        [
            pd.Index(["A", "B", "A"], name="asset"),
            pd.date_range("2024-01-01", periods=3, tz="UTC", name="time"),
        ],
        names=["asset", "time"],
    )
    frame = pd.DataFrame(
        np.array([[1.5, -0.0], [np.inf, np.nan], [3.25, -4.5]], dtype=np.float64),
        index=index,
        columns=pd.Index(["x", "x"], name="metric"),
    )
    frame.attrs = {"tags": ("research", "snapshot"), "version": 2}
    _assert_stream_matches_legacy(frame)
    _assert_stream_matches_legacy(index)
    _assert_stream_matches_legacy(
        pd.CategoricalIndex(["low", "high", "low"], ordered=True, name="q")
    )
    _assert_stream_matches_legacy(pd.timedelta_range("1D", periods=4, name="elapsed"))
    _assert_stream_matches_legacy(pd.RangeIndex(-2, 10, 2, name="range"))


def test_timezone_aware_timestamp_columns_match_object_conversion() -> None:
    timestamps = pd.date_range("2025-03-01", periods=7, freq="h", tz="America/New_York")
    frame = pd.DataFrame(
        {
            "time": timestamps,
            "signal": np.arange(7, dtype=np.float64),
            "label": pd.Series(["a", "b", "a", "c", "a", "b", "c"], dtype=object),
        },
        index=pd.RangeIndex(7),
    )
    frame.loc[3, "time"] = pd.NaT
    _assert_stream_matches_legacy(frame)
    _assert_stream_matches_legacy(frame["time"])


def test_integer_datetime_and_timestamp_index_bulk_frames_match_legacy() -> None:
    values = np.array(
        [0, 1, -1, 10, -10, 2**62, -(2**62), np.iinfo(np.int64).min + 1],
        dtype=np.int64,
    )
    _assert_stream_matches_legacy(pd.Series(values, name="integer"))
    _assert_stream_matches_legacy(pd.Index(values, name="integer-index"))

    timestamps = pd.date_range(
        "2024-01-01", periods=256, freq="s", tz="UTC", name="time"
    ).to_series(index=pd.RangeIndex(256))
    timestamps.iloc[31] = pd.NaT
    _assert_stream_matches_legacy(timestamps)
    _assert_stream_matches_legacy(timestamps.index)


@pytest.mark.parametrize(
    "dtype",
    [np.int8, np.int16, np.int32, np.int64, np.uint8, np.uint16, np.uint32, np.uint64],
)
def test_integer_dtype_bulk_frames_match_legacy(dtype: Any) -> None:
    info = np.iinfo(dtype)
    values = np.array([0, 1, info.max, info.min], dtype=dtype)
    _assert_stream_matches_legacy(pd.Series(values, name="integer"))
    _assert_stream_matches_legacy(pd.Index(values, name="integer-index"))


def test_nested_mappingproxy_and_dataclass_payload_match_legacy() -> None:
    @dataclass(frozen=True)
    class Report:
        tables: MappingProxyType
        details: tuple[object, ...]

    frame = pd.DataFrame(
        {
            "signal": np.array([1.0, 2.0, -0.0]),
            "label": np.array(["up", "down", "up"], dtype=object),
        },
        index=pd.date_range("2025-01-01", periods=3, name="date"),
    )
    value = Report(
        MappingProxyType({"z": frame, "a": pd.Series([1, 2, 3], name="count")}),
        ({"longer": "same", "x": 4}, "same", -1),
    )
    _assert_stream_matches_legacy(value)


@pytest.mark.parametrize(
    "value",
    [
        {
            "zeta": "text-key",
            -9: "negative-small",
            0: "zero",
            23: "positive-small",
            -(10**80): "negative-large",
            10**95: "positive-large",
        },
        {
            "nested": {
                2: pd.Series([1.25, -0.0], name="positive"),
                -12: pd.Series([np.inf, np.nan], name="negative"),
                "label": "mixed-keys",
            },
            "outer": 10**64,
        },
    ],
)
def test_exact_string_and_integer_mapping_keys_keep_legacy_sort_order(
    value: dict[object, object],
) -> None:
    expected = models._fingerprint_bytes(value)
    actual = _stream_bytes(value)

    assert actual == expected
    assert models.fingerprint_value(value) == sha256(expected).hexdigest()


def test_custom_mapping_key_and_encoded_key_collision_keep_legacy_fallback() -> None:
    class IntKey(int):
        pass

    class SameEncodingKey:
        pass

    first = SameEncodingKey()
    second = SameEncodingKey()
    assert first is not second
    assert models._fingerprint_bytes(first) == models._fingerprint_bytes(second)

    values = (
        {IntKey(7): "int-subclass"},
        {first: "z-value", second: "a-value"},
    )
    for value in values:
        expected = models._fingerprint_bytes(value)
        with pytest.raises(models._FingerprintNotStreamable):
            models._stream_fingerprint(value, _CaptureWriter())
        assert models.fingerprint_value(value) == sha256(expected).hexdigest()


def test_duplicate_encoded_builtin_keys_trigger_legacy_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    value = {1: "first", 2: "second"}
    expected = models._fingerprint_bytes(value)
    original_scalar_bytes = models._FingerprintStreamContext.scalar_bytes

    def colliding_scalar_bytes(
        context: models._FingerprintStreamContext, key: object
    ) -> bytes:
        if type(key) in (str, int):
            return b"same-encoded-key"
        return original_scalar_bytes(context, key)

    monkeypatch.setattr(
        models._FingerprintStreamContext, "scalar_bytes", colliding_scalar_bytes
    )
    with pytest.raises(models._FingerprintNotStreamable):
        models._stream_fingerprint(value, _CaptureWriter())
    assert models.fingerprint_value(value) == sha256(expected).hexdigest()


def test_prepared_report_result_with_integer_turnover_keys_streams_exactly(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from fincore.factor_analysis import analysis, data

    dates = pd.bdate_range("2024-01-02", periods=10, tz="UTC")
    assets = tuple(f"A{position:02d}" for position in range(8))
    factor_dates = dates[:-2]
    index = pd.MultiIndex.from_product(
        (factor_dates, assets), names=("date", "asset")
    )
    cross_section = np.linspace(-1.0, 1.0, len(assets))
    factor = pd.Series(
        np.tile(cross_section, len(factor_dates))
        + np.repeat(np.arange(len(factor_dates), dtype=float) * 0.025, len(assets)),
        index=index,
        name="factor",
    )
    prices = pd.DataFrame(
        {
            asset: (100.0 + position)
            * (1.0 + np.arange(len(dates), dtype=float) * (position + 1) * 0.0007)
            for position, asset in enumerate(assets)
        },
        index=dates,
    )
    prepared = data.prepare_factor_data(
        factor,
        prices,
        periods=(1,),
        quantiles=4,
        max_loss=1.0,
    )

    result_payloads: list[tuple[int, str]] = []

    def compare_streamed_result_payload(value: object) -> str:
        assert type(value) is dict
        payload = value
        turnover = payload["quantile_turnover"]
        assert type(turnover) is dict
        assert turnover
        assert all(type(key) is int for key in turnover)

        expected = models._fingerprint_bytes(payload)
        actual = _stream_bytes(payload)
        assert actual == expected
        digest = sha256(actual).hexdigest()
        assert digest == sha256(expected).hexdigest()
        result_payloads.append((len(actual), digest))
        return digest

    monkeypatch.setattr(analysis, "fingerprint_value", compare_streamed_result_payload)
    model = analysis.analyze_factor(
        prepared.data,
        periods=None,
        turnover_periods=(1, 2),
        time_aggregation=(),
        include_portfolio_inputs=True,
        portfolio_benchmark_period="1D",
    )

    assert len(result_payloads) == 1
    assert model.result_fingerprint == result_payloads[0][1]


@pytest.mark.parametrize(
    "value",
    [
        [],
        ["x", 1, []],
        [[], ["a", [1, 2]], {"nested": [[], "b"]}],
        {"empty": [], "nested": {"levels": [["x"], []]}},
        ([], ["inside-tuple", {"list": [1, []]}]),
    ],
)
def test_nested_lists_use_exact_legacy_envelopes(value: object) -> None:
    _assert_stream_matches_legacy(value)


def test_repository_frozen_mapping_and_defensive_report_fields_stream_without_copies() -> (
    None
):
    from fincore.factor_analysis.models import EventAnalysisModel

    event = EventAnalysisModel(
        event_windows=pd.DataFrame({"value": np.array([1.0, -0.0, np.nan])}),
        mean_returns=pd.Series([1.0, 2.0]),
        return_distribution=pd.Series([3.0, 4.0]),
        quantile_average_returns=pd.DataFrame({"q1": [0.1, 0.2]}),
        aggregate_quantile_average_returns=pd.DataFrame({"q1": [0.3, 0.4]}),
    )
    frozen = models.frozen_mapping(
        {1: pd.DataFrame({"turnover": [0.1, 0.2]}), 2: event.event_windows}
    )
    _assert_stream_matches_legacy(event)
    _assert_stream_matches_legacy(frozen)


def test_custom_object_cells_keep_defensive_dataclass_getter_semantics(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from fincore.factor_analysis.models import EventAnalysisModel

    event = EventAnalysisModel(
        event_windows=pd.DataFrame(
            {"custom": np.array([_CustomCell("event")], dtype=object)}
        ),
        mean_returns=pd.Series([1.0]),
        return_distribution=pd.Series([2.0]),
        quantile_average_returns=pd.DataFrame({"q1": [0.1]}),
        aggregate_quantile_average_returns=pd.DataFrame({"q1": [0.2]}),
    )
    expected = sha256(models._fingerprint_bytes(event)).hexdigest()
    real_snapshot_pandas = models.snapshot_pandas
    snapshots: list[int] = []

    def record_snapshot(value: pd.Series | pd.DataFrame) -> pd.Series | pd.DataFrame:
        snapshots.append(1)
        return real_snapshot_pandas(value)

    monkeypatch.setattr(models, "snapshot_pandas", record_snapshot)
    assert models.fingerprint_value(event) == expected
    assert snapshots


def test_dataclass_property_descriptor_keeps_single_legacy_getter_read() -> None:
    record = _DescriptorRecord()
    _CHANGING_DESCRIPTOR.reads = 0
    expected = sha256(models._fingerprint_bytes(record)).hexdigest()
    _CHANGING_DESCRIPTOR.reads = 0
    assert models.fingerprint_value(record) == expected
    assert _CHANGING_DESCRIPTOR.reads == 1


def test_extension_dtype_and_custom_mapping_keep_legacy_fallback() -> None:
    class CustomMapping(dict[str, object]):
        pass

    extension = pd.Series([1, None, 3], dtype="Int64")
    custom = CustomMapping({"frame": pd.DataFrame({"x": [1.0, 2.0]})})
    for value in (extension, custom):
        with pytest.raises(models._FingerprintNotStreamable):
            models._stream_fingerprint(value, _CaptureWriter())
        assert models.fingerprint_value(value) == sha256(
            models._fingerprint_bytes(value)
        ).hexdigest()


def test_series_subclass_override_uses_legacy_astype_semantics() -> None:
    class CustomSeries(pd.Series):
        calls = 0

        @property
        def _constructor(self) -> type[pd.Series]:
            return pd.Series

        def astype(self, *args: Any, **kwargs: Any) -> pd.Series:
            type(self).calls += 1
            return super().astype(*args, **kwargs)

    series = CustomSeries([1.0, 2.0, 3.0], name="subclass")
    expected = sha256(models._fingerprint_bytes(series)).hexdigest()
    CustomSeries.calls = 0
    assert models.fingerprint_value(series) == expected
    assert CustomSeries.calls == 1


class _UpdateSpy:
    def __init__(self) -> None:
        self.sizes: list[int] = []

    def update(self, data: bytes | bytearray | memoryview) -> None:
        self.sizes.append(len(data))


def test_digest_writer_bounds_each_update() -> None:
    digest = _UpdateSpy()
    writer = models._FingerprintDigestWriter(digest)
    writer.write(b"a" * (models._FINGERPRINT_CHUNK_BYTES * 2 + 17))
    writer.finish()
    assert digest.sizes
    assert max(digest.sizes) <= models._FINGERPRINT_CHUNK_BYTES
    assert sum(digest.sizes) == models._FINGERPRINT_CHUNK_BYTES * 2 + 17


def test_datetime_frames_are_bounded_before_writer_receives_them() -> None:
    series = pd.Series(
        pd.date_range("2024-01-01", periods=32_769, tz="UTC"),
        name="timestamps",
    )

    class _FrameSizeSpy:
        def __init__(self) -> None:
            self.sizes: list[int] = []

        def write(self, data: bytes | bytearray | memoryview) -> None:
            self.sizes.append(len(data))

    writer = _FrameSizeSpy()
    encoded_length = models._stream_fingerprint(series, writer)

    assert writer.sizes
    assert sum(writer.sizes) == encoded_length
    assert max(writer.sizes) <= models._FINGERPRINT_CHUNK_BYTES
    # UTC's v1 timezone envelope makes a fixed 32,768-row timestamp frame
    # exceed the limit even though the digest writer itself streams updates.
    assert sum(size > 1_000_000 for size in writer.sizes) >= 2


@pytest.mark.parametrize(
    ("unit", "timezone"),
    [
        ("ns", "UTC"),
        ("us", "America/New_York"),
        ("ms", "UTC"),
    ],
)
def test_datetime_adjacent_runs_match_legacy_bytes(
    unit: str, timezone: str
) -> None:
    raw_values = np.repeat(
        np.array(
            [
                1_700_000_000,
                -1,
                0,
                1_700_000_000,
                np.iinfo(np.int64).min,
                2,
                3,
            ],
            dtype=np.int64,
        ),
        [6, 4, 4, 4, 4, 4, 6],
    )
    timestamps = pd.to_datetime(raw_values, unit=unit, utc=True).tz_convert(timezone)
    series = pd.Series(timestamps, name="timestamps")
    frame = pd.DataFrame({"timestamps": timestamps})

    _assert_stream_matches_legacy(series)
    _assert_stream_matches_legacy(frame)


def test_datetime_all_nat_runs_match_legacy_bytes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    timestamps = pd.Series(
        pd.to_datetime(
            np.full(32, np.iinfo(np.int64).min, dtype=np.int64),
            unit="ns",
            utc=True,
        ),
        name="all-nat",
    )
    calls = 0
    original = models._stream_write_datetime_run_chunk

    def record_run_chunk(
        chunk: np.ndarray[Any, Any],
        unit: bytes,
        timezone: bytes,
        writer: Any,
    ) -> None:
        nonlocal calls
        calls += 1
        original(chunk, unit, timezone, writer)

    monkeypatch.setattr(models, "_stream_write_datetime_run_chunk", record_run_chunk)
    _assert_stream_matches_legacy(timestamps)

    assert calls > 0


def test_datetime_run_crossing_chunk_boundary_matches_legacy_bytes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    raw_values = np.repeat(np.array([1, 2, 3], dtype=np.int64), [4, 8, 4])
    series = pd.Series(pd.to_datetime(raw_values, unit="ns", utc=True))
    original_plan = models._stream_datetime_plan
    original_writer = models._stream_write_datetime_run_chunk
    written_chunks: list[tuple[int, ...]] = []

    def small_plan(
        values: Any,
        dtype: pd.DatetimeTZDtype,
        context: models._FingerprintStreamContext,
    ) -> tuple[np.ndarray[Any, Any], bytes, bytes, int, int]:
        raw, unit, timezone, length, _ = original_plan(values, dtype, context)
        return raw, unit, timezone, length, 8

    def record_run_chunk(
        chunk: np.ndarray[Any, Any],
        unit: bytes,
        timezone: bytes,
        writer: Any,
    ) -> None:
        written_chunks.append(tuple(int(value) for value in chunk))
        original_writer(chunk, unit, timezone, writer)

    monkeypatch.setattr(models, "_stream_datetime_plan", small_plan)
    monkeypatch.setattr(models, "_stream_write_datetime_run_chunk", record_run_chunk)
    _assert_stream_matches_legacy(series)

    assert written_chunks[0][-1] == written_chunks[1][0] == 2


def test_unique_heavy_datetime_chunk_keeps_vector_writer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    raw_values = np.arange(10_000, dtype=np.int64) * 1_000_000
    series = pd.Series(pd.to_datetime(raw_values, unit="ns", utc=True))
    run_writer_calls = 0
    original = models._stream_write_datetime_run_chunk

    def record_run_chunk(
        chunk: np.ndarray[Any, Any],
        unit: bytes,
        timezone: bytes,
        writer: Any,
    ) -> None:
        nonlocal run_writer_calls
        run_writer_calls += 1
        original(chunk, unit, timezone, writer)

    monkeypatch.setattr(models, "_stream_write_datetime_run_chunk", record_run_chunk)
    _assert_stream_matches_legacy(series)

    assert run_writer_calls == 0


def test_repeated_datetime_writes_remain_bounded() -> None:
    raw_values = np.repeat(np.arange(500, dtype=np.int64) * 1_000_000, 100)
    series = pd.Series(pd.to_datetime(raw_values, unit="ns", utc=True))
    expected = sha256(models._fingerprint_bytes(series)).hexdigest()

    class _DigestSizeSpy:
        def __init__(self) -> None:
            self.digest = sha256()
            self.sizes: list[int] = []
            self.total = 0

        def write(self, data: bytes | bytearray | memoryview) -> None:
            self.sizes.append(len(data))
            self.total += len(data)
            self.digest.update(data)

    writer = _DigestSizeSpy()
    encoded_length = models._stream_fingerprint(series, writer)

    assert writer.sizes
    assert writer.total == encoded_length
    assert max(writer.sizes) <= models._FINGERPRINT_CHUNK_BYTES
    assert writer.digest.hexdigest() == expected


def test_large_dataframe_uses_streaming_path_and_keeps_digest_updates_bounded(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import hashlib

    frame = pd.DataFrame({"signal": np.arange(260_000, dtype=np.float64)})
    expected = hashlib.sha256(models._fingerprint_bytes(frame)).hexdigest()
    real_sha256 = models.sha256
    updates: list[int] = []

    class RecordingDigest:
        def __init__(self) -> None:
            self.digest = real_sha256()

        def update(self, data: bytes | bytearray | memoryview) -> None:
            updates.append(len(data))
            self.digest.update(data)

        def hexdigest(self) -> str:
            return self.digest.hexdigest()

    monkeypatch.setattr(models, "sha256", RecordingDigest)
    monkeypatch.setattr(
        models,
        "_fingerprint_pandas",
        lambda _value: (_ for _ in ()).throw(
            AssertionError("legacy pandas encoder used")
        ),
    )
    assert models.fingerprint_value(frame) == expected
    assert len(updates) > 1
    assert max(updates) <= models._FINGERPRINT_CHUNK_BYTES


def test_research_result_fingerprinting_streams_production_payloads(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import hashlib

    from fincore.factor_analysis import research

    dates = pd.bdate_range("2024-01-02", periods=8, tz="UTC")
    assets = tuple(f"A{position:02d}" for position in range(12))
    grid = pd.bdate_range(dates[0], periods=len(dates) + 2, tz="UTC")
    index = pd.MultiIndex.from_product((dates, assets), names=("time", "asset"))
    cross_section = np.linspace(-0.05, 0.05, len(assets))
    factor_values = np.tile(cross_section, len(dates)) + np.repeat(
        np.arange(len(dates), dtype=float) * 0.01, len(assets)
    )
    factors = pd.DataFrame(
        {"momentum": factor_values}, index=index
    )
    opens = pd.DataFrame(index=grid, columns=list(assets), dtype=float)
    for position, asset in enumerate(assets):
        opens[asset] = 100.0 + position + np.arange(len(grid), dtype=float)
    closes = opens * 0.99
    spec = {
        "targets": assets,
        "start": dates[0],
        "end": dates[-1],
        "periods": (1,),
        "preprocess": "none",
        "min_assets": 10,
        "quantiles": 5,
        "stability_lags": (1,),
        "label": "close_to_close",
    }

    def forbidden_legacy_pandas(_value: object) -> bytes:
        raise AssertionError("production research payload used legacy pandas encoding")

    real_fingerprint_pandas = models._fingerprint_pandas
    real_sha256 = models.sha256
    update_sizes: list[int] = []

    class RecordingDigest:
        def __init__(self) -> None:
            self.digest = real_sha256()

        def update(self, data: bytes | bytearray | memoryview) -> None:
            update_sizes.append(len(data))
            self.digest.update(data)

        def hexdigest(self) -> str:
            return self.digest.hexdigest()

    monkeypatch.setattr(models, "_fingerprint_pandas", forbidden_legacy_pandas)
    monkeypatch.setattr(models, "sha256", RecordingDigest)
    result = research.evaluate_factor_research(
        factors,
        grid=pd.DataFrame({"open_time": grid}, index=grid),
        prices={"open": opens, "close": closes},
        universe=None,
        groups=None,
        precomputed_returns=None,
        spec=spec,
    )
    assert result.manifest["evaluation_id"]
    assert result.manifest["result_fingerprint"]
    assert update_sizes
    assert max(update_sizes) <= models._FINGERPRINT_CHUNK_BYTES

    monkeypatch.setattr(models, "_fingerprint_pandas", real_fingerprint_pandas)

    def legacy_fingerprint(value: object) -> str:
        return hashlib.sha256(models._fingerprint_bytes(value)).hexdigest()

    monkeypatch.setattr(research, "fingerprint_value", legacy_fingerprint)
    legacy_result = research.evaluate_factor_research(
        factors,
        grid=pd.DataFrame({"open_time": grid}, index=grid),
        prices={"open": opens, "close": closes},
        universe=None,
        groups=None,
        precomputed_returns=None,
        spec=spec,
    )
    assert result.manifest["evaluation_id"] == legacy_result.manifest["evaluation_id"]
    assert result.manifest["result_fingerprint"] == legacy_result.manifest[
        "result_fingerprint"
    ]
