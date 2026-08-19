"""Branch-completion tests for factor_analysis.models JSON handoff codec."""

from __future__ import annotations

import datetime as dt
from decimal import Decimal

import numpy as np
import pandas as pd
import pytest

from fincore.factor_analysis.models import (
    _WIRE_TYPE,
    _fixed_offset_timezone_from_payload,
    _frequency_from_payload,
    _index_from_payload,
    _numpy_array_from_payload,
    _numpy_dtype_from_payload,
    _numpy_scalar_from_payload,
    _pandas_from_payload,
    _pandas_timezone_from_payload,
    _restore_datetime_timezone,
    _restore_pandas_metadata,
    _restore_series_dtype,
    _unwire_scalar,
    _wire_scalar,
)

# ---------------------------------------------------------------------------
# _wire_scalar / _unwire_scalar
# ---------------------------------------------------------------------------


def test_wire_scalar_rejects_unsupported_type() -> None:
    with pytest.raises(TypeError, match="does not support scalar"):
        _wire_scalar(object())


def test_unwire_scalar_pandas_nat() -> None:
    result = _unwire_scalar({_WIRE_TYPE: "pandas-nat"})
    assert result is pd.NaT


def test_unwire_scalar_invalid_float_bits() -> None:
    with pytest.raises(ValueError, match="float handoff"):
        _unwire_scalar({_WIRE_TYPE: "float", "bits": "not-hex"})


def test_unwire_scalar_float_hex_fallback() -> None:
    result = _unwire_scalar({_WIRE_TYPE: "float", "hex": "0x1.0p+0"})
    assert result == 1.0


def test_unwire_scalar_invalid_complex() -> None:
    with pytest.raises(ValueError, match="complex handoff"):
        _unwire_scalar({_WIRE_TYPE: "complex", "real": {}, "imaginary": {}})


def test_unwire_scalar_invalid_bytes() -> None:
    with pytest.raises(ValueError, match="bytes handoff"):
        _unwire_scalar({_WIRE_TYPE: "bytes", "base64": "!!!not-base64!!!"})


def test_unwire_scalar_decimal_invalid_sign() -> None:
    with pytest.raises(ValueError, match="decimal handoff"):
        _unwire_scalar({_WIRE_TYPE: "decimal", "sign": 5, "digits": [1], "exponent": "0"})


def test_unwire_scalar_decimal_invalid_digit() -> None:
    with pytest.raises(ValueError, match="decimal handoff"):
        _unwire_scalar({_WIRE_TYPE: "decimal", "sign": 0, "digits": [10], "exponent": "0"})


def test_unwire_scalar_decimal_non_int_exponent() -> None:
    result = _unwire_scalar({_WIRE_TYPE: "decimal", "sign": 0, "digits": [1, 2], "exponent": "F"})
    assert isinstance(result, Decimal)


def test_unwire_scalar_unknown_wire_type() -> None:
    with pytest.raises(ValueError, match="unknown.*scalar wire type"):
        _unwire_scalar({_WIRE_TYPE: "no-such-type"})


# ---------------------------------------------------------------------------
# timezone codecs
# ---------------------------------------------------------------------------


def test_fixed_offset_timezone_from_payload_legacy_seconds() -> None:
    tz = _fixed_offset_timezone_from_payload({"seconds": "3600", "name": None})
    assert tz.utcoffset(None) == dt.timedelta(hours=1)


def test_datetime_timezone_unsupported_offset_none() -> None:
    class _BrokenTz(dt.tzinfo):
        def utcoffset(self, value):
            return None

        def dst(self, value):
            return dt.timedelta(0)

    broken = dt.datetime(2024, 1, 1, tzinfo=_BrokenTz())
    with pytest.raises(TypeError, match="timezone"):
        _wire_scalar(broken)


def test_pandas_timezone_payload_fallback_string() -> None:
    from fincore.factor_analysis.models import _pandas_timezone_payload

    class _OddTz:
        def __str__(self):
            return "OddZone"

    assert _pandas_timezone_payload(_OddTz()) == {"kind": "pandas-timezone", "name": "OddZone"}


def test_pandas_timezone_from_payload_rejects_non_mapping() -> None:
    with pytest.raises(ValueError, match="pandas timezone handoff"):
        _pandas_timezone_from_payload(42)


def test_pandas_timezone_from_payload_unknown_kind() -> None:
    with pytest.raises(ValueError, match="unknown.*pandas timezone"):
        _pandas_timezone_from_payload({"kind": "bogus"})


def test_restore_datetime_timezone_rejects_non_mapping() -> None:
    with pytest.raises(ValueError, match="datetime timezone handoff"):
        _restore_datetime_timezone(dt.datetime(2024, 1, 1), 42, fold=0)


def test_restore_datetime_timezone_unknown_kind() -> None:
    with pytest.raises(ValueError, match="unknown.*datetime timezone"):
        _restore_datetime_timezone(dt.datetime(2024, 1, 1), {"kind": "bogus"}, fold=0)


def test_restore_datetime_timezone_offset_mismatch() -> None:
    # A naive value combined with an IANA zone produces a restored offset that
    # differs from the naive source offset (None), so it must be rejected.
    naive = dt.datetime(2024, 1, 1, 12, 0)
    with pytest.raises(ValueError, match="offset does not match"):
        _restore_datetime_timezone(
            naive,
            {"kind": "iana-zone", "name": "America/New_York"},
            fold=0,
        )


# ---------------------------------------------------------------------------
# frequency codec
# ---------------------------------------------------------------------------


def test_frequency_from_payload_rejects_non_mapping() -> None:
    with pytest.raises(ValueError, match="frequency handoff"):
        _frequency_from_payload(42)


def test_frequency_from_payload_unknown_type() -> None:
    with pytest.raises(ValueError, match="unknown.*frequency wire type"):
        _frequency_from_payload({_WIRE_TYPE: "bogus"})


# ---------------------------------------------------------------------------
# index codec
# ---------------------------------------------------------------------------


def test_index_from_payload_invalid_datetime_value() -> None:
    payload = {
        _WIRE_TYPE: "datetime-index",
        "unit": "ns",
        "values": [42],  # not a Mapping
        "name": None,
        "timezone": None,
    }
    with pytest.raises(ValueError, match="datetime index value"):
        _index_from_payload(payload)


def test_index_from_payload_datetime_nat_value() -> None:
    payload = {
        _WIRE_TYPE: "datetime-index",
        "unit": "ns",
        "values": [{_WIRE_TYPE: "pandas-nat"}],
        "name": None,
        "timezone": None,
    }
    result = _index_from_payload(payload)
    assert len(result) == 1
    assert pd.isna(result[0])


def test_index_from_payload_datetime_unit_mismatch() -> None:
    payload = {
        _WIRE_TYPE: "datetime-index",
        "unit": "ns",
        "values": [{_WIRE_TYPE: "timestamp", "value": "0", "unit": "us"}],
        "name": None,
        "timezone": None,
    }
    with pytest.raises(ValueError, match="timestamp unit"):
        _index_from_payload(payload)


def test_index_from_payload_unknown_type() -> None:
    with pytest.raises(ValueError, match="unknown.*index wire type"):
        _index_from_payload({_WIRE_TYPE: "bogus"})


# ---------------------------------------------------------------------------
# numpy dtype codec
# ---------------------------------------------------------------------------


def test_numpy_dtype_from_payload_rejects_wrong_type() -> None:
    with pytest.raises(ValueError, match="NumPy dtype handoff"):
        _numpy_dtype_from_payload({_WIRE_TYPE: "not-numpy-dtype"})


def test_numpy_dtype_from_payload_rejects_non_mapping_metadata() -> None:
    with pytest.raises(ValueError, match="metadata"):
        _numpy_dtype_from_payload({_WIRE_TYPE: "numpy-dtype", "kind": "scalar", "dtype": "f8", "metadata": 42})


def test_numpy_dtype_from_payload_unknown_kind() -> None:
    with pytest.raises(ValueError, match="unknown.*NumPy dtype wire kind"):
        _numpy_dtype_from_payload({_WIRE_TYPE: "numpy-dtype", "kind": "bogus", "metadata": {}})


def test_numpy_dtype_from_payload_structured_bad_field() -> None:
    with pytest.raises(ValueError, match="structured NumPy dtype handoff field"):
        _numpy_dtype_from_payload(
            {
                _WIRE_TYPE: "numpy-dtype",
                "kind": "structured",
                "fields": [42],
                "itemsize": "1",
                "aligned": False,
                "metadata": {},
            }
        )


# ---------------------------------------------------------------------------
# numpy scalar / array codec
# ---------------------------------------------------------------------------


def test_numpy_scalar_from_payload_rejects_wrong_type() -> None:
    with pytest.raises(ValueError, match="NumPy scalar handoff"):
        _numpy_scalar_from_payload({_WIRE_TYPE: "not-numpy-scalar"})


def test_numpy_scalar_from_payload_invalid_base64() -> None:
    dtype = {"__fincore_factor_analysis_type__": "numpy-dtype", "kind": "scalar", "dtype": "f8", "metadata": {}}
    with pytest.raises(ValueError, match="NumPy scalar byte payload"):
        _numpy_scalar_from_payload({_WIRE_TYPE: "numpy-scalar", "dtype": dtype, "storage": "bytes", "base64": "!!!"})


def test_numpy_scalar_from_payload_wrong_byte_length() -> None:
    import base64

    dtype = {"__fincore_factor_analysis_type__": "numpy-dtype", "kind": "scalar", "dtype": "f8", "metadata": {}}
    with pytest.raises(ValueError, match="byte length"):
        _numpy_scalar_from_payload(
            {
                _WIRE_TYPE: "numpy-scalar",
                "dtype": dtype,
                "storage": "bytes",
                "base64": base64.b64encode(b"\x00").decode(),
            }
        )


def test_numpy_scalar_from_payload_unknown_storage() -> None:
    dtype = {"__fincore_factor_analysis_type__": "numpy-dtype", "kind": "scalar", "dtype": "f8", "metadata": {}}
    with pytest.raises(ValueError, match="unknown.*NumPy scalar storage"):
        _numpy_scalar_from_payload({_WIRE_TYPE: "numpy-scalar", "dtype": dtype, "storage": "bogus"})


def test_numpy_array_from_payload_rejects_wrong_type() -> None:
    with pytest.raises(ValueError, match="NumPy array handoff"):
        _numpy_array_from_payload({_WIRE_TYPE: "not-numpy-array"})


def test_numpy_array_from_payload_invalid_shape() -> None:
    dtype = {"__fincore_factor_analysis_type__": "numpy-dtype", "kind": "scalar", "dtype": "f8", "metadata": {}}
    with pytest.raises(ValueError, match="array shape"):
        _numpy_array_from_payload({_WIRE_TYPE: "numpy-array", "dtype": dtype, "shape": ["x"]})


def test_numpy_array_from_payload_negative_shape() -> None:
    dtype = {"__fincore_factor_analysis_type__": "numpy-dtype", "kind": "scalar", "dtype": "f8", "metadata": {}}
    with pytest.raises(ValueError, match="array shape"):
        _numpy_array_from_payload({_WIRE_TYPE: "numpy-array", "dtype": dtype, "shape": ["-1"]})


def test_numpy_array_from_payload_unknown_storage() -> None:
    dtype = {"__fincore_factor_analysis_type__": "numpy-dtype", "kind": "scalar", "dtype": "f8", "metadata": {}}
    with pytest.raises(ValueError, match="unknown.*NumPy array storage"):
        _numpy_array_from_payload(
            {_WIRE_TYPE: "numpy-array", "dtype": dtype, "shape": ["1"], "storage": "bogus"}
        )


def test_numpy_array_from_payload_wrong_item_count() -> None:
    dtype = {"__fincore_factor_analysis_type__": "numpy-dtype", "kind": "scalar", "dtype": "f8", "metadata": {}}
    with pytest.raises(ValueError, match="item count"):
        _numpy_array_from_payload(
            {
                _WIRE_TYPE: "numpy-array",
                "dtype": dtype,
                "shape": ["2"],
                "storage": "items",
                "items": [1.0],
            }
        )


# ---------------------------------------------------------------------------
# series dtype / pandas metadata / pandas codec
# ---------------------------------------------------------------------------


def test_restore_series_dtype_unknown_type() -> None:
    with pytest.raises(ValueError, match="unknown.*dtype wire type"):
        _restore_series_dtype(pd.Series([1.0]), {_WIRE_TYPE: "bogus"})


def test_restore_pandas_metadata_rejects_non_mapping_attrs() -> None:
    with pytest.raises(ValueError, match="attrs handoff"):
        _restore_pandas_metadata(pd.Series([1.0]), {"attrs": 42})


def test_restore_pandas_metadata_rejects_non_bool_duplicate() -> None:
    with pytest.raises(ValueError, match="duplicate-label"):
        _restore_pandas_metadata(pd.Series([1.0]), {"allows_duplicate_labels": "yes"})


def test_pandas_from_payload_rejects_unsupported_schema() -> None:
    with pytest.raises(ValueError, match="unsupported.*pandas handoff schema"):
        _pandas_from_payload({"schema": "not-the-right-schema"})


# ---------------------------------------------------------------------------
# serializable_value / deserialize_serializable_value — container types
# ---------------------------------------------------------------------------


def test_serializable_list_roundtrip() -> None:
    from fincore.factor_analysis.models import deserialize_serializable_value, serializable_value

    value = [1, "a", 2.5]
    assert deserialize_serializable_value(serializable_value(value)) == value


def test_serializable_tuple_roundtrip() -> None:
    from fincore.factor_analysis.models import deserialize_serializable_value, serializable_value

    value = (1, "a", 2.5)
    assert deserialize_serializable_value(serializable_value(value)) == value


def test_serializable_set_roundtrip() -> None:
    from fincore.factor_analysis.models import deserialize_serializable_value, serializable_value

    value = {1, 2, 3}
    assert deserialize_serializable_value(serializable_value(value)) == value


def test_serializable_frozenset_roundtrip() -> None:
    from fincore.factor_analysis.models import deserialize_serializable_value, serializable_value

    value = frozenset({1, 2, 3})
    assert deserialize_serializable_value(serializable_value(value)) == value


def test_serializable_mapping_roundtrip() -> None:
    from fincore.factor_analysis.models import deserialize_serializable_value, serializable_value

    value = {"a": 1, "b": [1, 2]}
    assert deserialize_serializable_value(serializable_value(value)) == value


def test_deserialize_plain_list() -> None:
    from fincore.factor_analysis.models import deserialize_serializable_value

    assert deserialize_serializable_value([1, 2]) == [1, 2]


def test_deserialize_mapping_rejects_bad_schema() -> None:
    from fincore.factor_analysis.models import _WIRE_TYPE, deserialize_serializable_value

    with pytest.raises(ValueError, match="mapping handoff"):
        deserialize_serializable_value({_WIRE_TYPE: "mapping", "schema": "wrong", "entries": []})


def test_pandas_from_payload_rejects_unknown_wire_type() -> None:
    from fincore.factor_analysis.models import _WIRE_SCHEMA, _WIRE_TYPE, _pandas_from_payload

    with pytest.raises(ValueError, match="unknown.*pandas wire type"):
        _pandas_from_payload({"schema": _WIRE_SCHEMA, _WIRE_TYPE: "bogus"})


def test_pandas_from_payload_rejects_bad_dataframe_shape() -> None:
    from fincore.factor_analysis.models import _WIRE_SCHEMA, _WIRE_TYPE, _pandas_from_payload

    idx_payload = {_WIRE_TYPE: "index", "name": None, "dtype": "object", "values": [0]}
    payload = {
        "schema": _WIRE_SCHEMA,
        _WIRE_TYPE: "dataframe",
        "index": idx_payload,
        "columns": idx_payload,
        "dtypes": [{_WIRE_TYPE: "dtype", "name": "object"}],
        "data": [[1.0, 2.0]],  # row longer than one column
    }
    with pytest.raises(ValueError, match="shape"):
        _pandas_from_payload(payload)


# ---------------------------------------------------------------------------
# fingerprint_value / _snapshot_value — container types
# ---------------------------------------------------------------------------


def test_fingerprint_set() -> None:
    from fincore.factor_analysis.models import fingerprint_value

    assert len(fingerprint_value({1, 2, 3})) == 64


def test_fingerprint_unsupported_type() -> None:
    from fincore.factor_analysis.models import fingerprint_value

    with pytest.raises(TypeError, match="does not support"):
        fingerprint_value(object())


def test_snapshot_value_containers() -> None:
    from fincore.factor_analysis.models import _snapshot_value

    assert _snapshot_value([1, 2]) == [1, 2]
    assert _snapshot_value((1, 2)) == (1, 2)
    assert _snapshot_value({1, 2}) == {1, 2}
    assert _snapshot_value(frozenset({1, 2})) == frozenset({1, 2})
    assert _snapshot_value({"a": 1}) == {"a": 1}


def test_fingerprint_slots_with_name_mangling() -> None:
    from fincore.factor_analysis.models import fingerprint_value

    class _Slotted:
        __slots__ = ("__value",)

        def __init__(self):
            self.__value = 1.0

    assert len(fingerprint_value(_Slotted())) == 64


def test_wire_scalar_naT() -> None:
    from fincore.factor_analysis.models import _WIRE_TYPE, _wire_scalar

    assert _wire_scalar(pd.NaT) == {_WIRE_TYPE: "pandas-nat"}
