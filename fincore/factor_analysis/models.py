"""Frozen, renderer-ready models for enhanced factor analysis.

These dataclasses deliberately contain only analytical data.  Matplotlib,
seaborn, IPython, and renderer callables belong to later adapter layers and
are intentionally absent from this module.
"""

from __future__ import annotations

import base64
import datetime as dt
import pickle  # nosec B403 # round-trips trusted in-process data only; never deserializes external input
import struct
from collections.abc import Hashable, Mapping, Sequence
from dataclasses import dataclass, field, fields, is_dataclass
from decimal import Decimal
from hashlib import sha256
from types import MappingProxyType
from typing import Any, Literal, TypeVar, cast

import numpy as np
import pandas as pd
from pandas.tseries.frequencies import to_offset
from pandas.tseries.offsets import BDay, CustomBusinessDay

from fincore.factor_analysis.portfolio import PyfolioFactorInputs

_MappingKey = TypeVar("_MappingKey", bound=Hashable)
_MappingValue = TypeVar("_MappingValue")

_WIRE_TYPE = "__fincore_factor_analysis_type__"
_WIRE_SCHEMA = "fincore-factor-analysis-json-v1"


class _FixedOffsetTimezone(dt.tzinfo):
    """A fixed offset whose absent display name differs from ``datetime.timezone``."""

    __slots__ = ("_name", "_offset")

    def __init__(self, offset: dt.timedelta, name: str | None) -> None:
        self._offset = offset
        self._name = name

    def utcoffset(self, value: dt.datetime | None) -> dt.timedelta:
        return self._offset

    def tzname(self, value: dt.datetime | None) -> str | None:
        return self._name

    def dst(self, value: dt.datetime | None) -> dt.timedelta:
        return dt.timedelta(0)


def _fixed_offset_timezone_payload(offset: dt.timedelta, name: str | None) -> dict[str, object]:
    """Encode every fixed-offset bit rather than rounding to seconds."""

    offset_microseconds = ((offset.days * 86_400 + offset.seconds) * 1_000_000) + offset.microseconds
    return {
        "kind": "fixed-offset",
        "microseconds": str(offset_microseconds),
        "name": name,
    }


def _fixed_offset_timezone_from_payload(payload: Mapping[str, object]) -> dt.tzinfo:
    """Restore a fixed-offset timezone, including sub-minute offsets."""

    name = payload.get("name")
    if "microseconds" in payload:
        offset = dt.timedelta(microseconds=int(cast("str", payload["microseconds"])))
    else:  # Backward-compatible decoder for the initial v1 whole-second representation.
        offset = dt.timedelta(seconds=int(cast("str", payload["seconds"])))
    return _FixedOffsetTimezone(offset, None) if name is None else dt.timezone(offset, cast("str", name))


def _datetime_timezone_payload(value: dt.datetime) -> dict[str, object] | None:
    """Encode a stdlib datetime timezone without collapsing named zones to offsets."""

    if value.tzinfo is None:
        return None
    name = _iana_timezone_name(value.tzinfo)
    if name is not None:
        return {"kind": "iana-zone", "name": name}
    offset = value.utcoffset()
    if offset is None:
        raise TypeError(f"factor-analysis JSON handoff does not support timezone {value.tzinfo!r}")
    return _fixed_offset_timezone_payload(offset, value.tzname())


def _iana_timezone_name(timezone: object) -> str | None:
    """Extract the stable IANA identity shared by common timezone providers."""

    for attribute in ("key", "zone"):
        name = getattr(timezone, attribute, None)
        if isinstance(name, str) and name:
            return name
    filename = getattr(timezone, "_filename", None)
    if isinstance(filename, str):
        _, marker, name = filename.rpartition("/zoneinfo/")
        if marker and name:
            return name
    return None


def _pandas_timezone_payload(timezone: object) -> dict[str, object] | None:
    """Encode pandas timezone identity without routing fixed offsets through strings."""

    if timezone is None:
        return None
    if isinstance(timezone, dt.timezone):
        offset = timezone.utcoffset(None)
        if offset is None:  # pragma: no cover - datetime.timezone invariant
            raise TypeError(f"factor-analysis JSON handoff does not support timezone {timezone!r}")
        return _fixed_offset_timezone_payload(offset, timezone.tzname(None))
    fixed_offset = getattr(timezone, "utcoffset", lambda _value: None)(None)
    if isinstance(fixed_offset, dt.timedelta):
        fixed_name = getattr(timezone, "tzname", lambda _value: None)(None)
        return _fixed_offset_timezone_payload(fixed_offset, fixed_name if isinstance(fixed_name, str) else None)
    name = _iana_timezone_name(timezone)
    if name is not None:
        return {"kind": "iana-zone", "name": name}
    return {"kind": "pandas-timezone", "name": str(timezone)}


def _resolve_named_timezone(name: str) -> object:
    """Resolve an IANA zone name to a ``ZoneInfo`` object when possible.

    CPython's automatic ``tzdata`` fallback is sysconfig-based and misses
    common Windows layouts, while pandas' own Windows fallback builds
    malformed ``tzdata.zoneinfo.tzfile(...)`` module paths.  Resolving the
    name here keeps both restore paths deterministic; unresolvable names are
    returned unchanged so pandas raises its own resolution error.
    """
    from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

    try:
        return ZoneInfo(name)
    except ZoneInfoNotFoundError:
        pass
    try:
        import zoneinfo as zoneinfo_module
        from importlib import resources

        import tzdata

        zoneinfo_module.reset_tzpath([str(resources.files(tzdata) / "zoneinfo")])
        return ZoneInfo(name)
    except (ImportError, ZoneInfoNotFoundError):
        return name


def _pandas_timezone_from_payload(payload: object) -> object | None:
    """Restore a timezone object/string accepted by ``Timestamp.tz_convert``."""

    if payload is None:
        return None
    if isinstance(payload, str):  # Backward-compatible v1 payloads.
        return payload
    if not isinstance(payload, Mapping):
        raise ValueError("invalid factor-analysis pandas timezone handoff payload")
    kind = payload.get("kind")
    if kind == "fixed-offset":
        return _fixed_offset_timezone_from_payload(payload)
    if kind in {"zoneinfo", "iana-zone"}:
        from zoneinfo import ZoneInfo

        field = "key" if kind == "zoneinfo" else "name"
        return ZoneInfo(cast("str", payload[field]))
    if kind == "pandas-timezone":
        return _resolve_named_timezone(cast("str", payload["name"]))
    raise ValueError(f"unknown factor-analysis pandas timezone wire type {kind!r}")


def _timestamp_from_wire(timestamp_value: int, unit: object, timezone_payload: object) -> pd.Timestamp:
    """Restore a timestamp without pandas normalizing a sub-minute fixed offset."""

    timezone = _pandas_timezone_from_payload(timezone_payload)
    if timezone is None:
        return pd.Timestamp(timestamp_value, unit=cast("Any", unit))
    if isinstance(timezone_payload, Mapping) and timezone_payload.get("kind") == "fixed-offset":
        fixed_timezone = cast("dt.tzinfo", timezone)
        offset = fixed_timezone.utcoffset(None)
        if offset is None:  # pragma: no cover - datetime.timezone invariant
            raise ValueError("invalid factor-analysis fixed timestamp timezone")
        local_naive = pd.Timestamp(timestamp_value, unit=cast("Any", unit)) + pd.Timedelta(offset)
        # ``Timestamp(..., tz=...)`` retains pandas' sub-microsecond payload;
        # routing through ``datetime`` would truncate its nanoseconds.
        restored = pd.Timestamp(local_naive, tz=fixed_timezone)
        # Adding the fixed offset can promote seconds/milliseconds to
        # microseconds.  The wire unit is part of the provenance contract, so
        # restore it when the represented instant permits it.
        return restored.as_unit(cast("Literal['s', 'ms', 'us', 'ns']", unit))
    return pd.Timestamp(timestamp_value, unit=cast("Any", unit), tz="UTC").tz_convert(cast("Any", timezone))


def _restore_datetime_timezone(
    value: dt.datetime,
    timezone_payload: object,
    *,
    fold: int,
) -> dt.datetime:
    """Restore the timezone identity encoded by :func:`_datetime_timezone_payload`."""

    if timezone_payload is None:
        return value.replace(fold=fold)
    if not isinstance(timezone_payload, Mapping):
        raise ValueError("invalid factor-analysis datetime timezone handoff payload")
    source_offset = value.utcoffset()
    kind = timezone_payload.get("kind")
    if kind in {"zoneinfo", "iana-zone"}:
        from zoneinfo import ZoneInfo

        field = "key" if kind == "zoneinfo" else "name"
        restored = value.replace(tzinfo=ZoneInfo(cast("str", timezone_payload[field])), fold=fold)
    elif kind == "fixed-offset":
        fixed_timezone = _fixed_offset_timezone_from_payload(timezone_payload)
        restored = value.replace(tzinfo=fixed_timezone, fold=fold)
        # ``datetime.fromisoformat`` normalizes a sub-second offset smaller
        # than one second to UTC.  The typed offset is the authoritative
        # source representation for fixed-offset zones.
        source_offset = fixed_timezone.utcoffset(None)
    else:
        raise ValueError(f"unknown factor-analysis datetime timezone wire type {kind!r}")
    if restored.utcoffset() != source_offset:
        raise ValueError("factor-analysis datetime timezone handoff offset does not match the encoded value")
    return restored


def _frequency_payload(frequency: object) -> object:
    """Encode a DateOffset without reducing a custom trading calendar to ``C``."""

    if frequency is None:
        return None
    if isinstance(frequency, CustomBusinessDay):
        custom_frequency = cast("Any", frequency)
        return {
            _WIRE_TYPE: "custom-business-day-offset",
            "n": str(custom_frequency.n),
            "normalize": custom_frequency.normalize,
            "weekmask": custom_frequency.weekmask,
            "holidays": [_wire_scalar(pd.Timestamp(holiday)) for holiday in custom_frequency.holidays],
            "offset": _wire_scalar(pd.Timedelta(custom_frequency.offset)),
        }
    if isinstance(frequency, BDay):
        business_frequency = cast("Any", frequency)
        return {
            _WIRE_TYPE: "business-day-offset",
            "n": str(business_frequency.n),
            "normalize": business_frequency.normalize,
            "offset": _wire_scalar(pd.Timedelta(business_frequency.offset)),
        }
    return {_WIRE_TYPE: "offset", "frequency": cast("Any", frequency).freqstr}


def _frequency_from_payload(payload: object) -> object:
    """Restore one DateOffset emitted by :func:`_frequency_payload`."""

    if payload is None:
        return None
    if not isinstance(payload, Mapping):
        raise ValueError("invalid factor-analysis frequency handoff payload")
    wire_type = payload.get(_WIRE_TYPE)
    if wire_type == "custom-business-day-offset":
        holidays = [_unwire_scalar(item) for item in cast("list[object]", payload["holidays"])]
        offset = _unwire_scalar(payload["offset"])
        custom_business_day = cast("Any", CustomBusinessDay)
        return custom_business_day(
            n=int(cast("str", payload["n"])),
            normalize=bool(payload["normalize"]),
            weekmask=cast("str", payload["weekmask"]),
            holidays=cast("list[object]", holidays),
            offset=cast("Any", offset),
        )
    if wire_type == "business-day-offset":
        offset = _unwire_scalar(payload["offset"])
        business_day = cast("Any", BDay)
        return business_day(
            n=int(cast("str", payload["n"])),
            normalize=bool(payload["normalize"]),
            offset=cast("Any", offset),
        )
    if wire_type == "offset":
        return to_offset(cast("str", payload["frequency"]))
    raise ValueError(f"unknown factor-analysis frequency wire type {wire_type!r}")


def _wire_scalar(value: object) -> object:
    """Encode one scalar without losing numeric bits or pandas sentinels."""

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
        return {_WIRE_TYPE: "float", "bits": struct.pack(">d", value).hex()}
    if isinstance(value, complex):
        return {
            _WIRE_TYPE: "complex",
            "real": _wire_scalar(value.real),
            "imaginary": _wire_scalar(value.imag),
        }
    if isinstance(value, bytes):
        return {_WIRE_TYPE: "bytes", "base64": base64.b64encode(value).decode("ascii")}
    if isinstance(value, Decimal):
        decimal_tuple = value.as_tuple()
        return {
            _WIRE_TYPE: "decimal",
            "sign": decimal_tuple.sign,
            "digits": list(decimal_tuple.digits),
            "exponent": str(decimal_tuple.exponent),
        }
    if isinstance(value, pd.Timestamp):
        unit = value.unit
        return {
            _WIRE_TYPE: "timestamp",
            "value": str(value.asm8.astype("int64")),
            "unit": unit,
            "timezone": _pandas_timezone_payload(value.tz),
        }
    if isinstance(value, pd.Timedelta):
        return {
            _WIRE_TYPE: "timedelta",
            "value": str(value.asm8.astype("int64")),
            "unit": value.unit,
        }
    if isinstance(value, pd.Period):
        return {
            _WIRE_TYPE: "period",
            "ordinal": str(value.ordinal),
            "frequency": value.freqstr,
        }
    if isinstance(value, pd.Interval):
        return {
            _WIRE_TYPE: "interval",
            "left": serializable_value(value.left),
            "right": serializable_value(value.right),
            "closed": value.closed,
        }
    if isinstance(value, dt.datetime):
        return {
            _WIRE_TYPE: "python-datetime",
            "value": value.isoformat(),
            "fold": value.fold,
            "timezone": _datetime_timezone_payload(value),
        }
    if isinstance(value, dt.date):
        return {_WIRE_TYPE: "python-date", "value": value.isoformat()}
    if isinstance(value, dt.timedelta):
        return {
            _WIRE_TYPE: "python-timedelta",
            "days": str(value.days),
            "seconds": str(value.seconds),
            "microseconds": str(value.microseconds),
        }
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
        if "bits" in value:
            try:
                return struct.unpack(">d", bytes.fromhex(cast("str", value["bits"])))[0]
            except (ValueError, struct.error) as error:
                raise ValueError("invalid factor-analysis float handoff payload") from error
        return float.fromhex(cast("str", value["hex"]))
    if wire_type == "complex":
        real = _unwire_scalar(value["real"])
        imaginary = _unwire_scalar(value["imaginary"])
        if not isinstance(real, (int, float)) or not isinstance(imaginary, (int, float)):
            raise ValueError("invalid factor-analysis complex handoff payload")
        return complex(real, imaginary)
    if wire_type == "bytes":
        try:
            return base64.b64decode(cast("str", value["base64"]), validate=True)
        except (TypeError, ValueError) as error:
            raise ValueError("invalid factor-analysis bytes handoff payload") from error
    if wire_type == "decimal":
        try:
            sign = value["sign"]
            raw_digits = cast("list[object]", value["digits"])
            if not isinstance(sign, int) or sign not in {0, 1}:
                raise ValueError
            if not all(isinstance(digit, int) and 0 <= digit <= 9 for digit in raw_digits):
                raise ValueError
            digits = tuple(cast("int", digit) for digit in raw_digits)
            exponent_text = cast("str", value["exponent"])
            exponent: int | str
            try:
                exponent = int(exponent_text)
            except ValueError:
                exponent = exponent_text
            return Decimal(cast("Any", (sign, digits, exponent)))
        except (TypeError, ValueError) as error:
            raise ValueError("invalid factor-analysis decimal handoff payload") from error
    if wire_type == "timestamp":
        timestamp_value = int(cast("str", value["value"]))
        return _timestamp_from_wire(timestamp_value, value["unit"], value.get("timezone"))
    if wire_type == "timedelta":
        return pd.Timedelta(int(cast("str", value["value"])), unit=cast("Any", value["unit"]))
    if wire_type == "period":
        return pd.Period(ordinal=int(cast("str", value["ordinal"])), freq=cast("str", value["frequency"]))
    if wire_type == "interval":
        return pd.Interval(
            cast("Any", deserialize_serializable_value(value["left"])),
            cast("Any", deserialize_serializable_value(value["right"])),
            closed=cast("Any", value["closed"]),
        )
    if wire_type == "python-datetime":
        return _restore_datetime_timezone(
            dt.datetime.fromisoformat(cast("str", value["value"])),
            value.get("timezone"),
            fold=int(cast("int", value["fold"])),
        )
    if wire_type == "python-date":
        return dt.date.fromisoformat(cast("str", value["value"]))
    if wire_type == "python-timedelta":
        return dt.timedelta(
            days=int(cast("str", value["days"])),
            seconds=int(cast("str", value["seconds"])),
            microseconds=int(cast("str", value["microseconds"])),
        )
    raise ValueError(f"unknown factor-analysis scalar wire type {wire_type!r}")


def _index_payload(index: pd.Index) -> dict[str, object]:
    """Encode an index with its names, levels, categories, timezone, and freq."""

    if isinstance(index, pd.MultiIndex):
        return {
            _WIRE_TYPE: "multiindex",
            "names": [serializable_value(name) for name in index.names],
            "levels": [_index_payload(level) for level in index.levels],
            "codes": [codes.tolist() for codes in index.codes],
        }
    if isinstance(index, pd.CategoricalIndex):
        return {
            _WIRE_TYPE: "categorical-index",
            "name": serializable_value(index.name),
            "categories": _index_payload(index.categories),
            "codes": index.codes.tolist(),
            "ordered": cast("pd.CategoricalDtype", index.dtype).ordered,
        }
    if isinstance(index, pd.RangeIndex):
        return {
            _WIRE_TYPE: "range-index",
            "name": serializable_value(index.name),
            "start": str(index.start),
            "stop": str(index.stop),
            "step": str(index.step),
        }
    if isinstance(index, pd.DatetimeIndex):
        return {
            _WIRE_TYPE: "datetime-index",
            "name": serializable_value(index.name),
            "values": [serializable_value(value) for value in index],
            "frequency": _frequency_payload(index.freq),
            "unit": index.unit,
            "timezone": _pandas_timezone_payload(index.tz),
        }
    if isinstance(index, pd.TimedeltaIndex):
        return {
            _WIRE_TYPE: "timedelta-index",
            "name": serializable_value(index.name),
            "values": [serializable_value(value) for value in index],
            "frequency": _frequency_payload(index.freq),
            "unit": index.unit,
        }
    return {
        _WIRE_TYPE: "index",
        "name": serializable_value(index.name),
        "dtype": _dtype_payload(index.dtype),
        "values": [serializable_value(value) for value in index],
    }


def _restore_frequency(
    index: pd.DatetimeIndex | pd.TimedeltaIndex, frequency_payload: object
) -> pd.DatetimeIndex | pd.TimedeltaIndex:
    """Restore a valid explicit frequency without rejecting irregular indexes."""

    frequency = _frequency_from_payload(frequency_payload)
    if frequency is None:
        return index
    try:
        return type(index)(index, freq=frequency, name=index.name)  # type: ignore[arg-type]
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
        names = [deserialize_serializable_value(item) for item in cast("list[object]", payload["names"])]
        codes = cast("list[list[int]]", payload["codes"])
        return pd.MultiIndex(levels=cast("Any", levels), codes=cast("Any", codes), names=names)
    if wire_type == "categorical-index":
        categories = _index_from_payload(cast("Mapping[str, object]", payload["categories"]))
        categorical = pd.Categorical.from_codes(
            cast("list[int]", payload["codes"]),
            categories=categories,
            ordered=bool(payload["ordered"]),
        )
        return pd.CategoricalIndex(categorical, name=deserialize_serializable_value(payload["name"]))
    if wire_type == "range-index":
        return pd.RangeIndex(
            start=int(cast("str", payload["start"])),
            stop=int(cast("str", payload["stop"])),
            step=int(cast("str", payload["step"])),
            name=deserialize_serializable_value(payload["name"]),
        )
    if wire_type == "datetime-index":
        unit = cast("str", payload["unit"])
        datetime_values: list[int] = []
        for item in cast("list[object]", payload["values"]):
            if not isinstance(item, Mapping):
                raise ValueError("invalid factor-analysis datetime index value")
            item_wire_type = item.get(_WIRE_TYPE)
            if item_wire_type == "pandas-nat":
                datetime_values.append(np.iinfo(np.int64).min)
                continue
            if item_wire_type != "timestamp" or item.get("unit") != unit:
                raise ValueError("invalid factor-analysis datetime index timestamp unit")
            datetime_values.append(int(cast("str", item["value"])))
        datetime_array = np.asarray(datetime_values, dtype=cast("Any", f"datetime64[{unit}]"))
        timezone = _pandas_timezone_from_payload(payload.get("timezone"))
        datetime_result = pd.DatetimeIndex(
            datetime_array,
            name=deserialize_serializable_value(payload["name"]),
            tz="UTC" if timezone is not None else None,
        )
        if timezone is not None:
            datetime_result = datetime_result.tz_convert(cast("Any", timezone))
        datetime_result = cast("pd.DatetimeIndex", _restore_datetime_unit(datetime_result, payload.get("unit")))
        return cast("pd.Index", _restore_frequency(datetime_result, payload.get("frequency")))
    if wire_type == "timedelta-index":
        values = [deserialize_serializable_value(item) for item in cast("list[object]", payload["values"])]
        timedelta_result = pd.TimedeltaIndex(
            values, name=cast("str | None", deserialize_serializable_value(payload["name"]))
        )
        timedelta_result = cast("pd.TimedeltaIndex", _restore_datetime_unit(timedelta_result, payload.get("unit")))
        return cast("pd.Index", _restore_frequency(timedelta_result, payload.get("frequency")))
    if wire_type == "index":
        generic_result = pd.Index(
            [deserialize_serializable_value(item) for item in cast("list[object]", payload["values"])],
            name=deserialize_serializable_value(payload["name"]),
        )
        dtype_payload = payload["dtype"]
        if isinstance(dtype_payload, Mapping):
            return _restore_index_dtype(generic_result, dtype_payload)
        try:  # pragma: no cover - compatibility for pre-typed handoff payloads
            return cast("pd.Index", generic_result.astype(cast("Any", dtype_payload)))
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
    if isinstance(dtype, pd.StringDtype):
        return {
            _WIRE_TYPE: "string-dtype",
            "storage": dtype.storage,
            "na_value": serializable_value(dtype.na_value),
        }
    if isinstance(dtype, pd.ArrowDtype):
        import pyarrow

        schema = pyarrow.schema([pyarrow.field("value", dtype.pyarrow_dtype)])
        return {
            _WIRE_TYPE: "arrow-dtype",
            "schema": base64.b64encode(schema.serialize().to_pybytes()).decode("ascii"),
        }
    return {_WIRE_TYPE: "dtype", "name": str(dtype)}


def _numpy_dtype_payload(dtype: np.dtype[Any]) -> dict[str, object]:
    """Encode NumPy dtype structure and public metadata without pickle formatting."""

    metadata = serializable_value({} if dtype.metadata is None else dict(dtype.metadata))
    if dtype.fields is not None:
        fields_payload: list[dict[str, object]] = []
        for name in dtype.names or ():
            field = cast("tuple[object, ...]", dtype.fields[name])
            fields_payload.append(
                {
                    "name": name,
                    "dtype": _numpy_dtype_payload(cast("np.dtype[Any]", field[0])),
                    "offset": str(cast("int", field[1])),
                    "title": serializable_value(field[2] if len(field) > 2 else None),
                }
            )
        return {
            _WIRE_TYPE: "numpy-dtype",
            "kind": "structured",
            "fields": fields_payload,
            "itemsize": str(dtype.itemsize),
            "aligned": dtype.isalignedstruct,
            "metadata": metadata,
        }
    if dtype.subdtype is not None:
        base_dtype, shape = dtype.subdtype
        return {
            _WIRE_TYPE: "numpy-dtype",
            "kind": "subarray",
            "base": _numpy_dtype_payload(cast("np.dtype[Any]", base_dtype)),
            "shape": [str(dimension) for dimension in cast("tuple[int, ...]", shape)],
            "metadata": metadata,
        }
    return {
        _WIRE_TYPE: "numpy-dtype",
        "kind": "scalar",
        "dtype": dtype.str,
        "metadata": metadata,
    }


def _numpy_dtype_from_payload(payload: Mapping[str, object]) -> np.dtype[Any]:
    """Restore a typed NumPy dtype envelope emitted by :func:`_numpy_dtype_payload`."""

    if payload.get(_WIRE_TYPE) != "numpy-dtype":
        raise ValueError("invalid factor-analysis NumPy dtype handoff payload")
    metadata = deserialize_serializable_value(payload["metadata"])
    if not isinstance(metadata, Mapping):
        raise ValueError("invalid factor-analysis NumPy dtype metadata handoff payload")
    # NumPy permits arbitrary hashable metadata keys.  Its public stub is
    # narrower than the runtime API, so retain the exact decoded mapping.
    typed_metadata = cast("dict[str, Any]", dict(metadata))
    kind = payload.get("kind")
    if kind == "scalar":
        return cast("np.dtype[Any]", np.dtype(cast("str", payload["dtype"]), metadata=typed_metadata))
    if kind == "subarray":
        base_dtype = _numpy_dtype_from_payload(cast("Mapping[str, object]", payload["base"]))
        shape = tuple(int(cast("str", dimension)) for dimension in cast("list[object]", payload["shape"]))
        return cast("np.dtype[Any]", np.dtype((base_dtype, shape), metadata=typed_metadata))
    if kind == "structured":
        fields_payload = cast("list[object]", payload["fields"])
        names: list[str] = []
        formats: list[np.dtype[Any]] = []
        offsets: list[int] = []
        titles: list[object] = []
        for item in fields_payload:
            if not isinstance(item, Mapping):
                raise ValueError("invalid factor-analysis structured NumPy dtype handoff field")
            names.append(cast("str", item["name"]))
            formats.append(_numpy_dtype_from_payload(cast("Mapping[str, object]", item["dtype"])))
            offsets.append(int(cast("str", item["offset"])))
            titles.append(deserialize_serializable_value(item["title"]))
        descriptor = {
            "names": names,
            "formats": formats,
            "offsets": offsets,
            "titles": titles,
            "itemsize": int(cast("str", payload["itemsize"])),
        }
        return cast(
            "np.dtype[Any]",
            np.dtype(cast("Any", descriptor), align=bool(payload["aligned"]), metadata=typed_metadata),
        )
    raise ValueError(f"unknown factor-analysis NumPy dtype wire kind {kind!r}")


def _numpy_array_payload(value: np.ndarray[Any, Any]) -> dict[str, object]:
    """Encode an ndarray without treating object pointers as portable bytes."""

    payload: dict[str, object] = {
        _WIRE_TYPE: "numpy-array",
        "dtype": _numpy_dtype_payload(value.dtype),
        "shape": [str(dimension) for dimension in value.shape],
    }
    if value.dtype.hasobject:
        payload["storage"] = "items"
        payload["items"] = [serializable_value(item) for item in value.ravel(order="C").tolist()]
    else:
        payload["storage"] = "bytes"
        payload["base64"] = base64.b64encode(np.ascontiguousarray(value).tobytes()).decode("ascii")
    return payload


def _numpy_scalar_payload(value: np.generic) -> dict[str, object]:
    """Encode a NumPy scalar with its dtype rather than collapsing to Python."""

    payload: dict[str, object] = {
        _WIRE_TYPE: "numpy-scalar",
        "dtype": _numpy_dtype_payload(value.dtype),
    }
    if value.dtype.hasobject:
        payload["storage"] = "item"
        payload["item"] = serializable_value(value.item())
    else:
        payload["storage"] = "bytes"
        payload["base64"] = base64.b64encode(value.tobytes()).decode("ascii")
    return payload


def _numpy_scalar_from_payload(payload: Mapping[str, object]) -> np.generic:
    """Restore one typed NumPy scalar emitted by :func:`_numpy_scalar_payload`."""

    if payload.get(_WIRE_TYPE) != "numpy-scalar":
        raise ValueError("invalid factor-analysis NumPy scalar handoff payload")
    dtype = _numpy_dtype_from_payload(cast("Mapping[str, object]", payload["dtype"]))
    storage = payload.get("storage")
    if storage == "bytes":
        try:
            raw = base64.b64decode(cast("str", payload["base64"]), validate=True)
        except (TypeError, ValueError) as error:
            raise ValueError("invalid factor-analysis NumPy scalar byte payload") from error
        if len(raw) != dtype.itemsize:
            raise ValueError("invalid factor-analysis NumPy scalar byte length")
        return cast("np.generic", np.frombuffer(raw, dtype=dtype, count=1).copy()[0])
    if storage == "item":
        result = np.empty((), dtype=dtype)
        result[()] = deserialize_serializable_value(payload["item"])
        return cast("np.generic", result[()])
    raise ValueError("unknown factor-analysis NumPy scalar storage")


def _numpy_array_from_payload(payload: Mapping[str, object]) -> np.ndarray[Any, Any]:
    """Restore an ndarray emitted by :func:`_numpy_array_payload`."""

    if payload.get(_WIRE_TYPE) != "numpy-array":
        raise ValueError("invalid factor-analysis NumPy array handoff payload")
    dtype = _numpy_dtype_from_payload(cast("Mapping[str, object]", payload["dtype"]))
    try:
        shape = tuple(int(cast("str", dimension)) for dimension in cast("list[object]", payload["shape"]))
    except (TypeError, ValueError) as error:
        raise ValueError("invalid factor-analysis NumPy array shape") from error
    if any(dimension < 0 for dimension in shape):
        raise ValueError("invalid factor-analysis NumPy array shape")
    size = 1
    for dimension in shape:
        size *= dimension
    storage = payload.get("storage")
    if storage == "bytes":
        try:
            raw = base64.b64decode(cast("str", payload["base64"]), validate=True)
        except (TypeError, ValueError) as error:
            raise ValueError("invalid factor-analysis NumPy array byte payload") from error
        if len(raw) != size * dtype.itemsize:
            raise ValueError("invalid factor-analysis NumPy array byte length")
        return np.frombuffer(raw, dtype=dtype, count=size).copy().reshape(shape)
    if storage == "items":
        items = [deserialize_serializable_value(item) for item in cast("list[object]", payload["items"])]
        if len(items) != size:
            raise ValueError("invalid factor-analysis NumPy array item count")
        result = np.empty(shape, dtype=dtype)
        flat_result = result.ravel(order="C")
        for position, item in enumerate(items):
            flat_result[position] = item
        return result
    raise ValueError("unknown factor-analysis NumPy array storage")


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
    if wire_type == "string-dtype":
        string_dtype = pd.StringDtype(
            storage=cast("Literal['python', 'pyarrow']", payload["storage"]),
            na_value=cast("Any", deserialize_serializable_value(payload["na_value"])),
        )
        return cast("pd.Series", series.astype(string_dtype))
    if wire_type == "arrow-dtype":
        try:
            import pyarrow
        except ImportError as error:  # pragma: no cover - exercised in extra-specific consumers
            raise ValueError("pyarrow is required to restore an ArrowDtype factor-analysis handoff") from error
        try:
            serialized_schema = base64.b64decode(cast("str", payload["schema"]), validate=True)
            schema = pyarrow.ipc.read_schema(pyarrow.py_buffer(serialized_schema))
        except (KeyError, ValueError) as error:
            raise ValueError("invalid factor-analysis ArrowDtype handoff payload") from error
        if len(schema) != 1 or schema.field("value").name != "value":
            raise ValueError("invalid factor-analysis ArrowDtype handoff schema")
        arrow_dtype = pd.ArrowDtype(schema.field("value").type)
        return cast("pd.Series", series.astype(arrow_dtype))
    if wire_type == "dtype":
        return cast("pd.Series", series.astype(cast("Any", payload["name"])))
    raise ValueError(f"unknown factor-analysis dtype wire type {wire_type!r}")


def _restore_index_dtype(index: pd.Index, payload: Mapping[str, object]) -> pd.Index:
    """Restore generic Index extension dtypes through the same typed codec as columns."""

    restored = _restore_series_dtype(pd.Series(index.tolist(), name=index.name), payload)
    return cast("pd.Index", pd.Index(restored.array, name=index.name))


def _pandas_payload(value: pd.Series | pd.DataFrame) -> dict[str, object]:
    """Return a standards-compliant, lossless table envelope for JSON handoff."""

    if isinstance(value, pd.Series):
        return {
            _WIRE_TYPE: "series",
            "schema": _WIRE_SCHEMA,
            "name": serializable_value(value.name),
            "index": _index_payload(value.index),
            "dtype": _dtype_payload(value.dtype),
            "attrs": serializable_value(dict(value.attrs)),
            "allows_duplicate_labels": value.flags.allows_duplicate_labels,
            "data": [serializable_value(item) for item in value.astype(object).tolist()],
        }
    rows = [list(row) for row in value.astype(object).itertuples(index=False, name=None)]
    return {
        _WIRE_TYPE: "dataframe",
        "schema": _WIRE_SCHEMA,
        "index": _index_payload(value.index),
        "columns": _index_payload(value.columns),
        "dtypes": [_dtype_payload(dtype) for dtype in value.dtypes],
        "attrs": serializable_value(dict(value.attrs)),
        "allows_duplicate_labels": value.flags.allows_duplicate_labels,
        "data": [[serializable_value(item) for item in row] for row in rows],
    }


def _restore_pandas_metadata(
    value: pd.Series | pd.DataFrame, payload: Mapping[str, object]
) -> pd.Series | pd.DataFrame:
    """Restore table attributes and duplicate-label policy after values and dtypes."""

    if "attrs" in payload:
        attrs = deserialize_serializable_value(payload["attrs"])
        if not isinstance(attrs, Mapping):
            raise ValueError("invalid factor-analysis pandas attrs handoff payload")
        value.attrs = dict(attrs)
    if "allows_duplicate_labels" in payload:
        allows_duplicate_labels = payload["allows_duplicate_labels"]
        if not isinstance(allows_duplicate_labels, bool):
            raise ValueError("invalid factor-analysis duplicate-label handoff payload")
        cast("Any", value.flags).allows_duplicate_labels = allows_duplicate_labels
    return value


def _pandas_from_payload(payload: Mapping[str, object]) -> pd.Series | pd.DataFrame:
    """Restore a Series or DataFrame encoded by :func:`_pandas_payload`."""

    if payload.get("schema") != _WIRE_SCHEMA:
        raise ValueError("unsupported factor-analysis pandas handoff schema")
    wire_type = payload.get(_WIRE_TYPE)
    if wire_type == "series":
        dtype_payload = cast("Mapping[str, object]", payload["dtype"])
        values = [deserialize_serializable_value(item) for item in cast("list[object]", payload["data"])]
        series_result = pd.Series(
            values,
            index=_index_from_payload(cast("Mapping[str, object]", payload["index"])),
            name=deserialize_serializable_value(payload["name"]),
            dtype=object
            if dtype_payload.get(_WIRE_TYPE) == "dtype" and dtype_payload.get("name") == "object"
            else None,
        )
        return _restore_pandas_metadata(
            _restore_series_dtype(series_result, dtype_payload),
            payload,
        )
    if wire_type == "dataframe":
        index = _index_from_payload(cast("Mapping[str, object]", payload["index"]))
        columns = _index_from_payload(cast("Mapping[str, object]", payload["columns"]))
        rows = cast("list[list[object]]", payload["data"])
        dtype_payloads = cast("list[object]", payload["dtypes"])
        if any(len(row) != len(columns) for row in rows) or len(dtype_payloads) != len(columns):
            raise ValueError("invalid factor-analysis DataFrame handoff shape")
        frame_result = pd.DataFrame(index=index, columns=columns)
        for position, dtype in enumerate(dtype_payloads):
            dtype_payload = cast("Mapping[str, object]", dtype)
            values = [deserialize_serializable_value(row[position]) for row in rows]
            source = pd.Series(
                values,
                index=index,
                name=columns[position],
                dtype=object
                if dtype_payload.get(_WIRE_TYPE) == "dtype" and dtype_payload.get("name") == "object"
                else None,
            )
            restored = _restore_series_dtype(source, dtype_payload)
            frame_result.isetitem(position, restored.array)
        return _restore_pandas_metadata(frame_result, payload)
    raise ValueError(f"unknown factor-analysis pandas wire type {wire_type!r}")


def serializable_value(value: object) -> object:
    """Convert analytical data to a lossless, standards-compliant JSON value."""

    if isinstance(value, (pd.Series, pd.DataFrame)):
        return _pandas_payload(value)
    if isinstance(value, np.generic):
        return _numpy_scalar_payload(value)
    if isinstance(value, np.ndarray):
        return _numpy_array_payload(value)
    if isinstance(value, np.dtype):
        return _numpy_dtype_payload(value)
    if is_dataclass(value) and not isinstance(value, type):
        return {item.name: serializable_value(getattr(value, item.name)) for item in fields(value)}
    if isinstance(value, Mapping):
        entries = [[serializable_value(key), serializable_value(item)] for key, item in value.items()]
        entries.sort(key=lambda entry: _fingerprint_bytes(entry[0]))
        return {
            _WIRE_TYPE: "mapping",
            "schema": _WIRE_SCHEMA,
            "entries": entries,
        }
    if isinstance(value, tuple):
        return {_WIRE_TYPE: "tuple", "items": [serializable_value(item) for item in value]}
    if isinstance(value, list):
        return {_WIRE_TYPE: "list", "items": [serializable_value(item) for item in value]}
    if isinstance(value, frozenset):
        items = [serializable_value(item) for item in value]
        items.sort(key=_fingerprint_bytes)
        return {_WIRE_TYPE: "frozenset", "items": items}
    if isinstance(value, set):
        items = [serializable_value(item) for item in value]
        items.sort(key=_fingerprint_bytes)
        return {_WIRE_TYPE: "set", "items": items}
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
    if wire_type == "list":
        return [deserialize_serializable_value(item) for item in cast("list[object]", value["items"])]
    if wire_type == "frozenset":
        return frozenset(deserialize_serializable_value(item) for item in cast("list[object]", value["items"]))
    if wire_type == "set":
        return {deserialize_serializable_value(item) for item in cast("list[object]", value["items"])}
    if wire_type == "numpy-dtype":
        return _numpy_dtype_from_payload(cast("Mapping[str, object]", value))
    if wire_type == "numpy-scalar":
        return _numpy_scalar_from_payload(cast("Mapping[str, object]", value))
    if wire_type == "numpy-array":
        return _numpy_array_from_payload(cast("Mapping[str, object]", value))
    if wire_type is not None:
        return _unwire_scalar(cast("dict[str, object]", value))
    return {key: deserialize_serializable_value(item) for key, item in value.items()}


def _fingerprint_envelope(tag: str, *parts: bytes) -> bytes:
    """Join typed fragments with lengths so distinct values cannot collide."""

    tag_bytes = tag.encode("utf-8")
    encoded = [len(tag_bytes).to_bytes(4, "big"), tag_bytes]
    for part in parts:
        encoded.extend((len(part).to_bytes(8, "big"), part))
    return b"".join(encoded)


def _fingerprint_dtype(dtype: object) -> bytes:
    """Encode a pandas dtype, including categorical categories and ordering."""

    if isinstance(dtype, pd.CategoricalDtype):
        return _fingerprint_envelope(
            "categorical-dtype",
            _fingerprint_index(dtype.categories),
            _fingerprint_bytes(dtype.ordered),
        )
    return _fingerprint_envelope(
        "dtype",
        type(dtype).__module__.encode("utf-8"),
        type(dtype).__qualname__.encode("utf-8"),
        str(dtype).encode("utf-8"),
        repr(dtype).encode("utf-8"),
    )


def _fingerprint_numpy_dtype(dtype: np.dtype[Any]) -> bytes:
    """Encode NumPy dtype structure, not only its raw storage representation."""

    metadata = {} if dtype.metadata is None else dict(dtype.metadata)
    fields_by_name = dtype.fields
    if fields_by_name is not None:
        fields_payload: list[bytes] = []
        for name in dtype.names or ():
            field = cast("tuple[object, ...]", fields_by_name[name])
            field_dtype = cast("np.dtype[Any]", field[0])
            title = field[2] if len(field) > 2 else None
            fields_payload.append(
                _fingerprint_envelope(
                    "numpy-dtype-field",
                    name.encode("utf-8"),
                    _fingerprint_numpy_dtype(field_dtype),
                    str(cast("int", field[1])).encode("ascii"),
                    _fingerprint_bytes(title),
                )
            )
        return _fingerprint_envelope(
            "numpy-structured-dtype",
            str(dtype.itemsize).encode("ascii"),
            b"1" if dtype.isalignedstruct else b"0",
            _fingerprint_bytes(metadata),
            *fields_payload,
        )
    if dtype.subdtype is not None:
        base_dtype, shape = dtype.subdtype
        return _fingerprint_envelope(
            "numpy-subarray-dtype",
            _fingerprint_numpy_dtype(cast("np.dtype[Any]", base_dtype)),
            _fingerprint_bytes(tuple(cast("tuple[int, ...]", shape))),
            _fingerprint_bytes(metadata),
        )
    return _fingerprint_envelope(
        "numpy-dtype",
        dtype.str.encode("ascii"),
        dtype.kind.encode("ascii"),
        str(dtype.itemsize).encode("ascii"),
        _fingerprint_bytes(metadata),
    )


def _fingerprint_index(index: pd.Index) -> bytes:
    """Encode index labels and metadata without depending on pandas pickle order."""

    if isinstance(index, pd.MultiIndex):
        return _fingerprint_envelope(
            "multiindex",
            _fingerprint_bytes(tuple(index.names)),
            _fingerprint_bytes(tuple(_fingerprint_index(level) for level in index.levels)),
            _fingerprint_bytes(tuple(tuple(int(code) for code in codes) for codes in index.codes)),
        )
    if isinstance(index, pd.CategoricalIndex):
        return _fingerprint_envelope(
            "categorical-index",
            _fingerprint_bytes(index.name),
            _fingerprint_index(index.categories),
            _fingerprint_bytes(cast("pd.CategoricalDtype", index.dtype).ordered),
            _fingerprint_bytes(tuple(int(code) for code in index.codes)),
        )
    if isinstance(index, pd.RangeIndex):
        return _fingerprint_envelope(
            "range-index",
            _fingerprint_bytes(index.name),
            _fingerprint_bytes(index.start),
            _fingerprint_bytes(index.stop),
            _fingerprint_bytes(index.step),
        )
    if isinstance(index, pd.DatetimeIndex):
        return _fingerprint_envelope(
            "datetime-index",
            _fingerprint_bytes(index.name),
            index.unit.encode("ascii"),
            _fingerprint_bytes(_pandas_timezone_payload(index.tz)),
            _fingerprint_bytes(_frequency_payload(index.freq)),
            _fingerprint_bytes(tuple(int(value) for value in cast("Any", index).asi8)),
        )
    if isinstance(index, pd.TimedeltaIndex):
        return _fingerprint_envelope(
            "timedelta-index",
            _fingerprint_bytes(index.name),
            str(index.dtype).encode("utf-8"),
            _fingerprint_bytes(_frequency_payload(index.freq)),
            _fingerprint_bytes(tuple(int(value) for value in cast("Any", index).asi8)),
        )
    return _fingerprint_envelope(
        "index",
        _fingerprint_bytes(index.name),
        _fingerprint_dtype(index.dtype),
        _fingerprint_bytes(tuple(index.tolist())),
    )


def _fingerprint_pandas(value: pd.Series | pd.DataFrame) -> bytes:
    """Encode pandas content in positional order while preserving metadata."""

    if isinstance(value, pd.Series):
        return _fingerprint_envelope(
            "series",
            _fingerprint_bytes(value.name),
            _fingerprint_index(value.index),
            _fingerprint_dtype(value.dtype),
            _fingerprint_bytes(value.attrs),
            _fingerprint_bytes(value.flags.allows_duplicate_labels),
            _fingerprint_bytes(tuple(value.astype(object).tolist())),
        )

    columns: list[bytes] = []
    for position in range(value.shape[1]):
        column = value.iloc[:, position]
        columns.append(
            _fingerprint_envelope(
                "dataframe-column",
                _fingerprint_dtype(column.dtype),
                _fingerprint_bytes(tuple(column.astype(object).tolist())),
            )
        )
    return _fingerprint_envelope(
        "dataframe",
        _fingerprint_index(value.index),
        _fingerprint_index(value.columns),
        _fingerprint_bytes(value.attrs),
        _fingerprint_bytes(value.flags.allows_duplicate_labels),
        _fingerprint_bytes(tuple(columns)),
    )


def _fingerprint_slots(value: object) -> Mapping[str, object] | None:
    """Return every inherited slot value, including alongside an instance dict."""

    slots: dict[str, object] = {}
    for owner in reversed(type(value).__mro__):
        names = owner.__dict__.get("__slots__", ())
        if isinstance(names, str):
            names = (names,)
        for name in cast("tuple[str, ...]", names):
            if name in {"__dict__", "__weakref__"}:
                continue
            resolved_name = name
            if name.startswith("__") and not name.endswith("__"):
                resolved_name = f"_{owner.__name__.lstrip('_')}{name}"
            if hasattr(value, resolved_name):
                slots[f"{owner.__module__}.{owner.__qualname__}.{name}"] = getattr(value, resolved_name)
    return slots or None


def _fingerprint_bytes(value: object) -> bytes:
    """Canonically encode every supported model value for cross-process provenance."""

    if isinstance(value, (pd.Series, pd.DataFrame)):
        return _fingerprint_pandas(value)
    if isinstance(value, pd.Index):
        return _fingerprint_index(value)
    if value is pd.NA:
        return _fingerprint_envelope("pandas-na")
    if value is pd.NaT:
        return _fingerprint_envelope("pandas-nat")
    if isinstance(value, np.dtype):
        return _fingerprint_numpy_dtype(value)
    if isinstance(value, np.generic):
        if value.dtype.hasobject:
            return _fingerprint_envelope(
                "object-numpy-scalar",
                _fingerprint_numpy_dtype(value.dtype),
                _fingerprint_bytes(value.item()),
            )
        return _fingerprint_envelope("numpy-scalar", _fingerprint_numpy_dtype(value.dtype), value.tobytes())
    if isinstance(value, np.ndarray):
        if value.dtype.hasobject:
            return _fingerprint_envelope(
                "object-array",
                _fingerprint_numpy_dtype(value.dtype),
                str(value.shape).encode("utf-8"),
                _fingerprint_bytes(tuple(value.ravel().tolist())),
            )
        return _fingerprint_envelope(
            "array",
            _fingerprint_numpy_dtype(value.dtype),
            str(value.shape).encode("utf-8"),
            value.tobytes(),
        )
    if is_dataclass(value) and not isinstance(value, type):
        parts = [type(value).__module__.encode("utf-8"), type(value).__qualname__.encode("utf-8")]
        for item in fields(value):
            parts.extend((item.name.encode("utf-8"), _fingerprint_bytes(getattr(value, item.name))))
        return _fingerprint_envelope("dataclass", *parts)
    if isinstance(value, Mapping):
        entries = sorted(
            (
                _fingerprint_envelope("mapping-entry", _fingerprint_bytes(key), _fingerprint_bytes(item))
                for key, item in value.items()
            )
        )
        return _fingerprint_envelope("mapping", *entries)
    if isinstance(value, tuple):
        return _fingerprint_envelope("tuple", *(_fingerprint_bytes(item) for item in value))
    if isinstance(value, list):
        return _fingerprint_envelope("list", *(_fingerprint_bytes(item) for item in value))
    if isinstance(value, frozenset):
        return _fingerprint_envelope("frozenset", *sorted(_fingerprint_bytes(item) for item in value))
    if isinstance(value, set):
        return _fingerprint_envelope("set", *sorted(_fingerprint_bytes(item) for item in value))
    if value is None:
        return _fingerprint_envelope("none")
    if isinstance(value, bool):
        return _fingerprint_envelope("bool", b"1" if value else b"0")
    if isinstance(value, int):
        return _fingerprint_envelope("int", str(value).encode("ascii"))
    if isinstance(value, float):
        return _fingerprint_envelope("float", struct.pack(">d", value))
    if isinstance(value, complex):
        return _fingerprint_envelope("complex", struct.pack(">d", value.real), struct.pack(">d", value.imag))
    if isinstance(value, Decimal):
        decimal_tuple = value.as_tuple()
        return _fingerprint_envelope(
            "decimal",
            str(decimal_tuple.sign).encode("ascii"),
            bytes(decimal_tuple.digits),
            str(decimal_tuple.exponent).encode("ascii"),
        )
    if isinstance(value, str):
        return _fingerprint_envelope("str", value.encode("utf-8"))
    if isinstance(value, bytes):
        return _fingerprint_envelope("bytes", value)
    if isinstance(value, pd.Timestamp):
        return _fingerprint_envelope(
            "timestamp",
            value.unit.encode("ascii"),
            str(value.asm8.astype("int64")).encode("ascii"),
            _fingerprint_bytes(_pandas_timezone_payload(value.tz)),
        )
    if isinstance(value, pd.Timedelta):
        return _fingerprint_envelope(
            "timedelta", value.unit.encode("ascii"), str(value.asm8.astype("int64")).encode("ascii")
        )
    if isinstance(value, pd.Period):
        return _fingerprint_envelope("period", str(value.ordinal).encode("ascii"), value.freqstr.encode("utf-8"))
    if isinstance(value, pd.Interval):
        return _fingerprint_envelope(
            "interval",
            _fingerprint_bytes(value.left),
            _fingerprint_bytes(value.right),
            value.closed.encode("ascii"),
        )
    if isinstance(value, dt.datetime):
        return _fingerprint_envelope(
            "python-datetime",
            value.isoformat().encode("utf-8"),
            str(value.fold).encode("ascii"),
            _fingerprint_bytes(_datetime_timezone_payload(value)),
        )
    if isinstance(value, dt.date):
        return _fingerprint_envelope("python-date", value.isoformat().encode("ascii"))
    if isinstance(value, dt.timedelta):
        return _fingerprint_envelope(
            "python-timedelta",
            str(value.days).encode("ascii"),
            str(value.seconds).encode("ascii"),
            str(value.microseconds).encode("ascii"),
        )
    object_dict = vars(value) if hasattr(value, "__dict__") else None
    slots = _fingerprint_slots(value)
    if object_dict is not None or slots is not None:
        return _fingerprint_envelope(
            "object",
            type(value).__module__.encode("utf-8"),
            type(value).__qualname__.encode("utf-8"),
            _fingerprint_bytes({} if object_dict is None else object_dict),
            _fingerprint_bytes({} if slots is None else slots),
        )
    raise TypeError(f"factor-analysis fingerprint does not support {value!r} ({type(value).__name__})")


def fingerprint_value(value: object) -> str:
    """Return a deterministic, lossless SHA-256 fingerprint for model data."""

    return sha256(_fingerprint_bytes(value)).hexdigest()


def snapshot_pandas(value: pd.Series | pd.DataFrame) -> pd.Series | pd.DataFrame:
    """Take an owned recursive pandas snapshot, including object-dtype cells."""

    try:
        snapshot = pickle.loads(pickle.dumps(value, protocol=5))  # nosec B301 # same-process round-trip snapshot
    except (AttributeError, pickle.PickleError, TypeError, ValueError) as error:
        raise TypeError("factor-analysis data must contain pickleable object values") from error
    if not isinstance(snapshot, (pd.Series, pd.DataFrame)):  # pragma: no cover - pickle protocol invariant
        raise TypeError("factor-analysis pandas snapshot changed its runtime type")
    return snapshot


def _snapshot_value(value: object) -> object:
    """Deep-copy analytical data before storage or public model exposure."""

    if isinstance(value, (pd.Series, pd.DataFrame)):
        return snapshot_pandas(value)
    if isinstance(value, Mapping):
        return frozen_mapping(value)
    if isinstance(value, tuple):
        return tuple(_snapshot_value(item) for item in value)
    if isinstance(value, list):
        return [_snapshot_value(item) for item in value]
    if isinstance(value, set):
        return {_snapshot_value(item) for item in value}
    if isinstance(value, frozenset):
        return frozenset(_snapshot_value(item) for item in value)
    if isinstance(value, PyfolioFactorInputs):
        return PyfolioFactorInputs(
            returns=cast("pd.Series", snapshot_pandas(value.returns)),
            positions=cast("pd.DataFrame", snapshot_pandas(value.positions)),
            benchmark_rets=None
            if value.benchmark_rets is None
            else cast("pd.Series", snapshot_pandas(value.benchmark_rets)),
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
            aggregate_quantile_average_returns=cast(
                "pd.DataFrame", _snapshot_value(value.aggregate_quantile_average_returns)
            ),
        )
    if value is None or isinstance(
        value,
        (str, bytes, bool, int, float, complex, Decimal, dt.date, dt.datetime, dt.timedelta, np.dtype, np.generic),
    ):
        return value
    try:
        return pickle.loads(pickle.dumps(value, protocol=5))  # nosec B301 # same-process round-trip snapshot
    except (AttributeError, pickle.PickleError, TypeError, ValueError) as error:
        raise TypeError("factor-analysis snapshot values must be pickleable") from error


def _snapshot_mapping_key(value: _MappingKey) -> _MappingKey:
    """Copy a mapping key so mutable-but-hashable labels cannot leak inward."""

    try:
        snapshot = pickle.loads(pickle.dumps(value, protocol=5))  # nosec B301 # same-process round-trip snapshot
    except (AttributeError, pickle.PickleError, TypeError, ValueError) as error:
        raise TypeError("factor-analysis mapping keys must be pickleable") from error
    if not isinstance(snapshot, Hashable):  # pragma: no cover - Mapping protocol invariant
        raise TypeError("factor-analysis mapping key snapshot is not hashable")
    try:
        preserves_lookup = hash(snapshot) == hash(value) and bool(snapshot == value) and bool(value == snapshot)
    except (TypeError, ValueError) as error:
        raise TypeError("factor-analysis mapping keys must preserve equality and hash after snapshotting") from error
    if not preserves_lookup:
        raise TypeError("factor-analysis mapping keys must preserve equality and hash after snapshotting")
    return cast("_MappingKey", snapshot)


class _FrozenMapping(Mapping[_MappingKey, _MappingValue]):
    """Private mapping storage that releases only fresh snapshots to callers."""

    __slots__ = ("__data",)

    def __init__(self, mapping: Mapping[_MappingKey, _MappingValue]) -> None:
        self.__data = MappingProxyType(
            {
                _snapshot_mapping_key(key): cast("_MappingValue", _snapshot_value(value))
                for key, value in mapping.items()
            }
        )

    def __getitem__(self, key: _MappingKey) -> _MappingValue:
        return cast("_MappingValue", _snapshot_value(self.__data[key]))

    def __iter__(self):  # type: ignore[override]
        return iter(tuple(_snapshot_mapping_key(key) for key in self.__data))

    def __len__(self) -> int:
        return len(self.__data)


def frozen_mapping(mapping: Mapping[_MappingKey, _MappingValue]) -> Mapping[_MappingKey, _MappingValue]:
    """Own mapping values and expose a fresh read-only cache boundary."""

    return _FrozenMapping(mapping)


def _typed_sequence(value: object, name: str) -> tuple[object, ...]:
    """Copy only deterministic public sequences, never strings or unordered sets."""

    if isinstance(value, (str, bytes, bytearray)) or not isinstance(value, Sequence):
        raise TypeError(f"{name} must be a sequence")
    return tuple(value)


@dataclass(frozen=True, slots=True)
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

        periods = _typed_sequence(self.periods, "periods")
        turnover_periods = _typed_sequence(self.turnover_periods, "turnover_periods")
        time_aggregation = _typed_sequence(self.time_aggregation, "time_aggregation")
        for name in ("long_short", "group_neutral", "equal_weight", "by_group", "include_pyfolio"):
            if not isinstance(getattr(self, name), bool):
                raise TypeError(f"{name} must be a bool")
        if any(not isinstance(period, str) for period in periods):
            raise TypeError("periods must contain strings")
        if any(not isinstance(period, int) or isinstance(period, bool) for period in turnover_periods):
            raise TypeError("turnover_periods must contain integers")
        if any(not isinstance(frequency, str) for frequency in time_aggregation):
            raise TypeError("time_aggregation must contain strings")
        for name, value in (("event_before", self.event_before), ("event_after", self.event_after)):
            if value is not None and (not isinstance(value, int) or isinstance(value, bool)):
                raise TypeError(f"{name} must be an int or None")
            if value is not None and value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        if not isinstance(self.pyfolio_benchmark_period, str):
            raise TypeError("pyfolio_benchmark_period must be a string")
        capital = self.pyfolio_capital
        if capital is not None:
            if not isinstance(capital, (int, float)) or isinstance(capital, bool):
                raise TypeError("pyfolio_capital must be an int, float, or None")
            if isinstance(capital, int):
                try:
                    exactly_representable = int(float(capital)) == capital
                except OverflowError:
                    exactly_representable = False
                if not exactly_representable:
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


@dataclass(frozen=True, slots=True)
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

        object.__setattr__(self, "group", _snapshot_mapping_key(self.group))
        for name in self._DATA_FIELDS:
            object.__setattr__(self, name, _snapshot_value(object.__getattribute__(self, name)))

    def __getattribute__(self, name: str) -> object:
        value = object.__getattribute__(self, name)
        if name in object.__getattribute__(self, "_DATA_FIELDS"):
            return _snapshot_value(value)
        return value


@dataclass(frozen=True, slots=True)
class EventAnalysisModel:
    """Frozen event-window values, averages, and distribution inputs."""

    event_windows: pd.DataFrame
    mean_returns: pd.Series
    return_distribution: pd.Series
    quantile_average_returns: pd.DataFrame
    aggregate_quantile_average_returns: pd.DataFrame

    _DATA_FIELDS = frozenset(
        {
            "event_windows",
            "mean_returns",
            "return_distribution",
            "quantile_average_returns",
            "aggregate_quantile_average_returns",
        }
    )

    def __post_init__(self) -> None:
        """Own independent event data rather than caller-visible work buffers."""

        for name in self._DATA_FIELDS:
            object.__setattr__(self, name, _snapshot_value(object.__getattribute__(self, name)))

    def __getattribute__(self, name: str) -> object:
        value = object.__getattribute__(self, name)
        if name in object.__getattribute__(self, "_DATA_FIELDS"):
            return _snapshot_value(value)
        return value


@dataclass(frozen=True, slots=True)
class FactorAnalysisModel:
    """One compute-once analytical snapshot consumed by later renderers."""

    config: FactorAnalysisConfig
    factor_data: pd.DataFrame
    forward_periods: tuple[str, ...]
    quantile_statistics: pd.DataFrame
    factor_weights: pd.DataFrame
    factor_returns: pd.DataFrame
    factor_cumulative_returns: Mapping[str, pd.Series]
    legacy_quantile_cumulative_returns: Mapping[str, pd.DataFrame]
    factor_positions: Mapping[str, pd.DataFrame]
    alpha_beta: pd.DataFrame
    mean_returns_by_quantile: pd.DataFrame
    std_error_by_quantile: pd.DataFrame
    mean_returns_by_date: pd.DataFrame
    std_error_by_date: pd.DataFrame
    aggregate_mean_returns_by_quantile: pd.DataFrame
    aggregate_std_error_by_quantile: pd.DataFrame
    aggregate_mean_returns_by_date: pd.DataFrame
    aggregate_std_error_by_date: pd.DataFrame
    aggregate_mean_return_spread: pd.DataFrame
    aggregate_mean_return_spread_std: pd.DataFrame | None
    mean_return_spread: pd.DataFrame
    mean_return_spread_std: pd.DataFrame | None
    information_coefficient: pd.DataFrame
    mean_information_coefficient: pd.Series | pd.DataFrame
    aggregate_information_coefficient: pd.DataFrame
    aggregate_mean_information_coefficient: pd.Series | pd.DataFrame
    summary_information_coefficient: pd.DataFrame
    quantile_turnover: Mapping[int, pd.DataFrame]
    rank_autocorrelation: pd.DataFrame
    grouped_results: Mapping[Hashable, FactorGroupAnalysis]
    time_aggregated_results: Mapping[str, pd.Series | pd.DataFrame]
    aggregate_time_aggregated_results: Mapping[str, pd.Series | pd.DataFrame]
    pyfolio_inputs: PyfolioFactorInputs | None
    event_input_snapshot: pd.DataFrame | None
    event_returns: EventAnalysisModel | None = None
    result_fingerprint: str = ""

    _DATA_FIELDS = frozenset(
        {
            "factor_data",
            "quantile_statistics",
            "factor_weights",
            "factor_returns",
            "factor_cumulative_returns",
            "legacy_quantile_cumulative_returns",
            "factor_positions",
            "alpha_beta",
            "mean_returns_by_quantile",
            "std_error_by_quantile",
            "mean_returns_by_date",
            "std_error_by_date",
            "aggregate_mean_returns_by_quantile",
            "aggregate_std_error_by_quantile",
            "aggregate_mean_returns_by_date",
            "aggregate_std_error_by_date",
            "aggregate_mean_return_spread",
            "aggregate_mean_return_spread_std",
            "mean_return_spread",
            "mean_return_spread_std",
            "information_coefficient",
            "mean_information_coefficient",
            "aggregate_information_coefficient",
            "aggregate_mean_information_coefficient",
            "summary_information_coefficient",
            "quantile_turnover",
            "rank_autocorrelation",
            "grouped_results",
            "time_aggregated_results",
            "aggregate_time_aggregated_results",
            "pyfolio_inputs",
            "event_input_snapshot",
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
    "snapshot_pandas",
]
