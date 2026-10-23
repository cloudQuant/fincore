"""Frozen, renderer-ready models for enhanced factor analysis.

These dataclasses deliberately contain only analytical data.  Matplotlib,
seaborn, IPython, and renderer callables belong to later adapter layers and
are intentionally absent from this module.
"""

from __future__ import annotations

import base64
import datetime as dt
import pickle  # nosec B403 # round-trips trusted in-process data only; never deserializes external input
import re
import struct
from collections.abc import Hashable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field, fields, is_dataclass
from decimal import Decimal
from hashlib import sha256
from types import MappingProxyType, MemberDescriptorType
from typing import Any, Literal, TypeVar, cast

import numpy as np
import pandas as pd
from pandas.tseries.frequencies import to_offset
from pandas.tseries.offsets import BDay, CustomBusinessDay

from fincore.factor_analysis.portfolio import FactorPortfolioInputs

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


_TZFILE_TEXT_FORM = re.compile(r"\Atzfile\('([^']*)'\)\Z")


def _iana_name_from_tzfile_text(value: object) -> str | None:
    """Extract the IANA name from dateutil's ``tzfile(...)`` string form.

    On Windows, dateutil resolves named zones through its bundled tarball and
    stringifies the result as ``tzfile('America/New_York')``; that text is the
    only stable identity available for the wire envelope.
    """
    match = _TZFILE_TEXT_FORM.match(str(value))
    if match is None:
        return None
    candidate = match.group(1)
    _, _, candidate = candidate.rpartition("/zoneinfo/")
    if not candidate or "\\" in candidate or candidate.endswith((".tar", ".tar.gz")):
        return None
    return candidate


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
    text_name = _iana_name_from_tzfile_text(timezone)
    if text_name is not None:
        return {"kind": "iana-zone", "name": text_name}
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

    unwrapped = _iana_name_from_tzfile_text(name)
    if unwrapped is not None:
        name = unwrapped
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


class _FingerprintNotStreamable(Exception):
    """Internal signal to use the byte-compatible legacy encoder."""


_FINGERPRINT_CHUNK_BYTES = 8 * 1024 * 1024
_FINGERPRINT_FLOAT_FRAME_BYTES = 33
_FINGERPRINT_LEAF_CACHE_ENTRIES = 2048
_FINGERPRINT_LEAF_CACHE_BYTES = 256 * 1024
_FINGERPRINT_LEAF_BATCH_BYTES = 1024 * 1024
_FINGERPRINT_LEAF_BATCH_ENTRIES = 8192
_FINGERPRINT_INTEGER_BATCH_ROWS = 32768


class _FingerprintDigestWriter:
    """Buffer small framing writes and cap every digest update at eight MiB."""

    __slots__ = ("_buffer", "_digest")

    def __init__(self, digest: Any) -> None:
        self._digest = digest
        self._buffer = bytearray()

    def write(self, data: bytes | bytearray | memoryview) -> None:
        view = memoryview(data).cast("B")
        offset = 0
        while offset < len(view):
            remaining = len(view) - offset
            if not self._buffer and remaining >= _FINGERPRINT_CHUNK_BYTES:
                end = offset + _FINGERPRINT_CHUNK_BYTES
                self._digest.update(view[offset:end])
                offset = end
                continue
            take = min(_FINGERPRINT_CHUNK_BYTES - len(self._buffer), remaining)
            self._buffer.extend(view[offset : offset + take])
            offset += take
            if len(self._buffer) == _FINGERPRINT_CHUNK_BYTES:
                self._digest.update(self._buffer)
                self._buffer.clear()

    def finish(self) -> None:
        if self._buffer:
            self._digest.update(self._buffer)
            self._buffer.clear()


class _FingerprintStreamContext:
    """Bounded caches for repeated builtin string and integer leaves."""

    __slots__ = (
        "_cache_bytes",
        "_datetime_plans",
        "_framed_parts",
        "_index_plans",
        "_int_tuple_lengths",
        "_integers",
        "_node_lengths",
        "_other_scalars",
        "_pandas_plans",
        "_strings",
        "_timestamps",
    )

    def __init__(self) -> None:
        self._strings: dict[str, bytes] = {}
        self._integers: dict[int, bytes] = {}
        self._other_scalars: dict[tuple[type[object], object], bytes] = {}
        self._timestamps: dict[tuple[bytes, bytes, int], bytes] = {}
        self._framed_parts: dict[bytes, bytes] = {}
        self._cache_bytes = 0
        self._node_lengths: dict[int, tuple[object, int]] = {}
        self._pandas_plans: dict[int, tuple[object, Any]] = {}
        self._datetime_plans: dict[int, tuple[object, Any]] = {}
        self._index_plans: dict[int, tuple[object, Any]] = {}
        self._int_tuple_lengths: dict[int, tuple[object, int]] = {}

    def node_length(self, value: object) -> int | None:
        cached = self._node_lengths.get(id(value))
        return cached[1] if cached is not None and cached[0] is value else None

    def remember_node(self, value: object, length: int) -> None:
        if type(value) in {pd.Series, pd.DataFrame} or isinstance(value, pd.Index):
            self._node_lengths[id(value)] = (value, length)

    def pandas_plan(self, value: object) -> Any:
        cached = self._pandas_plans.get(id(value))
        return cached[1] if cached is not None and cached[0] is value else None

    def remember_pandas_plan(self, value: object, plan: Any) -> None:
        self._pandas_plans[id(value)] = (value, plan)

    def datetime_plan(self, value: object) -> Any:
        cached = self._datetime_plans.get(id(value))
        return cached[1] if cached is not None and cached[0] is value else None

    def remember_datetime_plan(self, value: object, plan: Any) -> None:
        self._datetime_plans[id(value)] = (value, plan)

    def index_plan(self, value: object) -> Any:
        cached = self._index_plans.get(id(value))
        return cached[1] if cached is not None and cached[0] is value else None

    def remember_index_plan(self, value: object, plan: Any) -> None:
        self._index_plans[id(value)] = (value, plan)

    def int_tuple_length(self, values: object) -> int | None:
        cached = self._int_tuple_lengths.get(id(values))
        return cached[1] if cached is not None and cached[0] is values else None

    def remember_int_tuple_length(self, values: object, length: int) -> None:
        self._int_tuple_lengths[id(values)] = (values, length)

    def scalar_bytes(self, value: object) -> bytes:
        value_type = type(value)
        if value_type is str:
            cached = self._strings.get(cast("str", value))
            if cached is not None:
                return cached
            encoded = _fingerprint_envelope("str", cast("str", value).encode("utf-8"))
            self._cache("str", value, encoded)
            return encoded
        if value_type is int:
            cached = self._integers.get(cast("int", value))
            if cached is not None:
                return cached
            encoded = _fingerprint_envelope(
                "int", str(cast("int", value)).encode("ascii")
            )
            self._cache("int", value, encoded)
            return encoded
        if value_type is pd.Timestamp:
            timezone = _fingerprint_bytes(_pandas_timezone_payload(value.tz))
            return self.timestamp_bytes(
                int(value.asm8.astype("int64")), value.unit.encode("ascii"), timezone
            )
        if value_type in {type(None), bool, bytes}:
            key = (value_type, value)
            cached = self._other_scalars.get(key)
            if cached is not None:
                return cached
            encoded = _fingerprint_bytes(value)
            self._cache("other", key, encoded)
            return encoded
        if value_type in _STREAM_SCALAR_TYPES:
            return _fingerprint_bytes(value)
        if value is pd.NA or value is pd.NaT:
            return _fingerprint_bytes(value)
        raise _FingerprintNotStreamable

    def timestamp_bytes(self, raw_value: int, unit: bytes, timezone: bytes) -> bytes:
        if raw_value == np.iinfo(np.int64).min:
            return _fingerprint_bytes(pd.NaT)
        key = (unit, timezone, raw_value)
        cached = self._timestamps.get(key)
        if cached is not None:
            return cached
        encoded = _fingerprint_envelope(
            "timestamp",
            unit,
            str(raw_value).encode("ascii"),
            timezone,
        )
        self._cache("timestamp", key, encoded)
        return encoded

    def framed_part(self, encoded: bytes) -> bytes | None:
        if len(encoded) > 256:
            return None
        cached = self._framed_parts.get(encoded)
        if cached is not None:
            return cached
        framed = len(encoded).to_bytes(8, "big") + encoded
        entry_count = (
            len(self._strings)
            + len(self._integers)
            + len(self._other_scalars)
            + len(self._timestamps)
            + len(self._framed_parts)
        )
        if (
            entry_count >= _FINGERPRINT_LEAF_CACHE_ENTRIES
            or self._cache_bytes + len(encoded) + len(framed)
            > _FINGERPRINT_LEAF_CACHE_BYTES
        ):
            return None
        self._framed_parts[encoded] = framed
        self._cache_bytes += len(encoded) + len(framed)
        return framed

    def _cache(self, kind: str, value: object, encoded: bytes) -> None:
        if (
            len(encoded) > 256
            or (
                len(self._strings)
                + len(self._integers)
                + len(self._other_scalars)
                + len(self._timestamps)
                + len(self._framed_parts)
            )
            >= _FINGERPRINT_LEAF_CACHE_ENTRIES
        ):
            return
        if self._cache_bytes + len(encoded) > _FINGERPRINT_LEAF_CACHE_BYTES:
            return
        if kind == "str":
            self._strings[cast("str", value)] = encoded
        elif kind == "int":
            self._integers[cast("int", value)] = encoded
        elif kind == "timestamp":
            self._timestamps[cast("tuple[bytes, bytes, int]", value)] = encoded
        elif kind == "other":
            self._other_scalars[cast("tuple[type[object], object]", value)] = encoded
        self._cache_bytes += len(encoded)


_STREAM_SCALAR_TYPES = {
    type(None),
    bool,
    float,
    complex,
    Decimal,
    bytes,
    dt.datetime,
    dt.date,
    dt.timedelta,
    pd.Timestamp,
    pd.Timedelta,
    pd.Period,
    pd.Interval,
}


def _stream_envelope_length(tag: str, part_lengths: Sequence[int]) -> int:
    return (
        4
        + len(tag.encode("utf-8"))
        + sum(8 + part_length for part_length in part_lengths)
    )


def _stream_envelope_start(writer: Any, tag: str) -> None:
    tag_bytes = tag.encode("utf-8")
    writer.write(len(tag_bytes).to_bytes(4, "big"))
    writer.write(tag_bytes)


def _stream_part_start(writer: Any, length: int) -> None:
    writer.write(length.to_bytes(8, "big"))


def _stream_dtype_is_plain(dtype: object) -> bool:
    return (
        isinstance(dtype, np.dtype)
        and dtype.fields is None
        and dtype.subdtype is None
        and dtype.metadata is None
        and dtype.kind in "biufcMmOSU"
    )


def _stream_series_values(value: pd.Series) -> tuple[Any, Any]:
    dtype = value.dtype
    if type(dtype) is pd.DatetimeTZDtype:
        values = value._values
        if type(values) is not pd.arrays.DatetimeArray:
            raise _FingerprintNotStreamable
        return values, dtype
    if not _stream_dtype_is_plain(dtype):
        raise _FingerprintNotStreamable
    values = value._values
    if dtype.kind in "f" and dtype.itemsize in (4, 8):
        if not isinstance(values, np.ndarray) or values.dtype != dtype:
            raise _FingerprintNotStreamable
    elif dtype.kind not in "biucMmOSU":
        raise _FingerprintNotStreamable
    return values, cast("np.dtype[Any]", dtype)


def _stream_item_from_array(values: Any, dtype: np.dtype[Any], position: int) -> object:
    item = values[position]
    if dtype.kind in "biuc":
        return item.item()
    if dtype.kind in "SU":
        return item.item()
    return item


def _stream_integer_decimal_widths_chunk(
    values: np.ndarray[Any, Any],
) -> np.ndarray[Any, Any]:
    """Return decimal widths, including a minus sign, for one bounded chunk."""

    kind = values.dtype.kind
    if kind not in "iu":
        raise _FingerprintNotStreamable
    widths = np.ones(len(values), dtype=np.uint8)
    if kind == "i":
        negative = values < 0
        for exponent in range(1, 19):
            power = 10**exponent
            widths += values >= power
            widths += values <= -power
        widths += negative
    else:
        for exponent in range(1, 20):
            widths += values >= 10**exponent
    return widths


def _stream_integer_decimal_width_sum(values: np.ndarray[Any, Any]) -> int:
    width_sum = 0
    for start in range(0, len(values), _FINGERPRINT_INTEGER_BATCH_ROWS):
        chunk = values[start : start + _FINGERPRINT_INTEGER_BATCH_ROWS]
        widths = _stream_integer_decimal_widths_chunk(chunk)
        width_sum += int(widths.sum(dtype=np.uint64))
    return width_sum


def _stream_datetime_plan(
    values: Any, dtype: pd.DatetimeTZDtype, context: _FingerprintStreamContext
) -> tuple[np.ndarray[Any, Any], bytes, bytes, int, int]:
    cached = context.datetime_plan(values)
    if cached is not None:
        return cast("tuple[np.ndarray[Any, Any], bytes, bytes, int, int]", cached)
    if type(values) is not pd.arrays.DatetimeArray:
        raise _FingerprintNotStreamable
    raw_values = values.asi8
    unit = dtype.unit.encode("ascii")
    timezone = _fingerprint_bytes(_pandas_timezone_payload(dtype.tz))
    max_frame_bytes = 8 + 37 + len(unit) + len(timezone) + 20
    if max_frame_bytes > _FINGERPRINT_CHUNK_BYTES:
        # A single timestamp envelope cannot be split without changing its
        # framing. Use the compatibility encoder for this unsupported edge.
        raise _FingerprintNotStreamable
    rows_per_chunk = min(
        _FINGERPRINT_INTEGER_BATCH_ROWS,
        _FINGERPRINT_CHUNK_BYTES // max_frame_bytes,
    )
    count = len(raw_values)
    nat_count = int(np.count_nonzero(raw_values == np.iinfo(np.int64).min))
    decimal_width_sum = _stream_integer_decimal_width_sum(raw_values)
    # int64 NaT uses the -9223372036854775808 raw value, whose 20-character
    # representation is replaced by the 14-byte pandas-nat leaf.
    decimal_width_sum -= nat_count * 20
    regular_count = count - nat_count
    values_length = (
        9
        + 8 * count
        + 14 * nat_count
        + (37 + len(unit) + len(timezone)) * regular_count
        + decimal_width_sum
    )
    plan = (raw_values, unit, timezone, values_length, rows_per_chunk)
    context.remember_datetime_plan(values, plan)
    return plan


def _stream_write_encoded_part_batch(
    encoded_values: Any,
    writer: Any,
    context: _FingerprintStreamContext,
) -> None:
    parts: list[bytes] = []
    buffered_bytes = 0

    def flush() -> None:
        nonlocal buffered_bytes
        if parts:
            writer.write(b"".join(parts))
            parts.clear()
            buffered_bytes = 0

    for encoded in encoded_values:
        framed = context.framed_part(encoded)
        if framed is None or len(framed) > _FINGERPRINT_LEAF_BATCH_BYTES:
            flush()
            _stream_part_start(writer, len(encoded))
            writer.write(encoded)
            continue
        if parts and (
            buffered_bytes + len(framed) > _FINGERPRINT_LEAF_BATCH_BYTES
            or len(parts) >= _FINGERPRINT_LEAF_BATCH_ENTRIES
        ):
            flush()
        parts.append(framed)
        buffered_bytes += len(framed)
    flush()


def _stream_write_scalar_part_batch(
    values: Any,
    writer: Any,
    context: _FingerprintStreamContext,
) -> None:
    _stream_write_encoded_part_batch(
        (context.scalar_bytes(value) for value in values), writer, context
    )


def _stream_object_scalar_runs(
    values: np.ndarray[Any, Any],
) -> Iterator[tuple[object, int]]:
    """Yield adjacent exact string/integer runs without materializing a column."""

    position = 0
    count = len(values)
    while position < count:
        value = values[position]
        value_type = type(value)
        if value_type is not str and value_type is not int:
            yield value, 1
            position += 1
            continue

        end = position + 1
        while end < count:
            candidate = values[end]
            if type(candidate) is not value_type or candidate != value:
                break
            end += 1
        yield value, end - position
        position = end


def _stream_object_scalar_supported(value: object) -> bool:
    value_type = type(value)
    return (
        value_type in _STREAM_SCALAR_TYPES
        or value_type is str
        or value_type is int
        or value is pd.NA
        or value is pd.NaT
    )


def _stream_write_object_scalar_runs(
    values: np.ndarray[Any, Any],
    writer: Any,
    context: _FingerprintStreamContext,
) -> None:
    """Write object-cell leaf frames in bounded batches, repeating exact runs."""

    parts: list[bytes] = []
    buffered_bytes = 0

    def flush() -> None:
        nonlocal buffered_bytes
        if parts:
            writer.write(b"".join(parts))
            parts.clear()
            buffered_bytes = 0

    def append_repetitions(encoded: bytes, repetitions: int) -> None:
        nonlocal buffered_bytes
        framed = context.framed_part(encoded)
        if framed is None or len(framed) > _FINGERPRINT_LEAF_BATCH_BYTES:
            flush()
            for _ in range(repetitions):
                _stream_part_start(writer, len(encoded))
                writer.write(encoded)
            return

        while repetitions:
            if parts and (
                buffered_bytes + len(framed) > _FINGERPRINT_LEAF_BATCH_BYTES
                or len(parts) >= _FINGERPRINT_LEAF_BATCH_ENTRIES
            ):
                flush()
            available_bytes = _FINGERPRINT_LEAF_BATCH_BYTES - buffered_bytes
            available_entries = _FINGERPRINT_LEAF_BATCH_ENTRIES
            take = min(
                repetitions,
                available_entries,
                available_bytes // len(framed),
            )
            if take <= 0:
                flush()
                continue
            block = framed if take == 1 else framed * take
            parts.append(block)
            buffered_bytes += len(block)
            repetitions -= take

    for value, repetitions in _stream_object_scalar_runs(values):
        if not _stream_object_scalar_supported(value):
            raise _FingerprintNotStreamable
        encoded = context.scalar_bytes(value)
        append_repetitions(encoded, repetitions)
    flush()


def _stream_write_part_values(
    values: Any,
    writer: Any,
    context: _FingerprintStreamContext,
) -> None:
    parts: list[bytes] = []
    buffered_bytes = 0

    def flush() -> None:
        nonlocal buffered_bytes
        if parts:
            writer.write(b"".join(parts))
            parts.clear()
            buffered_bytes = 0

    for value in values:
        value_type = type(value)
        if (
            value_type in _STREAM_SCALAR_TYPES
            or value_type in {str, int}
            or value is pd.NA
            or value is pd.NaT
        ):
            encoded = context.scalar_bytes(value)
            framed = context.framed_part(encoded)
            if framed is None or len(framed) > _FINGERPRINT_LEAF_BATCH_BYTES:
                flush()
                _stream_part_start(writer, len(encoded))
                writer.write(encoded)
                continue
            if parts and (
                buffered_bytes + len(framed) > _FINGERPRINT_LEAF_BATCH_BYTES
                or len(parts) >= _FINGERPRINT_LEAF_BATCH_ENTRIES
            ):
                flush()
            parts.append(framed)
            buffered_bytes += len(framed)
            continue
        flush()
        _stream_emit_part_value(value, writer, context)
    flush()


def _stream_write_integer_tuple(
    values: np.ndarray[Any, Any], writer: Any
) -> None:
    writer.write((5).to_bytes(4, "big"))
    writer.write(b"tuple")
    for start in range(0, len(values), _FINGERPRINT_INTEGER_BATCH_ROWS):
        chunk = np.asarray(
            values[start : start + _FINGERPRINT_INTEGER_BATCH_ROWS]
        )
        widths = _stream_integer_decimal_widths_chunk(chunk)
        decimal_bytes = np.char.mod("%d", chunk).astype("S20", copy=False)
        decimal_matrix = decimal_bytes.view(np.uint8).reshape(len(chunk), 20)
        strides = 23 + widths.astype(np.int64)
        starts = np.empty(len(chunk), dtype=np.int64)
        starts[0] = 0
        if len(chunk) > 1:
            np.cumsum(strides[:-1], dtype=np.int64, out=starts[1:])
        frame = bytearray(int(strides.sum(dtype=np.int64)))
        frame_bytes = np.frombuffer(frame, dtype=np.uint8)
        frame_bytes[starts + 7] = 15 + widths
        frame_bytes[starts + 11] = 3
        frame_bytes[starts + 12] = ord("i")
        frame_bytes[starts + 13] = ord("n")
        frame_bytes[starts + 14] = ord("t")
        frame_bytes[starts + 22] = widths
        for digit in range(20):
            active = widths > digit
            frame_bytes[starts[active] + 23 + digit] = decimal_matrix[
                active, digit
            ]
        writer.write(frame)


def _stream_write_datetime_tuple(
    raw_values: np.ndarray[Any, Any],
    unit: bytes,
    timezone: bytes,
    rows_per_chunk: int,
    writer: Any,
) -> None:
    writer.write((5).to_bytes(4, "big"))
    writer.write(b"tuple")
    nat_value = np.iinfo(np.int64).min
    for start in range(0, len(raw_values), rows_per_chunk):
        chunk = np.asarray(raw_values[start : start + rows_per_chunk])
        widths = _stream_integer_decimal_widths_chunk(chunk)
        decimal_bytes = np.char.mod("%d", chunk).astype("S20", copy=False)
        decimal_matrix = decimal_bytes.view(np.uint8).reshape(len(chunk), 20)
        nat = chunk == nat_value
        leaf_lengths = np.where(
            nat,
            14,
            37 + len(unit) + len(timezone) + widths.astype(np.int64),
        ).astype(np.uint64)
        strides = (8 + leaf_lengths).astype(np.int64)
        starts = np.empty(len(chunk), dtype=np.int64)
        starts[0] = 0
        if len(chunk) > 1:
            np.cumsum(strides[:-1], dtype=np.int64, out=starts[1:])
        frame = bytearray(int(strides.sum(dtype=np.int64)))
        frame_bytes = np.frombuffer(frame, dtype=np.uint8)

        def write_uint64(offsets: np.ndarray[Any, Any], numbers: np.ndarray[Any, Any]) -> None:
            for byte_index in range(8):
                shift = 8 * (7 - byte_index)
                frame_bytes[offsets + byte_index] = (numbers >> shift) & 0xFF

        write_uint64(starts, leaf_lengths)
        regular = ~nat
        leaf_starts = starts[regular] + 8
        regular_widths = widths[regular].astype(np.uint64)
        frame_bytes[leaf_starts + 3] = 9
        tag = b"timestamp"
        for byte_index, byte_value in enumerate(tag):
            frame_bytes[leaf_starts + 4 + byte_index] = byte_value
        write_uint64(leaf_starts + 13, np.full(len(leaf_starts), len(unit), dtype=np.uint64))
        for byte_index, byte_value in enumerate(unit):
            frame_bytes[leaf_starts + 21 + byte_index] = byte_value
        width_offsets = leaf_starts + 21 + len(unit)
        write_uint64(width_offsets, regular_widths)
        digit_offsets = leaf_starts + 29 + len(unit)
        regular_matrix = decimal_matrix[regular]
        for digit in range(20):
            active = regular_widths > digit
            frame_bytes[digit_offsets[active] + digit] = regular_matrix[active, digit]
        timezone_offsets = leaf_starts + 29 + len(unit) + regular_widths.astype(np.int64)
        write_uint64(
            timezone_offsets,
            np.full(len(leaf_starts), len(timezone), dtype=np.uint64),
        )
        payload_offsets = leaf_starts + 37 + len(unit) + regular_widths.astype(np.int64)
        for byte_index, byte_value in enumerate(timezone):
            frame_bytes[payload_offsets + byte_index] = byte_value

        nat_starts = starts[nat] + 8
        nat_leaf = b"\x00\x00\x00\x0apandas-nat"
        for byte_index, byte_value in enumerate(nat_leaf):
            frame_bytes[nat_starts + byte_index] = byte_value
        writer.write(frame)


def _stream_tuple_array_length(
    values: Any, dtype: Any, context: _FingerprintStreamContext
) -> int:
    count = len(values)
    if dtype.kind == "f":
        return 9 + _FINGERPRINT_FLOAT_FRAME_BYTES * count
    if type(dtype) is pd.DatetimeTZDtype:
        return _stream_datetime_plan(values, dtype, context)[3]
    if dtype.kind in "iu" and type(values) is np.ndarray:
        return 9 + 23 * count + _stream_integer_decimal_width_sum(values)
    if dtype.kind == "b":
        return 9 + 25 * count
    if dtype.kind == "O" and type(values) is np.ndarray:
        # Exact ndarray object cells cannot override indexing. Keep the same
        # standard-scalar allowlist as emission, but avoid generic recursive
        # dispatch for every cell. Adjacent equal strings/integers share one
        # leaf-length lookup; no per-cell bytes or column-sized cache is kept.
        length = 9
        for item, repetitions in _stream_object_scalar_runs(values):
            if not _stream_object_scalar_supported(item):
                raise _FingerprintNotStreamable
            length += repetitions * (8 + len(context.scalar_bytes(item)))
        return length
    length = 9
    for position in range(count):
        item = _stream_item_from_array(values, dtype, position)
        if dtype.kind == "O" and not _stream_object_scalar_supported(item):
            raise _FingerprintNotStreamable
        length += 8 + _stream_length(item, context)
    return length


def _stream_write_float_tuple(values: np.ndarray[Any, Any], writer: Any) -> None:
    writer.write((5).to_bytes(4, "big"))
    writer.write(b"tuple")
    frame_rows = _FINGERPRINT_CHUNK_BYTES // _FINGERPRINT_FLOAT_FRAME_BYTES
    for start in range(0, len(values), frame_rows):
        source = values[start : start + frame_rows]
        as_float64 = np.asarray(source, dtype=np.float64)
        payload = as_float64.astype(">f8", copy=False).view(np.uint8).reshape(-1, 8)
        frames = np.empty((len(source), _FINGERPRINT_FLOAT_FRAME_BYTES), dtype=np.uint8)
        frames[:, 0:8] = np.frombuffer((25).to_bytes(8, "big"), dtype=np.uint8)
        frames[:, 8:12] = np.frombuffer((5).to_bytes(4, "big"), dtype=np.uint8)
        frames[:, 12:17] = np.frombuffer(b"float", dtype=np.uint8)
        frames[:, 17:25] = np.frombuffer((8).to_bytes(8, "big"), dtype=np.uint8)
        frames[:, 25:33] = payload
        writer.write(memoryview(frames).cast("B"))


def _stream_tuple_array(
    values: Any, dtype: Any, writer: Any, context: _FingerprintStreamContext
) -> None:
    if dtype.kind == "f":
        _stream_write_float_tuple(cast("np.ndarray[Any, Any]", values), writer)
        return
    if type(dtype) is pd.DatetimeTZDtype:
        raw_values, unit, timezone, _, rows_per_chunk = _stream_datetime_plan(
            values, dtype, context
        )
        _stream_write_datetime_tuple(
            raw_values, unit, timezone, rows_per_chunk, writer
        )
        return
    if dtype.kind in "iu" and type(values) is np.ndarray:
        _stream_write_integer_tuple(values, writer)
        return
    writer.write((5).to_bytes(4, "big"))
    writer.write(b"tuple")
    if dtype.kind == "O" and type(values) is np.ndarray:
        _stream_write_object_scalar_runs(values, writer, context)
        return
    _stream_write_scalar_part_batch(
        (_stream_item_from_array(values, dtype, position) for position in range(len(values))),
        writer,
        context,
    )


def _stream_index_length_uncached(
    index: pd.Index, context: _FingerprintStreamContext
) -> int:
    index_type = type(index)
    if index_type is pd.MultiIndex:
        names = tuple(index.names)
        names_length = _stream_length(names, context)
        levels = tuple(index.levels)
        level_lengths = tuple(_stream_index_length(level, context) for level in levels)
        levels_length = 9 + sum(8 + 17 + item for item in level_lengths)
        codes = tuple(index.codes)
        code_lengths = tuple(_stream_int_tuple_length(code, context) for code in codes)
        codes_length = 9 + sum(code_length + 8 for code_length in code_lengths)
        length = _stream_envelope_length(
            "multiindex", (names_length, levels_length, codes_length)
        )
        context.remember_index_plan(
            index,
            (
                "multiindex",
                names,
                names_length,
                levels,
                level_lengths,
                levels_length,
                codes,
                code_lengths,
                codes_length,
            ),
        )
        return length
    if index_type is pd.CategoricalIndex:
        name = index.name
        name_length = _stream_length(name, context)
        categories = index.categories
        categories_length = _stream_index_length(categories, context)
        ordered = cast("pd.CategoricalDtype", index.dtype).ordered
        ordered_length = _stream_length(ordered, context)
        codes = index.codes
        codes_length = _stream_int_tuple_length(codes, context)
        length = _stream_envelope_length(
            "categorical-index",
            (
                name_length,
                categories_length,
                ordered_length,
                codes_length,
            ),
        )
        context.remember_index_plan(
            index,
            (
                "categorical-index",
                name,
                name_length,
                categories,
                categories_length,
                ordered,
                ordered_length,
                codes,
                codes_length,
            ),
        )
        return length
    if index_type is pd.RangeIndex:
        parts = (index.name, index.start, index.stop, index.step)
        part_lengths = tuple(_stream_length(part, context) for part in parts)
        length = _stream_envelope_length("range-index", part_lengths)
        context.remember_index_plan(index, ("range-index", parts, part_lengths))
        return length
    if index_type is pd.DatetimeIndex:
        name = index.name
        name_length = _stream_length(name, context)
        unit = index.unit.encode("ascii")
        timezone = _fingerprint_bytes(_pandas_timezone_payload(index.tz))
        frequency = _fingerprint_bytes(_frequency_payload(index.freq))
        int_values = cast("Any", index).asi8
        int_length = _stream_int_tuple_length(int_values, context)
        parts = (
            name_length,
            len(unit),
            len(timezone),
            len(frequency),
            int_length,
        )
        length = _stream_envelope_length("datetime-index", parts)
        context.remember_index_plan(
            index,
            (
                "datetime-index",
                name,
                name_length,
                unit,
                timezone,
                frequency,
                int_values,
                int_length,
            ),
        )
        return length
    if index_type is pd.TimedeltaIndex:
        name = index.name
        name_length = _stream_length(name, context)
        dtype_bytes = str(index.dtype).encode("utf-8")
        frequency = _fingerprint_bytes(_frequency_payload(index.freq))
        int_values = cast("Any", index).asi8
        int_length = _stream_int_tuple_length(int_values, context)
        parts = (
            name_length,
            len(dtype_bytes),
            len(frequency),
            int_length,
        )
        length = _stream_envelope_length("timedelta-index", parts)
        context.remember_index_plan(
            index,
            (
                "timedelta-index",
                name,
                name_length,
                dtype_bytes,
                frequency,
                int_values,
                int_length,
            ),
        )
        return length
    if index_type is not pd.Index:
        raise _FingerprintNotStreamable
    dtype = index.dtype
    if not _stream_dtype_is_plain(dtype) or dtype.kind not in "biufcMmOSU":
        raise _FingerprintNotStreamable
    values = index._values
    name = index.name
    name_length = _stream_length(name, context)
    dtype_bytes = _fingerprint_dtype(dtype)
    value_length = _stream_tuple_array_length(
        values, cast("np.dtype[Any]", dtype), context
    )
    length = _stream_envelope_length(
        "index",
        (
            name_length,
            len(dtype_bytes),
            value_length,
        ),
    )
    context.remember_index_plan(
        index,
        ("index", name, name_length, dtype_bytes, values, dtype, value_length),
    )
    return length


def _stream_index_length(index: pd.Index, context: _FingerprintStreamContext) -> int:
    cached = context.node_length(index)
    if cached is not None:
        return cached
    length = _stream_index_length_uncached(index, context)
    context.remember_node(index, length)
    return length


def _stream_int_tuple_length(values: Any, context: _FingerprintStreamContext) -> int:
    if type(values) is np.ndarray and values.dtype.kind in "iu":
        cached = context.int_tuple_length(values)
        if cached is not None:
            return cached
        length = 9 + 23 * len(values) + _stream_integer_decimal_width_sum(values)
        context.remember_int_tuple_length(values, length)
        return length
    length = 9
    for item in values:
        encoded_length = len(context.scalar_bytes(int(item)))
        length += 8 + encoded_length
    return length


def _stream_index_bytes_tuple_length(
    levels: Sequence[pd.Index], context: _FingerprintStreamContext
) -> int:
    length = 9
    for level in levels:
        index_length = _stream_index_length(level, context)
        length += 8 + 17 + index_length
    return length


def _stream_codes_tuple_length(
    codes: Sequence[Any], context: _FingerprintStreamContext
) -> int:
    length = 9
    for code_array in codes:
        code_length = _stream_int_tuple_length(code_array, context)
        length += 8 + code_length
    return length


def _stream_values_length(value: pd.Series, context: _FingerprintStreamContext) -> int:
    values, dtype = _stream_series_values(value)
    return _stream_tuple_array_length(values, dtype, context)


def _stream_pandas_length(
    value: pd.Series | pd.DataFrame, context: _FingerprintStreamContext
) -> int:
    if type(value) is pd.Series:
        values, dtype = _stream_series_values(value)
        name_length = _stream_length(value.name, context)
        index_length = _stream_index_length(value.index, context)
        dtype_bytes = _fingerprint_dtype(value.dtype)
        attrs_length = _stream_length(value.attrs, context)
        flags_length = _stream_length(value.flags.allows_duplicate_labels, context)
        values_length = _stream_tuple_array_length(values, dtype, context)
        length = _stream_envelope_length(
            "series",
            (
                name_length,
                index_length,
                len(dtype_bytes),
                attrs_length,
                flags_length,
                values_length,
            ),
        )
        context.remember_pandas_plan(
            value,
            (
                "series",
                values,
                dtype,
                dtype_bytes,
                name_length,
                index_length,
                attrs_length,
                flags_length,
                values_length,
            ),
        )
        return length
    if type(value) is not pd.DataFrame:
        raise _FingerprintNotStreamable
    index_length = _stream_index_length(value.index, context)
    columns_length = _stream_index_length(value.columns, context)
    attrs_length = _stream_length(value.attrs, context)
    flags_length = _stream_length(value.flags.allows_duplicate_labels, context)
    tuple_columns_length = 9
    column_plan: list[tuple[pd.Series, Any, Any, bytes, int, int]] = []
    for position in range(value.shape[1]):
        column = value.iloc[:, position]
        values, dtype = _stream_series_values(column)
        dtype_bytes = _fingerprint_dtype(column.dtype)
        values_length = _stream_tuple_array_length(values, dtype, context)
        column_length = _stream_envelope_length(
            "dataframe-column", (len(dtype_bytes), values_length)
        )
        tuple_columns_length += 8 + 17 + column_length
        column_plan.append(
            (column, values, dtype, dtype_bytes, values_length, column_length)
        )
    length = _stream_envelope_length(
        "dataframe",
        (
            index_length,
            columns_length,
            attrs_length,
            flags_length,
            tuple_columns_length,
        ),
    )
    context.remember_pandas_plan(
        value,
        (
            "dataframe",
            index_length,
            columns_length,
            attrs_length,
            flags_length,
            tuple_columns_length,
            column_plan,
        ),
    )
    return length


def _stream_mapping_items(
    value: object, context: _FingerprintStreamContext
) -> list[tuple[object, object]]:
    if type(value) is dict:
        mapping = cast("dict[object, object]", value)
    elif type(value) is MappingProxyType:
        mapping = cast("Mapping[object, object]", value)
    elif type(value) is globals().get("_FrozenMapping"):
        # This repository-owned immutable mapping returns deep snapshots through
        # its public API. Its private backing proxy is the same canonical data
        # and avoids copying every table just to fingerprint it.
        mapping = cast(
            "Mapping[object, object]",
            object.__getattribute__(value, "_FrozenMapping__data"),
        )
    else:
        raise _FingerprintNotStreamable
    items = list(mapping.items())
    if type(value) is not globals().get("_FrozenMapping") and any(
        type(key) not in (str, int) for key, _ in items
    ):
        raise _FingerprintNotStreamable
    encoded_keys = [(context.scalar_bytes(key), key, item) for key, item in items]
    # The legacy encoder sorts complete mapping-entry envelopes. Every entry
    # has the same tag prefix, followed by the eight-byte key-part length and
    # encoded key, so unique key encodings establish the same order as sorting
    # the complete entries. Equal encodings are rejected because legacy then
    # compares the encoded values too.
    encoded_keys.sort(key=lambda row: (len(row[0]), row[0]))
    if any(
        encoded_keys[i - 1][0] == encoded_keys[i][0]
        for i in range(1, len(encoded_keys))
    ):
        raise _FingerprintNotStreamable
    return [(key, item) for _, key, item in encoded_keys]


def _stream_mapping_length(value: object, context: _FingerprintStreamContext) -> int:
    entries: list[int] = []
    for key, item in _stream_mapping_items(value, context):
        key_length = context.scalar_bytes(key)
        item_length = _stream_length(item, context)
        entries.append(
            _stream_envelope_length("mapping-entry", (len(key_length), item_length))
        )
    return _stream_envelope_length("mapping", entries)


def _stream_tuple_length(
    value: tuple[object, ...] | list[object], context: _FingerprintStreamContext
) -> int:
    tag = "tuple" if type(value) is tuple else "list"
    length = 4 + len(tag.encode("utf-8"))
    for item in value:
        length += 8 + _stream_length(item, context)
    return length


def _stream_dataclass_length(value: object, context: _FingerprintStreamContext) -> int:
    class_fields = fields(value)
    parts = [
        len(type(value).__module__.encode("utf-8")),
        len(type(value).__qualname__.encode("utf-8")),
    ]
    for item in class_fields:
        parts.append(len(item.name.encode("utf-8")))
        parts.append(_stream_length(_stream_dataclass_field(value, item.name), context))
    return _stream_envelope_length("dataclass", parts)


def _stream_dataclass_field(value: object, name: str) -> object:
    value_type = type(value)
    standard_getattribute = value_type.__getattribute__ is object.__getattribute__
    frozen_model = value_type in {
        globals().get("FactorGroupAnalysis"),
        globals().get("EventAnalysisModel"),
        globals().get("FactorAnalysisModel"),
    }
    if not standard_getattribute and not frozen_model:
        raise _FingerprintNotStreamable

    descriptor: object | None = None
    for owner in value_type.__mro__:
        if name in owner.__dict__:
            descriptor = owner.__dict__[name]
            break
    if type(descriptor) is MemberDescriptorType:
        return object.__getattribute__(value, name)
    if descriptor is not None and hasattr(type(descriptor), "__get__"):
        raise _FingerprintNotStreamable
    try:
        instance_values = object.__getattribute__(value, "__dict__")
    except AttributeError:
        raise _FingerprintNotStreamable from None
    if type(instance_values) is not dict or name not in instance_values:
        raise _FingerprintNotStreamable
    return instance_values[name]


def _stream_length(value: object, context: _FingerprintStreamContext) -> int:
    cached_length = context.node_length(value)
    if cached_length is not None:
        return cached_length
    if type(value) is pd.Series or type(value) is pd.DataFrame:
        length = _stream_pandas_length(cast("pd.Series | pd.DataFrame", value), context)
    elif isinstance(value, pd.Index):
        length = _stream_index_length(value, context)
    elif value is pd.NA or value is pd.NaT:
        return len(_fingerprint_bytes(value))
    elif type(value) in {tuple, list}:
        length = _stream_tuple_length(value, context)
    elif type(value) in {dict, MappingProxyType} or type(value) is globals().get(
        "_FrozenMapping"
    ):
        length = _stream_mapping_length(value, context)
    elif is_dataclass(value) and not isinstance(value, type):
        if type(value).__dict__.get("__dataclass_fields__") is None:
            raise _FingerprintNotStreamable
        length = _stream_dataclass_length(value, context)
    else:
        return len(context.scalar_bytes(value))
    context.remember_node(value, length)
    return length


def _stream_emit_envelope(
    writer: Any, tag: str, parts: Sequence[tuple[int, Any]]
) -> None:
    _stream_envelope_start(writer, tag)
    for length, emit_part in parts:
        _stream_part_start(writer, length)
        emit_part()


def _stream_emit_tuple_array(
    values: Any, dtype: np.dtype[Any], writer: Any, context: _FingerprintStreamContext
) -> None:
    _stream_tuple_array(values, dtype, writer, context)


def _stream_emit_part_value(
    value: object, writer: Any, context: _FingerprintStreamContext
) -> None:
    if (
        type(value) in _STREAM_SCALAR_TYPES
        or type(value) in {str, int}
        or value is pd.NA
        or value is pd.NaT
    ):
        encoded = context.scalar_bytes(value)
        _stream_part_start(writer, len(encoded))
        writer.write(encoded)
        return
    length = _stream_length(value, context)
    _stream_part_start(writer, length)
    _stream_emit(value, writer, context)


def _stream_emit_index(
    index: pd.Index, writer: Any, context: _FingerprintStreamContext
) -> None:
    index_type = type(index)
    if index_type is pd.MultiIndex:
        plan = context.index_plan(index)
        if plan is None:
            _stream_index_length(index, context)
            plan = context.index_plan(index)
        (
            _,
            names,
            names_length,
            levels,
            level_lengths,
            levels_length,
            code_arrays,
            code_lengths,
            codes_length,
        ) = plan
        _stream_envelope_start(writer, "multiindex")
        _stream_part_start(writer, names_length)
        _stream_emit(names, writer, context)
        _stream_part_start(writer, levels_length)
        writer.write((5).to_bytes(4, "big"))
        writer.write(b"tuple")
        for level, level_length in zip(levels, level_lengths, strict=True):
            _stream_part_start(writer, 17 + level_length)
            _stream_envelope_start(writer, "bytes")
            _stream_part_start(writer, level_length)
            _stream_emit_index(level, writer, context)
        _stream_part_start(writer, codes_length)
        writer.write((5).to_bytes(4, "big"))
        writer.write(b"tuple")
        for code_array, code_length in zip(code_arrays, code_lengths, strict=True):
            _stream_part_start(writer, code_length)
            _stream_emit_int_tuple(code_array, writer, context)
        return
    if index_type is pd.CategoricalIndex:
        plan = context.index_plan(index)
        if plan is None:
            _stream_index_length(index, context)
            plan = context.index_plan(index)
        (
            _,
            name,
            name_length,
            categories,
            categories_length,
            ordered,
            ordered_length,
            codes,
            codes_length,
        ) = plan
        _stream_envelope_start(writer, "categorical-index")
        _stream_part_start(writer, name_length)
        _stream_emit(name, writer, context)
        _stream_part_start(writer, categories_length)
        _stream_emit_index(categories, writer, context)
        _stream_part_start(writer, ordered_length)
        _stream_emit(ordered, writer, context)
        _stream_part_start(writer, codes_length)
        _stream_emit_int_tuple(codes, writer, context)
        return
    if index_type is pd.RangeIndex:
        plan = context.index_plan(index)
        if plan is None:
            _stream_index_length(index, context)
            plan = context.index_plan(index)
        _, parts, part_lengths = plan
        _stream_emit_envelope(
            writer,
            "range-index",
            tuple(
                (
                    part_length,
                    lambda part=part: _stream_emit(part, writer, context),
                )
                for part, part_length in zip(parts, part_lengths, strict=True)
            ),
        )
        return
    if index_type in {pd.DatetimeIndex, pd.TimedeltaIndex}:
        is_datetime = index_type is pd.DatetimeIndex
        plan = context.index_plan(index)
        if plan is None:
            _stream_index_length(index, context)
            plan = context.index_plan(index)
        if is_datetime:
            _, name, name_length, unit, encoded_timezone, frequency_payload, int_values, int_length = plan
        else:
            _, name, name_length, dtype_bytes, frequency_payload, int_values, int_length = plan
        tag = "datetime-index" if is_datetime else "timedelta-index"
        parts: list[tuple[int, Any]] = []
        parts.append(
            (
                name_length,
                lambda: _stream_emit(name, writer, context),
            )
        )
        if is_datetime:
            parts.append(
                (
                    len(unit),
                    lambda: writer.write(unit),
                )
            )
            parts.append(
                (len(encoded_timezone), lambda: writer.write(encoded_timezone))
            )
            parts.append(
                (len(frequency_payload), lambda: writer.write(frequency_payload))
            )
        else:
            parts.append((len(dtype_bytes), lambda: writer.write(dtype_bytes)))
            parts.append(
                (len(frequency_payload), lambda: writer.write(frequency_payload))
            )
        parts.append(
            (int_length, lambda: _stream_emit_int_tuple(int_values, writer, context))
        )
        _stream_emit_envelope(writer, tag, parts)
        return
    if index_type is not pd.Index:
        raise _FingerprintNotStreamable
    plan = context.index_plan(index)
    if plan is None:
        _stream_index_length(index, context)
        plan = context.index_plan(index)
    _, name, name_length, dtype_bytes, values, dtype, values_length = plan
    _stream_envelope_start(writer, "index")
    _stream_part_start(writer, name_length)
    _stream_emit(name, writer, context)
    _stream_part_start(writer, len(dtype_bytes))
    writer.write(dtype_bytes)
    _stream_part_start(writer, values_length)
    _stream_emit_tuple_array(values, dtype, writer, context)


def _stream_emit_int_tuple(
    values: Any, writer: Any, context: _FingerprintStreamContext
) -> None:
    if type(values) is np.ndarray and values.dtype.kind in "iu":
        _stream_write_integer_tuple(values, writer)
        return
    writer.write((5).to_bytes(4, "big"))
    writer.write(b"tuple")
    _stream_write_scalar_part_batch((int(item) for item in values), writer, context)


def _stream_emit_mapping(
    value: object, writer: Any, context: _FingerprintStreamContext
) -> None:
    items = _stream_mapping_items(value, context)
    entry_lengths = [
        _stream_envelope_length(
            "mapping-entry",
            (len(context.scalar_bytes(key)), _stream_length(item, context)),
        )
        for key, item in items
    ]
    _stream_envelope_start(writer, "mapping")
    for (key, item), entry_length in zip(items, entry_lengths, strict=True):
        _stream_part_start(writer, entry_length)
        key_bytes = context.scalar_bytes(key)
        item_length = _stream_length(item, context)
        _stream_envelope_start(writer, "mapping-entry")
        _stream_part_start(writer, len(key_bytes))
        writer.write(key_bytes)
        _stream_part_start(writer, item_length)
        _stream_emit(item, writer, context)


def _stream_emit_tuple(
    value: tuple[object, ...] | list[object],
    tag: str,
    writer: Any,
    context: _FingerprintStreamContext,
) -> None:
    _stream_envelope_start(writer, tag)
    _stream_write_part_values(value, writer, context)


def _stream_emit_dataclass(
    value: object, writer: Any, context: _FingerprintStreamContext
) -> None:
    parts: list[tuple[int, Any]] = [
        (
            len(type(value).__module__.encode("utf-8")),
            lambda: writer.write(type(value).__module__.encode("utf-8")),
        ),
        (
            len(type(value).__qualname__.encode("utf-8")),
            lambda: writer.write(type(value).__qualname__.encode("utf-8")),
        ),
    ]
    for item in fields(value):
        name_bytes = item.name.encode("utf-8")
        field_value = _stream_dataclass_field(value, item.name)
        field_length = _stream_length(field_value, context)
        parts.append(
            (len(name_bytes), lambda name_bytes=name_bytes: writer.write(name_bytes))
        )
        parts.append(
            (
                field_length,
                lambda field_value=field_value: _stream_emit(
                    field_value, writer, context
                ),
            )
        )
    _stream_emit_envelope(writer, "dataclass", parts)


def _stream_emit_scalar_bytes(
    value: object, encoded: bytes, writer: Any, context: _FingerprintStreamContext
) -> None:
    writer.write(encoded)


def _stream_emit_pandas(
    value: pd.Series | pd.DataFrame, writer: Any, context: _FingerprintStreamContext
) -> None:
    plan = context.pandas_plan(value)
    if plan is None:
        _stream_pandas_length(value, context)
        plan = context.pandas_plan(value)
    if type(value) is pd.Series:
        (
            _,
            values,
            dtype,
            dtype_bytes,
            name_length,
            index_length,
            attrs_length,
            flags_length,
            values_length,
        ) = plan
        _stream_envelope_start(writer, "series")
        _stream_part_start(writer, name_length)
        _stream_emit(value.name, writer, context)
        _stream_part_start(writer, index_length)
        _stream_emit_index(value.index, writer, context)
        _stream_part_start(writer, len(dtype_bytes))
        writer.write(dtype_bytes)
        _stream_part_start(writer, attrs_length)
        _stream_emit(value.attrs, writer, context)
        flags = value.flags.allows_duplicate_labels
        _stream_part_start(writer, flags_length)
        _stream_emit(flags, writer, context)
        _stream_part_start(writer, values_length)
        _stream_emit_tuple_array(values, dtype, writer, context)
        return
    (
        _,
        index_length,
        columns_index_length,
        attrs_length,
        flags_length,
        tuple_columns_length,
        column_plan,
    ) = plan
    _stream_envelope_start(writer, "dataframe")
    _stream_part_start(writer, index_length)
    _stream_emit_index(value.index, writer, context)
    _stream_part_start(writer, columns_index_length)
    _stream_emit_index(value.columns, writer, context)
    _stream_part_start(writer, attrs_length)
    _stream_emit(value.attrs, writer, context)
    flags = value.flags.allows_duplicate_labels
    _stream_part_start(writer, flags_length)
    _stream_emit(flags, writer, context)
    _stream_part_start(writer, tuple_columns_length)
    _stream_envelope_start(writer, "tuple")
    for (
        column,
        values,
        dtype,
        dtype_bytes,
        values_length,
        encoded_length,
    ) in column_plan:
        bytes_length = 17 + encoded_length
        _stream_part_start(writer, bytes_length)
        _stream_envelope_start(writer, "bytes")
        _stream_part_start(writer, encoded_length)
        _stream_envelope_start(writer, "dataframe-column")
        _stream_part_start(writer, len(dtype_bytes))
        writer.write(dtype_bytes)
        _stream_part_start(writer, values_length)
        _stream_emit_tuple_array(values, dtype, writer, context)


def _stream_emit(
    value: object, writer: Any, context: _FingerprintStreamContext
) -> None:
    if type(value) is pd.Series or type(value) is pd.DataFrame:
        _stream_emit_pandas(cast("pd.Series | pd.DataFrame", value), writer, context)
        return
    if isinstance(value, pd.Index):
        _stream_emit_index(value, writer, context)
        return
    if value is pd.NA or value is pd.NaT:
        writer.write(_fingerprint_bytes(value))
        return
    if type(value) is tuple:
        _stream_emit_tuple(cast("tuple[object, ...]", value), "tuple", writer, context)
        return
    if type(value) is list:
        _stream_emit_tuple(cast("list[object]", value), "list", writer, context)
        return
    if type(value) in {dict, MappingProxyType} or type(value) is globals().get(
        "_FrozenMapping"
    ):
        _stream_emit_mapping(value, writer, context)
        return
    if is_dataclass(value) and not isinstance(value, type):
        _stream_emit_dataclass(value, writer, context)
        return
    encoded = context.scalar_bytes(value)
    _stream_emit_scalar_bytes(value, encoded, writer, context)


def _stream_fingerprint(value: object, writer: Any) -> int:
    """Write the exact v1 encoding to ``writer`` without building a parent bytes object."""

    context = _FingerprintStreamContext()
    encoded_length = _stream_length(value, context)
    _stream_emit(value, writer, context)
    return encoded_length


def fingerprint_value(value: object) -> str:
    """Return a deterministic, lossless SHA-256 fingerprint for model data."""

    digest = sha256()
    writer = _FingerprintDigestWriter(digest)
    try:
        _stream_fingerprint(value, writer)
    except _FingerprintNotStreamable:
        return sha256(_fingerprint_bytes(value)).hexdigest()
    writer.finish()
    return digest.hexdigest()


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
    if isinstance(value, FactorPortfolioInputs):
        return FactorPortfolioInputs(
            returns=cast("pd.Series", snapshot_pandas(value.returns)),
            positions=cast("pd.DataFrame", snapshot_pandas(value.positions)),
            benchmark_returns=None
            if value.benchmark_returns is None
            else cast("pd.Series", snapshot_pandas(value.benchmark_returns)),
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
    include_portfolio_inputs: bool = True
    portfolio_capital: int | float | None = None
    portfolio_benchmark_period: str = "1D"
    fingerprint: str = field(init=False)

    def __post_init__(self) -> None:
        """Normalize public options before computing the frozen config digest."""

        periods = _typed_sequence(self.periods, "periods")
        turnover_periods = _typed_sequence(self.turnover_periods, "turnover_periods")
        time_aggregation = _typed_sequence(self.time_aggregation, "time_aggregation")
        for name in ("long_short", "group_neutral", "equal_weight", "by_group", "include_portfolio_inputs"):
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
        if not isinstance(self.portfolio_benchmark_period, str):
            raise TypeError("portfolio_benchmark_period must be a string")
        capital = self.portfolio_capital
        if capital is not None:
            if not isinstance(capital, (int, float)) or isinstance(capital, bool):
                raise TypeError("portfolio_capital must be an int, float, or None")
            if isinstance(capital, int):
                try:
                    exactly_representable = int(float(capital)) == capital
                except OverflowError:
                    exactly_representable = False
                if not exactly_representable:
                    raise ValueError("portfolio_capital integer must be representable exactly by float arithmetic")

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
            "include_portfolio_inputs": self.include_portfolio_inputs,
            "portfolio_capital": self.portfolio_capital,
            "portfolio_benchmark_period": self.portfolio_benchmark_period,
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
    quantile_cumulative_returns: Mapping[str, pd.DataFrame]
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
    portfolio_inputs: FactorPortfolioInputs | None
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
            "quantile_cumulative_returns",
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
            "portfolio_inputs",
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
