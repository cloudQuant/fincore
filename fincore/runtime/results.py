"""Portable result values and deterministic orchestration provenance."""

from __future__ import annotations

import base64
import copy
import hashlib
import json
import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import date, datetime
from decimal import Decimal
from types import MappingProxyType
from typing import Any

import numpy as np
import pandas as pd

RESULT_SCHEMA_VERSION = "0.5"
_TYPE_KEY = "$fincore_type"
_VOLATILE_METADATA_KEYS = frozenset(
    {
        "duration_ns",
        "finished_ns",
        "run_id",
        "started_ns",
        "temporary_path",
        "workspace_path",
    }
)


def _freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        if any(not isinstance(key, str) for key in value):
            raise TypeError("result metadata keys must be strings")
        return MappingProxyType({key: _freeze(item) for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(item) for item in value)
    if isinstance(value, set):
        return frozenset(_freeze(item) for item in value)
    return value


def _encode_mapping(value: Mapping[str, Any]) -> dict[str, Any]:
    if any(not isinstance(key, str) for key in value):
        raise TypeError("result mappings must use string keys")
    return {
        _TYPE_KEY: "mapping",
        "items": [[key, _encode(value[key])] for key in sorted(value)],
    }


def _encode(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if math.isnan(value):
            return {_TYPE_KEY: "float", "value": "nan"}
        if math.isinf(value):
            return {_TYPE_KEY: "float", "value": "inf" if value > 0 else "-inf"}
        return value
    if isinstance(value, np.generic):
        return {
            _TYPE_KEY: "numpy_scalar",
            "dtype": str(value.dtype),
            "value": _encode(value.item()),
        }
    if isinstance(value, np.ndarray):
        return {
            _TYPE_KEY: "numpy_ndarray",
            "dtype": str(value.dtype),
            "shape": list(value.shape),
            "values": _encode(value.tolist()),
        }
    if value is pd.NA:
        return {_TYPE_KEY: "pandas_na"}
    if isinstance(value, pd.Timestamp):
        return {_TYPE_KEY: "pandas_timestamp", "value": value.isoformat()}
    if isinstance(value, pd.Timedelta):
        return {_TYPE_KEY: "pandas_timedelta", "value": value.isoformat()}
    if isinstance(value, pd.Index):
        return {
            _TYPE_KEY: "pandas_index",
            "dtype": str(value.dtype),
            "name": _encode(value.name),
            "values": _encode(value.tolist()),
        }
    if isinstance(value, pd.Series):
        return {
            _TYPE_KEY: "pandas_series",
            "dtype": str(value.dtype),
            "name": _encode(value.name),
            "index": _encode(value.index),
            "values": _encode(value.tolist()),
        }
    if isinstance(value, pd.DataFrame):
        return {
            _TYPE_KEY: "pandas_frame",
            "columns": _encode(value.columns),
            "dtypes": [str(dtype) for dtype in value.dtypes],
            "index": _encode(value.index),
            "values": [_encode(value[column].tolist()) for column in value.columns],
        }
    if isinstance(value, Mapping):
        return _encode_mapping(value)
    if isinstance(value, tuple):
        return {_TYPE_KEY: "tuple", "items": [_encode(item) for item in value]}
    if isinstance(value, list):
        return [_encode(item) for item in value]
    if isinstance(value, (bytes, bytearray)):
        return {
            _TYPE_KEY: "bytes",
            "data": base64.b64encode(bytes(value)).decode("ascii"),
        }
    if isinstance(value, Decimal):
        return {_TYPE_KEY: "decimal", "value": str(value)}
    if isinstance(value, datetime):
        return {_TYPE_KEY: "datetime", "value": value.isoformat()}
    if isinstance(value, date):
        return {_TYPE_KEY: "date", "value": value.isoformat()}
    raise TypeError(f"unsupported result value type: {type(value).__module__}.{type(value).__qualname__}")


def _decode(value: Any) -> Any:
    if isinstance(value, list):
        return [_decode(item) for item in value]
    if not isinstance(value, dict):
        return value
    type_name = value.get(_TYPE_KEY)
    if type_name is None:
        return {key: _decode(item) for key, item in value.items()}
    if type_name == "mapping":
        items = value.get("items")
        if not isinstance(items, list):
            raise ValueError("serialized mapping items must be a list")
        decoded: dict[str, Any] = {}
        for item in items:
            if not isinstance(item, list) or len(item) != 2 or not isinstance(item[0], str):
                raise ValueError("serialized mapping item must contain a string key and value")
            decoded[item[0]] = _decode(item[1])
        return decoded
    if type_name == "tuple":
        items = value.get("items")
        if not isinstance(items, list):
            raise ValueError("serialized tuple items must be a list")
        return tuple(_decode(item) for item in items)
    if type_name == "float":
        encoded = value.get("value")
        if encoded == "nan":
            return float("nan")
        if encoded == "inf":
            return float("inf")
        if encoded == "-inf":
            return float("-inf")
        raise ValueError("serialized float value is invalid")
    if type_name == "numpy_scalar":
        return np.asarray(_decode(value.get("value")), dtype=value.get("dtype")).item()
    if type_name == "numpy_ndarray":
        shape = value.get("shape")
        if not isinstance(shape, list) or any(not isinstance(item, int) for item in shape):
            raise ValueError("serialized array shape must be a list of integers")
        return np.asarray(_decode(value.get("values")), dtype=value.get("dtype")).reshape(shape)
    if type_name == "pandas_na":
        return pd.NA
    if type_name == "pandas_timestamp":
        encoded = value.get("value")
        if encoded is None:
            raise ValueError("serialized timestamp value is required")
        return pd.Timestamp(encoded)
    if type_name == "pandas_timedelta":
        encoded = value.get("value")
        if encoded is None:
            raise ValueError("serialized timedelta value is required")
        return pd.Timedelta(encoded)
    if type_name == "pandas_index":
        return pd.Index(_decode(value.get("values")), name=_decode(value.get("name")), dtype=value.get("dtype"))
    if type_name == "pandas_series":
        index = _decode(value.get("index"))
        if not isinstance(index, pd.Index):
            raise ValueError("serialized series index must decode to a pandas Index")
        return pd.Series(
            _decode(value.get("values")),
            index=index,
            name=_decode(value.get("name")),
            dtype=value.get("dtype"),
        )
    if type_name == "pandas_frame":
        columns = _decode(value.get("columns"))
        index = _decode(value.get("index"))
        encoded_values = value.get("values")
        dtypes = value.get("dtypes")
        if not isinstance(columns, pd.Index) or not isinstance(index, pd.Index):
            raise ValueError("serialized frame indexes must decode to pandas Index values")
        if not isinstance(encoded_values, list) or not isinstance(dtypes, list) or len(encoded_values) != len(columns):
            raise ValueError("serialized frame columns, dtypes, and values must align")
        frame = pd.DataFrame(
            {column: _decode(column_values) for column, column_values in zip(columns, encoded_values, strict=True)},
            index=index,
        )
        for column, dtype in zip(columns, dtypes, strict=True):
            frame[column] = frame[column].astype(dtype)
        return frame
    if type_name == "bytes":
        data = value.get("data")
        if not isinstance(data, str):
            raise ValueError("serialized bytes data must be a string")
        return base64.b64decode(data.encode("ascii"), validate=True)
    if type_name == "decimal":
        encoded = value.get("value")
        if not isinstance(encoded, str):
            raise ValueError("serialized decimal value must be a string")
        return Decimal(encoded)
    if type_name == "datetime":
        encoded = value.get("value")
        if not isinstance(encoded, str):
            raise ValueError("serialized datetime value must be a string")
        return datetime.fromisoformat(encoded)
    if type_name == "date":
        encoded = value.get("value")
        if not isinstance(encoded, str):
            raise ValueError("serialized date value must be a string")
        return date.fromisoformat(encoded)
    raise ValueError(f"unknown serialized result type: {type_name}")


def _semantic_metadata(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {
            key: _semantic_metadata(item)
            for key, item in value.items()
            if key not in _VOLATILE_METADATA_KEYS and not key.endswith("_path")
        }
    if isinstance(value, tuple):
        return tuple(_semantic_metadata(item) for item in value)
    if isinstance(value, list):
        return [_semantic_metadata(item) for item in value]
    return value


@dataclass(frozen=True, slots=True)
class Result:
    """The natural domain value plus immutable, serializable provenance."""

    value: Any
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "metadata", _freeze(self.metadata))

    def with_metadata(self, **updates: Any) -> Result:
        """Return the same natural value with explicitly updated provenance."""
        return Result(value=self.value, metadata={**self.metadata, **updates})

    def copy_for_consumer(self, **metadata_updates: Any) -> Result:
        """Return an independent natural value for session-cache consumers."""
        return Result(
            value=copy.deepcopy(self.value),
            metadata={**self.metadata, **metadata_updates},
        )

    def to_payload(self) -> dict[str, Any]:
        """Return a versioned portable schema without pickling runtime state."""
        return {
            "schema_version": RESULT_SCHEMA_VERSION,
            "value": _encode(self.value),
            "metadata": _encode(self.metadata),
        }

    def to_json(self) -> str:
        """Encode the portable payload with deterministic JSON formatting."""
        return json.dumps(self.to_payload(), sort_keys=True, separators=(",", ":"), ensure_ascii=True)

    @property
    def semantic_digest(self) -> str:
        """Hash stable value and provenance while omitting runtime-only metadata."""
        payload = {
            "schema_version": RESULT_SCHEMA_VERSION,
            "value": _encode(self.value),
            "metadata": _encode(_semantic_metadata(self.metadata)),
        }
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> Result:
        """Restore a result from :meth:`to_payload`'s stable schema."""
        if not isinstance(payload, Mapping):
            raise TypeError("serialized result payload must be a mapping")
        if payload.get("schema_version") != RESULT_SCHEMA_VERSION:
            raise ValueError(f"unsupported result schema version: {payload.get('schema_version')!r}")
        if "value" not in payload or "metadata" not in payload:
            raise ValueError("serialized result payload must contain value and metadata")
        metadata = _decode(payload["metadata"])
        if not isinstance(metadata, Mapping):
            raise ValueError("serialized result metadata must decode to a mapping")
        return cls(value=_decode(payload["value"]), metadata=metadata)

    @classmethod
    def from_json(cls, encoded: str | bytes | bytearray) -> Result:
        """Restore a result from a JSON document created by :meth:`to_json`."""
        decoded = json.loads(encoded)
        if not isinstance(decoded, Mapping):
            raise ValueError("serialized result JSON must contain an object")
        return cls.from_payload(decoded)
