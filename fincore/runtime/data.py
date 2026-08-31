"""Immutable, copy-on-ingest input snapshots for runtime execution."""

from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import dataclass, field
from datetime import date, datetime
from types import MappingProxyType
from typing import Any, Mapping

import numpy as np
import pandas as pd


def _copy_value(value: Any) -> Any:
    if isinstance(value, (pd.Series, pd.DataFrame, pd.Index, np.ndarray)):
        return value.copy()
    if isinstance(value, Mapping):
        return {key: _copy_value(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(_copy_value(item) for item in value)
    if isinstance(value, list):
        return [_copy_value(item) for item in value]
    return copy.deepcopy(value)


def _label(value: Any) -> str:
    return f"{type(value).__module__}.{type(value).__qualname__}:{value!r}"


def _index_descriptor(index: pd.Index) -> dict[str, Any]:
    return {
        "type": type(index).__qualname__,
        "dtype": str(index.dtype),
        "names": [_label(name) for name in index.names],
        "timezone": str(getattr(index, "tz", None) or ""),
        "frequency": str(getattr(index, "freqstr", None) or ""),
    }


def _pandas_descriptor(value: pd.Series | pd.DataFrame) -> dict[str, Any]:
    if isinstance(value, pd.Series):
        schema: dict[str, Any] = {"kind": "series", "name": _label(value.name), "dtype": str(value.dtype)}
    else:
        schema = {
            "kind": "frame",
            "columns": [_label(column) for column in value.columns],
            "dtypes": [str(dtype) for dtype in value.dtypes],
        }
    values = pd.util.hash_pandas_object(value, index=True, categorize=True).to_numpy(dtype="uint64", copy=False)
    schema["index"] = _index_descriptor(value.index)
    schema["values_sha256"] = hashlib.sha256(values.tobytes()).hexdigest()
    return schema


def _descriptor(value: Any) -> Any:
    if isinstance(value, (pd.Series, pd.DataFrame)):
        return _pandas_descriptor(value)
    if isinstance(value, pd.Index):
        values = pd.util.hash_pandas_object(value.to_series(index=value), index=True, categorize=True).to_numpy(
            dtype="uint64", copy=False
        )
        return {
            "kind": "index",
            "schema": _index_descriptor(value),
            "values_sha256": hashlib.sha256(values.tobytes()).hexdigest(),
        }
    if isinstance(value, np.ndarray):
        return {
            "kind": "ndarray",
            "dtype": str(value.dtype),
            "shape": value.shape,
            "values_sha256": hashlib.sha256(value.tobytes()).hexdigest(),
        }
    if isinstance(value, Mapping):
        if any(not isinstance(key, str) for key in value):
            raise TypeError("nested snapshot mappings must use string keys")
        return {key: _descriptor(value[key]) for key in sorted(value)}
    if isinstance(value, (tuple, list)):
        return [_descriptor(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, (date, datetime, pd.Timestamp)):
        return {"kind": type(value).__qualname__, "value": value.isoformat()}
    raise TypeError(f"unsupported snapshot input type: {type(value).__module__}.{type(value).__qualname__}")


def _digest(inputs: Mapping[str, Any]) -> str:
    payload = json.dumps(_descriptor(inputs), sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


@dataclass(frozen=True, slots=True)
class AnalysisSnapshot:
    """Copy-on-ingest inputs with a deterministic digest and safe materialization."""

    _inputs: Mapping[str, Any] = field(repr=False)
    digest: str

    @classmethod
    def from_inputs(cls, inputs: Mapping[str, Any]) -> AnalysisSnapshot:
        if not isinstance(inputs, Mapping):
            raise TypeError("inputs must be a mapping")
        if not inputs:
            raise ValueError("inputs must not be empty")
        if any(not isinstance(key, str) or not key for key in inputs):
            raise TypeError("input names must be non-empty strings")
        copied = {key: _copy_value(value) for key, value in inputs.items()}
        return cls(_inputs=MappingProxyType(copied), digest=_digest(copied))

    def materialize(self) -> Mapping[str, Any]:
        """Return independent copies so a caller cannot mutate the snapshot."""
        return MappingProxyType({key: _copy_value(value) for key, value in self._inputs.items()})
