"""Versioned JSON serialization for AnalysisResult with NaN/inf policy.

The wire format is forward-compatible: readers ignore unknown top-level keys
and fail only on a missing ``schema_version`` or ``status``.  NaN and infinity
are encoded as JSON literals (``NaN``/``Infinity``/``-Infinity``), which the
``json`` module round-trips losslessly when ``allow_nan`` is enabled.
"""

from __future__ import annotations

import json
from typing import Any

import numpy as np
import pandas as pd

from fincore.exceptions import FincoreError
from fincore.results.base import AnalysisResult, ResultMetadata

SCHEMA_VERSION = "1.0"

__all__ = ["SCHEMA_VERSION", "from_json", "to_json"]


def _encode_value(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, np.ndarray):
        return {"__type__": "ndarray", "dtype": str(value.dtype), "values": value.tolist()}
    if isinstance(value, pd.Series):
        return {
            "__type__": "series",
            "index": [str(i) for i in value.index],
            "dtype": str(value.dtype),
            "values": value.tolist(),
        }
    if isinstance(value, pd.DataFrame):
        return {
            "__type__": "frame",
            "columns": [str(c) for c in value.columns],
            "index": [str(i) for i in value.index],
            "values": value.to_numpy().tolist(),
        }
    return str(value)


def _decode_value(payload: Any) -> Any:
    if not isinstance(payload, dict) or "__type__" not in payload:
        return payload
    kind = payload["__type__"]
    if kind == "ndarray":
        return np.asarray(payload["values"])
    if kind == "series":
        return pd.Series(payload["values"], index=payload["index"])
    if kind == "frame":
        return pd.DataFrame(payload["values"], columns=payload["columns"], index=payload["index"])
    return payload


def _metadata_to_dict(metadata: ResultMetadata | None) -> dict[str, Any] | None:
    if metadata is None:
        return None
    return {
        "operation": metadata.operation,
        "profile": metadata.profile,
        "schema_version": metadata.schema_version,
        "status": metadata.status,
        "units": metadata.units,
        "frequency": metadata.frequency,
        "sign": metadata.sign,
        "input_digest": metadata.input_digest,
        "config_digest": metadata.config_digest,
        "software": metadata.software,
        "dependency_provenance": metadata.dependency_provenance,
        "warnings": list(metadata.warnings),
        "diagnostics": metadata.diagnostics,
        "uncertainty": metadata.uncertainty,
    }


def _metadata_from_dict(payload: dict[str, Any]) -> ResultMetadata:
    return ResultMetadata(
        operation=payload["operation"],
        profile=payload["profile"],
        schema_version=payload.get("schema_version", SCHEMA_VERSION),
        status=payload.get("status", "success"),
        units=payload.get("units"),
        frequency=payload.get("frequency"),
        sign=payload.get("sign"),
        input_digest=payload.get("input_digest", ""),
        config_digest=payload.get("config_digest", ""),
        software=payload.get("software", {}),
        dependency_provenance=payload.get("dependency_provenance", {}),
        warnings=tuple(payload.get("warnings", [])),
        diagnostics=payload.get("diagnostics", {}),
        uncertainty=payload.get("uncertainty"),
    )


def to_json(result: AnalysisResult[Any]) -> str:
    """Serialize an AnalysisResult to a versioned JSON string."""
    payload = {
        "schema_version": SCHEMA_VERSION,
        "status": result.status,
        "value": _encode_value(result.value),
        "metadata": _metadata_to_dict(result.metadata),
        "error": str(result.error) if result.error is not None else None,
    }
    return json.dumps(payload, allow_nan=True, indent=2, sort_keys=True, default=str)


def from_json(text: str) -> AnalysisResult[Any]:
    """Deserialize an AnalysisResult, tolerating unknown future keys."""
    payload = json.loads(text)
    if "schema_version" not in payload:
        raise ValueError("missing schema_version in result payload")
    if "status" not in payload:
        raise ValueError("missing status in result payload")
    metadata = _metadata_from_dict(payload["metadata"]) if payload.get("metadata") else None
    value = _decode_value(payload.get("value"))
    error: FincoreError | None = None
    if payload.get("error"):
        error = FincoreError(payload["error"])
    return AnalysisResult(status=payload["status"], value=value, metadata=metadata, error=error)
