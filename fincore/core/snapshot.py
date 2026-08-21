"""AnalysisSnapshot: one validated, copy-on-ingest view of analysis inputs.

The snapshot holds the validated returns/benchmark/positions/transactions plus
their semantic metadata and a stable cache key.  Caller inputs are copied on
ingest, so downstream kernels can never mutate the caller's data.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field, is_dataclass
from pathlib import Path
from typing import Any, Mapping, cast

import pandas as pd

from fincore.contracts.analysis import SeriesSemantics

__all__ = ["AnalysisSnapshot"]


def _label(value: object) -> str:
    """Return a stable type-aware representation for a pandas label."""
    return f"{type(value).__module__}.{type(value).__qualname__}:{value!r}"


def _pandas_digest(value: pd.Series | pd.DataFrame) -> str:
    """Hash values *and* schema for a pandas object.

    CSV-based digests collapse several semantically relevant distinctions (for
    example nullable dtype, index timezone and duplicate column labels).  The
    pandas value hash is paired with explicit schema metadata so changing any
    analysis input always invalidates an :class:`AnalysisSnapshot` key.
    """
    values: pd.Series | pd.DataFrame
    if isinstance(value, pd.Series):
        values = value
        metadata: dict[str, Any] = {
            "kind": "Series",
            "name": _label(value.name),
            "dtype": str(value.dtype),
        }
    elif isinstance(value, pd.DataFrame):
        values = value
        metadata = {
            "kind": "DataFrame",
            "columns": [_label(column) for column in value.columns],
            "dtypes": [str(dtype) for dtype in value.dtypes],
        }
    else:  # pragma: no cover - protected by the public constructor.
        raise TypeError(f"expected pandas Series or DataFrame, got {type(value).__name__}")

    index = value.index
    metadata["index"] = {
        "kind": type(index).__qualname__,
        "dtype": str(index.dtype),
        "names": [_label(name) for name in index.names],
        "timezone": str(getattr(index, "tz", None) or ""),
        "frequency": str(getattr(index, "freqstr", None) or ""),
    }
    rows = pd.util.hash_pandas_object(values, index=True, categorize=True).to_numpy(dtype="uint64", copy=False)
    hasher = hashlib.sha256()
    hasher.update(json.dumps(metadata, sort_keys=True, separators=(",", ":")).encode("utf-8"))
    hasher.update(rows.tobytes())
    return hasher.hexdigest()


def _json_default(value: object) -> object:
    """Encode supported configuration values deterministically."""
    if is_dataclass(value):
        return asdict(cast(Any, value))
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, set):
        return sorted(value, key=repr)
    raise TypeError(f"unsupported snapshot configuration value: {type(value).__name__}")


def _object_digest(value: object) -> str:
    payload = json.dumps(
        value,
        default=_json_default,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _input_digest(
    *,
    returns: pd.Series,
    benchmark: pd.Series | None,
    positions: pd.DataFrame | None,
    transactions: pd.DataFrame | None,
) -> str:
    """Build a digest that changes for every cache-relevant analysis input."""
    return _object_digest(
        {
            "returns": _pandas_digest(returns),
            "benchmark": _pandas_digest(benchmark) if benchmark is not None else None,
            "positions": _pandas_digest(positions) if positions is not None else None,
            "transactions": _pandas_digest(transactions) if transactions is not None else None,
        }
    )


@dataclass(frozen=True)
class AnalysisSnapshot:
    """An immutable, validated analysis input snapshot."""

    returns: pd.Series
    benchmark: pd.Series | None = None
    positions: pd.DataFrame | None = None
    transactions: pd.DataFrame | None = None
    semantics: SeriesSemantics = field(default_factory=SeriesSemantics)
    profile: str = "enhanced_v1"
    data_digest: str = ""
    config_digest: str = ""
    backend: str = "pandas"
    overlay_generation: int = 0
    overlay_digest: str = ""
    operation_version: str = "1"
    cache_key: str = ""

    @classmethod
    def from_data(
        cls,
        returns: pd.Series,
        *,
        benchmark: pd.Series | None = None,
        positions: pd.DataFrame | None = None,
        transactions: pd.DataFrame | None = None,
        semantics: SeriesSemantics | None = None,
        profile: str = "enhanced_v1",
        backend: str = "pandas",
        overlay_generation: int = 0,
        overlay_digest: str = "",
        operation_version: str = "1",
        config: Mapping[str, Any] | None = None,
    ) -> AnalysisSnapshot:
        if not isinstance(returns, pd.Series):
            raise TypeError("returns must be a pandas Series")
        if benchmark is not None and not isinstance(benchmark, pd.Series):
            raise TypeError("benchmark must be a pandas Series or None")
        if positions is not None and not isinstance(positions, pd.DataFrame):
            raise TypeError("positions must be a pandas DataFrame or None")
        if transactions is not None and not isinstance(transactions, pd.DataFrame):
            raise TypeError("transactions must be a pandas DataFrame or None")
        snap_returns = returns.copy(deep=True)
        snap_benchmark = benchmark.copy(deep=True) if benchmark is not None else None
        snap_positions = positions.copy(deep=True) if positions is not None else None
        snap_transactions = transactions.copy(deep=True) if transactions is not None else None
        snap_semantics = semantics if semantics is not None else SeriesSemantics()
        data_digest = _input_digest(
            returns=snap_returns,
            benchmark=snap_benchmark,
            positions=snap_positions,
            transactions=snap_transactions,
        )
        config_digest = _object_digest(dict(config or {}))
        cache_key = _cache_key(
            data_digest=data_digest,
            profile=profile,
            semantics=snap_semantics,
            backend=backend,
            overlay_generation=overlay_generation,
            overlay_digest=overlay_digest,
            operation_version=operation_version,
            config_digest=config_digest,
        )
        return cls(
            returns=snap_returns,
            benchmark=snap_benchmark,
            positions=snap_positions,
            transactions=snap_transactions,
            semantics=snap_semantics,
            profile=profile,
            data_digest=data_digest,
            config_digest=config_digest,
            backend=backend,
            overlay_generation=overlay_generation,
            overlay_digest=overlay_digest,
            operation_version=operation_version,
            cache_key=cache_key,
        )


def _cache_key(
    *,
    data_digest: str,
    profile: str,
    semantics: SeriesSemantics,
    backend: str,
    overlay_generation: int,
    overlay_digest: str,
    operation_version: str,
    config_digest: str,
) -> str:
    return _object_digest(
        {
            "data_digest": data_digest,
            "profile": profile,
            "semantics": asdict(semantics),
            "backend": backend,
            "overlay_generation": overlay_generation,
            "overlay_digest": overlay_digest,
            "operation_version": operation_version,
            "config_digest": config_digest,
        }
    )
