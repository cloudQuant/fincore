"""AnalysisSnapshot: one validated, copy-on-ingest view of analysis inputs.

The snapshot holds the validated returns/benchmark/positions/transactions plus
their semantic metadata and a stable cache key.  Caller inputs are copied on
ingest, so downstream kernels can never mutate the caller's data.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field

import pandas as pd

from fincore.contracts.analysis import SeriesSemantics

__all__ = ["AnalysisSnapshot"]


def _series_digest(series: pd.Series) -> str:
    payload = series.to_frame(name="values").to_csv(index=True, lineterminator="\n").encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


@dataclass(frozen=True)
class AnalysisSnapshot:
    """An immutable, validated analysis input snapshot."""

    returns: pd.Series
    benchmark: pd.Series | None = None
    positions: pd.DataFrame | None = None
    transactions: pd.DataFrame | None = None
    semantics: SeriesSemantics = field(default_factory=SeriesSemantics)
    profile: str = "enhanced_v1"
    config_digest: str = ""
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
    ) -> AnalysisSnapshot:
        if not isinstance(returns, pd.Series):
            raise TypeError("returns must be a pandas Series")
        snap_returns = returns.copy(deep=True)
        snap_benchmark = benchmark.copy(deep=True) if benchmark is not None else None
        snap_positions = positions.copy(deep=True) if positions is not None else None
        snap_transactions = transactions.copy(deep=True) if transactions is not None else None
        snap_semantics = semantics if semantics is not None else SeriesSemantics()
        digest = _series_digest(snap_returns)
        cache_key = _cache_key(
            digest=digest,
            profile=profile,
            semantics=snap_semantics,
            backend=backend,
            overlay_generation=overlay_generation,
        )
        return cls(
            returns=snap_returns,
            benchmark=snap_benchmark,
            positions=snap_positions,
            transactions=snap_transactions,
            semantics=snap_semantics,
            profile=profile,
            config_digest=digest,
            cache_key=cache_key,
        )


def _cache_key(
    *,
    digest: str,
    profile: str,
    semantics: SeriesSemantics,
    backend: str,
    overlay_generation: int,
) -> str:
    payload = (
        f"{digest}|{profile}|{semantics.frequency}|{semantics.return_type}|"
        f"{semantics.timezone or ''}|{semantics.currency or ''}|{backend}|{overlay_generation}"
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
