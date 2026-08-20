"""Enhanced analysis input contracts.

``AnalysisInput`` separates validated data from its semantic metadata, so a
kernel receives both *what* it should compute on and *how* that data is to be
interpreted (frequency, return type, timezone, currency).  Strict compatibility
APIs never use these objects.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from fincore.contracts.profiles import ENHANCED_V1

__all__ = ["AnalysisInput", "PortfolioSemantics", "SeriesSemantics"]


@dataclass(frozen=True)
class SeriesSemantics:
    """Semantic metadata for a single return series."""

    frequency: str = "daily"
    return_type: str = "simple"
    timezone: str | None = None
    currency: str | None = None
    calendar: str | None = None


@dataclass(frozen=True)
class PortfolioSemantics:
    """Semantic metadata for a portfolio (positions/transactions)."""

    weight_timestamp_convention: str = "as_of"
    gross_net: str = "net"
    currency: str | None = None


@dataclass(frozen=True)
class AnalysisInput:
    """Validated data plus its semantic metadata and a config digest."""

    returns: pd.Series
    benchmark: pd.Series | None = None
    positions: pd.DataFrame | None = None
    transactions: pd.DataFrame | None = None
    semantics: SeriesSemantics = field(default_factory=SeriesSemantics)
    portfolio_semantics: PortfolioSemantics = field(default_factory=PortfolioSemantics)
    profile: str = ENHANCED_V1
    config_digest: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.returns, pd.Series):
            raise TypeError("returns must be a pandas Series")

    @classmethod
    def from_returns(
        cls,
        returns: pd.Series,
        *,
        benchmark: pd.Series | None = None,
        semantics: SeriesSemantics | None = None,
        profile: str = ENHANCED_V1,
    ) -> AnalysisInput:
        """Build an AnalysisInput from a returns series (copy-on-ingest)."""
        digest = _series_digest(returns)
        return cls(
            returns=returns.copy(deep=True),
            benchmark=benchmark.copy(deep=True) if benchmark is not None else None,
            semantics=semantics if semantics is not None else SeriesSemantics(),
            profile=profile,
            config_digest=digest,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "profile": self.profile,
            "frequency": self.semantics.frequency,
            "return_type": self.semantics.return_type,
            "timezone": self.semantics.timezone,
            "currency": self.semantics.currency,
            "config_digest": self.config_digest,
        }


def _series_digest(returns: pd.Series) -> str:
    payload = returns.to_frame(name="returns").to_csv(index=True, lineterminator="\n").encode("utf-8")
    return hashlib.sha256(payload).hexdigest()
