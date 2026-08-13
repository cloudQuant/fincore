"""Exceptions owned by the standalone factor-analysis kernel.

The compatibility facade re-exports the two legacy exception identities from
this module.  Keeping the types here lets enhanced callers retain structured
diagnostics without importing the facade.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from fincore.factor_analysis.data import FactorLossReport


class FactorDataError(ValueError):
    """Base class for invalid factor-analysis inputs."""


class NonMatchingTimezoneError(FactorDataError):
    """Factor and price timestamps use different timezone conventions."""


class FactorLossExceededError(FactorDataError):
    """The enhanced kernel dropped more input rows than the accepted limit."""

    def __init__(self, message: str = "", report: FactorLossReport | None = None) -> None:
        super().__init__(message)
        self.report = report


class MaxLossExceededError(FactorLossExceededError):
    """Legacy-compatible identity for a rejected cleaning-loss budget."""


__all__ = [
    "FactorDataError",
    "FactorLossExceededError",
    "MaxLossExceededError",
    "NonMatchingTimezoneError",
]
