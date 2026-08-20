"""Institution-grade performance semantics."""

from __future__ import annotations

from fincore.performance.disclosures import DisclosureContext, render_disclosure
from fincore.performance.inference import (
    sharpe_confidence_interval,
    sharpe_standard_error,
    standard_error_of_mean,
)
from fincore.performance.returns import mwr, twr, xirr

__all__ = [
    "DisclosureContext",
    "mwr",
    "render_disclosure",
    "sharpe_confidence_interval",
    "sharpe_standard_error",
    "standard_error_of_mean",
    "twr",
    "xirr",
]
