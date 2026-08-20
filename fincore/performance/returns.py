"""Institution-grade performance return measures.

Provides time-weighted (TWR) and money-weighted (MWR/XIRR) returns with
explicit, documented semantics.  All outputs carry their compounding and
frequency conventions in their docstrings; no implicit annualization.
"""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
from scipy import optimize

__all__ = ["mwr", "twr", "xirr"]


def twr(returns: pd.Series | np.ndarray) -> float:
    """Time-weighted return: the geometric compound of per-period returns.

    ``TWR = prod(1 + r_t) - 1``.  Returns are fractional (not percent).
    """
    r = np.asarray(returns, dtype=float)
    r = r[~np.isnan(r)]
    return float(np.prod(1.0 + r) - 1.0)


def _irr(cashflows: np.ndarray, periods: np.ndarray) -> float:
    """Solve ``sum CF_t / (1+r)^period_t = 0`` for the rate ``r``."""
    if len(cashflows) < 2:
        raise ValueError("at least two cashflows are required")

    def npv(rate: float) -> float:
        return float(np.sum(cashflows / (1.0 + rate) ** periods))

    try:
        return float(optimize.brentq(npv, -0.9999, 100.0))
    except ValueError:
        return float("nan")


def mwr(cashflows: pd.Series | np.ndarray, periods: int | None = None) -> float:
    """Money-weighted return (internal rate of return).

    ``cashflows`` is a sequence of net contributions (negative for outflow).
    ``periods`` is the length of each interval in years; when omitted, each
    cashflow is assumed one period apart.
    """
    cf = np.asarray(cashflows, dtype=float).flatten()
    cf = cf[~np.isnan(cf)]
    n = len(cf)
    times = np.arange(n, dtype=float) if periods is None else np.arange(n, dtype=float) * float(periods)
    return _irr(cf, times)


def xirr(cashflows: pd.Series, dates: pd.DatetimeIndex | pd.Series) -> float:
    """Money-weighted return with explicit dates (daily compounding).

    ``cashflows`` and ``dates`` must share an index; the initial date is time
    zero and each subsequent cashflow is discounted by ``days / 365``.
    """
    if len(cashflows) != len(dates):
        raise ValueError("cashflows and dates must have the same length")
    cf = np.asarray(cashflows, dtype=float).flatten()
    dt = pd.to_datetime(dates)
    start = dt[0]
    years = np.asarray([(d - start).days / 365.0 for d in dt], dtype=float)
    return _irr(cf, years)
