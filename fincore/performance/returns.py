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
    """Solve ``sum CF_t / (1+r)^period_t = 0`` for a unique rate.

    A sequence with zero or multiple sign changes can have no real IRR or
    more than one economically plausible IRR.  Returning an arbitrary root
    would make a single float misleading, so these cases return ``nan``.
    """
    if len(cashflows) < 2:
        raise ValueError("at least two cashflows are required")
    if len(cashflows) != len(periods):
        raise ValueError("cashflows and periods must have the same length")
    if not np.isfinite(cashflows).all():
        raise ValueError("cashflows must be finite")
    if not np.isfinite(periods).all() or np.any(periods < 0.0):
        raise ValueError("periods must be finite and non-negative")

    nonzero_cashflows = cashflows[cashflows != 0.0]
    sign_changes = np.count_nonzero(np.diff(np.sign(nonzero_cashflows)) != 0.0)
    if sign_changes != 1:
        return float("nan")

    def npv(rate: float) -> float:
        return float(np.sum(cashflows / (1.0 + rate) ** periods))

    lower = -0.9999
    upper = 1.0
    try:
        lower_npv = npv(lower)
        upper_npv = npv(upper)
        for _ in range(32):
            if lower_npv == 0.0:
                return lower
            if upper_npv == 0.0:
                return upper
            if np.signbit(lower_npv) != np.signbit(upper_npv):
                return float(optimize.brentq(npv, lower, upper))
            upper = (upper * 2.0) + 1.0
            upper_npv = npv(upper)
    except (FloatingPointError, OverflowError, ValueError):
        return float("nan")

    return float("nan")


def _as_cashflow_array(cashflows: pd.Series | np.ndarray) -> np.ndarray:
    """Return a finite, one-dimensional cashflow vector."""
    cashflow_array = np.asarray(cashflows, dtype=float).flatten()
    if not np.isfinite(cashflow_array).all():
        raise ValueError("cashflows must be finite")
    return cashflow_array


def _normalise_xirr_inputs(
    cashflows: pd.Series | np.ndarray,
    dates: pd.DatetimeIndex | pd.Series,
) -> tuple[np.ndarray, pd.DatetimeIndex]:
    """Align labelled inputs and produce deterministic dated cashflows."""
    cashflow_array = _as_cashflow_array(cashflows)

    if isinstance(cashflows, pd.Series) and isinstance(dates, pd.Series):
        if not cashflows.index.is_unique or not dates.index.is_unique:
            raise ValueError("cashflows and date series indexes must be unique")
        if not cashflows.index.isin(dates.index).all() or not dates.index.isin(cashflows.index).all():
            raise ValueError("cashflows and dates must have the same index labels")
        date_values = dates.reindex(cashflows.index).to_numpy()
    else:
        date_values = np.asarray(dates)

    if len(cashflow_array) != len(date_values):
        raise ValueError("cashflows and dates must have the same length")

    try:
        date_index = pd.DatetimeIndex(pd.to_datetime(date_values, errors="coerce", utc=True))
    except (TypeError, ValueError) as exc:
        raise ValueError("dates must be valid, non-missing dates") from exc
    if date_index.isna().any():
        raise ValueError("dates must be valid, non-missing dates")

    dated_cashflows = pd.DataFrame({"date": date_index, "cashflow": cashflow_array})
    dated_cashflows = (
        dated_cashflows.groupby("date", sort=True, as_index=False)["cashflow"].sum().reset_index(drop=True)
    )
    if len(dated_cashflows) < 2:
        raise ValueError("at least two distinct cashflow dates are required")

    return (
        dated_cashflows["cashflow"].to_numpy(dtype=float),
        pd.DatetimeIndex(dated_cashflows["date"]),
    )


def mwr(cashflows: pd.Series | np.ndarray, periods: float | None = None) -> float:
    """Money-weighted return (internal rate of return).

    ``cashflows`` is a sequence of net contributions (negative for outflow).
    ``periods`` is the length of each interval in years; when omitted, each
    cashflow is assumed one period apart.  The return is ``nan`` when there is
    no unique real IRR: inputs with zero or multiple cashflow sign changes are
    treated as indeterminate rather than selecting an arbitrary root.
    """
    cf = _as_cashflow_array(cashflows)
    n = len(cf)
    if periods is None:
        times = np.arange(n, dtype=float)
    else:
        interval = float(periods)
        if not np.isfinite(interval) or interval <= 0.0:
            raise ValueError("periods must be a positive finite interval")
        times = np.arange(n, dtype=float) * interval
    return _irr(cf, times)


def xirr(cashflows: pd.Series | np.ndarray, dates: pd.DatetimeIndex | pd.Series) -> float:
    """Money-weighted return with explicit dates (daily compounding).

    When both inputs are :class:`pandas.Series`, their unique index labels must
    match and dates are aligned to the cashflow labels.  Other input pairs use
    positional alignment.  Entries are sorted by date, and cashflows sharing
    a date are aggregated before solving; at least two distinct dates are
    required.  The earliest date is time zero and subsequent cashflows are
    discounted by elapsed days divided by 365.

    The return is ``nan`` when no unique real IRR can be identified.  In
    particular, sequences with zero or multiple cashflow sign changes are
    treated as indeterminate rather than returning an arbitrary root.
    """
    cf, dt = _normalise_xirr_inputs(cashflows, dates)
    years = (dt - dt[0]).total_seconds().to_numpy(dtype=float) / (365.0 * 24.0 * 60.0 * 60.0)
    return _irr(cf, years)
