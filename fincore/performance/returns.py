"""Institution-grade performance return measures.

Provides time-weighted (TWR) and money-weighted (MWR/XIRR) returns with
explicit, documented semantics.  All outputs carry their compounding and
frequency conventions in their docstrings; no implicit annualization.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from scipy import optimize

if TYPE_CHECKING:
    from collections.abc import Sequence

__all__ = ["mwr", "twr", "xirr"]


_LOG_GROSS_LOWER_BOUND = float(np.log1p(np.nextafter(-1.0, 0.0)))
_LOG_GROSS_UPPER_BOUND = float(np.log(np.finfo(float).max))


def twr(returns: pd.Series | np.ndarray) -> float:
    """Time-weighted return: the geometric compound of per-period returns.

    ``TWR = prod(1 + r_t) - 1``.  Returns are fractional (not percent).
    """
    r = np.asarray(returns, dtype=float)
    r = r[~np.isnan(r)]
    return float(np.prod(1.0 + r) - 1.0)


def _normalise_cashflow_polynomial(
    cashflows: np.ndarray,
    periods: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Aggregate equal periods and discard zero coefficients deterministically."""
    order = np.argsort(periods, kind="stable")
    sorted_periods = periods[order]
    sorted_cashflows = cashflows[order]
    group_starts = np.flatnonzero(np.r_[True, np.diff(sorted_periods) != 0.0])
    group_ends = np.r_[group_starts[1:], len(sorted_periods)]

    try:
        aggregated_cashflows = np.asarray(
            [math.fsum(sorted_cashflows[start:end]) for start, end in zip(group_starts, group_ends, strict=True)],
            dtype=float,
        )
    except OverflowError as exc:
        raise ValueError("cashflows cannot be aggregated without overflow") from exc
    if not np.isfinite(aggregated_cashflows).all():
        raise ValueError("cashflows cannot be aggregated without overflow")

    nonzero = aggregated_cashflows != 0.0
    return aggregated_cashflows[nonzero], sorted_periods[group_starts][nonzero]


def _scaled_npv(cashflows: np.ndarray, periods: np.ndarray, log_gross_rate: float) -> tuple[float, float]:
    """Return a sign-preserving NPV and scale without rate-space overflow."""
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        log_magnitudes = np.log(np.abs(cashflows)) - (periods * log_gross_rate)

    if not np.isfinite(log_magnitudes).all():
        return float("nan"), float("nan")

    anchor = float(np.max(log_magnitudes))
    normalized_terms = np.copysign(np.exp(log_magnitudes - anchor), cashflows)
    value = math.fsum(float(term) for term in normalized_terms)
    scale = math.fsum(abs(float(term)) for term in normalized_terms)
    return value, scale


def _has_single_cashflow_sign_change(cashflows: np.ndarray) -> bool:
    """Return whether a normalized cashflow stream is conventional.

    A conventional stream has exactly one sign transition after equal-period
    cashflows have been aggregated.  This is a deliberately conservative
    public contract: it gives the exponential NPV equation at most one real
    root, whereas non-conventional streams can have zero, several, or repeated
    roots that cannot be classified reliably from binary64 arithmetic alone.
    """
    if len(cashflows) < 2:
        return False
    return bool(np.count_nonzero(np.signbit(cashflows[1:]) != np.signbit(cashflows[:-1])) == 1)


def _solve_conventional_irr(cashflows: np.ndarray, periods: np.ndarray) -> float:
    """Solve a conventional cashflow stream over the full representable domain."""
    lower_value, _ = _scaled_npv(cashflows, periods, _LOG_GROSS_LOWER_BOUND)
    upper_value, _ = _scaled_npv(cashflows, periods, _LOG_GROSS_UPPER_BOUND)
    if not (np.isfinite(lower_value) and np.isfinite(upper_value)):
        return float("nan")

    # A conventional stream has opposite asymptotic signs.  Do not treat a
    # merely *small* endpoint residual as a root: that was the source of a
    # previous false-tangent-root bug.  brentq's sign-changing bracket is the
    # certificate of existence used by this scalar API.
    if np.signbit(lower_value) == np.signbit(upper_value):
        return float("nan")
    try:
        root = optimize.brentq(
            lambda log_rate: _scaled_npv(cashflows, periods, log_rate)[0],
            _LOG_GROSS_LOWER_BOUND,
            _LOG_GROSS_UPPER_BOUND,
            xtol=np.finfo(float).eps,
            rtol=4.0 * np.finfo(float).eps,
            maxiter=256,
        )
    except (RuntimeError, ValueError):
        return float("nan")
    return _rate_from_log_gross(float(root))


def _rate_from_log_gross(log_gross_rate: float) -> float:
    """Convert a bounded log gross rate to its representable periodic return."""
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        rate = np.expm1(log_gross_rate)
    return float(rate) if np.isfinite(rate) else float(np.finfo(float).max)


def _irr(cashflows: np.ndarray, periods: np.ndarray) -> float:
    """Solve a conventional ``sum CF_t / (1+r)^period_t = 0`` equation.

    The scalar API intentionally rejects non-conventional flows (anything
    other than exactly one post-aggregation sign transition) with ``nan``.
    Such inputs require an explicit multi-root analysis rather than selecting
    or inferring a root from a tolerance-sensitive numerical search.
    """
    cashflows = np.asarray(cashflows, dtype=float).reshape(-1)
    periods = np.asarray(periods, dtype=float).reshape(-1)
    if len(cashflows) < 2:
        raise ValueError("at least two cashflows are required")
    if len(cashflows) != len(periods):
        raise ValueError("cashflows and periods must have the same length")
    if not np.isfinite(cashflows).all():
        raise ValueError("cashflows must be finite")
    if not np.isfinite(periods).all() or np.any(periods < 0.0):
        raise ValueError("periods must be finite and non-negative")

    cashflows, periods = _normalise_cashflow_polynomial(cashflows, periods)
    if not _has_single_cashflow_sign_change(cashflows):
        return float("nan")
    rate = _solve_conventional_irr(cashflows, periods)
    return rate if rate > -1.0 else float("nan")


def _as_cashflow_array(cashflows: pd.Series | np.ndarray) -> np.ndarray:
    """Return a finite, one-dimensional cashflow vector."""
    cashflow_array = np.asarray(cashflows, dtype=float).flatten()
    if not np.isfinite(cashflow_array).all():
        raise ValueError("cashflows must be finite")
    return cashflow_array


def _normalise_xirr_inputs(
    cashflows: pd.Series | np.ndarray,
    dates: pd.DatetimeIndex | pd.Series | np.ndarray | Sequence[object],
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

    date_index = _local_calendar_dates(date_values)

    dated_cashflows = pd.DataFrame(
        pd.DataFrame({"date": date_index, "cashflow": cashflow_array})
        .groupby("date", sort=True, as_index=False)[["cashflow"]]
        .sum()
        .reset_index(drop=True)
    )
    if len(dated_cashflows) < 2:
        raise ValueError("at least two distinct cashflow dates are required")

    return (
        dated_cashflows["cashflow"].to_numpy(dtype=float),
        pd.DatetimeIndex(dated_cashflows["date"]),
    )


def _local_calendar_dates(date_values: np.ndarray) -> pd.DatetimeIndex:
    """Return timezone-consistent local calendar days for XIRR discounting."""
    local_days: list[pd.Timestamp] = []
    expected_timezone: str | None = None
    timezone_seen = False

    for value in date_values:
        try:
            timestamp = pd.Timestamp(value)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("dates must be valid, non-missing dates") from exc
        if pd.isna(timestamp):
            raise ValueError("dates must be valid, non-missing dates")

        timezone = timestamp.tzinfo
        timezone_name = None if timezone is None else str(timezone)
        if not timezone_seen:
            expected_timezone = timezone_name
            timezone_seen = True
        elif timezone_name != expected_timezone:
            raise ValueError("dates must use the same timezone or all be timezone-naive")

        local_day = timestamp.normalize()
        if timezone is not None:
            local_day = local_day.tz_localize(None)
        local_days.append(local_day)

    return pd.DatetimeIndex(local_days)


def mwr(cashflows: pd.Series | np.ndarray, periods: float | None = None) -> float:
    """Money-weighted return (internal rate of return).

    ``cashflows`` is a sequence of net contributions (negative for outflow).
    ``periods`` is the length of each interval in years; when omitted, each
    cashflow is assumed one period apart.  The return is ``nan`` when there is
    no conventional representable real IRR.  After same-period aggregation,
    non-conventional cashflows (anything other than one sign transition) are
    rejected with ``nan`` instead of choosing a tolerance-dependent root.
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


def xirr(
    cashflows: pd.Series | np.ndarray,
    dates: pd.DatetimeIndex | pd.Series | np.ndarray | Sequence[object],
) -> float:
    """Money-weighted return with explicit dates (daily compounding).

    When both inputs are :class:`pandas.Series`, their unique index labels must
    match and dates are aligned to the cashflow labels.  Other input pairs use
    positional alignment.  Every input must be timezone-naive or use the same
    timezone; mixed timezone inputs are rejected.  Time-of-day is ignored,
    cashflows sharing a *local calendar date* are aggregated, and entries are
    sorted before solving.  At least two distinct dates are required.  The
    earliest date is time zero and subsequent cashflows are discounted by
    calendar days divided by 365.

    The return is ``nan`` for no conventional real IRR or a non-conventional
    cashflow stream.  The latter is deliberately rejected rather than choosing
    a tolerance-dependent root from a potentially multi-root equation.
    """
    cf, dt = _normalise_xirr_inputs(cashflows, dates)
    years = np.asarray((dt - dt[0]).days, dtype=float) / 365.0
    return _irr(cf, years)
