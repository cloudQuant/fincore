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
_ROOT_RELATIVE_TOLERANCE = 64.0 * np.finfo(float).eps


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


def _is_numerical_zero(value: float, scale: float) -> bool:
    return bool(np.isfinite(value) and np.isfinite(scale) and abs(value) <= _ROOT_RELATIVE_TOLERANCE * scale)


def _append_unique_root(roots: list[float], candidate: float) -> None:
    tolerance = 32.0 * np.finfo(float).eps * max(1.0, abs(candidate))
    if not any(abs(candidate - existing) <= tolerance for existing in roots):
        roots.append(candidate)


def _enumerate_log_rate_roots(
    cashflows: np.ndarray,
    periods: np.ndarray,
    lower_bound: float = _LOG_GROSS_LOWER_BOUND,
    upper_bound: float = _LOG_GROSS_UPPER_BOUND,
) -> list[float]:
    """Enumerate NPV roots by recursively locating derivative roots.

    In log-gross-rate space, NPV is an exponential polynomial.  Its derivative
    has one fewer nonzero term, so recursively locating stationary points
    partitions the domain into monotone intervals.  Solving every interval and
    checking the stationary points detects both crossing and tangent roots.
    """
    if len(cashflows) < 2:
        return []

    # Factoring out the earliest-period exponential preserves root locations
    # and gives every recursive derivative a zero-period term to remove.
    periods = periods - np.min(periods)
    positive_periods = periods > 0.0
    if np.count_nonzero(positive_periods) < 2:
        stationary_points: list[float] = []
    else:
        time_scale = float(np.max(periods[positive_periods]))
        derivative_cashflows = -(periods[positive_periods] / time_scale) * cashflows[positive_periods]
        derivative_cashflows, derivative_periods = _normalise_cashflow_polynomial(
            derivative_cashflows,
            periods[positive_periods],
        )
        stationary_points = _enumerate_log_rate_roots(
            derivative_cashflows,
            derivative_periods,
            lower_bound,
            upper_bound,
        )

    boundaries = [lower_bound, *stationary_points, upper_bound]
    evaluations = [_scaled_npv(cashflows, periods, boundary) for boundary in boundaries]
    roots: list[float] = []

    for boundary, (value, scale) in zip(boundaries, evaluations, strict=True):
        if _is_numerical_zero(value, scale):
            _append_unique_root(roots, boundary)

    for left, right, left_evaluation, right_evaluation in zip(
        boundaries[:-1],
        boundaries[1:],
        evaluations[:-1],
        evaluations[1:],
        strict=True,
    ):
        left_value, left_scale = left_evaluation
        right_value, right_scale = right_evaluation
        if _is_numerical_zero(left_value, left_scale) or _is_numerical_zero(right_value, right_scale):
            continue
        if not (np.isfinite(left_value) and np.isfinite(right_value)):
            continue
        if np.signbit(left_value) == np.signbit(right_value):
            continue
        try:
            root = optimize.brentq(
                lambda log_rate: _scaled_npv(cashflows, periods, log_rate)[0],
                left,
                right,
                xtol=np.finfo(float).eps,
                rtol=4.0 * np.finfo(float).eps,
                maxiter=256,
            )
        except (RuntimeError, ValueError):
            continue
        _append_unique_root(roots, float(root))

    return roots


def _rate_from_log_gross(log_gross_rate: float) -> float:
    """Convert a bounded log gross rate to its representable periodic return."""
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        rate = np.expm1(log_gross_rate)
    return float(rate) if np.isfinite(rate) else float(np.finfo(float).max)


def _irr(cashflows: np.ndarray, periods: np.ndarray) -> float:
    """Solve ``sum CF_t / (1+r)^period_t = 0`` when exactly one rate exists.

    Candidate roots are enumerated over the full representable ``r > -1``
    domain in log-gross-rate space.  ``nan`` is returned only when no root or
    more than one distinct representable root is found.
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
    roots = _enumerate_log_rate_roots(cashflows, periods)
    if len(roots) != 1:
        return float("nan")

    rate = _rate_from_log_gross(roots[0])
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
    no unique representable real IRR.  It is determined by enumerating actual
    NPV roots, not by cashflow sign counts.
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

    The return is ``nan`` when actual root enumeration finds zero or multiple
    distinct representable real IRRs.
    """
    cf, dt = _normalise_xirr_inputs(cashflows, dates)
    years = np.asarray((dt - dt[0]).days, dtype=float) / 365.0
    return _irr(cf, years)
