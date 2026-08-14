"""Portfolio construction and Pyfolio bridge for the factor-analysis kernel.

The functions in this module are the enhanced, standalone implementation.
They deliberately do not inspect the compatibility fixture or carry a legacy
profile switch; the strict Alphalens facade projects its source-visible details
at its own boundary.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Any, Sequence, cast

import pandas as pd
from pandas.tseries.offsets import BDay

from fincore.factor_analysis import performance as _performance
from fincore.factor_analysis.calendar import add_custom_calendar_timedelta, get_forward_returns_columns


@dataclass(frozen=True)
class PyfolioFactorInputs:
    """Typed, enhanced representation of the three inputs expected by Pyfolio."""

    returns: pd.Series
    positions: pd.DataFrame
    benchmark_rets: pd.Series | None

    def as_legacy_tuple(self) -> tuple[pd.Series, pd.DataFrame, pd.Series | None]:
        """Return the immutable container's legacy Alphalens tuple projection."""

        return self.returns, self.positions, self.benchmark_rets


def _require_weight_series(weights: pd.Series) -> pd.Series:
    """Validate and copy a date/asset weight series before reshaping it."""

    if not isinstance(weights, pd.Series):
        raise TypeError("weights must be a pandas Series")
    if not isinstance(weights.index, pd.MultiIndex) or weights.index.nlevels != 2:
        raise ValueError("weights must use a two-level (date, asset) MultiIndex")
    copied = weights.copy(deep=True)
    copied.index = copied.index.set_names(("date", "asset"))
    return copied


def positions(weights: pd.Series, period: pd.Timedelta | object, freq: Any = None) -> pd.DataFrame:
    """Build gross-normalized active positions for overlapping factor trades.

    ``period`` may include an intraday remainder (for example ``"1D3h"``).
    Whole session days advance with the supplied trading calendar while the
    remainder stays at its wall-clock time.  The output includes every trade
    and expiry timestamp, with a zero row after the final holding expires.
    """

    copied = _require_weight_series(weights)
    weights_frame = copied.unstack()
    holding_period = period if isinstance(period, pd.Timedelta) else pd.Timedelta(cast("Any", period))

    if not isinstance(weights_frame.index, pd.DatetimeIndex):
        raise ValueError("weights date level must be a DatetimeIndex")
    calendar = freq if freq is not None else weights_frame.index.freq
    if calendar is None:
        calendar = BDay()
        warnings.warn("'freq' not set, using business day calendar", UserWarning, stacklevel=2)

    trades_index = weights_frame.index.copy()
    return_index = cast("pd.DatetimeIndex", add_custom_calendar_timedelta(trades_index, holding_period, calendar))
    position_index = trades_index.union(return_index)
    result = pd.DataFrame(0.0, index=position_index, columns=weights_frame.columns, dtype=float)
    active: list[tuple[pd.Timestamp, pd.Series]] = []

    for timestamp in position_index:
        if timestamp in weights_frame.index:
            assets_weights = weights_frame.loc[timestamp].copy(deep=True)
            if not isinstance(assets_weights, pd.Series):
                raise ValueError("weights must contain at most one row per date and asset")
            expires_at = cast("pd.Timestamp", add_custom_calendar_timedelta(timestamp, holding_period, calendar))
            active.append((expires_at, assets_weights))

        # Preserve the pinned lifecycle: a trade expiring at the current
        # timestamp is no longer active for that timestamp's output row.
        if active and active[0][0] <= timestamp:
            active.pop(0)
        if not active:
            continue

        total = pd.concat([entry[1] for entry in active], axis=1).sum(axis=1)
        gross = total.abs().sum()
        if gross != 0 and not pd.isna(gross):
            result.loc[timestamp, total.index] = total / gross

    return result.fillna(0.0)


def _filtered_portfolio_data(
    factor_data: pd.DataFrame,
    period: object,
    *,
    quantiles: Sequence[int] | None,
    groups: Sequence[str] | None,
) -> pd.DataFrame:
    """Copy, period-project, and optionally filter one factor portfolio input."""

    copied = _performance._copy_factor_data(factor_data)
    forward_columns = get_forward_returns_columns(copied.columns)
    if period not in forward_columns:
        raise ValueError(f"Period '{period}' not found")
    drop_columns = [column for column in forward_columns if column != period]
    portfolio_data = copied.drop(columns=drop_columns)
    if quantiles is not None:
        if "factor_quantile" not in portfolio_data.columns:
            raise KeyError("factor_quantile")
        portfolio_data = portfolio_data.loc[portfolio_data["factor_quantile"].isin(quantiles)].copy(deep=True)
    if groups is not None:
        if "group" not in portfolio_data.columns:
            raise KeyError("group")
        portfolio_data = portfolio_data.loc[portfolio_data["group"].isin(groups)].copy(deep=True)
    return portfolio_data


def _factor_portfolio_returns(
    factor_data: pd.DataFrame,
    period: object,
    *,
    long_short: bool,
    group_neutral: bool,
    equal_weight: bool,
    quantiles: Sequence[int] | None,
    groups: Sequence[str] | None,
) -> pd.Series:
    """Compute one non-cumulative factor-portfolio return series on a copy."""

    portfolio_data = _filtered_portfolio_data(
        factor_data,
        period,
        quantiles=quantiles,
        groups=groups,
    )
    returns = _performance.factor_returns(
        portfolio_data,
        demeaned=long_short,
        group_adjust=group_neutral,
        equal_weight=equal_weight,
    )
    selected = returns[period]
    if not isinstance(selected, pd.Series):
        raise ValueError(f"Period '{period}' must identify exactly one forward-return column")
    return selected.copy(deep=True)


def factor_cumulative_returns(
    factor_data: pd.DataFrame,
    period: object,
    long_short: bool = True,
    group_neutral: bool = False,
    equal_weight: bool = False,
    quantiles: Sequence[int] | None = None,
    groups: Sequence[str] | None = None,
) -> pd.Series:
    """Simulate and compound a filtered factor portfolio for one return period."""

    returns = _factor_portfolio_returns(
        factor_data,
        period,
        long_short=long_short,
        group_neutral=group_neutral,
        equal_weight=equal_weight,
        quantiles=quantiles,
        groups=groups,
    )
    return cast("pd.Series", _performance.cumulative_returns(returns))


def factor_positions(
    factor_data: pd.DataFrame,
    period: object,
    long_short: bool = True,
    group_neutral: bool = False,
    equal_weight: bool = False,
    quantiles: Sequence[int] | None = None,
    groups: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Simulate gross-normalized asset positions for a filtered factor portfolio."""

    portfolio_data = _filtered_portfolio_data(
        factor_data,
        period,
        quantiles=quantiles,
        groups=groups,
    )
    weights = _performance.factor_weights(
        portfolio_data,
        demeaned=long_short,
        group_adjust=group_neutral,
        equal_weight=equal_weight,
    )
    return positions(weights, period)


def _daily_cumulative_returns(cumulative: pd.Series) -> tuple[pd.Series, pd.Series]:
    """Resample a cumulative curve and derive Pyfolio-compatible daily returns."""

    daily_cumulative = cumulative.resample("1D").last().ffill()
    daily_returns = daily_cumulative.pct_change(fill_method=None).fillna(0.0)
    return daily_cumulative, daily_returns


def create_pyfolio_input(
    factor_data: pd.DataFrame,
    period: object,
    capital: float | None = None,
    long_short: bool = True,
    group_neutral: bool = False,
    equal_weight: bool = False,
    quantiles: Sequence[int] | None = None,
    groups: Sequence[str] | None = None,
    benchmark_period: object = "1D",
) -> PyfolioFactorInputs:
    """Build typed daily returns, positions, and optional benchmark for Pyfolio.

    The enhanced builder is independent of external Pyfolio.  It merely emits
    data in the workflow's canonical shape; rendering stays in
    :mod:`fincore.pyfolio` and remains lazily optional.
    """

    cumulative = factor_cumulative_returns(
        factor_data,
        period,
        long_short=long_short,
        group_neutral=group_neutral,
        equal_weight=equal_weight,
        quantiles=quantiles,
        groups=groups,
    )
    daily_cumulative, returns = _daily_cumulative_returns(cumulative)

    raw_positions = factor_positions(
        factor_data,
        period,
        long_short=long_short,
        group_neutral=group_neutral,
        equal_weight=equal_weight,
        quantiles=quantiles,
        groups=groups,
    )
    daily_positions = raw_positions.resample("1D").sum().ffill()
    asset_gross = daily_positions.abs().sum(axis=1)
    daily_positions = daily_positions.div(asset_gross, axis=0).fillna(0.0)
    daily_positions["cash"] = 1.0 - daily_positions.sum(axis=1)
    if capital is not None:
        # Holding periods can extend past the final factor-return observation.
        # The enhanced bridge keeps the last known portfolio value through
        # that active-position horizon instead of manufacturing trailing NaN
        # dollar positions during ``reindex``.
        capital_curve = daily_cumulative.reindex(daily_positions.index).ffill()
        daily_positions = daily_positions.mul(capital_curve * capital, axis=0)

    forward_columns = get_forward_returns_columns(factor_data.columns)
    benchmark_rets: pd.Series | None = None
    if benchmark_period in forward_columns:
        benchmark_data = _performance._copy_factor_data(factor_data)
        benchmark_data["factor"] = benchmark_data["factor"].abs()
        benchmark_cumulative = factor_cumulative_returns(
            benchmark_data,
            benchmark_period,
            long_short=False,
            group_neutral=False,
            equal_weight=True,
        )
        _, benchmark_rets = _daily_cumulative_returns(benchmark_cumulative)
        benchmark_rets.name = "benchmark"

    return PyfolioFactorInputs(
        returns=returns.copy(deep=True),
        positions=daily_positions.copy(deep=True),
        benchmark_rets=None if benchmark_rets is None else benchmark_rets.copy(deep=True),
    )


__all__ = [
    "PyfolioFactorInputs",
    "create_pyfolio_input",
    "factor_cumulative_returns",
    "factor_positions",
    "positions",
]
