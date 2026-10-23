"""Portfolio construction for the canonical factor-analysis kernel."""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Any, Sequence, cast

import numpy as np
import pandas as pd
from pandas.tseries.offsets import BDay

import fincore.factor_analysis.performance as _performance
from fincore.factor_analysis.calendar import add_custom_calendar_timedelta, get_forward_returns_columns


@dataclass(frozen=True)
class FactorPortfolioInputs:
    """Typed daily return, position, and benchmark inputs for factor workflows."""

    returns: pd.Series
    positions: pd.DataFrame
    benchmark_returns: pd.Series | None


def _require_weight_series(weights: pd.Series) -> pd.Series:
    """Validate and copy a date/asset weight series before reshaping it."""

    if not isinstance(weights, pd.Series):
        raise TypeError("weights must be a pandas Series")
    if not isinstance(weights.index, pd.MultiIndex) or weights.index.nlevels != 2:
        raise ValueError("weights must use a two-level (date, asset) MultiIndex")
    copied = weights.copy(deep=True)
    copied.index = copied.index.set_names(("date", "asset"))
    return copied


_POSITIONS_ACTIVE_SCRATCH_BUDGET = 8 * 1024 * 1024


def _float64_position_frame_values(
    weights: pd.Series, weights_frame: pd.DataFrame
) -> np.ndarray | None:
    """Return the frame's float64 view only for the standard pandas layout.

    The positions fast path depends on a single NumPy-backed pandas block and
    the ordinary Series/DataFrame constructors.  Keep subclasses, extension
    arrays, alternate managers, and any layout that would materialize another
    full panel on the legacy path.
    """

    # The exact reduction layout below is source-audited against pandas 2.2.3.
    # Keep every other release on the original public path until its concat
    # and axis-1 reduction layouts have been checked as well.
    if pd.__version__ != "2.2.3":
        return None
    if type(weights) is not pd.Series or type(weights.index) is not pd.MultiIndex:
        return None
    if type(weights_frame) is not pd.DataFrame:
        return None
    if type(weights_frame.index) is not pd.DatetimeIndex or not weights_frame.index.is_unique:
        return None
    if type(weights_frame.columns) not in (pd.Index, pd.MultiIndex):
        return None
    if any(type(asset) is not str for asset in weights_frame.columns):
        return None
    if not weights_frame.columns.is_unique:
        return None

    manager = getattr(weights_frame, "_mgr", None)
    blocks = getattr(manager, "blocks", None)
    if blocks is None or len(blocks) != 1:
        return None
    block_values = getattr(blocks[0], "values", None)
    if type(block_values) is not np.ndarray or block_values.dtype != np.dtype("float64"):
        return None
    expected_block_shape = (len(weights_frame.columns), len(weights_frame.index))
    if block_values.ndim != 2 or block_values.shape != expected_block_shape:
        return None

    frame_values = weights_frame.to_numpy(copy=False)
    if type(frame_values) is not np.ndarray or frame_values.dtype != np.dtype("float64"):
        return None
    if frame_values.shape != (len(weights_frame.index), len(weights_frame.columns)):
        return None
    if not np.shares_memory(frame_values, block_values):
        return None
    return frame_values


def _positions_float64_fastpath(
    weights_frame: pd.DataFrame,
    position_index: pd.Index,
    holding_period: pd.Timedelta,
    calendar: Any,
    frame_values: np.ndarray,
) -> pd.DataFrame:
    """Build positions with the same bounded NumPy reduction used by pandas.

    pandas 2.2's float64 ``concat(..., axis=1).sum(axis=1)`` transposes the
    homogeneous block and calls ``ndarray.sum`` on axis 1.  With no NaNs that
    view is Fortran-contiguous; when any NaN exists, nanops copies it in C
    order before replacing NaNs with zero.  Matching both layouts matters for
    exact float64 reduction bits.
    """

    columns = weights_frame.columns
    asset_count = len(columns)
    result_values = np.zeros((len(position_index), asset_count), dtype=np.float64)
    active: list[tuple[pd.Timestamp, pd.Timestamp, np.ndarray]] = []

    for output_position, timestamp in enumerate(position_index):
        if timestamp in weights_frame.index:
            row_position = weights_frame.index.get_loc(timestamp)
            expires_at = cast(
                "pd.Timestamp", add_custom_calendar_timedelta(timestamp, holding_period, calendar)
            )
            active.append((expires_at, timestamp, frame_values[row_position]))

        # Keep the existing lifecycle: at most one expired trade is removed
        # for each timestamp, even when more than one is already due.
        if active and active[0][0] <= timestamp:
            active.pop(0)
        if not active:
            continue

        active_count = len(active)
        # The NaN case temporarily holds the active matrix, its C-order
        # reduction copy, and a boolean mask. Include the output vectors in the
        # cap so a long holding period or wide universe falls back safely.
        scratch_bytes = active_count * asset_count * 17 + asset_count * 33
        if scratch_bytes <= _POSITIONS_ACTIVE_SCRATCH_BUDGET:
            active_matrix = np.empty((active_count, asset_count), dtype=np.float64)
            for active_position, (_expiry, _trade_time, row_values) in enumerate(active):
                active_matrix[active_position, :] = row_values

            missing = np.isnan(active_matrix)
            has_missing = bool(missing.any())
            if has_missing:
                # nanops copies the transposed block in C order after it finds
                # any NaN. Replacing in this source first preserves the same
                # finite values and sum order while avoiding a second mask.
                np.putmask(active_matrix, missing, 0.0)
            del missing

            transposed = active_matrix.T
            if has_missing:
                reduction_values = transposed.copy()
                total = reduction_values.sum(axis=1, dtype=np.float64)
                del reduction_values
            else:
                total = transposed.sum(axis=1, dtype=np.float64)
            del transposed, active_matrix
            total_series = pd.Series(total, index=columns, copy=False)
        else:
            # Preserve pandas' complete fallback reduction when the active
            # matrix would exceed the bounded scratch allowance.
            active_series = [
                pd.Series(row_values, index=columns, name=trade_time, copy=False)
                for _expiry, trade_time, row_values in active
            ]
            total_series = pd.concat(active_series, axis=1).sum(axis=1)

        # Keep pandas' public Series arithmetic for gross and normalization;
        # the optimization is the bounded active-window reduction and avoids
        # numexpr/version differences in scalar division.
        gross = total_series.abs().sum()
        if gross != 0 and not pd.isna(gross):
            normalized = total_series / gross
            result_row = result_values[output_position, :]
            result_row[:] = normalized.to_numpy(copy=False)
            np.putmask(result_row, np.isnan(result_row), 0.0)
            del result_row, normalized
        del total_series
        if scratch_bytes <= _POSITIONS_ACTIVE_SCRATCH_BUDGET:
            del total

    return pd.DataFrame(
        result_values,
        index=position_index,
        columns=columns,
        dtype=float,
        copy=False,
    )


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

    frame_values = _float64_position_frame_values(weights, weights_frame)
    # The legacy ``result.loc[timestamp, ...]`` assignment updates every row
    # sharing a duplicate expiry label. The fast path writes one positional
    # row, so retain the original path whenever unioned expiries are repeated.
    if frame_values is not None and position_index.is_unique:
        return _positions_float64_fastpath(
            weights_frame,
            position_index,
            holding_period,
            calendar,
            frame_values,
        )

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
    weights: "pd.Series | None" = None,
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
        weights=weights,
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
    weights: "pd.Series | None" = None,
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
        weights=weights,
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
    weights: "pd.Series | None" = None,
) -> pd.DataFrame:
    """Simulate gross-normalized asset positions for a filtered factor portfolio."""

    portfolio_data = _filtered_portfolio_data(
        factor_data,
        period,
        quantiles=quantiles,
        groups=groups,
    )
    if weights is None:
        weights = _performance.factor_weights(
            portfolio_data,
            demeaned=long_short,
            group_adjust=group_neutral,
            equal_weight=equal_weight,
        )
    return positions(weights, period)


def _daily_cumulative_returns(cumulative: pd.Series) -> tuple[pd.Series, pd.Series]:
    """Resample a cumulative curve and derive daily factor-portfolio returns."""

    daily_cumulative = cumulative.resample("1D").last().ffill()
    daily_returns = daily_cumulative.pct_change(fill_method=None).fillna(0.0)
    return daily_cumulative, daily_returns


def _build_factor_portfolio_inputs_from_outputs(
    factor_data: pd.DataFrame,
    daily_cumulative: pd.Series,
    returns: pd.Series,
    raw_positions: pd.DataFrame,
    *,
    capital: float | None,
    benchmark_period: object,
) -> FactorPortfolioInputs:
    """Project already-computed factor outputs into typed daily portfolio inputs."""

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
    benchmark_returns: pd.Series | None = None
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
        _, benchmark_returns = _daily_cumulative_returns(benchmark_cumulative)
        benchmark_returns.name = "benchmark"

    return FactorPortfolioInputs(
        returns=returns.copy(deep=True),
        positions=daily_positions.copy(deep=True),
        benchmark_returns=None if benchmark_returns is None else benchmark_returns.copy(deep=True),
    )


def build_factor_portfolio_inputs(
    factor_data: pd.DataFrame,
    period: object,
    capital: float | None = None,
    long_short: bool = True,
    group_neutral: bool = False,
    equal_weight: bool = False,
    quantiles: Sequence[int] | None = None,
    groups: Sequence[str] | None = None,
    benchmark_period: object = "1D",
    weights: "pd.Series | None" = None,
) -> FactorPortfolioInputs:
    """Build typed daily returns, positions, and an optional benchmark series."""

    cumulative = factor_cumulative_returns(
        factor_data,
        period,
        long_short=long_short,
        group_neutral=group_neutral,
        equal_weight=equal_weight,
        quantiles=quantiles,
        groups=groups,
        weights=weights,
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
        weights=weights,
    )
    return _build_factor_portfolio_inputs_from_outputs(
        factor_data,
        daily_cumulative,
        returns,
        raw_positions,
        capital=capital,
        benchmark_period=benchmark_period,
    )


__all__ = [
    "FactorPortfolioInputs",
    "build_factor_portfolio_inputs",
    "factor_cumulative_returns",
    "factor_positions",
    "positions",
]
