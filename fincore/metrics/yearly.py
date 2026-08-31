#
# Copyright 2016 Quantopian, Inc.
# Copyright 2025 CloudQuant Team
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Yearly aggregation metrics."""

from __future__ import annotations

from typing import cast

import numpy as np
import pandas as pd

from fincore.metrics._annual import annual_return
from fincore.metrics.basic import ensure_datetime_index_series
from fincore.metrics.drawdown import max_drawdown
from fincore.metrics.frequencies import DAILY
from fincore.metrics.ratios import sharpe_ratio
from fincore.metrics.risk import annual_volatility
from fincore.runtime.time_series import AlignmentPolicy, align_binary_metric_inputs

__all__ = [
    "annual_active_return",
    "annual_active_return_by_year",
    "annual_return",
    "annual_return_by_year",
    "annual_volatility_by_year",
    "information_ratio_by_year",
    "max_drawdown_by_year",
    "sharpe_ratio_by_year",
]


def annual_return_by_year(
    returns: pd.Series | np.ndarray,
    period: str = DAILY,
    annualization: float | None = None,
) -> pd.Series | np.ndarray:
    """Determine the annual return for each calendar year.

    Parameters
    ----------
    returns : pd.Series or np.ndarray
        Non-cumulative returns indexed by date.
    period : str, optional
        Frequency of the returns (default 'daily').
    annualization : int, optional
        Factor to convert period returns to yearly returns.

    Returns
    -------
    pd.Series or np.ndarray
        Annual return for each calendar year.
    """
    if len(returns) < 1:
        return_as_array = isinstance(returns, np.ndarray)
        if return_as_array:
            return np.array([])
        return pd.Series(dtype="float64")

    return_as_array = isinstance(returns, np.ndarray)

    returns = ensure_datetime_index_series(returns, period=period)

    # ensure_datetime_index_series guarantees a DatetimeIndex.
    annual_returns = returns.groupby(cast("pd.DatetimeIndex", returns.index).year).apply(
        lambda ret: annual_return(ret, period=period, annualization=annualization)
    )

    return np.asarray(annual_returns) if return_as_array else annual_returns


def sharpe_ratio_by_year(
    returns: pd.Series | np.ndarray,
    risk_free: float = 0,
    period: str = DAILY,
    annualization: float | None = None,
) -> pd.Series | np.ndarray:
    """Determine the Sharpe ratio for each calendar year.

    Parameters
    ----------
    returns : pd.Series or np.ndarray
        Non-cumulative returns indexed by date.
    risk_free : float, optional
        Risk-free rate (default 0).
    period : str, optional
        Frequency of the returns (default 'daily').
    annualization : int, optional
        Factor to convert period returns to yearly returns.

    Returns
    -------
    pd.Series or np.ndarray
        Sharpe ratio for each calendar year.
    """
    if len(returns) < 1:
        return_as_array = isinstance(returns, np.ndarray)
        return np.array([]) if return_as_array else pd.Series(dtype="float64")

    return_as_array = isinstance(returns, np.ndarray)

    returns = ensure_datetime_index_series(returns, period=period)

    # ensure_datetime_index_series guarantees a DatetimeIndex.
    sharpe_by_year = returns.groupby(cast("pd.DatetimeIndex", returns.index).year).apply(
        lambda ret: sharpe_ratio(ret, risk_free=risk_free, period=period, annualization=annualization)
    )

    return np.asarray(sharpe_by_year) if return_as_array else sharpe_by_year


def max_drawdown_by_year(returns: pd.Series | np.ndarray) -> pd.Series | np.ndarray:
    """Determine the maximum drawdown for each calendar year.

    Parameters
    ----------
    returns : pd.Series or np.ndarray
        Non-cumulative returns indexed by date.

    Returns
    -------
    pd.Series or np.ndarray
        Maximum drawdown for each calendar year.
    """
    if len(returns) < 1:
        return_as_array = isinstance(returns, np.ndarray)
        return np.array([]) if return_as_array else pd.Series(dtype="float64")

    return_as_array = isinstance(returns, np.ndarray)

    returns = ensure_datetime_index_series(returns, period=DAILY)

    # ensure_datetime_index_series guarantees a DatetimeIndex.
    max_dd_by_year = returns.groupby(cast("pd.DatetimeIndex", returns.index).year).apply(lambda ret: max_drawdown(ret))
    return np.asarray(max_dd_by_year) if return_as_array else max_dd_by_year


def annual_volatility_by_year(
    returns: pd.Series | np.ndarray,
    period: str = DAILY,
    annualization: float | None = None,
) -> pd.Series | np.ndarray:
    """Determine the annual volatility for each calendar year.

    Parameters
    ----------
    returns : pd.Series or np.ndarray
        Non-cumulative returns indexed by date.
    period : str, optional
        Frequency of the returns (default 'daily').
    annualization : int, optional
        Factor to convert period returns to yearly returns.

    Returns
    -------
    pd.Series or np.ndarray
        Annualized volatility for each calendar year.
    """
    if len(returns) < 1:
        return_as_array = isinstance(returns, np.ndarray)
        return np.array([]) if return_as_array else pd.Series(dtype="float64")

    return_as_array = isinstance(returns, np.ndarray)

    returns = ensure_datetime_index_series(returns, period=period)

    # ensure_datetime_index_series guarantees a DatetimeIndex.
    annual_vol_by_year = returns.groupby(cast("pd.DatetimeIndex", returns.index).year).apply(
        lambda ret: annual_volatility(ret, period=period, annualization=annualization)
    )

    return np.asarray(annual_vol_by_year) if return_as_array else annual_vol_by_year


def annual_active_return(
    returns: pd.Series | np.ndarray,
    factor_returns: pd.Series | np.ndarray,
    period: str = DAILY,
    annualization: float | None = None,
    *,
    alignment: AlignmentPolicy = "inner",
    normalize_tz: str | None = None,
) -> float:
    """Calculate annual active return (strategy minus benchmark).

    Parameters
    ----------
    returns : pd.Series or np.ndarray
        Non-cumulative strategy returns.
    factor_returns : pd.Series or np.ndarray
        Non-cumulative benchmark returns.
    period : str, optional
        Frequency of the returns (default 'daily').
    annualization : int, optional
        Factor to convert period returns to yearly returns.

    Returns
    -------
    float
        Annual active return, or ``NaN`` if insufficient data.
    """
    returns_aligned, factor_aligned = align_binary_metric_inputs(
        returns, factor_returns, alignment=alignment, normalize_tz=normalize_tz
    )
    if len(returns_aligned) < 1:
        return np.nan

    strategy_annual = annual_return(returns_aligned, period, annualization)
    benchmark_annual = annual_return(factor_aligned, period, annualization)

    if np.isnan(strategy_annual) or np.isnan(benchmark_annual):
        return np.nan  # pragma: no cover -- Defensive edge case

    return strategy_annual - benchmark_annual  # type: ignore[return-value]


def annual_active_return_by_year(
    returns: pd.Series,
    factor_returns: pd.Series,
    period: str = DAILY,
    annualization: float | None = None,
    *,
    alignment: AlignmentPolicy = "inner",
    normalize_tz: str | None = None,
) -> pd.Series:
    """Determine the annual active return for each calendar year.

    Parameters
    ----------
    returns : pd.Series
        Non-cumulative strategy returns indexed by date.
    factor_returns : pd.Series
        Non-cumulative benchmark returns indexed by date.
    period : str, optional
        Frequency of the returns (default 'daily').
    annualization : int, optional
        Factor to convert period returns to yearly returns.

    Returns
    -------
    pd.Series
        Annual active return for each calendar year.
    """
    aligned_returns, aligned_factor = align_binary_metric_inputs(
        returns, factor_returns, alignment=alignment, normalize_tz=normalize_tz
    )
    # align_binary_metric_inputs preserves Series containers for Series inputs.
    returns = cast("pd.Series", aligned_returns)
    factor_returns = cast("pd.Series", aligned_factor)
    if len(returns) < 1:
        return pd.Series([], dtype=float)

    if not isinstance(returns.index, pd.DatetimeIndex):
        return pd.Series([], dtype=float)

    grouped = returns.groupby(returns.index.year)
    factor_grouped = factor_returns.groupby(cast("pd.DatetimeIndex", factor_returns.index).year)

    annual_active_returns = []
    for year in grouped.groups:
        if year in factor_grouped.groups:
            year_returns = grouped.get_group(year)
            year_factor = factor_grouped.get_group(year)
            active_return = annual_active_return(
                year_returns,
                year_factor,
                period,
                annualization,
                alignment="strict",
            )
            annual_active_returns.append((year, active_return))

    if not annual_active_returns:
        return pd.Series([], dtype=float)

    years, active_returns = zip(*annual_active_returns, strict=False)
    return pd.Series(active_returns, index=years)


def information_ratio_by_year(
    returns: pd.Series | np.ndarray,
    factor_returns: pd.Series | np.ndarray,
    period: str = DAILY,
    annualization: float | None = None,
    *,
    alignment: AlignmentPolicy = "inner",
    normalize_tz: str | None = None,
) -> pd.Series | np.ndarray:
    """Determine the information ratio for each calendar year.

    Parameters
    ----------
    returns : pd.Series or np.ndarray
        Non-cumulative strategy returns indexed by date.
    factor_returns : pd.Series or np.ndarray
        Non-cumulative benchmark returns indexed by date.
    period : str, optional
        Frequency of the returns (default 'daily').
    annualization : int, optional
        Factor to convert period returns to yearly returns.

    Returns
    -------
    pd.Series or np.ndarray
        Information ratio for each calendar year.
    """
    from fincore.metrics.ratios import information_ratio as calc_ir

    return_as_array = isinstance(returns, np.ndarray)
    aligned_returns, aligned_factor = align_binary_metric_inputs(
        returns, factor_returns, alignment=alignment, normalize_tz=normalize_tz
    )
    # align_binary_metric_inputs preserves the Series | ndarray input kinds.
    returns = cast("pd.Series | np.ndarray", aligned_returns)
    factor_returns = cast("pd.Series | np.ndarray", aligned_factor)
    if len(returns) < 1:
        return np.array([]) if return_as_array else pd.Series(dtype="float64")

    returns = ensure_datetime_index_series(returns, period=period)
    factor_returns = ensure_datetime_index_series(factor_returns, period=period)

    def calc_ir_for_year(returns_group: pd.Series) -> float:
        """Calculate Information Ratio for a specific year.

        Parameters
        ----------
        returns_group : pd.Series
            Returns for a specific year.

        Returns
        -------
        float
            Information Ratio for the year.
        """
        factor_group = factor_returns.loc[returns_group.index]
        return cast("float", calc_ir(returns_group, factor_group, period, annualization, alignment="strict"))

    # ensure_datetime_index_series guarantees a DatetimeIndex.
    information_ratios = returns.groupby(cast("pd.DatetimeIndex", returns.index).year).apply(calc_ir_for_year)

    if hasattr(information_ratios, "name"):
        information_ratios.name = None

    if return_as_array:
        return np.asarray(information_ratios)
    return information_ratios
