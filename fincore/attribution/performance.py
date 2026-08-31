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

"""Direct performance-attribution kernels for the attribution domain."""

from __future__ import annotations

from collections import OrderedDict
from typing import cast

import numpy as np
import pandas as pd

from fincore.contracts.time_series import AlignmentPolicy, align_time_series
from fincore.exceptions import DataAlignmentError

__all__ = [
    "align_and_warn",
    "compute_exposures",
    "compute_exposures_internal",
    "create_perf_attrib_stats",
    "cumulative_returns_less_costs",
    "perf_attrib",
    "perf_attrib_core",
]


def perf_attrib_core(
    returns: pd.Series,
    positions: pd.Series | pd.DataFrame,
    factor_returns: pd.DataFrame,
    factor_loadings: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Core performance attribution computation.

    Computes risk exposures and performance attribution by factor.

    Parameters
    ----------
    returns : pd.Series
        Daily returns of the strategy, noncumulative.
    positions : pd.Series or pd.DataFrame
        Daily position values with MultiIndex (dt, ticker).
    factor_returns : pd.DataFrame
        Daily factor returns with dates as index and factors as columns.
    factor_loadings : pd.DataFrame
        Factor loadings with MultiIndex (dt, ticker) and factors as columns.

    Returns
    -------
    tuple
        (risk_exposures_portfolio, perf_attribution) - Risk exposures over
        time and performance attribution data.
    """
    if positions is None:
        raise ValueError("Either provide positions or set positions data")
    if factor_returns is None:
        raise ValueError("Either provide factor_returns or set factor_returns/benchmark_rets")
    if factor_loadings is None:
        raise ValueError("Either provide factor_loadings or set factor_loadings data")

    returns = returns.copy()
    factor_returns = factor_returns.copy()
    factor_loadings = factor_loadings.copy()
    if isinstance(factor_loadings.index, pd.MultiIndex):
        factor_loadings.index = factor_loadings.index.set_names(["dt", "ticker"])
    positions = positions.copy()
    if isinstance(positions.index, pd.MultiIndex):
        positions.index = positions.index.set_names(["dt", "ticker"])

    risk_exposures_portfolio = compute_exposures_internal(
        positions=positions,
        factor_loadings=factor_loadings,
    )
    risk_exposures_portfolio = risk_exposures_portfolio.loc[risk_exposures_portfolio.notna().any(axis="columns")]
    common_dates = returns.index.intersection(factor_returns.index, sort=False)
    common_dates = common_dates.intersection(risk_exposures_portfolio.index, sort=False).sort_values()
    if common_dates.empty:
        raise DataAlignmentError("performance attribution has no common dates")
    returns = returns.loc[common_dates]
    factor_returns = factor_returns.loc[common_dates]
    risk_exposures_portfolio = risk_exposures_portfolio.loc[common_dates]

    perf_attrib_by_factor = risk_exposures_portfolio.multiply(factor_returns)
    common_returns = perf_attrib_by_factor.sum(axis="columns", min_count=len(perf_attrib_by_factor.columns))
    tilt_exposure = risk_exposures_portfolio.mean()

    tilt_returns_raw = factor_returns.multiply(tilt_exposure)
    if isinstance(tilt_returns_raw, pd.DataFrame):
        tilt_returns = tilt_returns_raw.sum(axis="columns", min_count=len(tilt_returns_raw.columns))
    else:
        tilt_returns = tilt_returns_raw.sum()

    timing_returns = common_returns - tilt_returns
    specific_returns = returns - common_returns

    returns_df = pd.DataFrame(
        OrderedDict(
            [
                ("total_returns", returns),
                ("common_returns", common_returns),
                ("specific_returns", specific_returns),
                ("tilt_returns", tilt_returns),
                ("timing_returns", timing_returns),
            ]
        )
    )

    perf_attribution = pd.concat([perf_attrib_by_factor, returns_df], axis="columns", sort=False)

    return risk_exposures_portfolio, perf_attribution


def _date_index(value: pd.Series | pd.DataFrame) -> pd.Index:
    if isinstance(value.index, pd.MultiIndex):
        return value.index.get_level_values(0).unique()
    return value.index


def _normalize_date_index(index: pd.Index, normalize_tz: str | None) -> pd.Index:
    if normalize_tz is None:
        return index
    if normalize_tz.upper() != "UTC":
        raise ValueError("normalize_tz currently supports only 'UTC'")
    if not isinstance(index, pd.DatetimeIndex):
        return index
    if index.tz is None:
        return index.tz_localize("UTC")
    return index.tz_convert("UTC")


def _normalize_attribution_index(
    value: pd.Series | pd.DataFrame,
    normalize_tz: str | None,
) -> pd.Series | pd.DataFrame:
    result = value.copy()
    if isinstance(result.index, pd.MultiIndex):
        if result.index.has_duplicates:
            raise DataAlignmentError("duplicate attribution labels are ambiguous")
        arrays = [
            _normalize_date_index(result.index.get_level_values(0), normalize_tz),
            *[result.index.get_level_values(level) for level in range(1, result.index.nlevels)],
        ]
        result.index = pd.MultiIndex.from_arrays(arrays, names=result.index.names)
    else:
        result.index = _normalize_date_index(result.index, normalize_tz)
    return result


def _select_attribution_dates(
    value: pd.Series | pd.DataFrame,
    dates: pd.Index,
) -> pd.Series | pd.DataFrame:
    if isinstance(value.index, pd.MultiIndex):
        return value[value.index.get_level_values(0).isin(dates)]
    return value.loc[dates]


def _date_completeness(value: pd.Series | pd.DataFrame) -> pd.Series:
    complete = value.notna() if isinstance(value, pd.Series) else value.notna().all(axis="columns")
    if isinstance(value.index, pd.MultiIndex):
        complete = complete.groupby(level=0).all()
    return pd.Series(np.where(complete, 1.0, np.nan), index=complete.index)


def _align_factor_columns(
    factor_returns: pd.DataFrame,
    factor_loadings: pd.DataFrame,
    *,
    policy: AlignmentPolicy,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if factor_returns.columns.has_duplicates or factor_loadings.columns.has_duplicates:
        raise DataAlignmentError("duplicate factor columns are ambiguous")
    if policy == "strict":
        if not factor_returns.columns.equals(factor_loadings.columns):
            raise DataAlignmentError("strict attribution requires identical factor columns")
        return factor_returns, factor_loadings

    common = factor_returns.columns.intersection(factor_loadings.columns, sort=False)
    if common.empty:
        raise DataAlignmentError("performance attribution has no common factor columns")
    return factor_returns.loc[:, common], factor_loadings.loc[:, common]


def _align_attribution_dates(
    returns: pd.Series,
    positions: pd.Series | pd.DataFrame,
    factor_returns: pd.DataFrame,
    factor_loadings: pd.DataFrame,
    *,
    policy: AlignmentPolicy,
    normalize_tz: str | None,
) -> tuple[pd.Series, pd.Series | pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    normalized = tuple(
        _normalize_attribution_index(value, normalize_tz)
        for value in (returns, positions, factor_returns, factor_loadings)
    )
    date_carriers = tuple(_date_completeness(value) for value in normalized)
    aligned_carriers = align_time_series(
        *date_carriers,
        policy=policy,
        normalize_tz=None,
    )
    common_dates = aligned_carriers[0].index
    if common_dates.empty:
        raise DataAlignmentError("performance attribution has no common dates")
    return tuple(_select_attribution_dates(value, common_dates) for value in normalized)  # type: ignore[return-value]


def compute_exposures_internal(
    positions: pd.Series | pd.DataFrame,
    factor_loadings: pd.DataFrame,
) -> pd.DataFrame:
    """Compute exposures from positions and factor loadings.

    Parameters
    ----------
    positions : pd.Series or pd.DataFrame
        Daily position values with MultiIndex (dt, ticker).
    factor_loadings : pd.DataFrame
        Factor loadings with MultiIndex (dt, ticker) and factors as columns.

    Returns
    -------
    pd.DataFrame
        Portfolio risk exposures by factor and date.
    """
    if positions is None:
        raise ValueError("Either provide positions or set positions data")
    if factor_loadings is None:
        raise ValueError("Either provide factor_loadings or set factor_loadings data")
    risk_exposures = factor_loadings.multiply(positions, axis="index")
    return risk_exposures.groupby(level="dt").sum(min_count=1)


def perf_attrib(
    returns: pd.Series,
    positions: pd.Series | pd.DataFrame | None = None,
    factor_returns: pd.DataFrame | None = None,
    factor_loadings: pd.DataFrame | None = None,
    transactions: pd.DataFrame | None = None,
    pos_in_dollars: bool = True,
    regression_style: str = "OLS",
    *,
    alignment: AlignmentPolicy = "inner",
    normalize_tz: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Calculate performance attribution.

    Parameters
    ----------
    returns : pd.Series
        Daily returns of the strategy, noncumulative.
    positions : pd.Series or pd.DataFrame, optional
        Daily position values. If DataFrame, columns are tickers.
    factor_returns : pd.DataFrame, optional
        Daily factor returns.
    factor_loadings : pd.DataFrame, optional
        Factor loadings with MultiIndex (dt, ticker).
    transactions : pd.DataFrame, optional
        Transaction data for turnover checks.
    pos_in_dollars : bool, optional
        If True, positions are in dollars (default True).
    regression_style : str, optional
        Regression style for attribution (default "OLS").

    Returns
    -------
    tuple of (pd.DataFrame, pd.DataFrame)
        (risk_exposures, perf_attrib_data) - risk exposures portfolio and
        performance attribution data.
    """
    if positions is None or factor_returns is None or factor_loadings is None:
        raise ValueError("positions, factor_returns, and factor_loadings are required")
    if regression_style != "OLS":
        raise ValueError("regression_style must be 'OLS'")

    normalized = tuple(
        _normalize_attribution_index(value, normalize_tz)
        for value in (returns, positions, factor_returns, factor_loadings)
    )
    # _normalize_attribution_index preserves each input's container type.
    returns = cast("pd.Series", normalized[0])
    positions = cast("pd.Series | pd.DataFrame", normalized[1])
    factor_returns = cast("pd.DataFrame", normalized[2])
    factor_loadings = cast("pd.DataFrame", normalized[3])
    factor_returns, factor_loadings = _align_factor_columns(
        factor_returns,
        factor_loadings,
        policy=alignment,
    )

    if alignment == "strict":
        returns, positions, factor_returns, factor_loadings = _align_attribution_dates(
            returns,
            positions,
            factor_returns,
            factor_loadings,
            policy=alignment,
            normalize_tz=None,
        )

    # Align data and warn about missing values
    (returns, positions, factor_returns, factor_loadings) = align_and_warn(
        returns,
        positions,
        factor_returns,
        factor_loadings,
        transactions=transactions,
        pos_in_dollars=pos_in_dollars,
    )

    returns, positions, factor_returns, factor_loadings = _align_attribution_dates(
        returns,
        positions,
        factor_returns,
        factor_loadings,
        policy=alignment,
        normalize_tz=None,
    )

    # Stack positions if needed (convert from DataFrame to Series with MultiIndex)
    if not isinstance(positions, pd.Series):
        positions = normalize_and_stack_positions(positions, pos_in_dollars=pos_in_dollars)

    preview_exposures = compute_exposures_internal(positions, factor_loadings)
    usable_dates = preview_exposures.index[preview_exposures.notna().any(axis="columns")]
    if usable_dates.empty:
        raise DataAlignmentError("performance attribution has no dates with usable ticker coverage")
    if alignment == "strict" and not returns.index.equals(usable_dates):
        raise DataAlignmentError("strict attribution requires usable ticker coverage on every date")
    returns = returns.loc[returns.index.intersection(usable_dates, sort=False)]
    factor_returns = factor_returns.loc[factor_returns.index.intersection(usable_dates, sort=False)]
    positions = _select_attribution_dates(positions, usable_dates)
    # _select_attribution_dates preserves DataFrame containers for DataFrames.
    factor_loadings = cast("pd.DataFrame", _select_attribution_dates(factor_loadings, usable_dates))

    risk_exposures, perf_attrib_data = perf_attrib_core(returns, positions, factor_returns, factor_loadings)

    return risk_exposures, perf_attrib_data


def normalize_and_stack_positions(
    positions: pd.DataFrame,
    pos_in_dollars: bool = True,
) -> pd.Series:
    """Normalize dollar positions to percentage weights and stack.

    Unlike :func:`fincore.portfolio.positions.stack_positions`, this version
    converts dollar positions to percentage weights when ``pos_in_dollars``
    is True, which is required by the performance attribution pipeline.

    Parameters
    ----------
    positions : pd.DataFrame
        Daily holdings indexed by date with tickers as columns.
    pos_in_dollars : bool
        If True, convert positions from dollars to percentages.

    Returns
    -------
    pd.Series
        Stacked positions with MultiIndex (dt, ticker).
    """
    positions = positions.copy()

    if pos_in_dollars:
        total = positions.sum(axis="columns")
        positions = positions.divide(total, axis=0)
        positions = positions.replace([np.inf, -np.inf], np.nan)

    if "cash" in positions.columns:
        positions = positions.drop("cash", axis=1)

    stacked = cast("pd.Series", positions.stack())
    stacked.index = stacked.index.set_names(["dt", "ticker"])

    return stacked


def compute_exposures(
    positions: pd.Series | pd.DataFrame,
    factor_loadings: pd.DataFrame,
    stack_positions: bool = True,
    pos_in_dollars: bool = True,
) -> pd.DataFrame:
    """Compute factor exposures from positions and factor loadings.

    Parameters
    ----------
    positions : pd.Series or pd.DataFrame
        Daily position values with MultiIndex (dt, ticker).
    factor_loadings : pd.DataFrame
        Factor loadings with MultiIndex (dt, ticker) and factors as columns.

    Returns
    -------
    pd.DataFrame
        Portfolio risk exposures by factor and date.
    """
    if stack_positions:
        if isinstance(positions, pd.DataFrame):
            positions = normalize_and_stack_positions(positions, pos_in_dollars=pos_in_dollars)
        elif not isinstance(positions, pd.Series):
            raise TypeError("positions must be a DataFrame or a stacked Series")
    return compute_exposures_internal(positions, factor_loadings)


def create_perf_attrib_stats(
    perf_attrib_: pd.DataFrame,
    risk_exposures: pd.DataFrame,
) -> tuple[pd.Series, pd.DataFrame]:
    """Take perf attribution data and compute annualized statistics.

    Computes annualized multifactor alpha, multifactor sharpe, risk exposures.

    Parameters
    ----------
    perf_attrib_ : pd.DataFrame
        Performance attribution output with columns total_returns, specific_returns,
        common_returns, and factor contributions.
    risk_exposures : pd.DataFrame
        Risk exposures by factor and date.

    Returns
    -------
    tuple of (pd.Series, pd.DataFrame)
        (summary_stats, risk_exposure_summary) - summary performance stats and
        annualized return/exposure by factor.
    """
    from collections import OrderedDict

    from fincore.metrics.ratios import sharpe_ratio
    from fincore.metrics.returns import cum_returns_final
    from fincore.metrics.yearly import annual_return

    summary = OrderedDict()
    total_returns = perf_attrib_["total_returns"]
    specific_returns = perf_attrib_["specific_returns"]
    common_returns = perf_attrib_["common_returns"]

    summary["Annualized Specific Return"] = annual_return(specific_returns)
    summary["Annualized Common Return"] = annual_return(common_returns)
    summary["Annualized Total Return"] = annual_return(total_returns)

    summary["Specific Sharpe Ratio"] = sharpe_ratio(specific_returns)

    summary["Cumulative Specific Return"] = cum_returns_final(specific_returns)
    summary["Cumulative Common Return"] = cum_returns_final(common_returns)
    summary["Total Returns"] = cum_returns_final(total_returns)

    summary_series = pd.Series(summary, name="")

    annualized_returns_by_factor = [annual_return(perf_attrib_[c]) for c in risk_exposures.columns]
    cumulative_returns_by_factor = [cum_returns_final(perf_attrib_[c]) for c in risk_exposures.columns]

    risk_exposure_summary = pd.DataFrame(
        data=OrderedDict(
            [
                ("Average Risk Factor Exposure", risk_exposures.mean(axis="index")),
                ("Annualized Return", annualized_returns_by_factor),
                ("Cumulative Return", cumulative_returns_by_factor),
            ]
        ),
        index=risk_exposures.columns,
    )

    return summary_series, risk_exposure_summary


def align_and_warn(
    returns: pd.Series,
    positions: pd.Series | pd.DataFrame,
    factor_returns: pd.DataFrame,
    factor_loadings: pd.DataFrame,
    transactions: pd.DataFrame | None = None,
    pos_in_dollars: bool = True,
) -> tuple[pd.Series, pd.Series | pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Make sure that all inputs have matching dates and tickers. Raise warnings if necessary.

    Parameters
    ----------
    returns : pd.Series
        Strategy returns.
    positions : pd.Series or pd.DataFrame
        Position data.
    factor_returns : pd.DataFrame
        Factor returns.
    factor_loadings : pd.DataFrame
        Factor loadings.
    transactions : pd.DataFrame, optional
        Transaction data for turnover checks.
    pos_in_dollars : bool, optional
        If True, positions are in dollars.

    Returns
    -------
    tuple of (pd.Series, pd.Series|pd.DataFrame, pd.DataFrame, pd.DataFrame)
        (returns, positions, factor_returns, factor_loadings) aligned to common dates/tickers.
    """
    import warnings

    from fincore.constants.style import PERF_ATTRIB_TURNOVER_THRESHOLD

    # Handle both DataFrame (unstacked) and Series (stacked) positions
    if isinstance(positions, pd.Series):
        position_tickers = positions.index.get_level_values(1).unique()
    else:
        position_tickers = positions.columns

    missing_stocks = position_tickers.difference(factor_loadings.index.get_level_values(1).unique())

    # cash will not be in factor_loadings
    num_stocks = len(position_tickers) - (1 if "cash" in position_tickers else 0)
    if "cash" in missing_stocks:
        missing_stocks = missing_stocks.drop("cash")
    num_stocks_covered = num_stocks - len(missing_stocks)
    missing_ratio = round(len(missing_stocks) / num_stocks, ndigits=3) if num_stocks > 0 else 0.0

    if num_stocks_covered == 0:
        raise ValueError(
            "Could not perform performance attribution. "
            "No factor loadings were available for this algorithm's positions."
        )

    if len(missing_stocks) > 0:
        if len(missing_stocks) > 5:
            missing_stocks_displayed = (" {} assets were missing factor loadings, including: {}..{}").format(
                len(missing_stocks),
                ", ".join(missing_stocks[:5].map(str)),
                missing_stocks[-1],
            )
            avg_allocation_msg = "selected missing assets"
        else:
            missing_stocks_displayed = f"The following assets were missing factor loadings: {list(missing_stocks)}."
            avg_allocation_msg = "missing assets"

        # Calculate average allocation for warning message
        avg_alloc: float | pd.Series
        if isinstance(positions, pd.Series):
            # missing_stocks is guaranteed non-empty here, so the sample set is also non-empty.
            sample_stocks = missing_stocks[:5].union(missing_stocks[[-1]])
            avg_alloc = cast(
                "float | pd.Series", positions[positions.index.get_level_values(1).isin(sample_stocks)].mean()
            )
        else:
            avg_alloc = cast("float | pd.Series", positions[missing_stocks[:5].union(missing_stocks[[-1]])].mean())

        missing_stocks_warning_msg = (
            "Could not determine risk exposures for some of this algorithm's "
            "positions. Returns from the missing assets will not be properly "
            "accounted for in performance attribution.\n"
            "\n"
            f"{missing_stocks_displayed}. "
            "Ignoring for exposure calculation and performance attribution. "
            f"Ratio of assets missing: {missing_ratio}. Average allocation of {avg_allocation_msg}:\n"
            "\n"
            f"{avg_alloc}.\n"
        )

        warnings.warn(missing_stocks_warning_msg, stacklevel=2)

        # Drop missing stocks from positions
        if isinstance(positions, pd.Series):
            positions = positions[~positions.index.get_level_values(1).isin(missing_stocks)]
        else:
            positions = positions.drop(missing_stocks, axis="columns", errors="ignore")

    # Get date index from positions
    if isinstance(positions, pd.Series):
        positions_dates = positions.index.get_level_values(0).unique()
    else:
        positions_dates = positions.index

    missing_factor_loadings_index = positions_dates.difference(factor_loadings.index.get_level_values(0).unique())

    if len(missing_factor_loadings_index) > 0:
        if len(missing_factor_loadings_index) > 5:
            missing_dates_displayed = f"(first missing is {missing_factor_loadings_index[0]}, last missing is {missing_factor_loadings_index[-1]})"
        else:
            missing_dates_displayed = str(list(missing_factor_loadings_index))

        warning_msg = f"Could not find factor loadings for {len(missing_factor_loadings_index)} dates: {missing_dates_displayed}. Truncating date range for performance attribution. "

        warnings.warn(warning_msg, stacklevel=2)

        # Drop dates from positions
        if isinstance(positions, pd.Series):
            positions = positions[~positions.index.get_level_values(0).isin(missing_factor_loadings_index)]
        else:
            positions = positions.drop(missing_factor_loadings_index, errors="ignore")

        returns = returns.drop(missing_factor_loadings_index, errors="ignore")
        factor_returns = factor_returns.drop(missing_factor_loadings_index, errors="ignore")

    if transactions is not None and pos_in_dollars:
        from fincore.portfolio.transactions import get_turnover

        # get_turnover expects an unstacked DataFrame; Series inputs fail the
        # same way at runtime either way (preexisting contract).
        turnover = get_turnover(cast("pd.DataFrame", positions), transactions).mean()
        if turnover > PERF_ATTRIB_TURNOVER_THRESHOLD:
            warning_msg = (
                "This algorithm has relatively high turnover of its "
                "positions. As a result, performance attribution might not be "
                "fully accurate.\n"
                "\n"
                "Performance attribution is calculated based "
                "on end-of-day holdings and does not account for intraday "
                "activity. Algorithms that derive a high percentage of "
                "returns from buying and selling within the same day may "
                "receive inaccurate performance attribution.\n"
            )
            warnings.warn(warning_msg, stacklevel=2)

    return returns, positions, factor_returns, factor_loadings


def cumulative_returns_less_costs(
    returns: pd.Series,
    costs: pd.Series | None,
) -> pd.Series:
    """Compute cumulative returns, less costs.

    Parameters
    ----------
    returns : pd.Series
        Non-cumulative returns.
    costs : pd.Series or None
        Transaction costs to subtract from returns.

    Returns
    -------
    pd.Series
        Cumulative returns after subtracting costs.
    """
    from fincore.metrics.returns import cum_returns

    if costs is None:
        # cum_returns mirrors the input container type for Series inputs.
        return cast("pd.Series", cum_returns(returns))
    return cast("pd.Series", cum_returns(returns - costs))
