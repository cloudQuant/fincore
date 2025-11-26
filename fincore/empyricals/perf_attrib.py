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

"""绩效归因函数模块."""

import numpy as np
import pandas as pd
from collections import OrderedDict

__all__ = [
    'perf_attrib_core',
    'compute_exposures_internal',
    'perf_attrib',
    'compute_exposures',
    'create_perf_attrib_stats',
    'align_and_warn',
    'cumulative_returns_less_costs',
]


def perf_attrib_core(returns, positions, factor_returns, factor_loadings):
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

    start = returns.index[0]
    end = returns.index[-1]
    factor_returns = factor_returns.loc[start:end]
    factor_loadings = factor_loadings.loc[start:end]
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
    risk_exposures_portfolio.index = returns.index

    perf_attrib_by_factor = risk_exposures_portfolio.multiply(factor_returns)
    common_returns = perf_attrib_by_factor.sum(axis="columns")
    tilt_exposure = risk_exposures_portfolio.mean()

    tilt_returns_raw = factor_returns.multiply(tilt_exposure)
    if isinstance(tilt_returns_raw, pd.DataFrame):
        tilt_returns = tilt_returns_raw.sum(axis="columns")
    else:
        tilt_returns = tilt_returns_raw.sum()

    timing_returns = common_returns - tilt_returns
    specific_returns = returns - common_returns

    returns_df = pd.DataFrame(
        OrderedDict([
            ("total_returns", returns),
            ("common_returns", common_returns),
            ("specific_returns", specific_returns),
            ("tilt_returns", tilt_returns),
            ("timing_returns", timing_returns),
        ])
    )

    perf_attribution = pd.concat([perf_attrib_by_factor, returns_df], axis="columns")
    perf_attribution.index = returns.index

    return risk_exposures_portfolio, perf_attribution


def compute_exposures_internal(positions, factor_loadings):
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
    risk_exposures = factor_loadings.multiply(positions, axis="rows")
    return risk_exposures.groupby(level="dt").sum()


def perf_attrib(returns, positions=None, factor_returns=None, factor_loadings=None,
                transactions=None, pos_in_dollars=True, regression_style='OLS'):
    """Calculate performance attribution.
    
    Returns
    -------
    tuple
        (risk_exposures, perf_attrib_data) - risk exposures portfolio and performance attribution data
    """
    if positions is None or factor_returns is None or factor_loadings is None:
        raise ValueError("positions, factor_returns, and factor_loadings are required")

    # Align data and warn about missing values
    (returns, positions, factor_returns, factor_loadings) = align_and_warn(
        returns,
        positions,
        factor_returns,
        factor_loadings,
        transactions=transactions,
        pos_in_dollars=pos_in_dollars,
    )

    # Stack positions if needed (convert from DataFrame to Series with MultiIndex)
    if not isinstance(positions, pd.Series):
        positions = stack_positions(positions, pos_in_dollars=pos_in_dollars)

    risk_exposures, perf_attrib_data = perf_attrib_core(
        returns, positions, factor_returns, factor_loadings
    )

    return risk_exposures, perf_attrib_data


def stack_positions(positions, pos_in_dollars=True):
    """Stack positions from DataFrame to Series with MultiIndex.
    
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
    
    if 'cash' in positions.columns:
        positions = positions.drop('cash', axis=1)
    
    if pos_in_dollars:
        # Convert to percentages
        total = positions.abs().sum(axis=1)
        positions = positions.divide(total, axis=0)
    
    # Stack to get MultiIndex Series
    stacked = positions.stack()
    stacked.index = stacked.index.set_names(['dt', 'ticker'])
    
    return stacked


def compute_exposures(positions, factor_loadings):
    """Compute factor exposures from positions."""
    return compute_exposures_internal(positions, factor_loadings)


def create_perf_attrib_stats(perf_attrib_, risk_exposures):
    """Take perf attribution data and compute annualized statistics.

    Computes annualized multifactor alpha, multifactor sharpe, risk exposures.
    """
    from collections import OrderedDict
    from fincore.empyricals.returns import annual_return, cum_returns_final
    from fincore.empyricals.ratios import sharpe_ratio

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

    summary = pd.Series(summary, name="")

    annualized_returns_by_factor = [annual_return(perf_attrib_[c]) for c in risk_exposures.columns]
    cumulative_returns_by_factor = [cum_returns_final(perf_attrib_[c]) for c in risk_exposures.columns]

    risk_exposure_summary = pd.DataFrame(
        data=OrderedDict(
            [
                ("Average Risk Factor Exposure", risk_exposures.mean(axis="rows")),
                ("Annualized Return", annualized_returns_by_factor),
                ("Cumulative Return", cumulative_returns_by_factor),
            ]
        ),
        index=risk_exposures.columns,
    )

    return summary, risk_exposure_summary


def align_and_warn(returns, positions, factor_returns, factor_loadings,
                   transactions=None, pos_in_dollars=True):
    """Make sure that all inputs have matching dates and tickers. Raise warnings if necessary."""
    import warnings
    from fincore.constants.style import PERF_ATTRIB_TURNOVER_THRESHOLD

    # Handle both DataFrame (unstacked) and Series (stacked) positions
    if isinstance(positions, pd.Series):
        position_tickers = positions.index.get_level_values(1).unique()
    else:
        position_tickers = positions.columns

    missing_stocks = position_tickers.difference(
        factor_loadings.index.get_level_values(1).unique()
    )

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
            missing_stocks_displayed = (
                " {} assets were missing factor loadings, including: {}..{}"
            ).format(
                len(missing_stocks),
                ", ".join(missing_stocks[:5].map(str)),
                missing_stocks[-1],
            )
            avg_allocation_msg = "selected missing assets"
        else:
            missing_stocks_displayed = (
                "The following assets were missing factor loadings: {}."
            ).format(list(missing_stocks))
            avg_allocation_msg = "missing assets"

        # Calculate average allocation for warning message
        if isinstance(positions, pd.Series):
            sample_stocks = missing_stocks[:5].union(missing_stocks[[-1]]) if len(missing_stocks) > 0 else []
            if len(sample_stocks) > 0:
                avg_alloc = positions[positions.index.get_level_values(1).isin(sample_stocks)].mean()
            else:
                avg_alloc = 0.0
        else:
            avg_alloc = positions[missing_stocks[:5].union(missing_stocks[[-1]])].mean()

        missing_stocks_warning_msg = (
            "Could not determine risk exposures for some of this algorithm's "
            "positions. Returns from the missing assets will not be properly "
            "accounted for in performance attribution.\n"
            "\n"
            "{}. "
            "Ignoring for exposure calculation and performance attribution. "
            "Ratio of assets missing: {}. Average allocation of {}:\n"
            "\n"
            "{}.\n"
        ).format(
            missing_stocks_displayed,
            missing_ratio,
            avg_allocation_msg,
            avg_alloc,
        )

        warnings.warn(missing_stocks_warning_msg)

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

    missing_factor_loadings_index = positions_dates.difference(
        factor_loadings.index.get_level_values(0).unique()
    )

    if len(missing_factor_loadings_index) > 0:
        if len(missing_factor_loadings_index) > 5:
            missing_dates_displayed = (
                "(first missing is {}, last missing is {})"
            ).format(
                missing_factor_loadings_index[0], missing_factor_loadings_index[-1]
            )
        else:
            missing_dates_displayed = list(missing_factor_loadings_index)

        warning_msg = (
            "Could not find factor loadings for {} dates: {}. "
            "Truncating date range for performance attribution. "
        ).format(
            len(missing_factor_loadings_index),
            missing_dates_displayed
        )

        warnings.warn(warning_msg)

        # Drop dates from positions
        if isinstance(positions, pd.Series):
            positions = positions[~positions.index.get_level_values(0).isin(missing_factor_loadings_index)]
        else:
            positions = positions.drop(missing_factor_loadings_index, errors="ignore")

        returns = returns.drop(missing_factor_loadings_index, errors="ignore")
        factor_returns = factor_returns.drop(missing_factor_loadings_index, errors="ignore")

    if transactions is not None and pos_in_dollars:
        from fincore.empyricals.transactions import get_turnover
        turnover = get_turnover(positions, transactions).mean()
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
            warnings.warn(warning_msg)

    return returns, positions, factor_returns, factor_loadings


def cumulative_returns_less_costs(returns, costs):
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
    from fincore.empyricals.returns import cum_returns

    if costs is None:
        return cum_returns(returns)
    return cum_returns(returns - costs)
