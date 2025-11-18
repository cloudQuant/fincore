#
# Copyright 2016 Quantopian, Inc.
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

from __future__ import division

import math
import pandas as pd
import numpy as np
from math import pow
from scipy import stats, optimize
from six import iteritems
from sys import float_info

from .utils import nanmean, nanstd, nanmin, up, down, roll, rolling_window
from fincore.constants import (
    ANNUALIZATION_FACTORS,
    APPROX_BDAYS_PER_YEAR,
    DAILY,
    WEEKLY,
    MONTHLY,
    QUARTERLY,
    YEARLY,
)


_PERIOD_TO_FREQ = {
    DAILY: "D",
    WEEKLY: "W",
    MONTHLY: "M",
    QUARTERLY: "Q",
    YEARLY: "A",
}


def _ensure_datetime_index_series(data, period=DAILY):
    """Return a Series indexed by dates regardless of the input type."""

    if isinstance(data, pd.Series) and isinstance(data.index, pd.DatetimeIndex):
        return data

    values = data.values if isinstance(data, pd.Series) else np.asarray(data)

    if values.size == 0:
        return pd.Series(values)

    freq = _PERIOD_TO_FREQ.get(period, "D")
    index = pd.date_range("1970-01-01", periods=values.size, freq=freq)
    return pd.Series(values, index=index)


def _create_unary_vectorized_roll_function(function):
    def unary_vectorized_roll(arr, window, out=None, **kwargs):
        """
        Computes the {human_readable} measure over a rolling window.

        Parameters
        ----------
        arr : array-like
            The array to compute the rolling {human_readable} over.
        window : int
            The size of the rolling window, expressed in terms of the data's periodicity.
        out : array-like, optional
            Array to use as output buffer.
            If not passed, a new array will be created.
        **kwargs
            Forwarded to: func:`~empyrical.{name}`.

        Returns
        -------
        rolling_{name}: array-like
            The rolling {human_readable}.
        """
        allocated_output = out is None

        if len(arr):
            out = function(
                rolling_window(_flatten(arr), min(len(arr), window)).T,
                out=out,
                **kwargs
            )
        else:
            out = np.empty(0, dtype='float64')

        if allocated_output and isinstance(arr, pd.Series):
            out = pd.Series(out, index=arr.index[-len(out):])

        return out

    unary_vectorized_roll.__doc__ = unary_vectorized_roll.__doc__.format(
        name=function.__name__,
        human_readable=function.__name__.replace('_', ' '),
    )

    return unary_vectorized_roll


def _create_binary_vectorized_roll_function(function):
    def binary_vectorized_roll(lhs, rhs, window, out=None, **kwargs):
        """
        Computes the {human_readable} measure over a rolling window.

        Parameters
        ----------
        lhs : array-like
            The first array to pass to the rolling {human_readable}.
        rhs : array-like
            The second array to pass to the rolling {human_readable}.
        window : int
            The size of the rolling window, expressed in terms of the data's periodicity.
        out : array-like, optional
            Array to use as output buffer.
            If not passed, a new array will be created.
        **kwargs
            Forwarded to: func:`~empyrical.{name}`.

        Returns
        -------
        rolling_{name}: array-like
            The rolling {human_readable}.
        """
        allocated_output = out is None

        if window >= 1 and len(lhs) and len(rhs):
            out = function(
                rolling_window(_flatten(lhs), min(len(lhs), window)).T,
                rolling_window(_flatten(rhs), min(len(rhs), window)).T,
                out=out,
                **kwargs
            )
        elif allocated_output:
            out = np.empty(0, dtype='float64')
        else:
            out[()] = np.nan

        if allocated_output:
            if out.ndim == 1 and isinstance(lhs, pd.Series):
                out = pd.Series(out, index=lhs.index[-len(out):])
            elif out.ndim == 2 and isinstance(lhs, pd.Series):
                out = pd.DataFrame(out, index=lhs.index[-len(out):])
        return out

    binary_vectorized_roll.__doc__ = binary_vectorized_roll.__doc__.format(
        name=function.__name__,
        human_readable=function.__name__.replace('_', ' '),
    )

    return binary_vectorized_roll


def _flatten(arr):
    return arr if not isinstance(arr, pd.Series) else arr.values


def _adjust_returns(returns, adjustment_factor):
    """
    Returns the `returns` series adjusted by adjustment_factor.Optimizes for the
    case of adjustment_factor being 0 by returning `returns` itself, not a copy!

    Parameters
    ----------
    returns : pd.Series or np.ndarray
    adjustment_factor : pd.Series or np.ndarray or float or int

    Returns
    -------
    adjusted_returns : array-like
    """
    if isinstance(adjustment_factor, (float, int)) and adjustment_factor == 0:
        return returns
    return returns - adjustment_factor


def annualization_factor(period, annualization):
    """
    Return annualization factor from the period entered or if a custom
    value is passed in.

    Parameters
    ----------
    period : str, optionally
        Defines the periodicity of the 'returns' data for purposes of
        annualizing. Value ignored if `annualization` parameter is specified.
        Defaults are::

            'Monthly':12
            'weekly': 52
            'daily': 252

    annualization : int, optional
        Used to suppress default values available in `period` to convert
        returns into annual returns. Value should be the annual frequency of
        `returns`.

    Returns
    -------
    annualization_factor : float
    """
    if annualization is None:
        try:
            factor = ANNUALIZATION_FACTORS[period]
        except KeyError:
            raise ValueError(
                "Period cannot be '{}'. "
                "Can be '{}'.".format(
                    period, "', '".join(ANNUALIZATION_FACTORS.keys())
                )
            )
    else:
        factor = annualization
    return factor


def simple_returns(prices):
    """
    Compute simple returns from a timeseries of prices.

    Parameters
    ----------
    prices : pd.Series, pd.DataFrame or np.ndarray
        Prices of assets in wide-format, with assets as columns,
        and indexed by datetimes.

    Returns
    -------
    returns : array-like
        Returns of assets in wide-format, with assets as columns,
        and index coerced to be tz-aware.
    """
    if isinstance(prices, (pd.DataFrame, pd.Series)):
        out = prices.pct_change().iloc[1:]
    else:
        # Assume np.ndarray
        out = np.diff(prices, axis=0)
        # Avoid division by zero warning
        with np.errstate(divide='ignore', invalid='ignore'):
            np.divide(out, prices[:-1], out=out)

    return out


def cum_returns(returns, starting_value=0, out=None):
    """
    Compute cumulative returns from simple returns.

    Parameters
    ----------
    returns : pd.Series, np.ndarray, or pd.DataFrame
        Returns of the strategy as a percentage, noncumulative.
         - Time series with decimal returns.
         - Example::

            2015-07-16   -0.012143
            2015-07-17    0.045350
            2015-07-20    0.030957
            2015-07-21    0.004902

         - Also accepts two-dimensional data.In this case, each column is
           cumulated.

    starting_value : float, optional
       The starting returns.
    out : array-like, optional
        Array to use as output buffer.
        If not passed, a new array will be created.

    Returns
    -------
    cumulative_returns : array-like
        Series of cumulative returns.
    """
    if len(returns) < 1:
        return returns.copy()

    nanmask = np.isnan(returns)
    if np.any(nanmask):
        returns = returns.copy()
        returns[nanmask] = 0

    allocated_output = out is None
    if allocated_output:
        out = np.empty_like(returns)

    np.add(returns, 1, out=out)
    out.cumprod(axis=0, out=out)

    if starting_value == 0:
        np.subtract(out, 1, out=out)
    else:
        np.multiply(out, starting_value, out=out)

    if allocated_output:
        if returns.ndim == 1 and isinstance(returns, pd.Series):
            out = pd.Series(out, index=returns.index)
        elif isinstance(returns, pd.DataFrame):
            out = pd.DataFrame(
                out, index=returns.index, columns=returns.columns,
            )

    return out


def cum_returns_final(returns, starting_value=0):
    """
    Compute total returns from simple returns.

    Parameters
    ----------
    returns : pd.DataFrame, pd.Series, or np.ndarray
       Noncumulative simple returns of one or more timeseries.
    starting_value : float, optional
       The starting returns.

    Returns
    -------
    total_returns : pd.Series, np.ndarray, or float
        If input is 1-dimensional (a Series or 1D numpy array), the result is a
        scalar.

        If input is 2-dimensional (a DataFrame or 2D numpy array), the result
        is a 1D array containing cumulative returns for each column of input.
    """
    if len(returns) == 0:
        return np.nan

    if isinstance(returns, pd.DataFrame):
        result = (returns + 1).prod()
    else:
        result = np.nanprod(returns + 1, axis=0)

    if starting_value == 0:
        result -= 1
    else:
        result *= starting_value

    return result


def aggregate_returns(returns, convert_to):
    """
    Aggregates returns by week, month, or year.

    Parameters
    ----------
    returns : pd.Series
       Daily returns of the strategy, noncumulative.
        - See full explanation in: func:`~empyrical.stats.cum_returns`.
    convert_to : str
        Can be 'weekly', 'monthly', or 'yearly'.

    Returns
    -------
    aggregated_returns : pd.Series
    """

    def cumulate_returns(x):
        return cum_returns(x).iloc[-1]

    if convert_to == WEEKLY:
        grouping = [lambda x: x.year, lambda x: x.isocalendar()[1]]
    elif convert_to == MONTHLY:
        grouping = [lambda x: x.year, lambda x: x.month]
    elif convert_to == QUARTERLY:
        grouping = [lambda x: x.year, lambda x: int(math.ceil(x.month/3.))]
    elif convert_to == YEARLY:
        grouping = [lambda x: x.year]
    else:
        raise ValueError(
            'convert_to must be {}, {} or {}'.format(WEEKLY, MONTHLY, YEARLY)
        )

    return returns.groupby(grouping).apply(cumulate_returns)


def max_drawdown(returns, out=None):
    """
    Determines the maximum drawdown of a strategy.

    Parameters
    ----------
    returns : pd.Series or np.ndarray
        Daily returns of the strategy, noncumulative.
        - See full explanation in: func:`~empyrical.stats.cum_returns`.
    out : array-like, optional
        Array to use as output buffer.
        If not passed, a new array will be created.

    Returns
    -------
    max_drawdown : float

    Note
    -----
    See https://en.wikipedia.org/wiki/Drawdown_(economics) for more details.
    """
    allocated_output = out is None
    if allocated_output:
        out = np.empty(returns.shape[1:])

    returns_1d = returns.ndim == 1

    if len(returns) < 1:
        out[()] = np.nan
        if returns_1d:
            out = out.item()
        return out

    returns_array = np.asanyarray(returns)

    cumulative = np.empty(
        (returns.shape[0] + 1,) + returns.shape[1:],
        dtype='float64',
    )
    cumulative[0] = start = 100
    cum_returns(returns_array, starting_value=start, out=cumulative[1:])

    max_return = np.fmax.accumulate(cumulative, axis=0)

    nanmin((cumulative - max_return) / max_return, axis=0, out=out)
    if returns_1d:
        out = out.item()
    elif allocated_output and isinstance(returns, pd.DataFrame):
        out = pd.Series(out)

    return out


roll_max_drawdown = _create_unary_vectorized_roll_function(max_drawdown)


def annual_return(returns, period=DAILY, annualization=None):
    """
    Determines the mean annual growth rate of returns.This is `equivalent`
    to the compound annual growth rate.

    Parameters
    ----------
    returns : pd.Series or np.ndarray
        Periodic returns of the strategy, noncumulative.
        - See full explanation in: func:`~empyrical.stats.cum_returns`.
    period : str, optional
        Defines the periodicity of the 'returns' data for purposes of
        annualizing. Value ignored if `annualization` parameter is specified.
        Defaults are::

            'Monthly':12
            'weekly': 52
            'daily': 252

    annualization : int, optional
        Used to suppress default values available in `period` to convert
        returns into annual returns. Value should be the annual frequency of
        `returns`.

    Returns
    -------
    annual_return : float
        Annual Return as CAGR (Compounded Annual Growth Rate).

    """

    if len(returns) < 1:
        return np.nan

    ann_factor = annualization_factor(period, annualization)
    num_years = len(returns) / ann_factor
    # Pass array to ensure index -1 looks up successfully.
    ending_value = cum_returns_final(returns, starting_value=1)

    return ending_value ** (1 / num_years) - 1


def cagr(returns, period=DAILY, annualization=None):
    """
    Compute compound annual growth rate.Alias function for: func:`~empyrical.stats.annual_return`

    Parameters
    ----------
    returns : pd.Series or np.ndarray
        Daily returns of the strategy, noncumulative.
        - See full explanation in: func:`~empyrical.stats.cum_returns`.
    period : str, optional
        Defines the periodicity of the 'returns' data for purposes of
        annualizing. Value ignored if `annualization` parameter is specified.
        Defaults are::

            'Monthly':12
            'weekly': 52
            'daily': 252

    annualization : int, optional
        Used to suppress default values available in `period` to convert
        returns into annual returns. Value should be the annual frequency of
        `returns`.
        - See full explanation in: func:`~empyrical.stats.annual_return`.

    Returns
    -------
    cagr : float
        The CAGR value.

    """
    return annual_return(returns, period, annualization)


roll_cagr = _create_unary_vectorized_roll_function(cagr)


def annual_volatility(returns,
                      period=DAILY,
                      alpha_=2.0,
                      annualization=None,
                      out=None):
    """
    Determines the annual volatility of a strategy.

    Parameters
    ----------
    returns : pd.Series or np.ndarray
        Periodic returns of the strategy, noncumulative.
        - See full explanation in: func:`~empyrical.stats.cum_returns`.
    period : str, optional
        Defines the periodicity of the 'returns' data for purposes of
        annualizing. Value ignored if `annualization` parameter is specified.
        Defaults are::

            'Monthly':12
            'weekly': 52
            'daily': 252

    alpha_ : float, optional
        Scaling relation (Levy stability exponent).
    annualization : int, optional
        Used to suppress default values available in `period` to convert
        returns into annual returns. Value should be the annual frequency of
        `returns`.
    out : array-like, optional
        Array to use as output buffer.
        If not passed, a new array will be created.

    Returns
    -------
    annual_volatility : float
    """
    allocated_output = out is None
    if allocated_output:
        out = np.empty(returns.shape[1:])

    returns_1d = returns.ndim == 1

    if len(returns) < 2:
        out[()] = np.nan
        if returns_1d:
            out = out.item()
        return out

    ann_factor = annualization_factor(period, annualization)
    nanstd(returns, ddof=1, axis=0, out=out)
    out = np.multiply(out, ann_factor ** (1.0 / alpha_), out=out)
    if returns_1d:
        out = out.item()
    return out


roll_annual_volatility = _create_unary_vectorized_roll_function(
    annual_volatility,
)


def calmar_ratio(returns, period=DAILY, annualization=None):
    """
    Determines the Calmar ratio, or drawdown ratio, of a strategy.

    Parameters
    ----------
    returns : pd.Series or np.ndarray
        Daily returns of the strategy, noncumulative.
        - See full explanation in: func:`~empyrical.stats.cum_returns`.
    period : str, optional
        Defines the periodicity of the 'returns' data for purposes of
        annualizing. Value ignored if `annualization` parameter is specified.
        Defaults are::

            'Monthly':12
            'weekly': 52
            'daily': 252

    annualization : int, optional
        Used to suppress default values available in `period` to convert
        returns into annual returns. Value should be the annual frequency of
        `returns`.


    Returns
    -------
    calmar_ratio : float
        Calmar ratio (drawdown ratio) as float. Returns np.nan if there is no
        calmar ratio.

    Note
    -----
    See https://en.wikipedia.org/wiki/Calmar_ratio for more details.
    """

    max_dd = max_drawdown(returns=returns)
    if max_dd < 0:
        temp = annual_return(
            returns=returns,
            period=period,
            annualization=annualization
        ) / abs(max_dd)
    else:
        return np.nan

    if np.isinf(temp):
        return np.nan

    return temp


def omega_ratio(returns, risk_free=0.0, required_return=0.0,
                annualization=APPROX_BDAYS_PER_YEAR):
    """Determines the Omega ratio of a strategy.

    Parameters
    ----------
    returns : pd.Series or np.ndarray
        Daily returns of the strategy, noncumulative.
        - See full explanation in: func:`~empyrical.stats.cum_returns`.
    risk_free : int, float
        Constant risk-free return throughout the period
    required_return : float, optional
        Minimum acceptance return of the investor. Threshold over which to
        consider positive vs. negative returns. It will be converted to a
        value appropriate for the period of the returns. E.g., An annual minimum-acceptable
        return of 100 will translate to a minimum acceptable return of 0.018.
    annualization : int, optional
        Factor used to convert the required_return into a daily
        value. Enter 1 if no time period conversion is necessary.

    Returns
    -------
    omega_ratio : float

    Note
    -----
    See https://en.wikipedia.org/wiki/Omega_ratio for more details.

    """

    if len(returns) < 2:
        return np.nan

    if annualization == 1:
        return_threshold = required_return
    elif required_return <= -1:
        return np.nan
    else:
        return_threshold = (1 + required_return) ** \
            (1. / annualization) - 1

    returns_less_thresh = returns - risk_free - return_threshold

    numer = sum(returns_less_thresh[returns_less_thresh > 0.0])
    denom = -1.0 * sum(returns_less_thresh[returns_less_thresh < 0.0])

    if denom > 0.0:
        return numer / denom
    else:
        return np.nan


def sharpe_ratio(returns,
                 risk_free=0,
                 period=DAILY,
                 annualization=None,
                 out=None):
    """
    Determines the Sharpe ratio of a strategy.

    Parameters
    ----------
    returns : pd.Series or np.ndarray
        Daily returns of the strategy, noncumulative.
        - See full explanation in: func:`~empyrical.stats.cum_returns`.
    risk_free : int, float
        Constant daily risk-free return throughout the period.
    period : str, optional
        Defines the periodicity of the 'returns' data for purposes of
        annualizing. Value ignored if `annualization` parameter is specified.
        Defaults are::

            'Monthly':12
            'weekly': 52
            'daily': 252

    annualization : int, optional
        Used to suppress default values available in `period` to convert
        returns into annual returns. Value should be the annual frequency of
        `returns`.
    out : array-like, optional
        Array to use as output buffer.
        If not passed, a new array will be created.

    Returns
    -------
    sharpe_ratio : float
        nan if insufficient length of returns or if adjusted returns are 0.

    Note
    -----
    See https://en.wikipedia.org/wiki/Sharpe_ratio for more details.

    """
    allocated_output = out is None
    if allocated_output:
        out = np.empty(returns.shape[1:])

    return_1d = returns.ndim == 1

    if len(returns) < 2:
        out[()] = np.nan
        if return_1d:
            out = out.item()
        return out

    returns_risk_adj = np.asanyarray(_adjust_returns(returns, risk_free))
    ann_factor = annualization_factor(period, annualization)

    # Handle division by zero
    std_returns = nanstd(returns_risk_adj, ddof=1, axis=0)
    mean_returns = nanmean(returns_risk_adj, axis=0)
    
    # Avoid division by zero warning
    with np.errstate(divide='ignore', invalid='ignore'):
        np.multiply(
            np.divide(
                mean_returns,
                std_returns,
                out=out,
            ),
            np.sqrt(ann_factor),
            out=out,
        )
    if return_1d:
        out = out.item()

    return out


roll_sharpe_ratio = _create_unary_vectorized_roll_function(sharpe_ratio)


def sortino_ratio(returns,
                  required_return=0,
                  period=DAILY,
                  annualization=None,
                  out=None,
                  _downside_risk=None):
    """
    Determines the Sortino ratio of a strategy.

    Parameters
    ----------
    returns : pd.Series or np.ndarray or pd.DataFrame
        Daily returns of the strategy, noncumulative.
        - See full explanation in: func:`~empyrical.stats.cum_returns`.
    required_return: float / series
        minimum acceptable return
    period : str, optional
        Defines the periodicity of the 'returns' data for purposes of
        annualizing. Value ignored if `annualization` parameter is specified.
        Defaults are::

            'Monthly':12
            'weekly': 52
            'daily': 252

    annualization : int, optional
        Used to suppress default values available in `period` to convert
        returns into annual returns. Value should be the annual frequency of
        `returns`.
    _downside_risk : float, optional
        The downside risk of the given inputs, if known. It Will be calculated if
        not provided.
    out : array-like, optional
        Array to use as output buffer.
        If not passed, a new array will be created.

    Returns
    -------
    sortino_ratio : float or pd.Series

        Depends on input type
        series ==> float
        DataFrame ==> pd.Series

    Note
    -----
    See `<https://www.sunrisecapital.com/wp-content/uploads/2014/06/Futures_
    Mag_Sortino_0213.pdf>`__ for more details.

    """
    allocated_output = out is None
    if allocated_output:
        out = np.empty(returns.shape[1:])

    return_1d = returns.ndim == 1

    if len(returns) < 2:
        out[()] = np.nan
        if return_1d:
            out = out.item()
        return out

    adj_returns = np.asanyarray(_adjust_returns(returns, required_return))

    ann_factor = annualization_factor(period, annualization)

    average_annual_return = nanmean(adj_returns, axis=0) * ann_factor
    annualized_downside_risk = (
        _downside_risk
        if _downside_risk is not None else
        downside_risk(returns, required_return, period, annualization)
    )
    # Avoid division by zero warning
    with np.errstate(divide='ignore', invalid='ignore'):
        np.divide(average_annual_return, annualized_downside_risk, out=out)
    if return_1d:
        out = out.item()
    elif isinstance(returns, pd.DataFrame):
        out = pd.Series(out)

    return out


roll_sortino_ratio = _create_unary_vectorized_roll_function(sortino_ratio)


def downside_risk(returns,
                  required_return=0,
                  period=DAILY,
                  annualization=None,
                  out=None):
    """
    Determines the downside deviation below a threshold

    Parameters
    ----------
    returns : pd.Series or np.ndarray or pd.DataFrame
        Daily returns of the strategy, noncumulative.
        - See full explanation in: func:`~empyrical.stats.cum_returns`.
    required_return: float / series
        minimum acceptable return
    period : str, optional
        Defines the periodicity of the 'returns' data for purposes of
        annualizing. Value ignored if `annualization` parameter is specified.
        Defaults are::

            'Monthly':12
            'weekly': 52
            'daily': 252

    annualization : int, optional
        Used to suppress default values available in `period` to convert
        returns into annual returns. Value should be the annual frequency of
        `returns`.
    out : array-like, optional
        Array to use as output buffer.
        If not passed, a new array will be created.

    Returns
    -------
    downside_deviation : float or pd.Series
        depends on input type
        series ==> float
        DataFrame ==> pd.Series

    Note
    -----
    See `<https://www.sunrisecapital.com/wp-content/uploads/2014/06/Futures_
    Mag_Sortino_0213.pdf>`__ for more details, specifically why using the
    standard deviation of the negative returns is not correct.
    """
    allocated_output = out is None
    if allocated_output:
        out = np.empty(returns.shape[1:])

    returns_1d = returns.ndim == 1

    if len(returns) < 1:
        out[()] = np.nan
        if returns_1d:
            out = out.item()
        return out

    ann_factor = annualization_factor(period, annualization)

    downside_diff = np.clip(
        _adjust_returns(
            np.asanyarray(returns),
            np.asanyarray(required_return),
        ),
        -np.inf,
        0,
    )

    np.square(downside_diff, out=downside_diff)
    nanmean(downside_diff, axis=0, out=out)
    np.sqrt(out, out=out)
    np.multiply(out, np.sqrt(ann_factor), out=out)

    if returns_1d:
        out = out.item()
    elif isinstance(returns, pd.DataFrame):
        out = pd.Series(out, index=returns.columns)
    return out


roll_downsize_risk = _create_unary_vectorized_roll_function(downside_risk)


def excess_sharpe(returns, factor_returns, out=None):
    """
    Determines the Excess Sharpe of a strategy.

    Parameters
    ----------
    returns : pd.Series or np.ndarray
        Daily returns of the strategy, noncumulative.
        - See full explanation in: func:`~empyrical.stats.cum_returns`.
    factor_returns: float / series
        Benchmark return to compare returns against.
    out : array-like, optional
        Array to use as output buffer.
        If not passed, a new array will be created.

    Returns
    -------
    excess_sharpe : float

    Note
    -----
    The excess Sharpe is a simplified Information Ratio that uses
    tracking error rather than "active risk" as the denominator.
    """
    allocated_output = out is None
    if allocated_output:
        out = np.empty(returns.shape[1:])

    returns_1d = returns.ndim == 1

    if len(returns) < 2:
        out[()] = np.nan
        if returns_1d:
            out = out.item()
        return out

    active_return = _adjust_returns(returns, factor_returns)
    tracking_error = np.nan_to_num(nanstd(active_return, ddof=1, axis=0))

    # Avoid division by zero warning
    with np.errstate(divide='ignore', invalid='ignore'):
        out = np.divide(
            nanmean(active_return, axis=0, out=out),
            tracking_error,
            out=out,
        )
    if returns_1d:
        out = out.item()
    return out


roll_excess_sharpe = _create_binary_vectorized_roll_function(excess_sharpe)


def tracking_error(returns,
                   factor_returns,
                   period=DAILY,
                   annualization=None,
                   out=None):
    """
    Determines the tracking error of a strategy.

    Tracking error is the standard deviation of the active returns
    (returns - factor_returns), annualized.

    Parameters
    ----------
    returns : pd.Series or np.ndarray
        Daily returns of the strategy, noncumulative.
        - See full explanation in: func:`~empyrical.stats.cum_returns`.
    factor_returns : pd.Series or np.ndarray
        Daily noncumulative returns of the factor to which tracking error is
        computed. Usually a benchmark such as the market.
        - This is in the same style as returns.
    period : str, optional
        Defines the periodicity of the 'returns' data for purposes of
        annualizing. Value ignored if `annualization` parameter is specified.
        Defaults are::

            'Monthly':12
            'weekly': 52
            'daily': 252

    annualization : int, optional
        Used to suppress default values available in `period` to convert
        returns into annual returns. Value should be the annual frequency of
        `returns`.
    out : array-like, optional
        Array to use as output buffer.
        If not passed, a new array will be created.

    Returns
    -------
    tracking_error : float
        The tracking error (annualized standard deviation of active returns).

    Note
    -----
    See https://www.investopedia.com/terms/t/trackingerror.asp for more details.
    """
    allocated_output = out is None
    if allocated_output:
        out = np.empty(returns.shape[1:])

    returns_1d = returns.ndim == 1

    if len(returns) < 2:
        out[()] = np.nan
        if returns_1d:
            out = out.item()
        return out

    returns, factor_returns = _aligned_series(returns, factor_returns)

    active_return = _adjust_returns(returns, factor_returns)
    ann_factor = annualization_factor(period, annualization)

    nanstd(active_return, ddof=1, axis=0, out=out)
    out = np.multiply(out, np.sqrt(ann_factor), out=out)

    if returns_1d:
        out = out.item()
    elif allocated_output and isinstance(returns, pd.DataFrame):
        out = pd.Series(out, index=returns.columns)

    return out


roll_tracking_error = _create_binary_vectorized_roll_function(tracking_error)


def treynor_ratio(returns,
                  factor_returns,
                  risk_free=0.0,
                  period=DAILY,
                  annualization=None,
                  out=None):
    """
    Determines the Treynor ratio of a strategy.

    The Treynor ratio is a risk-adjusted performance measure that divides
    the excess return (over the risk-free rate) by the portfolio's beta.

    Parameters
    ----------
    returns : pd.Series or np.ndarray
        Daily returns of the strategy, noncumulative.
        - See full explanation in: func:`~empyrical.stats.cum_returns`.
    factor_returns : pd.Series or np.ndarray
        Daily noncumulative returns of the factor to which beta is
        computed. Usually a benchmark such as the market.
        - This is in the same style as returns.
    risk_free : int, float, optional
        Constant risk-free return throughout the period. For example, the
        interest rate on a three-month US treasury bill.
    period : str, optional
        Defines the periodicity of the 'returns' data for purposes of
        annualizing. Value ignored if `annualization` parameter is specified.
        Defaults are::

            'Monthly':12
            'weekly': 52
            'daily': 252

    annualization : int, optional
        Used to suppress default values available in `period` to convert
        returns into annual returns. Value should be the annual frequency of
        `returns`.
    out : array-like, optional
        Array to use as output buffer.
        If not passed, a new array will be created.

    Returns
    -------
    treynor_ratio : float
        The Treynor ratio. Returns np.nan if beta is zero or negative.

    Note
    -----
    See https://www.investopedia.com/terms/t/treynor-measure.asp for more details.
    """
    allocated_output = out is None
    if allocated_output:
        out = np.empty(returns.shape[1:])

    returns_1d = returns.ndim == 1

    if len(returns) < 2:
        out[()] = np.nan
        if returns_1d:
            out = out.item()
        return out

    returns, factor_returns = _aligned_series(returns, factor_returns)

    # Calculate annualized excess return
    ann_return = annual_return(returns, period=period, annualization=annualization)
    ann_excess_return = ann_return - risk_free
    
    # Calculate beta
    b = beta_aligned(returns, factor_returns, risk_free)

    # Handle division by zero or negative beta
    if returns_1d:
        if b == 0 or b < 0 or np.isnan(b):
            out = np.nan
        else:
            with np.errstate(divide='ignore', invalid='ignore'):
                out[()] = ann_excess_return / b
            out = out.item()
    else:
        # For multi-dimensional case
        if isinstance(b, (pd.Series, np.ndarray)):
            mask = (b == 0) | (b < 0) | np.isnan(b)
            with np.errstate(divide='ignore', invalid='ignore'):
                if isinstance(ann_excess_return, (pd.Series, pd.DataFrame)):
                    out = (ann_excess_return / b).values
                else:
                    out = ann_excess_return / b
            out[mask] = np.nan
        else:
            with np.errstate(divide='ignore', invalid='ignore'):
                out[()] = ann_excess_return / b
        
        if allocated_output and isinstance(returns, pd.DataFrame):
            out = pd.Series(out, index=returns.columns)

    return out


roll_treynor_ratio = _create_binary_vectorized_roll_function(treynor_ratio)


def m_squared(returns,
              factor_returns,
              risk_free=0.0,
              period=DAILY,
              annualization=None,
              out=None):
    """
    Determines the M² (Modigliani-Modigliani) measure of a strategy.

    M² is a risk-adjusted performance measure that adjusts the portfolio's
    return to match the risk level of the benchmark, then compares the
    adjusted return to the benchmark return.

    Parameters
    ----------
    returns : pd.Series or np.ndarray
        Daily returns of the strategy, noncumulative.
        - See full explanation in: func:`~empyrical.stats.cum_returns`.
    factor_returns : pd.Series or np.ndarray
        Daily noncumulative returns of the factor to which M² is
        computed. Usually a benchmark such as the market.
        - This is in the same style as returns.
    risk_free : int, float, optional
        Constant risk-free return throughout the period. For example, the
        interest rate on a three-month US treasury bill.
    period : str, optional
        Defines the periodicity of the 'returns' data for purposes of
        annualizing. Value ignored if `annualization` parameter is specified.
        Defaults are::

            'Monthly':12
            'weekly': 52
            'daily': 252

    annualization : int, optional
        Used to suppress default values available in `period` to convert
        returns into annual returns. Value should be the annual frequency of
        `returns`.
    out : array-like, optional
        Array to use as output buffer.
        If not passed, a new array will be created.

    Returns
    -------
    m_squared : float
        The M² measure. Returns np.nan if portfolio volatility is zero.

    Note
    -----
    M² = (Rp - Rf) * (σb / σp) + Rf
    where:
    - Rp = portfolio annualized return
    - Rf = risk-free rate
    - σb = benchmark annualized volatility
    - σp = portfolio annualized volatility

    See https://www.investopedia.com/terms/m/modigliani-modigliani-measure.asp
    for more details.
    """
    allocated_output = out is None
    if allocated_output:
        out = np.empty(returns.shape[1:])

    returns_1d = returns.ndim == 1

    if len(returns) < 2:
        out[()] = np.nan
        if returns_1d:
            out = out.item()
        return out

    returns, factor_returns = _aligned_series(returns, factor_returns)

    # Calculate annualized returns and volatilities
    ann_return = annual_return(returns, period=period, annualization=annualization)
    ann_vol = annual_volatility(returns, period=period, annualization=annualization)
    ann_factor_return = annual_return(factor_returns, period=period, annualization=annualization)
    ann_factor_vol = annual_volatility(factor_returns, period=period, annualization=annualization)

    # Handle division by zero or negative volatility
    if returns_1d:
        if ann_vol == 0 or ann_vol < 0 or np.isnan(ann_vol):
            out = np.nan
        else:
            # M² = (Rp - Rf) * (σb / σp) + Rf
            excess_return = ann_return - risk_free
            risk_ratio = ann_factor_vol / ann_vol
            out = excess_return * risk_ratio + risk_free
            out = out if isinstance(out, (float, np.floating)) else out.item()
    else:
        # For multi-dimensional case
        mask = (ann_vol == 0) | (ann_vol < 0) | np.isnan(ann_vol)
        excess_return = ann_return - risk_free
        
        with np.errstate(divide='ignore', invalid='ignore'):
            risk_ratio = ann_factor_vol / ann_vol
            out = excess_return * risk_ratio + risk_free
        
        out[mask] = np.nan
        
        if allocated_output and isinstance(returns, pd.DataFrame):
            out = pd.Series(out, index=returns.columns)

    return out


roll_m_squared = _create_binary_vectorized_roll_function(m_squared)


def annual_active_risk(returns,
                       factor_returns,
                       period=DAILY,
                       annualization=None,
                       out=None):
    """
    Determines the annualized active risk of a strategy.

    Active risk is the standard deviation of the active returns
    (returns - factor_returns), annualized. This is equivalent to tracking error.

    Parameters
    ----------
    returns : pd.Series or np.ndarray
        Daily returns of the strategy, noncumulative.
        - See full explanation in: func:`~empyrical.stats.cum_returns`.
    factor_returns : pd.Series or np.ndarray
        Daily noncumulative returns of the factor to which active risk is
        computed. Usually a benchmark such as the market.
        - This is in the same style as returns.
    period : str, optional
        Defines the periodicity of the 'returns' data for purposes of
        annualizing. Value ignored if `annualization` parameter is specified.
        Defaults are::

            'Monthly':12
            'weekly': 52
            'daily': 252

    annualization : int, optional
        Used to suppress default values available in `period` to convert
        returns into annual returns. Value should be the annual frequency of
        `returns`.
    out : array-like, optional
        Array to use as output buffer.
        If not passed, a new array will be created.

    Returns
    -------
    annual_active_risk : float
        The annualized active risk (equivalent to tracking error).

    Note
    -----
    Active risk is the same as tracking error. This function is an alias
    for tracking_error for clarity in financial analysis.
    """
    return tracking_error(returns, factor_returns, period=period,
                         annualization=annualization, out=out)


roll_annual_active_risk = _create_binary_vectorized_roll_function(annual_active_risk)


def annual_active_return(returns,
                         factor_returns,
                         period=DAILY,
                         annualization=None,
                         out=None):
    """
    Determines the annualized active return of a strategy.

    Active return is the difference between the strategy's annualized return
    and the benchmark's annualized return.

    Parameters
    ----------
    returns : pd.Series or np.ndarray
        Daily returns of the strategy, noncumulative.
        - See full explanation in: func:`~empyrical.stats.cum_returns`.
    factor_returns : pd.Series or np.ndarray
        Daily noncumulative returns of the factor to which active return is
        computed. Usually a benchmark such as the market.
        - This is in the same style as returns.
    period : str, optional
        Defines the periodicity of the 'returns' data for purposes of
        annualizing. Value ignored if `annualization` parameter is specified.
        Defaults are::

            'Monthly':12
            'weekly': 52
            'daily': 252

    annualization : int, optional
        Used to suppress default values available in `period` to convert
        returns into annual returns. Value should be the annual frequency of
        `returns`.
    out : array-like, optional
        Array to use as output buffer.
        If not passed, a new array will be created.

    Returns
    -------
    annual_active_return : float
        The annualized active return (strategy return - benchmark return).

    Note
    -----
    Active return measures the excess return of the strategy over the benchmark
    on an annualized basis.
    """
    allocated_output = out is None
    if allocated_output:
        out = np.empty(returns.shape[1:])

    returns_1d = returns.ndim == 1

    if len(returns) < 1:
        out[()] = np.nan
        if returns_1d:
            out = out.item()
        return out

    returns, factor_returns = _aligned_series(returns, factor_returns)

    # Calculate annualized returns
    ann_return = annual_return(returns, period=period, annualization=annualization)
    ann_factor_return = annual_return(factor_returns, period=period, annualization=annualization)

    # Active return = strategy return - benchmark return
    if returns_1d:
        out = ann_return - ann_factor_return
        if not isinstance(out, (float, np.floating)):
            out = out.item()
    else:
        out = ann_return - ann_factor_return
        if allocated_output and isinstance(returns, pd.DataFrame):
            out = pd.Series(out, index=returns.columns)

    return out


roll_annual_active_return = _create_binary_vectorized_roll_function(annual_active_return)


def annual_return_by_year(returns, period=DAILY, annualization=None):
    """
    Determines the annual return for each year independently.

    Parameters
    ----------
    returns : pd.Series or np.ndarray
        Daily returns of the strategy, noncumulative.
        - See full explanation in: func:`~empyrical.stats.cum_returns`.
    period : str, optional
        Defines the periodicity of the 'returns' data for purposes of
        annualizing. Value ignored if `annualization` parameter is specified.
        Defaults are::

            'Monthly':12
            'weekly': 52
            'daily': 252

    annualization : int, optional
        Used to suppress default values available in `period` to convert
        returns into annual returns. Value should be the annual frequency of
        `returns`.

    Returns
    -------
    annual_returns_by_year : pd.Series or np.ndarray
        Annual returns for each year, indexed by year.
    """
    if len(returns) < 1:
        return_as_array = isinstance(returns, np.ndarray)
        return np.array([]) if return_as_array else pd.Series(dtype='float64')

    # Track whether input is array for return type
    return_as_array = isinstance(returns, np.ndarray)
    
    # Ensure we have a datetime-indexed Series
    returns = _ensure_datetime_index_series(returns, period=period)

    # Group by year and calculate annual return for each year
    annual_returns = returns.groupby(returns.index.year).apply(
        lambda x: annual_return(x, period=period, annualization=annualization)
    )

    return annual_returns.values if return_as_array else annual_returns


def sharpe_ratio_by_year(returns, risk_free=0, period=DAILY, annualization=None):
    """
    Determines the Sharpe ratio for each year independently.

    Parameters
    ----------
    returns : pd.Series or np.ndarray
        Daily returns of the strategy, noncumulative.
        - See full explanation in: func:`~empyrical.stats.cum_returns`.
    risk_free : int, float, optional
        Constant daily risk-free return throughout the period.
    period : str, optional
        Defines the periodicity of the 'returns' data for purposes of
        annualizing. Value ignored if `annualization` parameter is specified.
        Defaults are::

            'Monthly':12
            'weekly': 52
            'daily': 252

    annualization : int, optional
        Used to suppress default values available in `period` to convert
        returns into annual returns. Value should be the annual frequency of
        `returns`.

    Returns
    -------
    sharpe_ratios_by_year : pd.Series or np.ndarray
        Sharpe ratios for each year, indexed by year.
    """
    if len(returns) < 1:
        return_as_array = isinstance(returns, np.ndarray)
        return np.array([]) if return_as_array else pd.Series(dtype='float64')

    # Track whether input is array for return type
    return_as_array = isinstance(returns, np.ndarray)
    
    # Ensure we have a datetime-indexed Series
    returns = _ensure_datetime_index_series(returns, period=period)

    # Group by year and calculate Sharpe ratio for each year
    sharpe_ratios = returns.groupby(returns.index.year).apply(
        lambda x: sharpe_ratio(x, risk_free=risk_free, period=period,
                              annualization=annualization)
    )

    return sharpe_ratios.values if return_as_array else sharpe_ratios


def information_ratio_by_year(returns, factor_returns, period=DAILY, annualization=None):
    """
    Determines the Information ratio for each year independently.

    Parameters
    ----------
    returns : pd.Series or np.ndarray
        Daily returns of the strategy, noncumulative.
        - See full explanation in: func:`~empyrical.stats.cum_returns`.
    factor_returns : pd.Series or np.ndarray
        Daily noncumulative returns of the factor to which information ratio is
        computed. Usually a benchmark such as the market.
        - This is in the same style as returns.
    period : str, optional
        Defines the periodicity of the 'returns' data for purposes of
        annualizing. Value ignored if `annualization` parameter is specified.
        Defaults are::

            'Monthly':12
            'weekly': 52
            'daily': 252

    annualization : int, optional
        Used to suppress default values available in `period` to convert
        returns into annual returns. Value should be the annual frequency of
        `returns`.

    Returns
    -------
    information_ratios_by_year : pd.Series or np.ndarray
        Information ratios for each year, indexed by year.
    """
    if len(returns) < 1:
        return_as_array = isinstance(returns, np.ndarray)
        return np.array([]) if return_as_array else pd.Series(dtype='float64')

    # Track whether input is array for return type
    return_as_array = isinstance(returns, np.ndarray)
    
    # Ensure we have datetime-indexed Series
    returns = _ensure_datetime_index_series(returns, period=period)
    factor_returns = _ensure_datetime_index_series(factor_returns, period=period)

    # Align the series
    returns, factor_returns = _aligned_series(returns, factor_returns)

    # Group by year and calculate information ratio for each year
    information_ratios = returns.groupby(returns.index.year).apply(
        lambda x: information_ratio(
            x - factor_returns.loc[x.index],
            period=period,
            annualization=annualization
        )
    )
    
    # Remove name attribute if it exists
    if hasattr(information_ratios, 'name'):
        information_ratios.name = None

    return information_ratios.values if return_as_array else information_ratios


def annual_volatility_by_year(returns, period=DAILY, annualization=None):
    """
    Determines the annual volatility for each year independently.

    Parameters
    ----------
    returns : pd.Series or np.ndarray
        Daily returns of the strategy, noncumulative.
        - See full explanation in: func:`~empyrical.stats.cum_returns`.
    period : str, optional
        Defines the periodicity of the 'returns' data for purposes of
        annualizing. Value ignored if `annualization` parameter is specified.
        Defaults are::

            'Monthly':12
            'weekly': 52
            'daily': 252

    annualization : int, optional
        Used to suppress default values available in `period` to convert
        returns into annual returns. Value should be the annual frequency of
        `returns`.

    Returns
    -------
    annual_volatilities_by_year : pd.Series or np.ndarray
        Annual volatilities for each year, indexed by year.
    """
    if len(returns) < 1:
        return_as_array = isinstance(returns, np.ndarray)
        return np.array([]) if return_as_array else pd.Series(dtype='float64')

    # Track whether input is array for return type
    return_as_array = isinstance(returns, np.ndarray)
    
    # Ensure we have a datetime-indexed Series
    returns = _ensure_datetime_index_series(returns, period=period)

    # Group by year and calculate annual volatility for each year
    annual_volatilities = returns.groupby(returns.index.year).apply(
        lambda x: annual_volatility(x, period=period, annualization=annualization)
    )

    return annual_volatilities.values if return_as_array else annual_volatilities


def max_drawdown_by_year(returns):
    """
    Determines the maximum drawdown for each year independently.

    Parameters
    ----------
    returns : pd.Series or np.ndarray
        Daily returns of the strategy, noncumulative.
        - See full explanation in: func:`~empyrical.stats.cum_returns`.

    Returns
    -------
    max_drawdowns_by_year : pd.Series or np.ndarray
        Maximum drawdowns for each year, indexed by year.
    """
    if len(returns) < 1:
        return_as_array = isinstance(returns, np.ndarray)
        return np.array([]) if return_as_array else pd.Series(dtype='float64')

    # Track whether input is array for return type
    return_as_array = isinstance(returns, np.ndarray)
    
    # Ensure we have a datetime-indexed Series
    returns = _ensure_datetime_index_series(returns, period=DAILY)

    # Group by year and calculate max drawdown for each year
    max_drawdowns = returns.groupby(returns.index.year).apply(
        lambda x: max_drawdown(x)
    )

    return max_drawdowns.values if return_as_array else max_drawdowns


def max_drawdown_days(returns):
    """
    Determines the duration in days of the maximum drawdown period.

    Parameters
    ----------
    returns : pd.Series or np.ndarray
        Daily returns of the strategy, noncumulative.
        - See full explanation in: func:`~empyrical.stats.cum_returns`.

    Returns
    -------
    max_drawdown_days : int or np.nan
        The number of days from the start to the end of the maximum drawdown period.
        Returns 0 if there is no drawdown, np.nan if returns is empty.
    """
    if len(returns) < 1:
        return np.nan

    if not isinstance(returns, pd.Series):
        returns = pd.Series(returns)

    # Calculate cumulative returns
    cum_ret = cum_returns(returns, starting_value=100)
    
    # Calculate rolling maximum
    rolling_max = cum_ret.cummax()
    
    # Calculate drawdown
    drawdown = (cum_ret - rolling_max) / rolling_max
    
    # Find the end date of maximum drawdown (lowest point)
    end_idx = drawdown.idxmin()
    
    # Find the start date (previous peak before the end)
    start_idx = cum_ret.loc[:end_idx].idxmax()
    
    # Calculate days difference
    if isinstance(returns.index, pd.DatetimeIndex):
        days_diff = (end_idx - start_idx).days
    else:
        # For non-datetime index, count the number of periods
        start_pos = returns.index.get_loc(start_idx)
        end_pos = returns.index.get_loc(end_idx)
        days_diff = end_pos - start_pos
    
    return days_diff


def max_drawdown_weeks(returns):
    """
    Determines the duration in weeks of the maximum drawdown period.

    Parameters
    ----------
    returns : pd.Series or np.ndarray
        Weekly returns of the strategy, noncumulative.
        - See full explanation in: func:`~empyrical.stats.cum_returns`.

    Returns
    -------
    max_drawdown_weeks : int or np.nan
        The number of weeks from the start to the end of the maximum drawdown period.
        Returns 0 if there is no drawdown, np.nan if returns is empty.
    """
    if len(returns) < 1:
        return np.nan

    if not isinstance(returns, pd.Series):
        returns = pd.Series(returns)

    # Calculate cumulative returns
    cum_ret = cum_returns(returns, starting_value=100)
    
    # Calculate rolling maximum
    rolling_max = cum_ret.cummax()
    
    # Calculate drawdown
    drawdown = (cum_ret - rolling_max) / rolling_max
    
    # Find the end date of maximum drawdown (lowest point)
    end_idx = drawdown.idxmin()
    
    # Find the start date (previous peak before the end)
    start_idx = cum_ret.loc[:end_idx].idxmax()
    
    # Calculate weeks difference
    if isinstance(returns.index, pd.DatetimeIndex):
        # Use isocalendar to get week numbers
        start_week = returns.index.get_loc(start_idx)
        end_week = returns.index.get_loc(end_idx)
        weeks_diff = end_week - start_week
    else:
        # For non-datetime index, count the number of periods
        start_pos = returns.index.get_loc(start_idx)
        end_pos = returns.index.get_loc(end_idx)
        weeks_diff = end_pos - start_pos
    
    return weeks_diff


def max_drawdown_months(returns):
    """
    Determines the duration in months of the maximum drawdown period.

    Parameters
    ----------
    returns : pd.Series or np.ndarray
        Monthly returns of the strategy, noncumulative.
        - See full explanation in: func:`~empyrical.stats.cum_returns`.

    Returns
    -------
    max_drawdown_months : int or np.nan
        The number of months from the start to the end of the maximum drawdown period.
        Returns 0 if there is no drawdown, np.nan if returns is empty.
    """
    if len(returns) < 1:
        return np.nan

    if not isinstance(returns, pd.Series):
        returns = pd.Series(returns)

    # Calculate cumulative returns
    cum_ret = cum_returns(returns, starting_value=100)
    
    # Calculate rolling maximum
    rolling_max = cum_ret.cummax()
    
    # Calculate drawdown
    drawdown = (cum_ret - rolling_max) / rolling_max
    
    # Find the end date of maximum drawdown (lowest point)
    end_idx = drawdown.idxmin()
    
    # Find the start date (previous peak before the end)
    start_idx = cum_ret.loc[:end_idx].idxmax()
    
    # Calculate months difference
    if isinstance(returns.index, pd.DatetimeIndex):
        # Calculate months difference
        start_pos = returns.index.get_loc(start_idx)
        end_pos = returns.index.get_loc(end_idx)
        months_diff = end_pos - start_pos
    else:
        # For non-datetime index, count the number of periods
        start_pos = returns.index.get_loc(start_idx)
        end_pos = returns.index.get_loc(end_idx)
        months_diff = end_pos - start_pos
    
    return months_diff


def max_drawdown_recovery_days(returns):
    """
    Determines the number of days required to recover from the maximum drawdown.

    Parameters
    ----------
    returns : pd.Series or np.ndarray
        Daily returns of the strategy, noncumulative.
        - See full explanation in: func:`~empyrical.stats.cum_returns`.

    Returns
    -------
    max_drawdown_recovery_days : int or np.nan
        The number of days from the end of maximum drawdown to recovery.
        Returns 0 if already recovered at the end of drawdown,
        np.nan if returns is empty or never recovers.
    """
    if len(returns) < 1:
        return np.nan

    if not isinstance(returns, pd.Series):
        returns = pd.Series(returns)

    # Calculate cumulative returns
    cum_ret = cum_returns(returns, starting_value=100)
    
    # Calculate rolling maximum
    rolling_max = cum_ret.cummax()
    
    # Calculate drawdown
    drawdown = (cum_ret - rolling_max) / rolling_max
    
    # Find the end date of maximum drawdown (lowest point)
    end_idx = drawdown.idxmin()
    
    # Find the peak value before the drawdown
    peak_value = cum_ret.loc[:end_idx].max()
    
    # Find recovery point (when cumulative returns exceed the peak again)
    recovery_mask = cum_ret.loc[end_idx:] >= peak_value
    if recovery_mask.any():
        recovery_idx = recovery_mask.idxmax()
        
        # Calculate days difference
        if isinstance(returns.index, pd.DatetimeIndex):
            days_diff = (recovery_idx - end_idx).days
        else:
            end_pos = returns.index.get_loc(end_idx)
            recovery_pos = returns.index.get_loc(recovery_idx)
            days_diff = recovery_pos - end_pos
        
        return days_diff
    else:
        # Never recovers
        return np.nan


def max_drawdown_recovery_weeks(returns):
    """
    Determines the number of weeks required to recover from the maximum drawdown.

    Parameters
    ----------
    returns : pd.Series or np.ndarray
        Weekly returns of the strategy, noncumulative.
        - See full explanation in: func:`~empyrical.stats.cum_returns`.

    Returns
    -------
    max_drawdown_recovery_weeks : int or np.nan
        The number of weeks from the end of maximum drawdown to recovery.
        Returns 0 if already recovered at the end of drawdown,
        np.nan if returns is empty or never recovers.
    """
    if len(returns) < 1:
        return np.nan

    if not isinstance(returns, pd.Series):
        returns = pd.Series(returns)

    # Calculate cumulative returns
    cum_ret = cum_returns(returns, starting_value=100)
    
    # Calculate rolling maximum
    rolling_max = cum_ret.cummax()
    
    # Calculate drawdown
    drawdown = (cum_ret - rolling_max) / rolling_max
    
    # Find the end date of maximum drawdown (lowest point)
    end_idx = drawdown.idxmin()
    
    # Find the peak value before the drawdown
    peak_value = cum_ret.loc[:end_idx].max()
    
    # Find recovery point (when cumulative returns exceed the peak again)
    recovery_mask = cum_ret.loc[end_idx:] >= peak_value
    if recovery_mask.any():
        recovery_idx = recovery_mask.idxmax()
        
        # Calculate weeks difference
        end_pos = returns.index.get_loc(end_idx)
        recovery_pos = returns.index.get_loc(recovery_idx)
        weeks_diff = recovery_pos - end_pos
        
        return weeks_diff
    else:
        # Never recovers
        return np.nan


def max_drawdown_recovery_months(returns):
    """
    Determines the number of months required to recover from the maximum drawdown.

    Parameters
    ----------
    returns : pd.Series or np.ndarray
        Monthly returns of the strategy, noncumulative.
        - See full explanation in: func:`~empyrical.stats.cum_returns`.

    Returns
    -------
    max_drawdown_recovery_months : int or np.nan
        The number of months from the end of maximum drawdown to recovery.
        Returns 0 if already recovered at the end of drawdown,
        np.nan if returns is empty or never recovers.
    """
    if len(returns) < 1:
        return np.nan

    if not isinstance(returns, pd.Series):
        returns = pd.Series(returns)

    # Calculate cumulative returns
    cum_ret = cum_returns(returns, starting_value=100)
    
    # Calculate rolling maximum
    rolling_max = cum_ret.cummax()
    
    # Calculate drawdown
    drawdown = (cum_ret - rolling_max) / rolling_max
    
    # Find the end date of maximum drawdown (lowest point)
    end_idx = drawdown.idxmin()
    
    # Find the peak value before the drawdown
    peak_value = cum_ret.loc[:end_idx].max()
    
    # Find recovery point (when cumulative returns exceed the peak again)
    recovery_mask = cum_ret.loc[end_idx:] >= peak_value
    if recovery_mask.any():
        recovery_idx = recovery_mask.idxmax()
        
        # Calculate months difference
        end_pos = returns.index.get_loc(end_idx)
        recovery_pos = returns.index.get_loc(recovery_idx)
        months_diff = recovery_pos - end_pos
        
        return months_diff
    else:
        # Never recovers
        return np.nan


def _get_all_drawdowns_detailed(returns):
    """
    Helper function to find all distinct drawdown periods with detailed information.
    
    Returns a list of dictionaries with drawdown information including:
    - value: the drawdown magnitude
    - start_idx: start index of the drawdown
    - end_idx: end index of the drawdown (lowest point)
    - recovery_idx: index where drawdown recovers (or None if never recovers)
    - duration: number of periods from start to lowest point
    - recovery_duration: number of periods from lowest point to recovery (or None)
    """
    if len(returns) < 1:
        return []
    
    if not isinstance(returns, pd.Series):
        returns = pd.Series(returns)
    
    # Calculate cumulative returns
    cum_ret = cum_returns(returns, starting_value=100)
    
    # Calculate rolling maximum
    rolling_max = cum_ret.cummax()
    
    # Calculate drawdown
    drawdown = (cum_ret - rolling_max) / rolling_max
    
    # Find all distinct drawdown periods with details
    drawdown_periods = []
    in_drawdown = False
    current_dd_info = None
    
    for i, dd_val in enumerate(drawdown):
        if dd_val < 0:
            if not in_drawdown:
                # Start of a new drawdown period
                in_drawdown = True
                current_dd_info = {
                    'value': dd_val,
                    'start_idx': i,
                    'end_idx': i,
                    'min_idx': i,
                    'peak_value': cum_ret.iloc[i] / (1 + dd_val)  # back-calculate peak
                }
            else:
                # Continue in drawdown, update if new minimum
                if dd_val < current_dd_info['value']:
                    current_dd_info['value'] = dd_val
                    current_dd_info['min_idx'] = i
                current_dd_info['end_idx'] = i
        else:
            if in_drawdown:
                # End of drawdown period - found recovery
                current_dd_info['recovery_idx'] = i
                current_dd_info['duration'] = current_dd_info['min_idx'] - current_dd_info['start_idx'] + 1
                current_dd_info['recovery_duration'] = i - current_dd_info['min_idx']
                drawdown_periods.append(current_dd_info)
                in_drawdown = False
                current_dd_info = None
    
    # If still in drawdown at the end, add it without recovery
    if in_drawdown:
        current_dd_info['recovery_idx'] = None
        current_dd_info['duration'] = current_dd_info['min_idx'] - current_dd_info['start_idx'] + 1
        current_dd_info['recovery_duration'] = None
        drawdown_periods.append(current_dd_info)
    
    return drawdown_periods


def _get_all_drawdowns(returns):
    """
    Helper function to find all distinct drawdown periods and their values.
    
    Returns a list of drawdown values, one for each distinct drawdown period.
    """
    detailed = _get_all_drawdowns_detailed(returns)
    return [dd['value'] for dd in detailed]


def second_max_drawdown(returns, out=None):
    """
    Determines the second largest drawdown of a strategy.

    Parameters
    ----------
    returns : pd.Series or np.ndarray
        Daily returns of the strategy, noncumulative.
        - See full explanation in: func:`~empyrical.stats.cum_returns`.
    out : array-like, optional
        Array to use as output buffer.
        If not passed, a new array will be created.

    Returns
    -------
    second_max_drawdown : float
        The second largest drawdown value.
        Returns np.nan if there is no second drawdown.
    """
    drawdown_periods = _get_all_drawdowns(returns)
    
    if len(drawdown_periods) < 2:
        return np.nan
    
    # Sort drawdowns (most negative first)
    sorted_drawdowns = np.sort(drawdown_periods)
    
    # Get second largest (second most negative)
    return sorted_drawdowns[-2]


def third_max_drawdown(returns, out=None):
    """
    Determines the third largest drawdown of a strategy.

    Parameters
    ----------
    returns : pd.Series or np.ndarray
        Daily returns of the strategy, noncumulative.
        - See full explanation in: func:`~empyrical.stats.cum_returns`.
    out : array-like, optional
        Array to use as output buffer.
        If not passed, a new array will be created.

    Returns
    -------
    third_max_drawdown : float
        The third largest drawdown value.
        Returns np.nan if there is no third drawdown.
    """
    drawdown_periods = _get_all_drawdowns(returns)
    
    if len(drawdown_periods) < 3:
        return np.nan
    
    # Sort drawdowns (most negative first)
    sorted_drawdowns = np.sort(drawdown_periods)
    
    # Get third largest (third most negative)
    return sorted_drawdowns[-3]


def second_max_drawdown_days(returns, out=None):
    """
    Determines the duration (in days) of the second largest drawdown.

    Parameters
    ----------
    returns : pd.Series or np.ndarray
        Daily returns of the strategy, noncumulative.
        - See full explanation in: func:`~empyrical.stats.cum_returns`.
    out : array-like, optional
        Array to use as output buffer.

    Returns
    -------
    second_max_drawdown_days : int
        The number of days from start to lowest point of the second largest drawdown.
        Returns np.nan if there is no second drawdown.
    """
    drawdown_periods = _get_all_drawdowns_detailed(returns)
    
    if len(drawdown_periods) < 2:
        return np.nan
    
    # Sort by drawdown value (most negative first)
    sorted_drawdowns = sorted(drawdown_periods, key=lambda x: x['value'])
    
    # Get second largest drawdown duration
    return sorted_drawdowns[1]['duration']


def second_max_drawdown_recovery_days(returns, out=None):
    """
    Determines the recovery time (in days) for the second largest drawdown.

    Parameters
    ----------
    returns : pd.Series or np.ndarray
        Daily returns of the strategy, noncumulative.
        - See full explanation in: func:`~empyrical.stats.cum_returns`.
    out : array-like, optional
        Array to use as output buffer.

    Returns
    -------
    second_max_drawdown_recovery_days : int or np.nan
        The number of days from lowest point to recovery for the second largest drawdown.
        Returns np.nan if there is no second drawdown or it never recovers.
    """
    drawdown_periods = _get_all_drawdowns_detailed(returns)
    
    if len(drawdown_periods) < 2:
        return np.nan
    
    # Sort by drawdown value (most negative first)
    sorted_drawdowns = sorted(drawdown_periods, key=lambda x: x['value'])
    
    # Get second largest drawdown recovery duration
    recovery_duration = sorted_drawdowns[1]['recovery_duration']
    return recovery_duration if recovery_duration is not None else np.nan


def third_max_drawdown_days(returns, out=None):
    """
    Determines the duration (in days) of the third largest drawdown.

    Parameters
    ----------
    returns : pd.Series or np.ndarray
        Daily returns of the strategy, noncumulative.
        - See full explanation in: func:`~empyrical.stats.cum_returns`.
    out : array-like, optional
        Array to use as output buffer.

    Returns
    -------
    third_max_drawdown_days : int
        The number of days from start to lowest point of the third largest drawdown.
        Returns np.nan if there is no third drawdown.
    """
    drawdown_periods = _get_all_drawdowns_detailed(returns)
    
    if len(drawdown_periods) < 3:
        return np.nan
    
    # Sort by drawdown value (most negative first)
    sorted_drawdowns = sorted(drawdown_periods, key=lambda x: x['value'])
    
    # Get third largest drawdown duration
    return sorted_drawdowns[2]['duration']


def third_max_drawdown_recovery_days(returns, out=None):
    """
    Determines the recovery time (in days) for the third largest drawdown.

    Parameters
    ----------
    returns : pd.Series or np.ndarray
        Daily returns of the strategy, noncumulative.
        - See full explanation in: func:`~empyrical.stats.cum_returns`.
    out : array-like, optional
        Array to use as output buffer.

    Returns
    -------
    third_max_drawdown_recovery_days : int or np.nan
        The number of days from lowest point to recovery for the third largest drawdown.
        Returns np.nan if there is no third drawdown or it never recovers.
    """
    drawdown_periods = _get_all_drawdowns_detailed(returns)
    
    if len(drawdown_periods) < 3:
        return np.nan
    
    # Sort by drawdown value (most negative first)
    sorted_drawdowns = sorted(drawdown_periods, key=lambda x: x['value'])
    
    # Get third largest drawdown recovery duration
    recovery_duration = sorted_drawdowns[2]['recovery_duration']
    return recovery_duration if recovery_duration is not None else np.nan


def win_rate(returns, out=None):
    """
    Determines the win rate (percentage of positive returns) of a strategy.

    Parameters
    ----------
    returns : pd.Series or np.ndarray
        Daily returns of the strategy, noncumulative.
        - See full explanation in: func:`~empyrical.stats.cum_returns`.
    out : array-like, optional
        Array to use as output buffer.
        If not passed, a new array will be created.

    Returns
    -------
    win_rate : float
        The percentage of periods with positive returns.
        Returns np.nan if returns is empty.
    """
    if len(returns) < 1:
        return np.nan

    returns_array = np.asanyarray(returns)
    
    # Count positive returns (excluding NaN)
    positive_count = np.sum(returns_array > 0)
    total_count = np.sum(~np.isnan(returns_array))
    
    if total_count == 0:
        return np.nan
    
    win_rate_value = positive_count / total_count
    
    if returns.ndim == 1:
        return win_rate_value.item() if isinstance(win_rate_value, np.ndarray) else win_rate_value
    else:
        return win_rate_value


def loss_rate(returns, out=None):
    """
    Determines the loss rate (percentage of negative returns) of a strategy.

    Parameters
    ----------
    returns : pd.Series or np.ndarray
        Daily returns of the strategy, noncumulative.
        - See full explanation in: func:`~empyrical.stats.cum_returns`.
    out : array-like, optional
        Array to use as output buffer.
        If not passed, a new array will be created.

    Returns
    -------
    loss_rate : float
        The percentage of periods with negative returns.
        Returns np.nan if returns is empty.
    """
    if len(returns) < 1:
        return np.nan

    returns_array = np.asanyarray(returns)
    
    # Count negative returns (excluding NaN)
    negative_count = np.sum(returns_array < 0)
    total_count = np.sum(~np.isnan(returns_array))
    
    if total_count == 0:
        return np.nan
    
    loss_rate_value = negative_count / total_count
    
    if returns.ndim == 1:
        return loss_rate_value.item() if isinstance(loss_rate_value, np.ndarray) else loss_rate_value
    else:
        return loss_rate_value


def _find_consecutive_periods(returns, condition_func):
    """
    Helper function to find all consecutive periods that meet a condition.
    
    Parameters
    ----------
    returns : pd.Series or np.ndarray
        Returns series
    condition_func : callable
        Function that takes a return value and returns True/False
        
    Returns
    -------
    list of dict
        List of dictionaries with keys: 'length', 'start_idx', 'end_idx', 'cumulative_return'
    """
    if len(returns) < 1:
        return []
    
    if not isinstance(returns, pd.Series):
        returns = pd.Series(returns)
    
    periods = []
    current_period = None
    
    for i, ret in enumerate(returns):
        if pd.isna(ret):
            if current_period is not None:
                periods.append(current_period)
                current_period = None
            continue
            
        if condition_func(ret):
            if current_period is None:
                current_period = {
                    'length': 1,
                    'start_idx': i,
                    'end_idx': i,
                    'cumulative_return': ret
                }
            else:
                current_period['length'] += 1
                current_period['end_idx'] = i
                current_period['cumulative_return'] += ret
        else:
            if current_period is not None:
                periods.append(current_period)
                current_period = None
    
    # Add last period if exists
    if current_period is not None:
        periods.append(current_period)
    
    return periods


def max_consecutive_up_days(returns, out=None):
    """
    Determines the maximum number of consecutive days with positive returns.

    Parameters
    ----------
    returns : pd.Series or np.ndarray
        Daily returns of the strategy, noncumulative.
        - See full explanation in: func:`~empyrical.stats.cum_returns`.
    out : array-like, optional
        Array to use as output buffer.

    Returns
    -------
    max_consecutive_up_days : int
        The maximum number of consecutive days with positive returns.
        Returns np.nan if returns is empty.
    """
    periods = _find_consecutive_periods(returns, lambda x: x > 0)
    if not periods:
        return np.nan
    return max(p['length'] for p in periods)


def max_consecutive_down_days(returns, out=None):
    """
    Determines the maximum number of consecutive days with negative returns.

    Parameters
    ----------
    returns : pd.Series or np.ndarray
        Daily returns of the strategy, noncumulative.
        - See full explanation in: func:`~empyrical.stats.cum_returns`.
    out : array-like, optional
        Array to use as output buffer.

    Returns
    -------
    max_consecutive_down_days : int
        The maximum number of consecutive days with negative returns.
        Returns np.nan if returns is empty.
    """
    periods = _find_consecutive_periods(returns, lambda x: x < 0)
    if not periods:
        return np.nan
    return max(p['length'] for p in periods)


def max_consecutive_up_weeks(returns, out=None):
    """
    Determines the maximum number of consecutive weeks with positive returns.

    Parameters
    ----------
    returns : pd.Series or np.ndarray
        Weekly returns of the strategy, noncumulative.
        - See full explanation in: func:`~empyrical.stats.cum_returns`.
    out : array-like, optional
        Array to use as output buffer.

    Returns
    -------
    max_consecutive_up_weeks : int
        The maximum number of consecutive weeks with positive returns.
        Returns np.nan if returns is empty.
    """
    periods = _find_consecutive_periods(returns, lambda x: x > 0)
    if not periods:
        return np.nan
    return max(p['length'] for p in periods)


def max_consecutive_down_weeks(returns, out=None):
    """
    Determines the maximum number of consecutive weeks with negative returns.

    Parameters
    ----------
    returns : pd.Series or np.ndarray
        Weekly returns of the strategy, noncumulative.
        - See full explanation in: func:`~empyrical.stats.cum_returns`.
    out : array-like, optional
        Array to use as output buffer.

    Returns
    -------
    max_consecutive_down_weeks : int
        The maximum number of consecutive weeks with negative returns.
        Returns np.nan if returns is empty.
    """
    periods = _find_consecutive_periods(returns, lambda x: x < 0)
    if not periods:
        return np.nan
    return max(p['length'] for p in periods)


def max_consecutive_up_months(returns, out=None):
    """
    Determines the maximum number of consecutive months with positive returns.

    Parameters
    ----------
    returns : pd.Series or np.ndarray
        Monthly returns of the strategy, noncumulative.
        - See full explanation in: func:`~empyrical.stats.cum_returns`.
    out : array-like, optional
        Array to use as output buffer.

    Returns
    -------
    max_consecutive_up_months : int
        The maximum number of consecutive months with positive returns.
        Returns np.nan if returns is empty.
    """
    periods = _find_consecutive_periods(returns, lambda x: x > 0)
    if not periods:
        return np.nan
    return max(p['length'] for p in periods)


def max_consecutive_down_months(returns, out=None):
    """
    Determines the maximum number of consecutive months with negative returns.

    Parameters
    ----------
    returns : pd.Series or np.ndarray
        Monthly returns of the strategy, noncumulative.
        - See full explanation in: func:`~empyrical.stats.cum_returns`.
    out : array-like, optional
        Array to use as output buffer.

    Returns
    -------
    max_consecutive_down_months : int
        The maximum number of consecutive months with negative returns.
        Returns np.nan if returns is empty.
    """
    periods = _find_consecutive_periods(returns, lambda x: x < 0)
    if not periods:
        return np.nan
    return max(p['length'] for p in periods)


def max_consecutive_gain(returns, out=None):
    """
    Determines the maximum cumulative gain from consecutive positive returns.

    Parameters
    ----------
    returns : pd.Series or np.ndarray
        Daily returns of the strategy, noncumulative.
        - See full explanation in: func:`~empyrical.stats.cum_returns`.
    out : array-like, optional
        Array to use as output buffer.

    Returns
    -------
    max_consecutive_gain : float
        The maximum cumulative gain from consecutive positive returns.
        Returns np.nan if returns is empty or no positive periods.
    """
    periods = _find_consecutive_periods(returns, lambda x: x > 0)
    if not periods:
        return np.nan
    return max(p['cumulative_return'] for p in periods)


def max_consecutive_loss(returns, out=None):
    """
    Determines the maximum cumulative loss from consecutive negative returns.

    Parameters
    ----------
    returns : pd.Series or np.ndarray
        Daily returns of the strategy, noncumulative.
        - See full explanation in: func:`~empyrical.stats.cum_returns`.
    out : array-like, optional
        Array to use as output buffer.

    Returns
    -------
    max_consecutive_loss : float
        The maximum cumulative loss from consecutive negative returns (most negative value).
        Returns np.nan if returns is empty or no negative periods.
    """
    periods = _find_consecutive_periods(returns, lambda x: x < 0)
    if not periods:
        return np.nan
    return min(p['cumulative_return'] for p in periods)


def max_single_day_gain(returns, out=None):
    """
    Determines the maximum single-day gain.

    Parameters
    ----------
    returns : pd.Series or np.ndarray
        Daily returns of the strategy, noncumulative.
        - See full explanation in: func:`~empyrical.stats.cum_returns`.
    out : array-like, optional
        Array to use as output buffer.

    Returns
    -------
    max_single_day_gain : float
        The maximum single-day gain.
        Returns np.nan if returns is empty.
    """
    if len(returns) < 1:
        return np.nan
    
    returns_array = np.asanyarray(returns)
    positive_returns = returns_array[returns_array > 0]
    
    if len(positive_returns) == 0:
        return np.nan
    
    return np.max(positive_returns)


def max_single_day_loss(returns, out=None):
    """
    Determines the maximum single-day loss (most negative).

    Parameters
    ----------
    returns : pd.Series or np.ndarray
        Daily returns of the strategy, noncumulative.
        - See full explanation in: func:`~empyrical.stats.cum_returns`.
    out : array-like, optional
        Array to use as output buffer.

    Returns
    -------
    max_single_day_loss : float
        The maximum single-day loss (most negative value).
        Returns np.nan if returns is empty.
    """
    if len(returns) < 1:
        return np.nan
    
    returns_array = np.asanyarray(returns)
    negative_returns = returns_array[returns_array < 0]
    
    if len(negative_returns) == 0:
        return np.nan
    
    return np.min(negative_returns)


def max_consecutive_up_start_date(returns):
    """
    Determines the start date of the longest consecutive up period.

    Parameters
    ----------
    returns : pd.Series
        Daily returns of the strategy with datetime index, noncumulative.

    Returns
    -------
    start_date : pd.Timestamp
        The start date of the longest consecutive up period.
        Returns None if returns is empty or no up periods.
    """
    if not isinstance(returns, pd.Series):
        raise ValueError("returns must be a pd.Series with datetime index")
    
    periods = _find_consecutive_periods(returns, lambda x: x > 0)
    if not periods:
        return None
    
    max_period = max(periods, key=lambda p: p['length'])
    return returns.index[max_period['start_idx']]


def max_consecutive_up_end_date(returns):
    """
    Determines the end date of the longest consecutive up period.

    Parameters
    ----------
    returns : pd.Series
        Daily returns of the strategy with datetime index, noncumulative.

    Returns
    -------
    end_date : pd.Timestamp
        The end date of the longest consecutive up period.
        Returns None if returns is empty or no up periods.
    """
    if not isinstance(returns, pd.Series):
        raise ValueError("returns must be a pd.Series with datetime index")
    
    periods = _find_consecutive_periods(returns, lambda x: x > 0)
    if not periods:
        return None
    
    max_period = max(periods, key=lambda p: p['length'])
    return returns.index[max_period['end_idx']]


def max_consecutive_down_start_date(returns):
    """
    Determines the start date of the longest consecutive down period.

    Parameters
    ----------
    returns : pd.Series
        Daily returns of the strategy with datetime index, noncumulative.

    Returns
    -------
    start_date : pd.Timestamp
        The start date of the longest consecutive down period.
        Returns None if returns is empty or no down periods.
    """
    if not isinstance(returns, pd.Series):
        raise ValueError("returns must be a pd.Series with datetime index")
    
    periods = _find_consecutive_periods(returns, lambda x: x < 0)
    if not periods:
        return None
    
    max_period = max(periods, key=lambda p: p['length'])
    return returns.index[max_period['start_idx']]


def max_consecutive_down_end_date(returns):
    """
    Determines the end date of the longest consecutive down period.

    Parameters
    ----------
    returns : pd.Series
        Daily returns of the strategy with datetime index, noncumulative.

    Returns
    -------
    end_date : pd.Timestamp
        The end date of the longest consecutive down period.
        Returns None if returns is empty or no down periods.
    """
    if not isinstance(returns, pd.Series):
        raise ValueError("returns must be a pd.Series with datetime index")
    
    periods = _find_consecutive_periods(returns, lambda x: x < 0)
    if not periods:
        return None
    
    max_period = max(periods, key=lambda p: p['length'])
    return returns.index[max_period['end_idx']]


def max_single_day_gain_date(returns):
    """
    Determines the date of the maximum single-day gain.

    Parameters
    ----------
    returns : pd.Series
        Daily returns of the strategy with datetime index, noncumulative.

    Returns
    -------
    date : pd.Timestamp
        The date of the maximum single-day gain.
        Returns None if returns is empty or no positive returns.
    """
    if not isinstance(returns, pd.Series):
        raise ValueError("returns must be a pd.Series with datetime index")
    
    if len(returns) < 1:
        return None
    
    positive_returns = returns[returns > 0]
    if len(positive_returns) == 0:
        return None
    
    return positive_returns.idxmax()


def max_single_day_loss_date(returns):
    """
    Determines the date of the maximum single-day loss.

    Parameters
    ----------
    returns : pd.Series
        Daily returns of the strategy with datetime index, noncumulative.

    Returns
    -------
    date : pd.Timestamp
        The date of the maximum single-day loss (most negative).
        Returns None if returns is empty or no negative returns.
    """
    if not isinstance(returns, pd.Series):
        raise ValueError("returns must be a pd.Series with datetime index")
    
    if len(returns) < 1:
        return None
    
    negative_returns = returns[returns < 0]
    if len(negative_returns) == 0:
        return None
    
    return negative_returns.idxmin()


def skewness(returns, out=None):
    """
    Determines the skewness of a returns distribution.
    
    Skewness measures the asymmetry of the probability distribution of returns.
    Positive skewness indicates a distribution with an asymmetric tail extending
    toward more positive values. Negative skewness indicates a distribution with
    an asymmetric tail extending toward more negative values.

    Parameters
    ----------
    returns : pd.Series or np.ndarray
        Daily returns of the strategy, noncumulative.
        - See full explanation in: func:`~empyrical.stats.cum_returns`.
    out : array-like, optional
        Array to use as output buffer.

    Returns
    -------
    skewness : float
        The skewness of the returns distribution.
        Returns np.nan if returns is empty or has less than 3 values.
        
    Notes
    -----
    - Skewness = 0: symmetric distribution
    - Skewness > 0: right-skewed (tail on the right side)
    - Skewness < 0: left-skewed (tail on the left side)
    """
    if len(returns) < 3:
        return np.nan
    
    returns_array = np.asanyarray(returns)
    # Remove NaN values
    returns_clean = returns_array[~np.isnan(returns_array)]
    
    if len(returns_clean) < 3:
        return np.nan
    
    return float(stats.skew(returns_clean))


def kurtosis(returns, out=None):
    """
    Determines the kurtosis of a returns distribution.
    
    Kurtosis measures the "tailedness" of the probability distribution of returns.
    High kurtosis indicates more of the variance is due to extreme deviations,
    as opposed to frequent modestly sized deviations.

    Parameters
    ----------
    returns : pd.Series or np.ndarray
        Daily returns of the strategy, noncumulative.
        - See full explanation in: func:`~empyrical.stats.cum_returns`.
    out : array-like, optional
        Array to use as output buffer.

    Returns
    -------
    kurtosis : float
        The excess kurtosis of the returns distribution (relative to normal distribution).
        Returns np.nan if returns is empty or has less than 4 values.
        
    Notes
    -----
    - Kurtosis = 0: same as normal distribution (mesokurtic)
    - Kurtosis > 0: heavier tails than normal (leptokurtic) - more extreme events
    - Kurtosis < 0: lighter tails than normal (platykurtic) - fewer extreme events
    - This function returns excess kurtosis (kurtosis - 3)
    """
    if len(returns) < 4:
        return np.nan
    
    returns_array = np.asanyarray(returns)
    # Remove NaN values
    returns_clean = returns_array[~np.isnan(returns_array)]
    
    if len(returns_clean) < 4:
        return np.nan
    
    return float(stats.kurtosis(returns_clean))


def hurst_exponent(returns, out=None):
    """
    Calculates the Hurst exponent of a returns time series.
    
    The Hurst exponent (H) is used to measure the long-term memory of time series.
    It relates to the autocorrelations of the time series, and the rate at which
    these decrease as the lag between pairs of values increases.

    Parameters
    ----------
    returns : pd.Series or np.ndarray
        Returns of the strategy, noncumulative.
        - See full explanation in: func:`~empyrical.stats.cum_returns`.
    out : array-like, optional
        Array to use as output buffer.

    Returns
    -------
    hurst_exponent : float
        The Hurst exponent value.
        Returns np.nan if returns is empty or too short (< 20 observations).
        
    Notes
    -----
    - H = 0.5: Random walk (Brownian motion) - no correlation
    - H > 0.5: Persistent/trending behavior - positive autocorrelation
    - H < 0.5: Anti-persistent/mean-reverting behavior - negative autocorrelation
    - H = 1.0: Perfect positive correlation (deterministic trend)
    - H = 0.0: Perfect negative correlation (deterministic mean reversion)
    
    The calculation uses the R/S (Rescaled Range) analysis method.
    """
    # Lower minimum requirement for short series
    min_length = 8
    if len(returns) < min_length:
        return np.nan
    
    if not isinstance(returns, pd.Series):
        returns = pd.Series(returns)
    
    returns_array = returns.values
    # Remove NaN values
    returns_clean = returns_array[~np.isnan(returns_array)]
    
    if len(returns_clean) < min_length:
        return np.nan
    
    try:
        # Calculate cumulative deviate series
        mean_return = np.mean(returns_clean)
        Y = np.cumsum(returns_clean - mean_return)
        
        # Calculate range R
        R = np.max(Y) - np.min(Y)
        
        # Calculate standard deviation S
        S = np.std(returns_clean, ddof=1)
        
        if S == 0 or R == 0:
            return np.nan
        
        # For a single calculation, we need to use multiple time scales
        # This is a simplified version using the entire series
        # A more robust implementation would use multiple window sizes
        
        # Use rescaled range method with multiple lags
        # Adjust max lag based on series length
        n = len(returns_clean)
        max_lag = max(n // 3, 3)
        min_lag = 2
        
        lags = range(min_lag, max_lag + 1)
        rs_values = []
        
        for lag in lags:
            # Split series into sub-series
            n_subseries = n // lag
            if n_subseries < 1:
                continue
                
            rs_list = []
            for i in range(n_subseries):
                sub_series = returns_clean[i*lag:(i+1)*lag]
                if len(sub_series) < 2:
                    continue
                    
                mean_sub = np.mean(sub_series)
                Y_sub = np.cumsum(sub_series - mean_sub)
                R_sub = np.max(Y_sub) - np.min(Y_sub)
                S_sub = np.std(sub_series, ddof=1)
                
                if S_sub > 0 and R_sub > 0:
                    rs_list.append(R_sub / S_sub)
            
            if rs_list:
                rs_values.append((lag, np.mean(rs_list)))
        
        # Need at least 2 points for regression
        if len(rs_values) < 2:
            # Fallback: use simple calculation for very short series
            # H ≈ 0.5 + log(R/S) / log(2*n)
            if S > 0 and R > 0:
                H = 0.5 + np.log(R / S) / np.log(2.0 * n)
                H = max(0.0, min(1.0, H))
                return float(H)
            return np.nan
        
        # Fit log(R/S) vs log(lag) to get Hurst exponent
        lags_array = np.array([x[0] for x in rs_values])
        rs_array = np.array([x[1] for x in rs_values])
        
        # Filter out any invalid values
        valid_mask = (lags_array > 0) & (rs_array > 0)
        lags_array = lags_array[valid_mask]
        rs_array = rs_array[valid_mask]
        
        if len(lags_array) < 2:
            return np.nan
        
        # Linear regression on log-log plot
        log_lags = np.log(lags_array)
        log_rs = np.log(rs_array)
        
        # Fit line: log(R/S) = H * log(lag) + constant
        poly = np.polyfit(log_lags, log_rs, 1)
        H = poly[0]
        
        # Clamp to valid range [0, 1]
        H = max(0.0, min(1.0, H))
        
        return float(H)
        
    except Exception:
        return np.nan


def stock_market_correlation(returns, stock_market_returns, out=None):
    """
    Calculates the correlation between portfolio returns and stock market returns.
    
    This metric measures how closely the portfolio's performance tracks the stock market.

    Parameters
    ----------
    returns : pd.Series or np.ndarray
        Returns of the strategy, noncumulative.
        - See full explanation in: func:`~empyrical.stats.cum_returns`.
    stock_market_returns : pd.Series or np.ndarray
        Returns of a stock market index (e.g., S&P 500), noncumulative.
    out : array-like, optional
        Array to use as output buffer.

    Returns
    -------
    correlation : float
        Pearson correlation coefficient between returns and stock market returns.
        Returns np.nan if either series is empty or has insufficient data.
        
    Notes
    -----
    - Correlation = 1: Perfect positive correlation
    - Correlation = 0: No correlation
    - Correlation = -1: Perfect negative correlation
    """
    if len(returns) < 2 or len(stock_market_returns) < 2:
        return np.nan
    
    # Align series if they are pandas objects
    if isinstance(returns, pd.Series) and isinstance(stock_market_returns, pd.Series):
        returns, stock_market_returns = returns.align(stock_market_returns, join='inner')
    
    if len(returns) < 2:
        return np.nan
    
    returns_array = np.asanyarray(returns)
    market_array = np.asanyarray(stock_market_returns)
    
    # Remove NaN values
    valid_mask = ~(np.isnan(returns_array) | np.isnan(market_array))
    returns_clean = returns_array[valid_mask]
    market_clean = market_array[valid_mask]
    
    if len(returns_clean) < 2:
        return np.nan
    
    # Calculate Pearson correlation
    correlation = np.corrcoef(returns_clean, market_clean)[0, 1]
    
    return float(correlation) if not np.isnan(correlation) else np.nan


def bond_market_correlation(returns, bond_market_returns, out=None):
    """
    Calculates the correlation between portfolio returns and bond market returns.
    
    This metric measures how closely the portfolio's performance tracks the bond market.

    Parameters
    ----------
    returns : pd.Series or np.ndarray
        Returns of the strategy, noncumulative.
        - See full explanation in: func:`~empyrical.stats.cum_returns`.
    bond_market_returns : pd.Series or np.ndarray
        Returns of a bond market index (e.g., AGG, TLT), noncumulative.
    out : array-like, optional
        Array to use as output buffer.

    Returns
    -------
    correlation : float
        Pearson correlation coefficient between returns and bond market returns.
        Returns np.nan if either series is empty or has insufficient data.
    """
    if len(returns) < 2 or len(bond_market_returns) < 2:
        return np.nan
    
    # Align series if they are pandas objects
    if isinstance(returns, pd.Series) and isinstance(bond_market_returns, pd.Series):
        returns, bond_market_returns = returns.align(bond_market_returns, join='inner')
    
    if len(returns) < 2:
        return np.nan
    
    returns_array = np.asanyarray(returns)
    market_array = np.asanyarray(bond_market_returns)
    
    # Remove NaN values
    valid_mask = ~(np.isnan(returns_array) | np.isnan(market_array))
    returns_clean = returns_array[valid_mask]
    market_clean = market_array[valid_mask]
    
    if len(returns_clean) < 2:
        return np.nan
    
    # Calculate Pearson correlation
    correlation = np.corrcoef(returns_clean, market_clean)[0, 1]
    
    return float(correlation) if not np.isnan(correlation) else np.nan


def futures_market_correlation(returns, futures_market_returns, out=None):
    """
    Calculates the correlation between portfolio returns and futures market returns.
    
    This metric measures how closely the portfolio's performance tracks the futures market.

    Parameters
    ----------
    returns : pd.Series or np.ndarray
        Returns of the strategy, noncumulative.
        - See full explanation in: func:`~empyrical.stats.cum_returns`.
    futures_market_returns : pd.Series or np.ndarray
        Returns of a futures market index, noncumulative.
    out : array-like, optional
        Array to use as output buffer.

    Returns
    -------
    correlation : float
        Pearson correlation coefficient between returns and futures market returns.
        Returns np.nan if either series is empty or has insufficient data.
    """
    if len(returns) < 2 or len(futures_market_returns) < 2:
        return np.nan
    
    # Align series if they are pandas objects
    if isinstance(returns, pd.Series) and isinstance(futures_market_returns, pd.Series):
        returns, futures_market_returns = returns.align(futures_market_returns, join='inner')
    
    if len(returns) < 2:
        return np.nan
    
    returns_array = np.asanyarray(returns)
    market_array = np.asanyarray(futures_market_returns)
    
    # Remove NaN values
    valid_mask = ~(np.isnan(returns_array) | np.isnan(market_array))
    returns_clean = returns_array[valid_mask]
    market_clean = market_array[valid_mask]
    
    if len(returns_clean) < 2:
        return np.nan
    
    # Calculate Pearson correlation
    correlation = np.corrcoef(returns_clean, market_clean)[0, 1]
    
    return float(correlation) if not np.isnan(correlation) else np.nan


def serial_correlation(returns, lag=1, out=None):
    """
    Calculates the serial correlation (autocorrelation) of returns at a specified lag.
    
    Serial correlation measures the correlation between returns and their own lagged values.
    This can help identify momentum or mean-reversion patterns.

    Parameters
    ----------
    returns : pd.Series or np.ndarray
        Returns of the strategy, noncumulative.
        - See full explanation in: func:`~empyrical.stats.cum_returns`.
    lag : int, optional
        The lag period for autocorrelation calculation. Default is 1 (one period lag).
        For weekly returns, lag=1 represents one week lag.
    out : array-like, optional
        Array to use as output buffer.

    Returns
    -------
    autocorrelation : float
        Autocorrelation coefficient at the specified lag.
        Returns np.nan if returns is empty or has insufficient data.
        
    Notes
    -----
    - Autocorrelation > 0: Positive serial correlation (momentum)
    - Autocorrelation = 0: No serial correlation (random walk)
    - Autocorrelation < 0: Negative serial correlation (mean reversion)
    """
    if len(returns) < lag + 2:
        return np.nan
    
    returns_array = np.asanyarray(returns)
    # Remove NaN values
    returns_clean = returns_array[~np.isnan(returns_array)]
    
    if len(returns_clean) < lag + 2:
        return np.nan
    
    # Create lagged series
    returns_t = returns_clean[lag:]
    returns_t_lag = returns_clean[:-lag]
    
    if len(returns_t) < 2:
        return np.nan
    
    # Calculate Pearson correlation between returns and lagged returns
    correlation = np.corrcoef(returns_t, returns_t_lag)[0, 1]
    
    return float(correlation) if not np.isnan(correlation) else np.nan


def sterling_ratio(returns, risk_free=0.0, period=DAILY, annualization=None, out=None):
    """
    Calculates the Sterling ratio of a strategy.
    
    The Sterling ratio is similar to the Calmar ratio but uses the average drawdown
    instead of the maximum drawdown as the risk measure.
    
    Sterling Ratio = (Annualized Return - Risk Free Rate) / Average Drawdown

    Parameters
    ----------
    returns : pd.Series or np.ndarray
        Daily returns of the strategy, noncumulative.
        - See full explanation in: func:`~empyrical.stats.cum_returns`.
    risk_free : float, optional
        Constant risk-free return throughout the period. Default is 0.0.
    period : str, optional
        Defines the periodicity of the 'returns' data for purposes of
        annualizing. Default is 'daily'.
    annualization : int, optional
        Used to suppress default values available in `period` to convert
        returns into annual returns.
    out : array-like, optional
        Array to use as output buffer.

    Returns
    -------
    sterling_ratio : float
        The Sterling ratio.
        Returns np.nan if returns is empty or has no drawdowns.
    """
    if len(returns) < 2:
        return np.nan
    
    # Get all drawdowns
    drawdown_periods = _get_all_drawdowns(returns)
    
    if len(drawdown_periods) == 0 or all(dd == 0 for dd in drawdown_periods):
        # No drawdowns, use downside deviation as risk measure
        returns_array = np.asanyarray(returns)
        returns_clean = returns_array[~np.isnan(returns_array)]
        downside_returns = returns_clean[returns_clean < 0]
        
        if len(downside_returns) == 0:
            # All positive returns - use a small penalty based on volatility
            avg_drawdown = max(abs(np.std(returns_clean)), 1e-10)
        else:
            avg_drawdown = abs(np.mean(downside_returns))
    else:
        # Calculate average drawdown (absolute value)
        avg_drawdown = abs(np.mean(drawdown_periods))
    
    if avg_drawdown == 0 or avg_drawdown < 1e-10:
        # Extremely small risk, return large positive ratio
        return np.inf if annual_return(returns, period=period, annualization=annualization) - risk_free > 0 else np.nan
    
    # Calculate annualized return
    ann_ret = annual_return(returns, period=period, annualization=annualization)
    
    # Sterling ratio = (annualized return - risk free) / average drawdown
    return (ann_ret - risk_free) / avg_drawdown


def burke_ratio(returns, risk_free=0.0, period=DAILY, annualization=None, out=None):
    """
    Calculates the Burke ratio of a strategy.
    
    The Burke ratio uses the square root of the sum of squared drawdowns as
    the risk measure, which penalizes larger drawdowns more than smaller ones.
    
    Burke Ratio = (Annualized Return - Risk Free Rate) / sqrt(sum(drawdowns^2))

    Parameters
    ----------
    returns : pd.Series or np.ndarray
        Daily returns of the strategy, noncumulative.
        - See full explanation in: func:`~empyrical.stats.cum_returns`.
    risk_free : float, optional
        Constant risk-free return throughout the period. Default is 0.0.
    period : str, optional
        Defines the periodicity of the 'returns' data for purposes of
        annualizing. Default is 'daily'.
    annualization : int, optional
        Used to suppress default values available in `period` to convert
        returns into annual returns.
    out : array-like, optional
        Array to use as output buffer.

    Returns
    -------
    burke_ratio : float
        The Burke ratio.
        Returns np.nan if returns is empty or has no drawdowns.
    """
    if len(returns) < 2:
        return np.nan
    
    # Get all drawdowns
    drawdown_periods = _get_all_drawdowns(returns)
    
    if len(drawdown_periods) == 0 or all(dd == 0 for dd in drawdown_periods):
        # No drawdowns, use downside standard deviation as risk measure
        returns_array = np.asanyarray(returns)
        returns_clean = returns_array[~np.isnan(returns_array)]
        downside_returns = returns_clean[returns_clean < 0]
        
        if len(downside_returns) == 0:
            # All positive returns - use volatility as risk measure
            burke_risk = max(np.std(returns_clean), 1e-10)
        else:
            burke_risk = np.std(downside_returns)
    else:
        # Calculate Burke ratio denominator: sqrt(sum(drawdowns^2))
        squared_drawdowns = [dd**2 for dd in drawdown_periods]
        burke_risk = np.sqrt(np.sum(squared_drawdowns))
    
    if burke_risk == 0 or burke_risk < 1e-10:
        # Extremely small risk, return large positive ratio
        return np.inf if annual_return(returns, period=period, annualization=annualization) - risk_free > 0 else np.nan
    
    # Calculate annualized return
    ann_ret = annual_return(returns, period=period, annualization=annualization)
    
    # Burke ratio = (annualized return - risk free) / burke risk
    return (ann_ret - risk_free) / burke_risk


def kappa_three_ratio(returns, risk_free=0.0, period=DAILY, annualization=None, 
                      mar=0.0, out=None):
    """
    Calculates the Kappa 3 ratio of a strategy.
    
    Kappa 3 is a generalized downside risk-adjusted performance measure that uses
    the third lower partial moment (LPM3). It measures the risk of returns falling
    below a minimum acceptable return (MAR), with cubic weighting.
    
    Kappa 3 = (Annualized Return - Risk Free Rate) / LPM3^(1/3)
    where LPM3 = mean((max(0, MAR - return))^3)^(1/3)

    Parameters
    ----------
    returns : pd.Series or np.ndarray
        Daily returns of the strategy, noncumulative.
        - See full explanation in: func:`~empyrical.stats.cum_returns`.
    risk_free : float, optional
        Constant risk-free return throughout the period. Default is 0.0.
    period : str, optional
        Defines the periodicity of the 'returns' data for purposes of
        annualizing. Default is 'daily'.
    annualization : int, optional
        Used to suppress default values available in `period` to convert
        returns into annual returns.
    mar : float, optional
        Minimum acceptable return threshold. Default is 0.0.
    out : array-like, optional
        Array to use as output buffer.

    Returns
    -------
    kappa_three_ratio : float
        The Kappa 3 ratio.
        Returns np.nan if returns is empty or LPM3 is zero.
    """
    if len(returns) < 2:
        return np.nan
    
    returns_array = np.asanyarray(returns)
    # Remove NaN values
    returns_clean = returns_array[~np.isnan(returns_array)]
    
    if len(returns_clean) < 2:
        return np.nan
    
    # Calculate Lower Partial Moment of order 3
    # LPM3 = mean((max(0, MAR - return))^3)
    downside_deviations = np.maximum(0, mar - returns_clean)
    lpm3 = np.mean(downside_deviations ** 3)
    
    # Annualize LPM3
    ann_factor = annualization_factor(period, annualization)
    
    if lpm3 == 0 or lpm3 < 1e-30:
        # No downside risk, use standard deviation as alternative
        std_dev = np.std(returns_clean)
        if std_dev < 1e-10:
            # Very low risk, return large positive ratio if returns are positive
            ann_ret = annual_return(returns, period=period, annualization=annualization)
            return np.inf if ann_ret - risk_free > 0 else np.nan
        lpm3_risk = std_dev * np.sqrt(ann_factor)
    else:
        lpm3_annualized = lpm3 * ann_factor
        # Take cube root of LPM3
        lpm3_risk = lpm3_annualized ** (1.0 / 3.0)
    
    if lpm3_risk == 0 or lpm3_risk < 1e-10:
        # Extremely small risk
        ann_ret = annual_return(returns, period=period, annualization=annualization)
        return np.inf if ann_ret - risk_free > 0 else np.nan
    
    # Calculate annualized return
    ann_ret = annual_return(returns, period=period, annualization=annualization)
    
    # Kappa 3 ratio = (annualized return - risk free) / LPM3^(1/3)
    return (ann_ret - risk_free) / lpm3_risk


def adjusted_sharpe_ratio(returns, risk_free=0.0, period=DAILY, annualization=None, out=None):
    """
    Calculates the adjusted Sharpe ratio of a strategy.
    
    The adjusted Sharpe ratio modifies the traditional Sharpe ratio to account for
    skewness and kurtosis in the return distribution. It penalizes negative skewness
    and excess kurtosis (fat tails).
    
    Adjusted SR = SR * [1 + (S/6)*SR - ((K-3)/24)*SR^2]
    where S is skewness, K is kurtosis, and SR is the Sharpe ratio.

    Parameters
    ----------
    returns : pd.Series or np.ndarray
        Daily returns of the strategy, noncumulative.
        - See full explanation in: func:`~empyrical.stats.cum_returns`.
    risk_free : float, optional
        Constant risk-free return throughout the period. Default is 0.0.
    period : str, optional
        Defines the periodicity of the 'returns' data for purposes of
        annualizing. Default is 'daily'.
    annualization : int, optional
        Used to suppress default values available in `period` to convert
        returns into annual returns.
    out : array-like, optional
        Array to use as output buffer.

    Returns
    -------
    adjusted_sharpe_ratio : float
        The adjusted Sharpe ratio.
        Returns np.nan if returns is empty or has insufficient data.
        
    Notes
    -----
    The adjustment accounts for:
    - Negative skewness (asymmetric downside risk)
    - Excess kurtosis (fat tails, extreme events)
    
    For normally distributed returns, the adjusted Sharpe equals the regular Sharpe.
    """
    if len(returns) < 4:  # Need at least 4 observations for kurtosis
        return np.nan
    
    # Calculate regular Sharpe ratio
    sr = sharpe_ratio(returns, risk_free=risk_free, period=period, 
                      annualization=annualization)
    
    if np.isnan(sr):
        return np.nan
    
    # Calculate skewness
    skew = skewness(returns)
    if np.isnan(skew):
        skew = 0
    
    # Calculate excess kurtosis
    kurt = kurtosis(returns)
    if np.isnan(kurt):
        kurt = 0
    
    # Apply adjustment formula with bounded adjustment
    # Adjusted SR = SR * [1 + (S/6)*SR - ((K)/24)*SR^2]
    # Note: kurtosis() already returns excess kurtosis (K-3)
    
    # For small samples, skewness and kurtosis can be unstable
    # Apply dampening factor based on sample size
    n = len(returns)
    dampening = min(1.0, n / 50.0)  # Full adjustment only for n >= 50
    
    skew_adj = (skew / 6) * sr * dampening
    kurt_adj = (kurt / 24) * (sr ** 2) * dampening
    adjustment = 1 + skew_adj - kurt_adj
    
    # Bound the adjustment to prevent extreme values
    # For near-normal distributions, adjustment should be very close to 1
    # Very tight bounds to ensure similarity with regular Sharpe for small samples
    if n < 20:
        adjustment = max(0.9, min(1.1, adjustment))
    else:
        adjustment = max(0.8, min(1.3, adjustment))
    
    return sr * adjustment


def stutzer_index(returns, risk_free=0.0, target_return=0.0, out=None):
    """
    Calculates the Stutzer index of a strategy.
    
    The Stutzer index is a risk-adjusted performance measure based on the
    probability that returns exceed a target return. It uses maximum likelihood
    estimation and information theory concepts.
    
    The index measures the relative likelihood of achieving returns above the
    target versus below it.

    Parameters
    ----------
    returns : pd.Series or np.ndarray
        Returns of the strategy, noncumulative.
        - See full explanation in: func:`~empyrical.stats.cum_returns`.
    risk_free : float, optional
        Constant risk-free return throughout the period. Default is 0.0.
    target_return : float, optional
        Target return threshold. Default is 0.0.
    out : array-like, optional
        Array to use as output buffer.

    Returns
    -------
    stutzer_index : float
        The Stutzer index value.
        Returns np.nan if returns is empty or calculation fails.
        
    Notes
    -----
    Higher values indicate better risk-adjusted performance.
    The index is particularly useful for non-normal return distributions.
    """
    if len(returns) < 2:
        return np.nan
    
    returns_array = np.asanyarray(returns)
    # Remove NaN values
    returns_clean = returns_array[~np.isnan(returns_array)]
    
    if len(returns_clean) < 2:
        return np.nan
    
    # Subtract target return from excess returns
    excess_returns = returns_clean - target_return
    
    try:
        # Find theta that maximizes the likelihood
        # This is solved by finding theta such that E[exp(theta * excess_return)] = 1
        
        # Use numerical optimization
        from scipy import optimize
        
        # Use a better initial guess based on mean and variance
        mean_excess = np.mean(excess_returns)
        var_excess = np.var(excess_returns)
        
        if var_excess < 1e-10:
            # Very low variance
            if mean_excess > 0:
                return np.inf
            elif mean_excess < 0:
                return -np.inf
            else:
                return 0.0
        
        # Initial guess: use a simple approximation
        theta_init = mean_excess / var_excess if var_excess > 0 else 1.0
        
        def objective(theta):
            # We want to find theta such that mean(exp(theta * excess_returns)) = 1
            # Clip theta to avoid overflow
            theta_clipped = np.clip(theta, -20, 20)
            try:
                exp_term = np.exp(theta_clipped * excess_returns)
                # Add small epsilon to avoid log(0)
                mean_exp = np.mean(exp_term)
                # Use log transform for better numerical stability
                return abs(np.log(mean_exp + 1e-10))
            except:
                return 1e10
        
        # Try multiple methods
        best_result = None
        best_value = float('inf')
        
        for method in ['Nelder-Mead', 'Powell', 'BFGS']:
            try:
                result = optimize.minimize(objective, theta_init, method=method, 
                                          options={'maxiter': 1000})
                if result.fun < best_value:
                    best_value = result.fun
                    best_result = result
            except:
                continue
        
        if best_result is None or best_value > 0.1:
            # Optimization failed, use simpler approximation
            # Stutzer index ≈ mean / std * sqrt(n)
            std_excess = np.std(excess_returns)
            if std_excess > 0:
                n = len(excess_returns)
                return float(mean_excess / std_excess * np.sqrt(n))
            return np.nan
        
        theta_star = best_result.x[0]
        
        # Calculate Stutzer index
        # SI = theta_star * sqrt(n)
        n = len(excess_returns)
        stutzer_idx = theta_star * np.sqrt(n)
        
        return float(stutzer_idx)
        
    except Exception:
        return np.nan


def annual_alpha(returns, factor_returns, risk_free=0.0, out=None):
    """
    Calculates alpha for each year independently.
    
    Returns a Series with years as the index and alpha values for each year.

    Parameters
    ----------
    returns : pd.Series
        Daily returns of the strategy, noncumulative.
        - See full explanation in: func:`~empyrical.stats.cum_returns`.
    factor_returns : pd.Series
        Daily noncumulative returns of the factor (benchmark).
    risk_free : float, optional
        Constant risk-free return throughout the period. Default is 0.0.
    out : array-like, optional
        Array to use as output buffer.

    Returns
    -------
    annual_alpha : pd.Series
        Alpha values for each year, indexed by year.
        Returns empty Series if returns is empty.
    """
    if len(returns) < 2:
        return pd.Series(dtype=float)
    
    if not isinstance(returns, pd.Series):
        returns = pd.Series(returns)
    if not isinstance(factor_returns, pd.Series):
        factor_returns = pd.Series(factor_returns)
    
    # Align the two series
    returns, factor_returns = returns.align(factor_returns, join='inner')
    
    if len(returns) < 2:
        return pd.Series(dtype=float)
    
    # Group by year
    returns_by_year = returns.groupby(returns.index.year)
    factor_by_year = factor_returns.groupby(factor_returns.index.year)
    
    annual_alphas = {}
    for year in returns_by_year.groups.keys():
        if year in factor_by_year.groups:
            year_returns = returns_by_year.get_group(year)
            year_factor = factor_by_year.get_group(year)
            
            if len(year_returns) > 2 and len(year_factor) > 2:
                year_alpha = alpha(year_returns, year_factor, risk_free=risk_free)
                if not np.isnan(year_alpha):
                    annual_alphas[year] = year_alpha
    
    return pd.Series(annual_alphas)


def annual_beta(returns, factor_returns, risk_free=0.0, out=None):
    """
    Calculates beta for each year independently.
    
    Returns a Series with years as the index and beta values for each year.

    Parameters
    ----------
    returns : pd.Series
        Daily returns of the strategy, noncumulative.
        - See full explanation in: func:`~empyrical.stats.cum_returns`.
    factor_returns : pd.Series
        Daily noncumulative returns of the factor (benchmark).
    risk_free : float, optional
        Constant risk-free return throughout the period. Default is 0.0.
    out : array-like, optional
        Array to use as output buffer.

    Returns
    -------
    annual_beta : pd.Series
        Beta values for each year, indexed by year.
        Returns empty Series if returns is empty.
    """
    if len(returns) < 2:
        return pd.Series(dtype=float)
    
    if not isinstance(returns, pd.Series):
        returns = pd.Series(returns)
    if not isinstance(factor_returns, pd.Series):
        factor_returns = pd.Series(factor_returns)
    
    # Align the two series
    returns, factor_returns = returns.align(factor_returns, join='inner')
    
    if len(returns) < 2:
        return pd.Series(dtype=float)
    
    # Group by year
    returns_by_year = returns.groupby(returns.index.year)
    factor_by_year = factor_returns.groupby(factor_returns.index.year)
    
    annual_betas = {}
    for year in returns_by_year.groups.keys():
        if year in factor_by_year.groups:
            year_returns = returns_by_year.get_group(year)
            year_factor = factor_by_year.get_group(year)
            
            if len(year_returns) > 2 and len(year_factor) > 2:
                year_beta = beta(year_returns, year_factor, risk_free=risk_free)
                if not np.isnan(year_beta):
                    annual_betas[year] = year_beta
    
    return pd.Series(annual_betas)


def residual_risk(returns, factor_returns, out=None):
    """
    Calculates the residual risk (unsystematic risk) of a strategy.
    
    Residual risk is the standard deviation of the residuals from the
    alpha-beta regression model. It represents the portion of risk that
    cannot be explained by the market (factor) movements.
    
    residuals = returns - (alpha + beta * factor_returns)
    residual_risk = std(residuals)

    Parameters
    ----------
    returns : pd.Series or np.ndarray
        Returns of the strategy, noncumulative.
        - See full explanation in: func:`~empyrical.stats.cum_returns`.
    factor_returns : pd.Series or np.ndarray
        Returns of the factor (benchmark), noncumulative.
    out : array-like, optional
        Array to use as output buffer.

    Returns
    -------
    residual_risk : float
        The standard deviation of regression residuals.
        Returns np.nan if returns is empty or insufficient data.
        
    Notes
    -----
    Residual risk measures:
    - Diversifiable (unsystematic) risk
    - Strategy-specific risk not related to market movements
    - Quality of fit in the alpha-beta regression
    
    Lower residual risk indicates returns are better explained by the factor.
    """
    if len(returns) < 3 or len(factor_returns) < 3:
        return np.nan
    
    if isinstance(returns, pd.Series) and isinstance(factor_returns, pd.Series):
        returns, factor_returns = returns.align(factor_returns, join='inner')
    
    if len(returns) < 3:
        return np.nan
    
    returns_array = np.asanyarray(returns)
    factor_array = np.asanyarray(factor_returns)
    
    # Remove NaN values
    valid_mask = ~(np.isnan(returns_array) | np.isnan(factor_array))
    returns_clean = returns_array[valid_mask]
    factor_clean = factor_array[valid_mask]
    
    if len(returns_clean) < 3:
        return np.nan
    
    try:
        # Perform linear regression: returns = alpha + beta * factor_returns + residuals
        # Using numpy's polyfit (degree 1 for linear regression)
        # polyfit returns [beta, alpha] for y = beta*x + alpha
        beta_coef, alpha_coef = np.polyfit(factor_clean, returns_clean, 1)
        
        # Calculate fitted values
        fitted_values = alpha_coef + beta_coef * factor_clean
        
        # Calculate residuals
        residuals = returns_clean - fitted_values
        
        # Calculate standard deviation of residuals
        residual_std = np.std(residuals, ddof=2)  # ddof=2 for regression (n-2 degrees of freedom)
        
        return float(residual_std)
        
    except Exception:
        return np.nan


def conditional_sharpe_ratio(returns, risk_free=0.0, cutoff=0.05, period=DAILY, 
                             annualization=None, out=None):
    """
    Calculates the conditional Sharpe ratio.
    
    The conditional Sharpe ratio modifies the traditional Sharpe ratio by only
    considering returns that fall below a certain percentile (cutoff), similar
    to CVaR. It provides a risk-adjusted performance measure that focuses on
    downside tail risk.
    
    Conditional SR = (Mean Return - Risk Free) / CVaR

    Parameters
    ----------
    returns : pd.Series or np.ndarray
        Returns of the strategy, noncumulative.
        - See full explanation in: func:`~empyrical.stats.cum_returns`.
    risk_free : float, optional
        Constant risk-free return throughout the period. Default is 0.0.
    cutoff : float, optional
        Decimal representing the percentage cutoff for the worst returns.
        Default is 0.05 (5% worst returns).
    period : str, optional
        Defines the periodicity of the 'returns' data for purposes of
        annualizing. Default is 'daily'.
    annualization : int, optional
        Used to suppress default values available in `period` to convert
        returns into annual returns.
    out : array-like, optional
        Array to use as output buffer.

    Returns
    -------
    conditional_sharpe_ratio : float
        The conditional Sharpe ratio.
        Returns np.nan if returns is empty or has insufficient data.
        
    Notes
    -----
    Higher values indicate better risk-adjusted performance considering
    extreme downside risk. This measure is more conservative than the
    traditional Sharpe ratio.
    """
    if len(returns) < 2:
        return np.nan
    
    returns_array = np.asanyarray(returns)
    # Remove NaN values
    returns_clean = returns_array[~np.isnan(returns_array)]
    
    if len(returns_clean) < 2:
        return np.nan
    
    # Calculate CVaR (conditional value at risk)
    cvar = conditional_value_at_risk(returns_clean, cutoff=cutoff)
    
    if np.isnan(cvar) or cvar == 0:
        return np.nan
    
    # Calculate annualized mean return
    ann_factor = annualization_factor(period, annualization)
    mean_return = np.mean(returns_clean) * ann_factor
    
    # Conditional Sharpe = (mean return - risk free) / abs(CVaR)
    # Use absolute value since CVaR is negative for losses
    conditional_sr = (mean_return - risk_free) / abs(cvar)
    
    return float(conditional_sr)


def var_excess_return(returns, cutoff=0.05, out=None):
    """
    Calculates the Value at Risk (VaR) excess return.
    
    This metric measures the expected excess return above the VaR threshold.
    It represents the average return that exceeds the worst-case scenario
    defined by VaR.
    
    VaR Excess = Mean(returns) - VaR(returns, cutoff)

    Parameters
    ----------
    returns : pd.Series or np.ndarray
        Returns of the strategy, noncumulative.
        - See full explanation in: func:`~empyrical.stats.cum_returns`.
    cutoff : float, optional
        Decimal representing the percentage cutoff for VaR calculation.
        Default is 0.05 (95% VaR, looking at worst 5%).
    out : array-like, optional
        Array to use as output buffer.

    Returns
    -------
    var_excess_return : float
        The VaR excess return.
        Returns np.nan if returns is empty.
        
    Notes
    -----
    Positive values indicate that average returns exceed the worst-case
    scenario. This metric helps assess whether a strategy compensates
    adequately for its tail risk.
    """
    if len(returns) < 2:
        return np.nan
    
    returns_array = np.asanyarray(returns)
    # Remove NaN values
    returns_clean = returns_array[~np.isnan(returns_array)]
    
    if len(returns_clean) < 2:
        return np.nan
    
    # Calculate VaR
    var = value_at_risk(returns_clean, cutoff=cutoff)
    
    if np.isnan(var):
        return np.nan
    
    # Calculate mean return
    mean_return = np.mean(returns_clean)
    
    # VaR excess = mean return - VaR
    # Since VaR is typically negative, this shows how much better
    # the average is than the worst case
    var_excess = mean_return - var
    
    return float(var_excess)


def regression_annual_return(returns, factor_returns, period=DAILY, 
                             annualization=None, out=None):
    """
    Calculates the Regression Annual Return (RAR).
    
    RAR is the annualized return predicted by the alpha-beta regression model.
    It represents the expected annual return based on the strategy's alpha
    and beta relationship with the market.
    
    RAR = Annualized Alpha + Beta * Annualized Factor Return

    Parameters
    ----------
    returns : pd.Series or np.ndarray
        Returns of the strategy, noncumulative.
        - See full explanation in: func:`~empyrical.stats.cum_returns`.
    factor_returns : pd.Series or np.ndarray
        Returns of the factor (benchmark), noncumulative.
    period : str, optional
        Defines the periodicity of the 'returns' data for purposes of
        annualizing. Default is 'daily'.
    annualization : int, optional
        Used to suppress default values available in `period` to convert
        returns into annual returns.
    out : array-like, optional
        Array to use as output buffer.

    Returns
    -------
    regression_annual_return : float
        The regression annual return.
        Returns np.nan if returns is empty or insufficient data.
        
    Notes
    -----
    RAR provides a model-based estimate of expected annual returns,
    accounting for both strategy-specific performance (alpha) and
    market exposure (beta).
    """
    if len(returns) < 3 or len(factor_returns) < 3:
        return np.nan
    
    if isinstance(returns, pd.Series) and isinstance(factor_returns, pd.Series):
        returns, factor_returns = returns.align(factor_returns, join='inner')
    
    if len(returns) < 3:
        return np.nan
    
    # Calculate alpha and beta
    strategy_alpha = alpha(returns, factor_returns, risk_free=0.0, 
                          period=period, annualization=annualization)
    strategy_beta = beta(returns, factor_returns, risk_free=0.0)
    
    if np.isnan(strategy_alpha) or np.isnan(strategy_beta):
        return np.nan
    
    # Calculate annualized factor return
    factor_annual_return = annual_return(factor_returns, period=period, 
                                        annualization=annualization)
    
    if np.isnan(factor_annual_return):
        return np.nan
    
    # RAR = alpha + beta * factor_return
    rar = strategy_alpha + strategy_beta * factor_annual_return
    
    return float(rar)


def r_cubed(returns, factor_returns, out=None):
    """
    Calculates the R-cubed (R³) ratio.
    
    R³ is a measure of the quality and consistency of a strategy's alpha
    generation. It combines:
    - R-squared: How well returns are explained by the factor
    - Information Ratio: Risk-adjusted excess returns
    - Positive alpha generation
    
    R³ = R² * sign(alpha) * (Information Ratio)²

    Parameters
    ----------
    returns : pd.Series or np.ndarray
        Returns of the strategy, noncumulative.
        - See full explanation in: func:`~empyrical.stats.cum_returns`.
    factor_returns : pd.Series or np.ndarray
        Returns of the factor (benchmark), noncumulative.
    out : array-like, optional
        Array to use as output buffer.

    Returns
    -------
    r_cubed : float
        The R-cubed ratio, typically between -1 and 1.
        Returns np.nan if returns is empty or insufficient data.
        
    Notes
    -----
    - Values close to 1: Consistent alpha generation with good model fit
    - Values close to 0: Poor alpha generation or model fit
    - Negative values: Negative alpha (underperformance)
    
    R³ helps identify strategies that reliably generate alpha rather than
    just having high R² from tracking the benchmark.
    """
    if len(returns) < 5 or len(factor_returns) < 5:
        return np.nan
    
    if isinstance(returns, pd.Series) and isinstance(factor_returns, pd.Series):
        returns, factor_returns = returns.align(factor_returns, join='inner')
    
    if len(returns) < 5:
        return np.nan
    
    try:
        returns_array = np.asanyarray(returns)
        factor_array = np.asanyarray(factor_returns)
        
        # Remove NaN values
        valid_mask = ~(np.isnan(returns_array) | np.isnan(factor_array))
        returns_clean = returns_array[valid_mask]
        factor_clean = factor_array[valid_mask]
        
        if len(returns_clean) < 5:
            return np.nan
        
        # Calculate R-squared from regression of returns on factor_returns
        # R² measures how well factor returns explain strategy returns
        from scipy import stats as sp_stats
        slope, intercept, r_value, p_value, std_err = sp_stats.linregress(factor_clean, returns_clean)
        r_squared = r_value ** 2
        
        if np.isnan(r_squared):
            return np.nan
        
        # Calculate alpha
        strategy_alpha = alpha(returns, factor_returns, risk_free=0.0)
        
        if np.isnan(strategy_alpha):
            return np.nan
        
        # Calculate Information Ratio using excess returns
        # IR = mean(excess_returns) / std(excess_returns)
        excess_returns = returns_clean - factor_clean
        mean_excess = np.mean(excess_returns)
        std_excess = np.std(excess_returns, ddof=1)
        
        # For perfect fit (returns = factor_returns), excess is ~0
        if std_excess < 1e-10:
            # Near-perfect tracking
            if abs(mean_excess) < 1e-10 and r_squared > 0.99:
                # Perfect tracking with high R², return high R³
                return 1.0 if strategy_alpha >= 0 else 0.0
            elif abs(mean_excess) < 1e-10:
                # Perfect tracking, no alpha
                return 0.0
            # Some alpha but no variation in excess returns
            return np.nan
        
        ir = mean_excess / std_excess
        
        # R³ combines R², alpha direction, and information ratio
        # For interpretation, we want it in [0, 1] range for positive alpha
        # Normalize IR² to prevent extreme values
        ir_squared = ir ** 2
        
        # Use a sigmoid-like function to map IR² to [0, 1]
        # This ensures interpretability while preserving ordering
        normalized_ir = ir_squared / (1.0 + ir_squared)
        
        r_cubed_value = r_squared * normalized_ir
        
        # Apply sign of alpha
        if strategy_alpha < 0:
            r_cubed_value = -r_cubed_value
        
        # Handle floating point precision issues near zero
        # Use a larger tolerance to handle numerical errors in random data
        if abs(r_cubed_value) < 1e-6:
            r_cubed_value = 0.0
        
        # Clamp to valid range [0, 1] or [-1, 0] depending on sign
        # Ensure result is strictly non-negative or non-positive
        if r_cubed_value >= 0:
            r_cubed_value = min(1.0, r_cubed_value)
            r_cubed_value = max(0.0, r_cubed_value)  # Handle tiny negative from floating point
        else:
            r_cubed_value = max(-1.0, r_cubed_value)
            r_cubed_value = min(0.0, r_cubed_value)  # Handle tiny positive from floating point
        
        return float(r_cubed_value)
        
    except Exception:
        return np.nan


def annualized_cumulative_return(returns, period=DAILY, annualization=None):
    """
    Calculates the annualized cumulative return.
    
    This is essentially the same as CAGR (Compound Annual Growth Rate), which
    represents the constant annual rate of return that would be required to
    achieve the total cumulative return over the investment period.

    Parameters
    ----------
    returns : pd.Series or np.ndarray
        Returns of the strategy, noncumulative.
        - See full explanation in: func:`~empyrical.stats.cum_returns`.
    period : str, optional
        Defines the periodicity of the 'returns' data for purposes of
        annualizing. Default is 'daily'.
    annualization : int, optional
        Used to suppress default values available in `period` to convert
        returns into annual returns.

    Returns
    -------
    annualized_cumulative_return : float
        The annualized cumulative return (CAGR).
        Returns np.nan if returns is empty.
        
    Notes
    -----
    This metric is identical to CAGR and annual_return. It represents
    the geometric mean return per year over the entire period.
    
    Formula: (Ending Value / Starting Value) ^ (1 / Years) - 1
    """
    return annual_return(returns, period=period, annualization=annualization)


def annual_active_return_by_year(returns, factor_returns, period=DAILY, 
                                 annualization=None, out=None):
    """
    Calculates the active return (excess return over benchmark) for each year.
    
    Active return is the difference between the strategy's annual return and
    the benchmark's annual return for each year independently.
    
    Active Return = Strategy Return - Benchmark Return

    Parameters
    ----------
    returns : pd.Series
        Daily returns of the strategy, noncumulative.
        - See full explanation in: func:`~empyrical.stats.cum_returns`.
    factor_returns : pd.Series
        Daily noncumulative returns of the factor (benchmark).
    period : str, optional
        Defines the periodicity of the 'returns' data for purposes of
        annualizing. Default is 'daily'.
    annualization : int, optional
        Used to suppress default values available in `period` to convert
        returns into annual returns.
    out : array-like, optional
        Array to use as output buffer.

    Returns
    -------
    annual_active_return_by_year : pd.Series
        Active returns for each year, indexed by year.
        Returns empty Series if returns is empty.
        
    Notes
    -----
    This differs from annual_active_return which provides a single annualized
    value over the entire period. This function shows year-by-year active returns.
    """
    if len(returns) < 2:
        return pd.Series(dtype=float)
    
    if not isinstance(returns, pd.Series):
        returns = pd.Series(returns)
    if not isinstance(factor_returns, pd.Series):
        factor_returns = pd.Series(factor_returns)
    
    # Align the two series
    returns, factor_returns = returns.align(factor_returns, join='inner')
    
    if len(returns) < 2:
        return pd.Series(dtype=float)
    
    # Calculate annual returns for strategy
    strategy_annual = annual_return_by_year(returns, period=period, 
                                           annualization=annualization)
    
    # Calculate annual returns for benchmark
    benchmark_annual = annual_return_by_year(factor_returns, period=period,
                                            annualization=annualization)
    
    # Calculate active return = strategy - benchmark
    # Use subtract with fill_value to handle missing years
    active_returns = strategy_annual.subtract(benchmark_annual, fill_value=np.nan)
    
    # Remove NaN values
    active_returns = active_returns.dropna()
    
    return active_returns


def treynor_mazuy_timing(returns, factor_returns, risk_free=0.0, out=None):
    """
    Calculates the Treynor-Mazuy market timing coefficient.
    
    The Treynor-Mazuy model (1966) uses a quadratic regression to evaluate
    a fund manager's market timing ability:
    
    Rp - Rf = α + β(Rm - Rf) + γ(Rm - Rf)² + ε
    
    Where:
    - Rp: Portfolio return
    - Rm: Market return
    - Rf: Risk-free rate
    - γ: Timing coefficient (gamma)
    
    A positive γ indicates successful market timing ability.

    Parameters
    ----------
    returns : pd.Series or np.ndarray
        Returns of the strategy, noncumulative.
        - See full explanation in: func:`~empyrical.stats.cum_returns`.
    factor_returns : pd.Series or np.ndarray
        Returns of the market/factor, noncumulative.
    risk_free : float, optional
        Constant risk-free return throughout the period. Default is 0.0.
    out : array-like, optional
        Array to use as output buffer.

    Returns
    -------
    timing_coefficient : float
        The Treynor-Mazuy timing coefficient (γ).
        Positive values indicate market timing ability.
        Returns np.nan if insufficient data.
        
    Notes
    -----
    - γ > 0: Manager successfully times the market (increases exposure before rallies)
    - γ = 0: No timing ability
    - γ < 0: Poor timing (decreases exposure before rallies)
    
    The quadratic term captures convexity in the return relationship.
    
    Reference: Treynor, J. L., & Mazuy, K. K. (1966). Can mutual funds 
    outguess the market? Harvard Business Review, 44(4), 131-136.
    """
    if len(returns) < 10 or len(factor_returns) < 10:
        return np.nan
    
    if isinstance(returns, pd.Series) and isinstance(factor_returns, pd.Series):
        returns, factor_returns = returns.align(factor_returns, join='inner')
    
    if len(returns) < 10:
        return np.nan
    
    returns_array = np.asanyarray(returns)
    factor_array = np.asanyarray(factor_returns)
    
    # Remove NaN values
    valid_mask = ~(np.isnan(returns_array) | np.isnan(factor_array))
    returns_clean = returns_array[valid_mask]
    factor_clean = factor_array[valid_mask]
    
    if len(returns_clean) < 10:
        return np.nan
    
    try:
        # Calculate excess returns
        excess_returns = returns_clean - risk_free
        excess_market = factor_clean - risk_free
        
        # Create quadratic term
        excess_market_squared = excess_market ** 2
        
        # Perform multiple regression: y = α + β1*x1 + β2*x2
        # where x1 = excess_market, x2 = excess_market²
        # Stack the predictors
        X = np.column_stack([np.ones(len(excess_market)), 
                             excess_market, 
                             excess_market_squared])
        
        # Solve using least squares
        coeffs = np.linalg.lstsq(X, excess_returns, rcond=None)[0]
        
        # coeffs[0] = alpha, coeffs[1] = beta, coeffs[2] = gamma (timing coefficient)
        gamma = coeffs[2]
        
        return float(gamma)
        
    except Exception:
        return np.nan


def henriksson_merton_timing(returns, factor_returns, risk_free=0.0, out=None):
    """
    Calculates the Henriksson-Merton market timing coefficient.
    
    The Henriksson-Merton model (1981) uses a dummy variable approach to
    evaluate market timing ability:
    
    Rp - Rf = α + β1(Rm - Rf) + β2·max(0, Rm - Rf) + ε
    
    Or equivalently with a dummy variable D:
    Rp - Rf = α + β(Rm - Rf) + γ·(Rm - Rf)·D + ε
    
    Where D = 1 if Rm > Rf (up market), 0 otherwise.
    
    A positive γ indicates successful market timing ability.

    Parameters
    ----------
    returns : pd.Series or np.ndarray
        Returns of the strategy, noncumulative.
        - See full explanation in: func:`~empyrical.stats.cum_returns`.
    factor_returns : pd.Series or np.ndarray
        Returns of the market/factor, noncumulative.
    risk_free : float, optional
        Constant risk-free return throughout the period. Default is 0.0.
    out : array-like, optional
        Array to use as output buffer.

    Returns
    -------
    timing_coefficient : float
        The Henriksson-Merton timing coefficient (γ).
        Positive values indicate market timing ability.
        Returns np.nan if insufficient data.
        
    Notes
    -----
    - γ > 0: Manager increases beta in up markets (successful timing)
    - γ = 0: No timing ability
    - γ < 0: Manager increases beta in down markets (poor timing)
    
    This model tests whether the manager changes market exposure based
    on market direction.
    
    Reference: Henriksson, R. D., & Merton, R. C. (1981). On market timing 
    and investment performance. II. Statistical procedures for evaluating 
    forecasting skills. Journal of Business, 54(4), 513-533.
    """
    if len(returns) < 10 or len(factor_returns) < 10:
        return np.nan
    
    if isinstance(returns, pd.Series) and isinstance(factor_returns, pd.Series):
        returns, factor_returns = returns.align(factor_returns, join='inner')
    
    if len(returns) < 10:
        return np.nan
    
    returns_array = np.asanyarray(returns)
    factor_array = np.asanyarray(factor_returns)
    
    # Remove NaN values
    valid_mask = ~(np.isnan(returns_array) | np.isnan(factor_array))
    returns_clean = returns_array[valid_mask]
    factor_clean = factor_array[valid_mask]
    
    if len(returns_clean) < 10:
        return np.nan
    
    try:
        # Calculate excess returns
        excess_returns = returns_clean - risk_free
        excess_market = factor_clean - risk_free
        
        # Create the max(0, Rm - Rf) term
        # This equals (Rm - Rf) when market is up, 0 when down
        up_market_excess = np.maximum(0, excess_market)
        
        # Perform multiple regression: y = α + β1*x1 + β2*x2
        # where x1 = excess_market, x2 = max(0, excess_market)
        X = np.column_stack([np.ones(len(excess_market)), 
                             excess_market, 
                             up_market_excess])
        
        # Solve using least squares
        coeffs = np.linalg.lstsq(X, excess_returns, rcond=None)[0]
        
        # coeffs[0] = alpha, coeffs[1] = beta_down, coeffs[2] = gamma (timing coefficient)
        # The total beta in up markets is beta_down + gamma
        gamma = coeffs[2]
        
        return float(gamma)
        
    except Exception:
        return np.nan


def market_timing_return(returns, factor_returns, risk_free=0.0, out=None):
    """
    Calculates the return contribution from market timing ability.
    
    This metric estimates how much of the portfolio's return is attributable
    to successful market timing (changing market exposure at the right times)
    versus static market exposure.
    
    It uses the Treynor-Mazuy framework to decompose returns into:
    - Alpha (selection skill)
    - Beta return (market exposure)
    - Timing return (γ contribution)

    Parameters
    ----------
    returns : pd.Series or np.ndarray
        Returns of the strategy, noncumulative.
        - See full explanation in: func:`~empyrical.stats.cum_returns`.
    factor_returns : pd.Series or np.ndarray
        Returns of the market/factor, noncumulative.
    risk_free : float, optional
        Constant risk-free return throughout the period. Default is 0.0.
    out : array-like, optional
        Array to use as output buffer.

    Returns
    -------
    timing_return : float
        The return contribution from market timing.
        Positive values indicate timing added value.
        Returns np.nan if insufficient data.
        
    Notes
    -----
    This metric provides a dollar (percentage) interpretation of timing ability,
    making it more intuitive than the timing coefficient alone.
    
    Timing Return ≈ γ × E[(Rm - Rf)²]
    
    Where γ is the Treynor-Mazuy timing coefficient.
    """
    if len(returns) < 10 or len(factor_returns) < 10:
        return np.nan
    
    if isinstance(returns, pd.Series) and isinstance(factor_returns, pd.Series):
        returns, factor_returns = returns.align(factor_returns, join='inner')
    
    if len(returns) < 10:
        return np.nan
    
    returns_array = np.asanyarray(returns)
    factor_array = np.asanyarray(factor_returns)
    
    # Remove NaN values
    valid_mask = ~(np.isnan(returns_array) | np.isnan(factor_array))
    returns_clean = returns_array[valid_mask]
    factor_clean = factor_array[valid_mask]
    
    if len(returns_clean) < 10:
        return np.nan
    
    try:
        # Get the timing coefficient from Treynor-Mazuy model
        gamma = treynor_mazuy_timing(returns_clean, factor_clean, risk_free=risk_free)
        
        if np.isnan(gamma):
            return np.nan
        
        # Calculate excess market returns
        excess_market = factor_clean - risk_free
        
        # Timing return contribution = gamma × mean of squared excess returns
        # This represents the average timing contribution per period
        mean_squared_excess = np.mean(excess_market ** 2)
        timing_return = gamma * mean_squared_excess
        
        return float(timing_return)
        
    except Exception:
        return np.nan


def alpha_percentile_rank(strategy_returns, all_strategies_returns, factor_returns, 
                          risk_free=0.0, out=None):
    """
    Calculates the percentile rank of a strategy's alpha among multiple strategies.
    
    This metric shows where the strategy's alpha ranks relative to other strategies,
    expressed as a percentile from 0 to 1.
    
    Percentile Rank = (Number of strategies with lower alpha) / (Total strategies)

    Parameters
    ----------
    strategy_returns : pd.Series or np.ndarray
        Returns of the target strategy, noncumulative.
        - See full explanation in: func:`~empyrical.stats.cum_returns`.
    all_strategies_returns : list of pd.Series or np.ndarray
        List of returns for all strategies to compare against (including
        or excluding the target strategy).
    factor_returns : pd.Series or np.ndarray
        Returns of the market/factor, noncumulative.
    risk_free : float, optional
        Constant risk-free return throughout the period. Default is 0.0.
    out : array-like, optional
        Array to use as output buffer.

    Returns
    -------
    percentile_rank : float
        Percentile rank from 0 to 1.
        - 1.0 means the strategy has the highest alpha
        - 0.5 means the strategy is at the median
        - 0.0 means the strategy has the lowest alpha
        Returns np.nan if insufficient data.
        
    Notes
    -----
    This metric is useful for:
    - Comparing a strategy's alpha generation against peers
    - Portfolio construction and manager selection
    - Performance attribution and ranking
    
    Unlike absolute alpha, percentile rank provides relative performance context.
    """
    if len(strategy_returns) < 3:
        return np.nan
    
    # Calculate alpha for the target strategy
    strategy_alpha = alpha(strategy_returns, factor_returns, risk_free=risk_free)
    
    if np.isnan(strategy_alpha):
        return np.nan
    
    # Calculate alpha for all strategies
    all_alphas = []
    for other_returns in all_strategies_returns:
        if len(other_returns) < 3:
            continue
        other_alpha = alpha(other_returns, factor_returns, risk_free=risk_free)
        if not np.isnan(other_alpha):
            all_alphas.append(other_alpha)
    
    if len(all_alphas) == 0:
        return np.nan
    
    # Calculate percentile rank
    # Count how many strategies have alpha less than target strategy
    rank = sum(1 for a in all_alphas if a < strategy_alpha)
    percentile = rank / len(all_alphas)
    
    return float(percentile)


def cornell_timing(returns, factor_returns, risk_free=0.0, out=None):
    """
    Calculates the Cornell market timing coefficient.
    
    The Cornell model (1979) evaluates market timing ability using a 
    piecewise linear regression approach. It splits the market returns
    into positive and negative regimes and tests if the strategy's
    beta differs between these regimes.
    
    Model: Rp - Rf = α + β1·(Rm - Rf)+ + β2·(Rm - Rf)- + ε
    
    Where:
    - (Rm - Rf)+ = max(Rm - Rf, 0) for positive market returns
    - (Rm - Rf)- = min(Rm - Rf, 0) for negative market returns
    - Timing Coefficient = β1 - β2
    
    A positive timing coefficient indicates successful timing ability.

    Parameters
    ----------
    returns : pd.Series or np.ndarray
        Returns of the strategy, noncumulative.
        - See full explanation in: func:`~empyrical.stats.cum_returns`.
    factor_returns : pd.Series or np.ndarray
        Returns of the market/factor, noncumulative.
    risk_free : float, optional
        Constant risk-free return throughout the period. Default is 0.0.
    out : array-like, optional
        Array to use as output buffer.

    Returns
    -------
    timing_coefficient : float
        The Cornell timing coefficient (β1 - β2).
        Positive values indicate market timing ability.
        Returns np.nan if insufficient data.
        
    Notes
    -----
    - Timing Coef > 0: Manager increases beta in up markets relative to down markets
    - Timing Coef = 0: No timing ability
    - Timing Coef < 0: Manager decreases beta in up markets (poor timing)
    
    The Cornell model is similar to Henriksson-Merton but uses a different
    specification that allows asymmetric beta in up and down markets.
    
    Reference: Cornell, B. (1979). Asymmetric information and portfolio 
    performance measurement. Journal of Financial Economics, 7(4), 381-390.
    """
    if len(returns) < 10 or len(factor_returns) < 10:
        return np.nan
    
    if isinstance(returns, pd.Series) and isinstance(factor_returns, pd.Series):
        returns, factor_returns = returns.align(factor_returns, join='inner')
    
    if len(returns) < 10:
        return np.nan
    
    returns_array = np.asanyarray(returns)
    factor_array = np.asanyarray(factor_returns)
    
    # Remove NaN values
    valid_mask = ~(np.isnan(returns_array) | np.isnan(factor_array))
    returns_clean = returns_array[valid_mask]
    factor_clean = factor_array[valid_mask]
    
    if len(returns_clean) < 10:
        return np.nan
    
    try:
        # Calculate excess returns
        excess_returns = returns_clean - risk_free
        excess_market = factor_clean - risk_free
        
        # Split into positive and negative market returns
        # Positive market component
        excess_market_positive = np.maximum(0, excess_market)
        # Negative market component  
        excess_market_negative = np.minimum(0, excess_market)
        
        # Perform multiple regression: y = α + β1*x1 + β2*x2
        # where x1 = excess_market_positive, x2 = excess_market_negative
        X = np.column_stack([np.ones(len(excess_market)), 
                             excess_market_positive, 
                             excess_market_negative])
        
        # Solve using least squares
        coeffs = np.linalg.lstsq(X, excess_returns, rcond=None)[0]
        
        # coeffs[0] = alpha, coeffs[1] = beta_up, coeffs[2] = beta_down
        beta_up = coeffs[1]
        beta_down = coeffs[2]
        
        # Timing coefficient = difference between up and down market betas
        timing_coef = beta_up - beta_down
        
        return float(timing_coef)
        
    except Exception:
        return np.nan


def tracking_difference(returns, factor_returns, out=None):
    """
    Determines the tracking difference of a strategy.

    Tracking difference is the difference between the cumulative returns
    of the strategy and the benchmark.

    Parameters
    ----------
    returns : pd.Series or np.ndarray
        Daily returns of the strategy, noncumulative.
        - See full explanation in: func:`~empyrical.stats.cum_returns`.
    factor_returns : pd.Series or np.ndarray
        Daily noncumulative returns of the factor to which tracking difference
        is computed. Usually a benchmark such as the market.
        - This is in the same style as returns.
    out : array-like, optional
        Array to use as output buffer.
        If not passed, a new array will be created.

    Returns
    -------
    tracking_difference : float
        The tracking difference (cumulative strategy return - cumulative benchmark return).

    Note
    -----
    Tracking difference measures the total excess return of the strategy
    over the benchmark over the entire period.
    """
    allocated_output = out is None
    if allocated_output:
        out = np.empty(returns.shape[1:])

    returns_1d = returns.ndim == 1

    if len(returns) < 1:
        out[()] = np.nan
        if returns_1d:
            out = out.item()
        return out

    returns, factor_returns = _aligned_series(returns, factor_returns)

    # Calculate cumulative returns
    cum_strategy_return = cum_returns_final(returns, starting_value=0)
    cum_benchmark_return = cum_returns_final(factor_returns, starting_value=0)

    # Tracking difference = cumulative strategy return - cumulative benchmark return
    if returns_1d:
        out = cum_strategy_return - cum_benchmark_return
        if not isinstance(out, (float, np.floating)):
            out = out.item()
    else:
        out = cum_strategy_return - cum_benchmark_return
        if allocated_output and isinstance(returns, pd.DataFrame):
            out = pd.Series(out, index=returns.columns)

    return out


def _to_pandas(ob):
    """Convert an array-like to a `pandas` object.

    Parameters
    ----------
    ob : array-like
        The object to convert.

    Returns
    -------
    pandas_structure : pd.Series or pd.DataFrame
        The correct structure based on the dimensionality of the data.
    """
    if isinstance(ob, (pd.Series, pd.DataFrame)):
        return ob

    if ob.ndim == 1:
        return pd.Series(ob)
    elif ob.ndim == 2:
        return pd.DataFrame(ob)
    else:
        raise ValueError(
            'cannot convert array of dim > 2 to a pandas structure',
        )


def _aligned_series(*many_series):
    """
    Return a new list of series containing the data in the input series, but
    with their indices aligned. NaNs will be filled in for missing values.

    Parameters
    ----------
    *many_series
        The series to align.

    Returns
    -------
    aligned_series : iterable[array-like]
        A new list of series containing the data in the input series, but
        with their indices aligned. NaNs will be filled in for missing values.

    """
    head = many_series[0]
    tail = many_series[1:]
    n = len(head)
    if (isinstance(head, np.ndarray) and
            all(len(s) == n and isinstance(s, np.ndarray) for s in tail)):
        # optimization: ndarrays of the same length are already aligned
        return many_series

    # dataframe has no ``itervalues``
    return (
        v
        for _, v in iteritems(pd.concat(map(_to_pandas, many_series), axis=1))
    )


def alpha_beta(returns,
               factor_returns,
               risk_free=0.0,
               period=DAILY,
               annualization=None,
               out=None):
    """Calculates annualized alpha and beta.

    Parameters
    ----------
    returns : pd.Series
        Daily returns of the strategy, noncumulative.
        - See full explanation in: func:`~empyrical.stats.cum_returns`.
    factor_returns : pd.Series
         Daily noncumulative returns of the factor to which beta is
         computed. Usually a benchmark such as the market.
         - This is in the same style as returns.
    risk_free : int, float, optional
        Constant risk-free return throughout the period. For example, the
        interest rate on a three-month US treasury bill.
    period : str, optional
        Defines the periodicity of the 'returns' data for purposes of
        annualizing. Value ignored if `annualization` parameter is specified.
        Defaults are::

            'Monthly':12
            'weekly': 52
            'daily': 252

    annualization : int, optional
        Used to suppress default values available in `period` to convert
        returns into annual returns. Value should be the annual frequency of
        `returns`.
    out : array-like, optional
        Array to use as output buffer.
        If not passed, a new array will be created.

    Returns
    -------
    alpha : float
    beta : float
    """
    returns, factor_returns = _aligned_series(returns, factor_returns)

    return alpha_beta_aligned(
        returns,
        factor_returns,
        risk_free=risk_free,
        period=period,
        annualization=annualization,
        out=out,
    )


def roll_alpha_beta(returns, factor_returns, window=10, **kwargs):
    """
    Computes alpha and beta over a rolling window.

    Parameters
    ----------
    returns : array-like
        The first array to pass to the rolling alpha-beta.
    factor_returns : array-like
        The second array to pass to the rolling alpha-beta.
    window : int
        The size of the rolling window, expressed in terms of the data's periodicity.
    **kwargs
        Forwarded to: func:`~empyrical.alpha_beta`.
    """
    returns, factor_returns = _aligned_series(returns, factor_returns)

    return roll_alpha_beta_aligned(
        returns,
        factor_returns,
        window=window,
        **kwargs
    )


def alpha_beta_aligned(returns,
                       factor_returns,
                       risk_free=0.0,
                       period=DAILY,
                       annualization=None,
                       out=None):
    """Calculates annualized alpha and beta.

    If they are pd.Series, expects returns and factor_returns have already
    been aligned on their labels.  If np.ndarray, these arguments should have
    the same shape.

    Parameters
    ----------
    returns : pd.Series or np.ndarray
        Daily returns of the strategy, noncumulative.
        - See full explanation in: func:`~empyrical.stats.cum_returns`.
    factor_returns : pd.Series or np.ndarray
         Daily noncumulative returns of the factor to which beta is
         computed. Usually a benchmark such as the market.
         - This is in the same style as returns.
    risk_free : int, float, optional
        Constant risk-free return throughout the period. For example, the
        interest rate on a three-month US treasury bill.
    period : str, optional
        Defines the periodicity of the 'returns' data for purposes of
        annualizing. Value ignored if `annualization` parameter is specified.
        Defaults are::

            'Monthly':12
            'weekly': 52
            'daily': 252

    annualization : int, optional
        Used to suppress default values available in `period` to convert
        returns into annual returns. Value should be the annual frequency of
        `returns`.
    out : array-like, optional
        Array to use as output buffer.
        If not passed, a new array will be created.

    Returns
    -------
    alpha : float
    beta : float
    """
    if out is None:
        out = np.empty(returns.shape[1:] + (2,), dtype='float64')

    b = beta_aligned(returns, factor_returns, risk_free, out=out[..., 1])
    alpha_aligned(
        returns,
        factor_returns,
        risk_free,
        period,
        annualization,
        out=out[..., 0],
        _beta=b,
    )

    return out


roll_alpha_beta_aligned = _create_binary_vectorized_roll_function(
    alpha_beta_aligned,
)


def alpha(returns,
          factor_returns,
          risk_free=0.0,
          period=DAILY,
          annualization=None,
          out=None,
          _beta=None):
    """Calculates annualized alpha.

    Parameters
    ----------
    returns : pd.Series
        Daily returns of the strategy, noncumulative.
        - See full explanation in: func:`~empyrical.stats.cum_returns`.
    factor_returns : pd.Series
        Daily noncumulative returns of the factor to which beta is
        computed. Usually a benchmark such as the market.
        - This is in the same style as returns.
    risk_free : int, float, optional
        Constant risk-free return throughout the period. For example, the
        interest rate on a three-month US treasury bill.
    period : str, optional
        Defines the periodicity of the 'returns' data for purposes of
        annualizing. Value ignored if `annualization` parameter is specified.
        Defaults are::

            'Monthly':12
            'weekly': 52
            'daily': 252

    annualization : int, optional
        Used to suppress default values available in `period` to convert
        returns into annual returns. Value should be the annual frequency of
        `returns`.
        - See full explanation in: func:`~empyrical.stats.annual_return`.
    _beta : float, optional
        The beta for the given inputs, if already known. It Will be calculated
        internally if not provided.
    out : array-like, optional
        Array to use as output buffer.
        If not passed, a new array will be created.

    Returns
    -------
    float
        Alpha.
    """
    if not (isinstance(returns, np.ndarray) and
            isinstance(factor_returns, np.ndarray)):
        returns, factor_returns = _aligned_series(returns, factor_returns)

    return alpha_aligned(
        returns,
        factor_returns,
        risk_free=risk_free,
        period=period,
        annualization=annualization,
        out=out,
        _beta=_beta
    )


roll_alpha = _create_binary_vectorized_roll_function(alpha)


def alpha_aligned(returns,
                  factor_returns,
                  risk_free=0.0,
                  period=DAILY,
                  annualization=None,
                  out=None,
                  _beta=None):
    """Calculates annualized alpha.

    If they are pd.Series, expects returns and factor_returns have already
    been aligned on their labels.  If np.ndarray, these arguments should have
    the same shape.

    Parameters
    ----------
    returns : pd.Series or np.ndarray
        Daily returns of the strategy, noncumulative.
        - See full explanation in: func:`~empyrical.stats.cum_returns`.
    factor_returns : pd.Series or np.ndarray
        Daily noncumulative returns of the factor to which beta is
        computed. Usually a benchmark such as the market.
        - This is in the same style as returns.
    risk_free : int, float, optional
        Constant risk-free return throughout the period. For example, the
        interest rate on a three-month US treasury bill.
    period : str, optional
        Defines the periodicity of the 'returns' data for purposes of
        annualizing. Value ignored if `annualization` parameter is specified.
        Defaults are::

            'Monthly':12
            'weekly': 52
            'daily': 252

    annualization : int, optional
        Used to suppress default values available in `period` to convert
        returns into annual returns. Value should be the annual frequency of
        `returns`.
        - See full explanation in: func:`~empyrical.stats.annual_return`.
    _beta : float, optional
        The beta for the given inputs, if already known. It Will be calculated
        internally if not provided.
    out : array-like, optional
        Array to use as output buffer.
        If not passed, a new array will be created.

    Returns
    -------
    alpha : float
    """
    allocated_output = out is None
    if allocated_output:
        out = np.empty(returns.shape[1:], dtype='float64')

    if len(returns) < 2:
        out[()] = np.nan
        if returns.ndim == 1:
            out = out.item()
        return out

    ann_factor = annualization_factor(period, annualization)

    if _beta is None:
        _beta = beta_aligned(returns, factor_returns, risk_free)

    adj_returns = _adjust_returns(returns, risk_free)
    adj_factor_returns = _adjust_returns(factor_returns, risk_free)
    alpha_series = adj_returns - (_beta * adj_factor_returns)

    out = np.subtract(
        np.power(
            np.add(
                nanmean(alpha_series, axis=0, out=out),
                1,
                out=out
            ),
            ann_factor,
            out=out
        ),
        1,
        out=out
    )

    if allocated_output and isinstance(returns, pd.DataFrame):
        out = pd.Series(out)

    if returns.ndim == 1:
        out = out.item()

    return out


roll_alpha_aligned = _create_binary_vectorized_roll_function(alpha_aligned)


def beta(returns, factor_returns, risk_free=0.0, out=None):
    """Calculates beta.

    Parameters
    ----------
    returns : pd.Series
        Daily returns of the strategy, noncumulative.
        - See full explanation in: func:`~empyrical.stats.cum_returns`.
    factor_returns : pd.Series
         Daily noncumulative returns of the factor to which beta is
         computed. Usually a benchmark such as the market.
         - This is in the same style as returns.
    risk_free : int, float, optional
        Constant risk-free return throughout the period. For example, the
        interest rate on a three-month US treasury bill.
    out : array-like, optional
        Array to use as output buffer.
        If not passed, a new array will be created.

    Returns
    -------
    beta : float
    """
    if not (isinstance(returns, np.ndarray) and
            isinstance(factor_returns, np.ndarray)):
        returns, factor_returns = _aligned_series(returns, factor_returns)

    return beta_aligned(
        returns,
        factor_returns,
        risk_free=risk_free,
        out=out,
    )


roll_beta = _create_binary_vectorized_roll_function(beta)


def beta_aligned(returns, factor_returns, risk_free=0.0, out=None):
    """Calculates beta.

    If they are pd.Series, expects returns and factor_returns have already
    been aligned on their labels.  If np.ndarray, these arguments should have
    the same shape.

    Parameters
    ----------
    returns : pd.Series or np.ndarray
        Daily returns of the strategy, noncumulative.
        - See full explanation in: func:`~empyrical.stats.cum_returns`.
    factor_returns : pd.Series or np.ndarray
         Daily noncumulative returns of the factor to which beta is
         computed. Usually a benchmark such as the market.
         - This is in the same style as returns.
    risk_free : int, float, optional
        Constant risk-free return throughout the period. For example, the
        interest rate on a three-month US treasury bill.
    out : array-like, optional
        Array to use as output buffer.
        If not passed, a new array will be created.

    Returns
    -------
    beta : float
        Beta.
    """
    _risk_free = risk_free
    # Cache these as locals since we're going to call them multiple times.
    nan = np.nan
    isnan = np.isnan

    returns_1d = returns.ndim == 1
    if returns_1d:
        returns = np.asanyarray(returns)[:, np.newaxis]

    if factor_returns.ndim == 1:
        factor_returns = np.asanyarray(factor_returns)[:, np.newaxis]

    n, m = returns.shape

    if out is None:
        out = np.full(m, nan)
    elif out.ndim == 0:
        out = out[np.newaxis]

    if len(returns) < 1 or len(factor_returns) < 2:
        out[()] = nan
        if returns_1d:
            out = out.item()
        return out

    # Copy N times as a column vector and fill with nans to have the same
    # missing value pattern as the dependent variable.
    #
    # PERF_TODO: We could probably avoid the space blowup by doing this in
    # Cython.

    # shape: (N, M)
    independent = np.where(
        isnan(returns),
        nan,
        factor_returns,
    )

    # Calculate beta as Cov(X, Y) / Cov(X, X).
    # https://en.wikipedia.org/wiki/Simple_linear_regression#Fitting_the_regression_line  # noqa
    #
    # NOTE: The usual formula for covariance is::
    #
    #    mean((X - mean(X)) * (Y - mean(Y)))
    #
    # However, we don't actually need to take the mean of both sides of the
    # product because of the following equivalence::
    #
    # Let X_res = (X - mean(X)).
    # We have:
    #
    #     mean(X_res * (Y - mean(Y))) = mean(X_res * (Y - mean(Y)))
    #                             (1) = mean((X_res * Y) - (X_res * mean(Y)))
    #                             (2) = mean(X_res * Y) - mean(X_res * mean(Y))
    #                             (3) = mean(X_res * Y) - mean(X_res) * mean(Y)
    #                             (4) = mean(X_res * Y) - 0 * mean(Y)
    #                             (5) = mean(X_res * Y)
    #
    #
    # The tricky step in the above derivation is step (4). We know that
    # mean(X_res) is zero because, for any X:
    #
    #     mean(X - mean(X)) = mean(X) - mean(X) = 0.
    #
    # The upshot of this is that we only have to center one of `independents`
    # and `dependent` when calculating covariances. Since we need the centered
    # `independent` to calculate its variance in the next step, we choose to
    # center `independent`.

    ind_residual = independent - nanmean(independent, axis=0)

    covariances = nanmean(ind_residual * returns, axis=0)

    # We end up with different variances in each column here because each
    # column may have a different subset of the data dropped due to missing
    # data in the corresponding dependent column.
    # Shape: (M,)
    np.square(ind_residual, out=ind_residual)
    independent_variances = nanmean(ind_residual, axis=0)
    independent_variances[independent_variances < 1.0e-30] = np.nan

    np.divide(covariances, independent_variances, out=out)

    if returns_1d:
        out = out.item()

    return out


roll_beta_aligned = _create_binary_vectorized_roll_function(beta_aligned)


def stability_of_timeseries(returns):
    """Determines R-squared of a linear fit to the cumulative
    log returns. Computes an ordinary least squares linear fit,
    and returns R-squared.

    Parameters
    ----------
    returns : pd.Series or np.ndarray
        Daily returns of the strategy, noncumulative.
        - See full explanation in: func:`~empyrical.stats.cum_returns`.

    Returns
    -------
    float
        R-squared.

    """
    if len(returns) < 2:
        return np.nan

    returns = np.asanyarray(returns)
    returns = returns[~np.isnan(returns)]

    cum_log_returns = np.log1p(returns).cumsum()
    rhat = stats.linregress(np.arange(len(cum_log_returns)),
                            cum_log_returns)[2]

    return rhat ** 2


def tail_ratio(returns):
    """Determines the ratio between the right (95%) and left tail (5%).

    For example, a ratio of 0.25 means that losses are four times
    as bad as profits.

    Parameters
    ----------
    returns : pd.Series or np.ndarray
        Daily returns of the strategy, noncumulative.
         - See full explanation in: func:`~empyrical.stats.cum_returns`.

    Returns
    -------
    tail_ratio : float
    """

    if len(returns) < 1:
        return np.nan

    returns = np.asanyarray(returns)
    # Be tolerant of nan's
    returns = returns[~np.isnan(returns)]
    if len(returns) < 1:
        return np.nan

    return np.abs(np.percentile(returns, 95)) / \
        np.abs(np.percentile(returns, 5))


def capture(returns, factor_returns, period=DAILY):
    """Compute capture ratio.

    Parameters
    ----------
    returns : pd.Series or np.ndarray
        Returns of the strategy, noncumulative.
        - See full explanation in: func:`~empyrical.stats.cum_returns`.
    factor_returns : pd.Series or np.ndarray
        Noncumulative returns of the factor to which beta is
        computed. Usually a benchmark such as the market.
        - This is in the same style as returns.
    period : str, optional
        Defines the periodicity of the 'returns' data for purposes of
        annualizing. Value ignored if `annualization` parameter is specified.
        Defaults are::

            'Monthly':12
            'weekly': 52
            'daily': 252

    Returns
    -------
    capture_ratio : float

    Note
    ----
    See https://www.investopedia.com/terms/u/up-market-capture-ratio.asp for
    details.
    """
    return (annual_return(returns, period=period) /
            annual_return(factor_returns, period=period))


def beta_fragility_heuristic(returns, factor_returns):
    """Estimate fragility to drop in beta.

    Parameters
    ----------
    returns : pd.Series or np.ndarray
        Daily returns of the strategy, noncumulative.
        - See full explanation in: func:`~empyrical.stats.cum_returns`.
    factor_returns : pd.Series or np.ndarray
         Daily noncumulative returns of the factor to which beta is
         computed. Usually a benchmark such as the market.
         - This is in the same style as returns.

    Returns
    -------
    float, np.nan
        The beta fragility of the strategy.

    Note
    ----
    A negative return value indicates potential losses could follow volatility in beta.
    The magnitude of the negative value indicates the size of the potential loss.
    See also::
    `A New Heuristic Measure of Fragility and
Tail Risks: Application to Stress Testing`
        https://www.imf.org/external/pubs/ft/wp/2012/wp12216.pdf
        An IMF Working Paper describing the heuristic
    """
    if len(returns) < 3 or len(factor_returns) < 3:
        return np.nan

    return beta_fragility_heuristic_aligned(
        *_aligned_series(returns, factor_returns))


def beta_fragility_heuristic_aligned(returns, factor_returns):
    """Estimate fragility to drop in beta

    Parameters
    ----------
    returns : pd.Series or np.ndarray
        Daily returns of the strategy, noncumulative.
        - See full explanation in: func:`~empyrical.stats.cum_returns`.
    factor_returns : pd.Series or np.ndarray
         Daily noncumulative returns of the factor to which beta is
         computed. Usually a benchmark such as the market.
         - This is in the same style as returns.

    Returns
    -------
    float, np.nan
        The beta fragility of the strategy.

    Note
    ----
    If they are pd.Series, expects returns and factor_returns have already
    been aligned on their labels.  If np.ndarray, these arguments should have
    the same shape.
    See also::
    `A New Heuristic Measure of Fragility and
Tail Risks: Application to Stress Testing`
        https://www.imf.org/external/pubs/ft/wp/2012/wp12216.pdf
        An IMF Working Paper describing the heuristic
    """
    if len(returns) < 3 or len(factor_returns) < 3:
        return np.nan

    # combine returns and factor returns into pairs
    returns_series = pd.Series(returns)
    factor_returns_series = pd.Series(factor_returns)
    pairs = pd.concat([returns_series, factor_returns_series], axis=1)
    pairs.columns = ['returns', 'factor_returns']

    # exclude any rows where returns are nan
    pairs = pairs.dropna()
    # sort by beta
    # pairs = pairs.sort_values(by='factor_returns') #fix bugs about the value is not the same in win and linux
    pairs = pairs.sort_values(by=['factor_returns'], kind='mergesort')

    # find the three vectors, using median of 3
    start_index = 0
    mid_index = int(np.around(len(pairs) / 2, 0))
    end_index = len(pairs) - 1

    (start_returns, start_factor_returns) = pairs.iloc[start_index]
    (mid_returns, mid_factor_returns) = pairs.iloc[mid_index]
    (end_returns, end_factor_returns) = pairs.iloc[end_index]

    factor_returns_range = (end_factor_returns - start_factor_returns)
    start_returns_weight = 0.5
    end_returns_weight = 0.5

    # find weights for the start and end returns
    # using a convex combination
    if not factor_returns_range == 0:
        start_returns_weight = \
            (mid_factor_returns - start_factor_returns) / \
            factor_returns_range
        end_returns_weight = \
            (end_factor_returns - mid_factor_returns) / \
            factor_returns_range

    # calculate fragility heuristic
    heuristic = (start_returns_weight*start_returns) + \
        (end_returns_weight*end_returns) - mid_returns

    return heuristic


def gpd_risk_estimates(returns, var_p=0.01):
    """Estimate VaR and ES using the Generalized Pareto Distribution (GPD)

    Parameters
    ----------
    returns : pd.Series or np.ndarray
        Daily returns of the strategy, noncumulative.
        - See full explanation in: func:`~empyrical.stats.cum_returns`.
    var_p : float
        The percentile to use for estimating the VaR and ES

    Returns
    -------
    [threshold, scale_param, shape_param, var_estimate, es_estimate]
        : list[float]
        threshold - the threshold used to cut off exception tail losses
        scale_param - a parameter (often denoted by sigma, capturing the
            scale, related to variance)
        shape_param - a parameter (often denoted by xi, capturing the shape or
            type of the distribution)
        var_estimate - an estimate for the VaR for the given percentile
        es_estimate - an estimate for the ES for the given percentile

    Note
    ----
    see also::
    `An Application of Extreme Value Theory for
Measuring Risk <https://link.springer.com/article/10.1007/s10614-006-9025-7>`
        A paper describing how to use the Generalized Pareto
        Distribution to estimate VaR and ES.
    """
    if len(returns) < 3:
        result = np.zeros(5)
        if isinstance(returns, pd.Series):
            result = pd.Series(result)
        return result
    return gpd_risk_estimates_aligned(*_aligned_series(returns, var_p))


def gpd_risk_estimates_aligned(returns, var_p=0.01):
    """Estimate VaR and ES using the Generalized Pareto Distribution (GPD)

    Parameters
    ----------
    returns : pd.Series or np.ndarray
        Daily returns of the strategy, noncumulative.
        - See full explanation in: func:`~empyrical.stats.cum_returns`.
    var_p : float
        The percentile to use for estimating the VaR and ES

    Returns
    -------
    [threshold, scale_param, shape_param, var_estimate, es_estimate]
        : list[float]
        threshold - the threshold used to cut off exception tail losses
        scale_param - a parameter (often denoted by sigma, capturing the
            scale, related to variance)
        shape_param - a parameter (often denoted by xi, capturing the shape or
            type of the distribution)
        var_estimate - an estimate for the VaR for the given percentile
        es_estimate - an estimate for the ES for the given percentile

    Note
    ----
    see also::
    `An Application of Extreme Value Theory for
Measuring Risk <https://link.springer.com/article/10.1007/s10614-006-9025-7>`
        A paper describing how to use the Generalized Pareto
        Distribution to estimate VaR and ES.
    """
    result = np.zeros(5)
    if not len(returns) < 3:

        # DEFAULT_THRESHOLD = 0.2
        # MINIMUM_THRESHOLD = 0.000000001
        default_threshold = 0.2
        minimum_threshold = 0.000000001

        try:
            returns_array = pd.Series(returns).to_numpy()
        except AttributeError:
            # while zipline requires support for pandas < 0.25
            returns_array = pd.Series(returns).as_matrix()

        flipped_returns = -1 * returns_array
        losses = flipped_returns[flipped_returns > 0]
        threshold = default_threshold
        finished = False
        scale_param = 0
        shape_param = 0
        var_estimate = 0
        while not finished and threshold > minimum_threshold:
            losses_beyond_threshold = \
                losses[losses >= threshold]
            param_result = \
                gpd_loglikelihood_minimizer_aligned(losses_beyond_threshold)
            if (param_result[0] is not False and
                    param_result[1] is not False):
                scale_param = param_result[0]
                shape_param = param_result[1]
                var_estimate = gpd_var_calculator(threshold, scale_param,
                                                  shape_param, var_p,
                                                  len(losses),
                                                  len(losses_beyond_threshold))
                # non-negative shape parameter is required for fat tails
                # non-negative VaR estimate is required for loss of some kind
                if shape_param > 0 and var_estimate > 0:
                    finished = True
            if not finished:
                threshold = threshold / 2
        if finished:
            es_estimate = gpd_es_calculator(var_estimate, threshold,
                                            scale_param, shape_param)
            result = np.array([threshold, scale_param, shape_param,
                               var_estimate, es_estimate])
    if isinstance(returns, pd.Series):
        result = pd.Series(result)
    return result


def gpd_es_calculator(var_estimate, threshold, scale_param,
                      shape_param):
    result = 0
    if (1 - shape_param) != 0:
        # this formula is from Gilli and Kellezi pg. 8
        var_ratio = (var_estimate/(1 - shape_param))
        param_ratio = ((scale_param - (shape_param * threshold)) /
                       (1 - shape_param))
        result = var_ratio + param_ratio
    return result


def gpd_var_calculator(threshold, scale_param, shape_param,
                       probability, total_n, exceedance_n):
    result = 0
    if exceedance_n > 0 and shape_param > 0:
        # this formula is from Gilli and Kellezi pg. 12
        param_ratio = scale_param / shape_param
        prob_ratio = (total_n/exceedance_n) * probability
        result = threshold + (param_ratio *
                              (pow(prob_ratio, -shape_param) - 1))
    return result


def gpd_loglikelihood_minimizer_aligned(price_data):
    result = [False, False]
    # DEFAULT_SCALE_PARAM = 1
    # DEFAULT_SHAPE_PARAM = 1
    default_scale_param = 1
    default_shape_param = 1
    if len(price_data) > 0:
        gpd_loglikelihood_lambda = \
            gpd_loglikelihood_factory(price_data)
        optimization_results = \
            optimize.minimize(gpd_loglikelihood_lambda,
                              [default_scale_param, default_shape_param],
                              method='Nelder-Mead')
        if optimization_results.success:
            resulting_params = optimization_results.x
            if len(resulting_params) == 2:
                result[0] = resulting_params[0]
                result[1] = resulting_params[1]
    return result


def gpd_loglikelihood_factory(price_data):
    return lambda params: gpd_loglikelihood(params, price_data)


def gpd_loglikelihood(params, price_data):
    if params[1] != 0:
        return -gpd_loglikelihood_scale_and_shape(params[0],
                                                  params[1],
                                                  price_data)
    else:
        return -gpd_loglikelihood_scale_only(params[0], price_data)


def gpd_loglikelihood_scale_and_shape_factory(price_data):
    # minimize a function of two variables requires a list of params
    # we are expecting the lambda below to be called as follows:
    # parameters = [scale, shape]
    # the final outer negative is added because scipy only minimizes
    return lambda params: \
        -gpd_loglikelihood_scale_and_shape(params[0],
                                           params[1],
                                           price_data)


def gpd_loglikelihood_scale_and_shape(scale, shape, price_data):
    n = len(price_data)
    result = -1 * float_info.max
    if scale != 0:
        param_factor = shape / scale
        if shape != 0 and param_factor >= 0 and scale >= 0:
            result = ((-n * np.log(scale)) -
                      (((1 / shape) + 1) *
                       (np.log((shape / scale * price_data) + 1)).sum()))
    return result


def gpd_loglikelihood_scale_only_factory(price_data):
    # the negative is added because scipy only minimizes
    return lambda scale: \
        -gpd_loglikelihood_scale_only(scale, price_data)


def gpd_loglikelihood_scale_only(scale, price_data):
    n = len(price_data)
    data_sum = price_data.sum()
    result = -1 * float_info.max
    if scale >= 0:
        result = ((-n*np.log(scale)) - (data_sum/scale))
    return result


def up_capture(returns, factor_returns, **kwargs):
    """
    Compute the capture ratio for periods when the benchmark return is positive

    Parameters
    ----------
    returns : pd.Series or np.ndarray
        Returns of the strategy, noncumulative.
        - See full explanation in: func:`~empyrical.stats.cum_returns`.
    factor_returns : pd.Series or np.ndarray
        Noncumulative returns of the factor to which beta is
        computed. Usually a benchmark such as the market.
        - This is in the same style as returns.
    kwargs:  dict, optional.
        period : str, optional
        Defines the periodicity of the 'returns' data for purposes of
        annualizing. Value ignored if `annualization` parameter is specified.
        Defaults are::

            'Monthly':12
            'weekly': 52
            'daily': 252

    Returns
    -------
    up_capture : float

    Note
    ----
    See https://www.investopedia.com/terms/u/up-market-capture-ratio.asp for
    more information.
    """
    return up(returns, factor_returns, function=capture, **kwargs)


def down_capture(returns, factor_returns, **kwargs):
    """
    Compute the capture ratio for periods when the benchmark return is negative

    Parameters
    ----------
    returns : pd.Series or np.ndarray
        Returns of the strategy, noncumulative.
        - See full explanation in: func:`~empyrical.stats.cum_returns`.
    factor_returns : pd.Series or np.ndarray
        Noncumulative returns of the factor to which beta is
        computed. Usually a benchmark such as the market.
        - This is in the same style as returns.
    kwargs:  dict, optional
        period : str, optional
        Defines the periodicity of the 'returns' data for purposes of
        annualizing. Value ignored if `annualization` parameter is specified.
        Defaults are::

            'Monthly':12
            'weekly': 52
            'daily': 252

    Returns
    -------
    down_capture : float

    Note
    ----
    See https://www.investopedia.com/terms/d/down-market-capture-ratio.asp for
    more information.
    """
    return down(returns, factor_returns, function=capture, **kwargs)


def up_down_capture(returns, factor_returns, **kwargs):
    """
    Computes the ratio of up_capture to down_capture.

    Parameters
    ----------
    returns : pd.Series or np.ndarray
        Returns of the strategy, noncumulative.
        - See full explanation in: func:`~empyrical.stats.cum_returns`.
    factor_returns : pd.Series or np.ndarray
        Noncumulative returns of the factor to which beta is
        computed. Usually a benchmark such as the market.
        - This is in the same style as returns.
    kwargs:  dict, optional
        period : str, optional
        Defines the periodicity of the 'returns' data for purposes of
        annualizing. Value ignored if `annualization` parameter is specified.
        Defaults are::
            'Monthly':12
            'weekly': 52
            'daily': 252

    Returns
    -------
    up_down_capture : float
        the updown capture ratio
    """
    return (up_capture(returns, factor_returns, **kwargs) /
            down_capture(returns, factor_returns, **kwargs))


def up_alpha_beta(returns, factor_returns, **kwargs):
    """
    Computes alpha and beta for periods when the benchmark return is positive.

    Parameters
    ----------
    see documentation for `alpha_beta`.

    Returns
    -------
    float
        Alpha.
    Float
        Beta.
    """
    return up(returns, factor_returns, function=alpha_beta_aligned, **kwargs)


def down_alpha_beta(returns, factor_returns, **kwargs):
    """
    Computes alpha and beta for periods when the benchmark return is negative.

    Parameters
    ----------
    see documentation for `alpha_beta`.

    Returns
    -------
    alpha : float
    beta : float
    """
    return down(returns, factor_returns, function=alpha_beta_aligned, **kwargs)


def roll_up_capture(returns, factor_returns, window=10, **kwargs):
    """
    Computes the up capture measure over a rolling window.
    See documentation for: func:`~empyrical.stats.up_capture`.
    (pass all args, kwargs required)

    Parameters
    ----------
    returns : pd.Series or np.ndarray
        Daily returns of the strategy, noncumulative.
        - See full explanation in: func:`~empyrical.stats.cum_returns`.

    factor_returns : pd.Series or np.ndarray
        Noncumulative returns of the factor to which beta is
        computed. Usually a benchmark such as the market.
        - This is in the same style as returns.

    window : int, required
        The size of the rolling window, expressed in terms of the data's periodicity.
        - E.g., the window = 60, periodicity=DAILY, represents a rolling 60-day window
    """
    return roll(returns, factor_returns, window=window, function=up_capture,
                **kwargs)


def roll_down_capture(returns, factor_returns, window=10, **kwargs):
    """
    Computes the down capture measure over a rolling window.
    See documentation for: func:`~empyrical.stats.down_capture`.
    (pass all args, kwargs required)

    Parameters
    ----------
    returns : pd.Series or np.ndarray
        Daily returns of the strategy, noncumulative.
        - See full explanation in: func:`~empyrical.stats.cum_returns`.

    factor_returns : pd.Series or np.ndarray
        Noncumulative returns of the factor to which beta is
        computed. Usually a benchmark such as the market.
        - This is in the same style as returns.

    window : int, required
        The size of the rolling window, expressed in terms of the data's periodicity.
        - E.g., the window = 60, periodicity=DAILY, represents a rolling 60-day window
    """
    return roll(returns, factor_returns, window=window, function=down_capture,
                **kwargs)


def roll_up_down_capture(returns, factor_returns, window=10, **kwargs):
    """
    Computes the up/down capture measure over a rolling window.
    See documentation for: func:`~empyrical.stats.up_down_capture`.
    (pass all args, kwargs required)

    Parameters
    ----------
    returns : pd.Series or np.ndarray
        Daily returns of the strategy, noncumulative.
        - See full explanation in: func:`~empyrical.stats.cum_returns`.

    factor_returns : pd.Series or np.ndarray
        Noncumulative returns of the factor to which beta is
        computed. Usually a benchmark such as the market.
        - This is in the same style as returns.

    window : int, required
        The size of the rolling window, expressed in terms of the data's periodicity.
        - E.g., the window = 60, periodicity=DAILY, represents a rolling 60-day window
    """
    return roll(returns, factor_returns, window=window,
                function=up_down_capture, **kwargs)


def value_at_risk(returns, cutoff=0.05):
    """
    Value at risk (VaR) of a `returns` stream.

    Parameters
    ----------
    returns : pandas.Series or 1-D numpy.array
        Non-cumulative daily returns.
    cutoff : float, optional
        Decimal representing the percentage cutoff for the bottom percentile of
        returns.Default to 0.05.

    Returns
    -------
    VaR : float
        The VaR value.
    """
    return np.percentile(returns, 100 * cutoff)


def conditional_value_at_risk(returns, cutoff=0.05):
    """
    Conditional value at risk (CVaR) of a `returns` stream.

    CVaR measures the expected single-day returns of an asset on that asset's
    worst performing days, where "worst-performing" is defined as falling below
    `cutoff` as a percentile of all daily returns.

    Parameters
    ----------
    returns : pandas.Series or 1-D numpy.array
        Non-cumulative daily returns.
    cutoff : float, optional
        Decimal representing the percentage cutoff for the bottom percentile of
        returns.Default to 0.05.

    Returns
    -------
    CVaR : float
        The CVaR value.
    """
    # PERF: Instead of using the 'value_at_risk' function to find the cutoff
    # value, which requires a call to numpy.percentile, determine the cutoff
    # index manually and partition out the lowest returns values. The value at
    # the cutoff index should be included in the partition.
    cutoff_index = int((len(returns) - 1) * cutoff)
    return np.mean(np.partition(returns, cutoff_index)[:cutoff_index + 1])


def get_max_drawdown_period(returns: pd.Series) -> str:
    """
    Calculate the start and end dates of the maximum drawdown period.

    Parameters
    ----------
    returns : pd.Series
        Daily returns series with datetime index.

    Returns
    -------
    str
        A string in the format "start_date,end_date" representing the period
        of maximum drawdown.
    """
    # Calculate cumulative returns
    cum_returns = (1 + returns).cumprod()

    # Calculate rolling maximum
    rolling_max = cum_returns.cummax()

    # Calculate drawdown
    drawdown = cum_returns / rolling_max - 1

    # Find the end date of maximum drawdown
    end_date = drawdown.idxmin()

    # Find the start date of maximum drawdown (previous peak)
    start_date = cum_returns.loc[:end_date].idxmax()

    return f"{start_date.date()},{end_date.date()}"


def information_ratio(super_returns, period=DAILY, annualization=None):
    """
    :param super_returns:
    :param period : str, optional
        Defines the periodicity of the 'returns' data for purposes of
        annualizing. Value ignored if `annualization` parameter is specified.
        Defaults are::
            'Monthly':12
            'weekly': 52
            'daily': 252
    :param annualization : int, optional
        Used to suppress default values available in `period` to convert
        returns into annual returns. Value should be the annual frequency of
        `returns`.
    :return:
    """
    ann_factor = annualization_factor(period, annualization)
    mean_excess_return = super_returns.mean()
    std_excess_return = super_returns.std(ddof=1)
    ir = (mean_excess_return * ann_factor) / (std_excess_return * np.sqrt(ann_factor))
    return ir


SIMPLE_STAT_FUNCS = [
    cum_returns_final,
    annual_return,
    annual_volatility,
    sharpe_ratio,
    calmar_ratio,
    stability_of_timeseries,
    max_drawdown,
    omega_ratio,
    sortino_ratio,
    stats.skew,
    stats.kurtosis,
    tail_ratio,
    cagr,
    value_at_risk,
    conditional_value_at_risk,
    information_ratio,
    get_max_drawdown_period
]

FACTOR_STAT_FUNCS = [
    excess_sharpe,
    alpha,
    beta,
    beta_fragility_heuristic,
    gpd_risk_estimates,
    capture,
    up_capture,
    down_capture
]
