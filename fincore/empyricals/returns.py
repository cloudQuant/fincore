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

"""收益计算函数模块."""

import math
import numpy as np
import pandas as pd
from fincore.constants import DAILY, WEEKLY, MONTHLY, QUARTERLY, YEARLY

__all__ = [
    'simple_returns',
    'cum_returns',
    'cum_returns_final',
    'aggregate_returns',
    'normalize',
]


def simple_returns(prices):
    """Compute simple returns from a time series of prices.

    Parameters
    ----------
    prices : array-like or pd.Series or pd.DataFrame
        Time series of prices.

    Returns
    -------
    array-like or pd.Series or pd.DataFrame
        Simple returns computed as ``(price[t] - price[t-1])`` divided by
        ``price[t-1]``. The first observation is dropped. For pandas
        inputs, the index is preserved (excluding the first element).
    """
    if isinstance(prices, (pd.DataFrame, pd.Series)):
        out = prices.pct_change().iloc[1:]
    else:
        # Assume np.ndarray
        out = np.diff(prices, axis=0)
        # Avoid division by zero warning
        with np.errstate(divide="ignore", invalid="ignore"):
            np.divide(out, prices[:-1], out=out)

    return out


def cum_returns(returns, starting_value=0, out=None):
    """Compute cumulative returns from simple returns.

    This compounds the input returns and optionally scales by an initial
    ``starting_value``.

    Parameters
    ----------
    returns : array-like or pd.Series or pd.DataFrame
        Non-cumulative simple returns.
    starting_value : float, optional
        Initial portfolio value. If ``0`` (default), the result is
        returned as a pure cumulative return. Otherwise, the result is
        scaled so that ``starting_value * cumprod(1 + returns)`` is
        returned.
    out : np.ndarray, optional
        pre-allocated output array. If given, the result is
        written in-place into this array.

    Returns
    -------
    array-like or pd.Series or pd.DataFrame
        Cumulative returns. The type mirrors the input type.
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
                out,
                index=returns.index,
                columns=returns.columns,
            )

    return out


def cum_returns_final(returns, starting_value=0):
    """Compute total cumulative return from a series of simple returns.

    This computes the cumulative return by compounding simple period
    returns and optionally scaling by an initial ``starting_value``.

    Parameters
    ----------
    returns : array-like or pd.Series or pd.DataFrame
        Non-cumulative simple returns. For a DataFrame, the computation is
        performed column-wise.
    starting_value : float, optional
        Initial portfolio value. If ``0`` (default), the result is
        returned as a pure return (ending_value - 1). Otherwise, the result
        is scaled so that ``starting_value * cumprod(1 + returns)`` is
        returned.

    Returns
    -------
    float or np.ndarray or pd.Series or pd.DataFrame
        Final cumulative return (or value, depending on
        ``starting_value``). The type mirrors the input type.
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


def aggregate_returns(returns, convert_to="monthly"):
    """Aggregate returns at a weekly/monthly/quarterly/yearly frequency.

    The function groups the input return series by calendar period and
    compounds the returns within each group using
    :func:`cum_returns`.

    Parameters
    ----------
    returns : pd.Series
        Non-cumulative returns with a ``DatetimeIndex``.
    convert_to : {"weekly", "monthly", "quarterly", "yearly"}, optional
        Target aggregation frequency. Case-insensitive. Default is
        ``"monthly"``.

    Returns
    -------
    pd.Series
        Aggregated returns at the requested frequency, indexed by
        (year, period) where period is week number, month, quarter, or
        year depending on ``convert_to``.
    """

    def cumulate_returns(ret):
        return cum_returns(ret).iloc[-1]

    if convert_to == WEEKLY:
        grouping = [lambda dt: dt.year, lambda dt: dt.isocalendar()[1]]
    elif convert_to == MONTHLY:
        grouping = [lambda dt: dt.year, lambda dt: dt.month]
    elif convert_to == QUARTERLY:
        grouping = [lambda dt: dt.year, lambda dt: int(
            math.ceil(dt.month / 3.0))]
    elif convert_to == YEARLY:
        grouping = [lambda dt: dt.year]
    else:
        raise ValueError(
            "convert_to must be {}, {} or {}".format(
                WEEKLY, MONTHLY, YEARLY)
        )

    return returns.groupby(grouping).apply(cumulate_returns)


def normalize(returns, starting_value=1):
    """Normalize cumulative returns to start at a given value.

    Parameters
    ----------
    returns : pd.Series or np.ndarray
        Non-cumulative returns.
    starting_value : float, optional
        The value at which to start the normalized series. Default is 1.

    Returns
    -------
    pd.Series or np.ndarray
        Cumulative returns normalized to start at ``starting_value``.
    """
    return cum_returns(returns, starting_value=starting_value)
