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

"""收益计算函数模块.

本模块包含用于计算收益率相关指标的核心函数，包括：
- 简单收益率计算
- 累积收益率计算
- 收益率聚合
- 收益率标准化
"""

from __future__ import annotations

import math
from typing import Optional, Union

import numpy as np
import pandas as pd

from fincore.constants import DAILY, MONTHLY, QUARTERLY, WEEKLY, YEARLY

__all__ = [
    "simple_returns",
    "cum_returns",
    "cum_returns_final",
    "aggregate_returns",
    "normalize",
]


def _get_annual_return():
    """延迟导入以避免循环依赖.

    Returns
    -------
    function
        annual_return 函数.
    """
    from fincore.metrics.yearly import annual_return

    return annual_return


# Re-export annual_return for backwards compatibility
def annual_return(*args, **kwargs):
    """计算年化收益率(CAGR)的包装函数.

    此函数为向后兼容而保留，实际实现委托给 yearly 模块.

    Parameters
    ----------
    *args : tuple
        传递给 yearly.annual_return 的位置参数.
    **kwargs : dict
        传递给 yearly.annual_return 的关键字参数.

    Returns
    -------
    float
        年化收益率.

    See Also
    --------
    fincore.metrics.yearly.annual_return : 实际实现函数.
    """
    return _get_annual_return()(*args, **kwargs)


def simple_returns(
    prices: pd.Series | pd.DataFrame | np.ndarray,
) -> pd.Series | pd.DataFrame | np.ndarray:
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


def cum_returns(
    returns: pd.Series | pd.DataFrame | np.ndarray,
    starting_value: float = 0,
    out: np.ndarray | None = None,
) -> pd.Series | pd.DataFrame | np.ndarray:
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


def cum_returns_final(
    returns: pd.Series | pd.DataFrame | np.ndarray,
    starting_value: float = 0,
) -> float | np.ndarray | pd.Series:
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


def aggregate_returns(
    returns: pd.Series,
    convert_to: str = "monthly",
) -> pd.Series:
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

    if not isinstance(returns.index, pd.DatetimeIndex):
        raise ValueError(
            f"aggregate_returns requires returns with a DatetimeIndex, got {type(returns.index).__name__} instead."
        )

    if convert_to == WEEKLY:
        grouping = [lambda dt: dt.year, lambda dt: dt.isocalendar()[1]]
    elif convert_to == MONTHLY:
        grouping = [lambda dt: dt.year, lambda dt: dt.month]
    elif convert_to == QUARTERLY:
        grouping = [lambda dt: dt.year, lambda dt: int(math.ceil(dt.month / 3.0))]
    elif convert_to == YEARLY:
        grouping = [lambda dt: dt.year]
    else:
        raise ValueError(f"convert_to must be {WEEKLY}, {MONTHLY}, {QUARTERLY} or {YEARLY}")

    return returns.groupby(grouping).apply(cumulate_returns)


def normalize(
    returns: pd.Series,
    starting_value: float = 1,
) -> pd.Series:
    """将收益时间序列标准化为从指定值开始.

    此函数将收益序列按其第一个值进行缩放，使序列从
    ``starting_value`` 开始。常用于将不同策略的收益
    曲线对齐到同一起点进行比较。

    Parameters
    ----------
    returns : pd.Series
        收益时间序列（通常为累积收益或价格序列）.
    starting_value : float, optional
        标准化后的起始值，默认为 1.

    Returns
    -------
    pd.Series
        标准化后的收益序列，第一个值等于 ``starting_value``.

    Examples
    --------
    >>> import pandas as pd
    >>> returns = pd.Series([100, 110, 105, 115])
    >>> normalize(returns, starting_value=1)
    0    1.00
    1    1.10
    2    1.05
    3    1.15
    dtype: float64
    """
    if len(returns) < 1:
        return returns.copy()

    first_value = returns.iloc[0]
    if first_value == 0:
        import warnings

        warnings.warn("First value of returns is 0, normalization will produce inf/nan values.")
        return returns * np.nan
    return starting_value * (returns / first_value)
