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

"""Rolling-window metrics."""

from __future__ import annotations

import numpy as np
import pandas as pd

from fincore.constants import DAILY
from fincore.contracts.time_series import AlignmentPolicy, align_binary_metric_inputs
from fincore.core.rolling_moments import roll_alpha_beta_vectorized, roll_max_drawdown_chunked
from fincore.metrics.basic import annualization_factor
from fincore.metrics.ratios import down_capture, sortino_ratio, up_capture
from fincore.metrics.risk import annual_volatility

__all__ = [
    "roll_alpha",
    "roll_alpha_aligned",
    "roll_alpha_beta",
    "roll_alpha_beta_aligned",
    "roll_annual_volatility",
    "roll_beta",
    "roll_beta_aligned",
    "roll_down_capture",
    "roll_max_drawdown",
    "roll_sharpe_ratio",
    "roll_sortino_ratio",
    "roll_up_capture",
    "roll_up_down_capture",
    "rolling_beta",
    "rolling_regression",
    "rolling_sharpe",
    "rolling_volatility",
]


def _write_rolling_out(result, out: np.ndarray | None):
    if out is None:
        return result
    out[...] = np.asarray(result)
    return out


def roll_alpha_aligned(lhs, rhs, window, out=None, **kwargs):
    """Empyrical-compatible aligned rolling alpha kernel."""

    return _write_rolling_out(roll_alpha(lhs, rhs, window, **kwargs), out)


def roll_beta_aligned(lhs, rhs, window, out=None, **kwargs):
    """Empyrical-compatible aligned rolling beta kernel."""

    return _write_rolling_out(roll_beta(lhs, rhs, window, **kwargs), out)


def roll_alpha_beta_aligned(lhs, rhs, window, out=None, **kwargs):
    """Empyrical-compatible aligned rolling alpha/beta kernel."""

    return _write_rolling_out(roll_alpha_beta(lhs, rhs, window, **kwargs), out)


def roll_annual_volatility(arr, window, out=None, **kwargs):
    """Calculate annual volatility for each complete rolling window."""

    values = np.asanyarray(arr)
    result = np.asarray(
        [annual_volatility(values[index : index + window], **kwargs) for index in range(len(values) - window + 1)]
    )
    return _write_rolling_out(result, out)


def roll_sortino_ratio(arr, window, out=None, **kwargs):
    """Calculate the Sortino ratio for each complete rolling window."""

    values = np.asanyarray(arr)
    result = np.asarray(
        [sortino_ratio(values[index : index + window], **kwargs) for index in range(len(values) - window + 1)]
    )
    return _write_rolling_out(result, out)


def roll_alpha(
    returns: pd.Series | np.ndarray,
    factor_returns: pd.Series | np.ndarray,
    window: int = 252,
    risk_free: float = 0.0,
    period: str = DAILY,
    annualization: float | None = None,
    *,
    alignment: AlignmentPolicy = "inner",
    normalize_tz: str | None = None,
) -> pd.Series | np.ndarray:
    """Calculate rolling alpha over a specified window.

    Parameters
    ----------
    returns : pd.Series or np.ndarray
        Non-cumulative returns of the strategy.
    factor_returns : pd.Series or np.ndarray
        Non-cumulative benchmark returns to calculate alpha against.
    window : int, optional
        Length of the rolling window (default 252).
    risk_free : float, optional
        Risk-free rate (default 0.0).
    period : str, optional
        Frequency of the returns (default 'daily').
    annualization : int, optional
        Factor to convert period returns to yearly returns.

    Returns
    -------
    pd.Series or np.ndarray
        Rolling alpha values.
    """
    returns_aligned, factor_aligned = align_binary_metric_inputs(
        returns, factor_returns, alignment=alignment, normalize_tz=normalize_tz
    )

    is_series = isinstance(returns_aligned, pd.Series)

    if len(returns_aligned) < window:
        if isinstance(returns_aligned, pd.Series):
            if isinstance(returns_aligned.index, pd.DatetimeIndex):
                return pd.Series([], dtype=float, index=returns_aligned.index[:0])
            return pd.Series([], dtype=float)
        return np.array([], dtype=float)

    if not is_series:
        returns_aligned = pd.Series(returns_aligned)
        factor_aligned = pd.Series(factor_aligned)

    assert isinstance(returns_aligned, pd.Series) and isinstance(factor_aligned, pd.Series)
    ann_factor = annualization_factor(period, annualization)
    alpha, _beta = roll_alpha_beta_vectorized(
        returns_aligned, factor_aligned, window, risk_free=risk_free, ann_factor=ann_factor
    )

    if is_series:
        return pd.Series(alpha, index=returns_aligned.index[window - 1 :])
    return alpha


def roll_beta(
    returns: pd.Series | np.ndarray,
    factor_returns: pd.Series | np.ndarray,
    window: int = 252,
    risk_free: float = 0.0,
    period: str = DAILY,
    annualization: float | None = None,
    *,
    alignment: AlignmentPolicy = "inner",
    normalize_tz: str | None = None,
) -> pd.Series | np.ndarray:
    """Calculate rolling beta over a specified window.

    Parameters
    ----------
    returns : pd.Series or np.ndarray
        Non-cumulative returns of the strategy.
    factor_returns : pd.Series or np.ndarray
        Non-cumulative benchmark returns to calculate beta against.
    window : int, optional
        Length of the rolling window (default 252).
    risk_free : float, optional
        Risk-free rate (default 0.0).
    period : str, optional
        Frequency of the returns (default 'daily').
    annualization : int, optional
        Factor to convert period returns to yearly returns.

    Returns
    -------
    pd.Series or np.ndarray
        Rolling beta values.
    """
    returns_aligned, factor_aligned = align_binary_metric_inputs(
        returns, factor_returns, alignment=alignment, normalize_tz=normalize_tz
    )

    is_series = isinstance(returns_aligned, pd.Series)

    if len(returns_aligned) < window:
        if isinstance(returns_aligned, pd.Series):
            if isinstance(returns_aligned.index, pd.DatetimeIndex):
                return pd.Series([], dtype=float, index=returns_aligned.index[:0])
            return pd.Series([], dtype=float)
        return np.array([], dtype=float)

    if not is_series:
        returns_aligned = pd.Series(returns_aligned)
        factor_aligned = pd.Series(factor_aligned)

    assert isinstance(returns_aligned, pd.Series) and isinstance(factor_aligned, pd.Series)
    ret_adj = returns_aligned - risk_free
    fac_adj = factor_aligned - risk_free
    rolling_cov = ret_adj.rolling(window).cov(fac_adj)
    rolling_var = fac_adj.rolling(window).var()
    with np.errstate(divide="ignore", invalid="ignore"):
        result = rolling_cov / rolling_var
    result = result.iloc[window - 1 :]

    if is_series:
        return result
    return result.values


def roll_alpha_beta(
    returns: pd.Series | np.ndarray,
    factor_returns: pd.Series | np.ndarray,
    window: int = 252,
    risk_free: float = 0.0,
    period: str = DAILY,
    annualization: float | None = None,
    *,
    alignment: AlignmentPolicy = "inner",
    normalize_tz: str | None = None,
) -> pd.DataFrame | np.ndarray:
    """Calculate rolling alpha and beta over a specified window.

    Parameters
    ----------
    returns : pd.Series or np.ndarray
        Non-cumulative returns of the strategy.
    factor_returns : pd.Series or np.ndarray
        Non-cumulative benchmark returns to calculate alpha and beta against.
    window : int, optional
        Length of the rolling window (default 252).
    risk_free : float, optional
        Risk-free rate (default 0.0).
    period : str, optional
        Frequency of the returns (default 'daily').
    annualization : int, optional
        Factor to convert period returns to yearly returns.

    Returns
    -------
    pd.DataFrame or np.ndarray
        Rolling alpha and beta values with columns ['alpha', 'beta'].
    """
    returns_aligned, factor_aligned = align_binary_metric_inputs(
        returns, factor_returns, alignment=alignment, normalize_tz=normalize_tz
    )

    is_series = isinstance(returns_aligned, pd.Series)

    if len(returns_aligned) < window:
        if isinstance(returns_aligned, pd.Series):
            if isinstance(returns_aligned.index, pd.DatetimeIndex):
                return pd.DataFrame(columns=["alpha", "beta"], index=returns_aligned.index[:0])
            return pd.DataFrame(columns=["alpha", "beta"])
        return pd.DataFrame(columns=["alpha", "beta"])

    if not is_series:
        returns_aligned = pd.Series(returns_aligned)
        factor_aligned = pd.Series(factor_aligned)

    assert isinstance(returns_aligned, pd.Series) and isinstance(factor_aligned, pd.Series)
    ann_factor = annualization_factor(period, annualization)
    out_alpha, out_beta = roll_alpha_beta_vectorized(
        returns_aligned, factor_aligned, window, risk_free=risk_free, ann_factor=ann_factor
    )

    idx = returns_aligned.index[window - 1 :]
    if is_series:
        return pd.DataFrame({"alpha": out_alpha, "beta": out_beta}, index=idx)
    return pd.DataFrame({"alpha": out_alpha, "beta": out_beta})


def roll_sharpe_ratio(
    returns: pd.Series | np.ndarray,
    window: int = 252,
    risk_free: float = 0.0,
    period: str = DAILY,
    annualization: float | None = None,
) -> pd.Series | np.ndarray:
    """Calculate rolling Sharpe ratio over a specified window.

    Parameters
    ----------
    returns : pd.Series or np.ndarray
        Non-cumulative returns of the strategy.
    window : int, optional
        Length of the rolling window (default 252).
    risk_free : float, optional
        Risk-free rate (default 0.0).
    period : str, optional
        Frequency of the returns (default 'daily').
    annualization : int, optional
        Factor to convert period returns to yearly returns.

    Returns
    -------
    pd.Series or np.ndarray
        Rolling Sharpe ratio values.
    """
    is_series = isinstance(returns, pd.Series)

    if len(returns) < window:
        if isinstance(returns, pd.Series):
            if isinstance(returns.index, pd.DatetimeIndex):
                return pd.Series([], dtype=float, index=returns.index[:0])
            return pd.Series([], dtype=float)
        return np.array([], dtype=float)

    if not is_series:
        returns = pd.Series(returns)

    assert isinstance(returns, pd.Series)
    ann_factor = annualization_factor(period, annualization)
    sqrt_ann = np.sqrt(ann_factor)

    ret_adj = returns - risk_free
    rolling_mean = ret_adj.rolling(window, min_periods=1).mean()
    rolling_std = ret_adj.rolling(window, min_periods=1).std(ddof=1)

    with np.errstate(divide="ignore", invalid="ignore"):
        result = (rolling_mean / rolling_std) * sqrt_ann

    result = result.iloc[window - 1 :]

    if is_series:
        return result
    return result.values


def roll_max_drawdown(
    returns: pd.Series | np.ndarray,
    window: int = 252,
) -> pd.Series | np.ndarray:
    """Calculate rolling maximum drawdown over a specified window.

    Parameters
    ----------
    returns : pd.Series or np.ndarray
        Non-cumulative returns of the strategy.
    window : int, optional
        Length of the rolling window (default 252).

    Returns
    -------
    pd.Series or np.ndarray
        Rolling maximum drawdown values.
    """
    if len(returns) < window:
        if isinstance(returns, pd.Series):
            if isinstance(returns.index, pd.DatetimeIndex):
                return pd.Series([], dtype=float, index=returns.index[:0])
            return pd.Series([], dtype=float)
        return np.array([], dtype=float)

    ret_arr = np.asanyarray(returns, dtype=np.float64)

    # Bounded-memory chunked kernel: peak scratch memory is
    # O(block_rows * window) instead of O(n * window), with results
    # bit-identical to the legacy full-matrix implementation.
    out = roll_max_drawdown_chunked(ret_arr, window)

    if isinstance(returns, pd.Series):
        return pd.Series(out, index=returns.index[window - 1 :])
    return out


def roll_up_capture(
    returns: pd.Series | np.ndarray,
    factor_returns: pd.Series | np.ndarray,
    window: int = 252,
    *,
    alignment: AlignmentPolicy = "inner",
    normalize_tz: str | None = None,
) -> pd.Series | np.ndarray:
    """Calculate rolling up capture over a specified window.

    Parameters
    ----------
    returns : pd.Series or np.ndarray
        Non-cumulative returns of the strategy.
    factor_returns : pd.Series or np.ndarray
        Non-cumulative benchmark returns.
    window : int, optional
        Length of the rolling window (default 252).

    Returns
    -------
    pd.Series or np.ndarray
        Rolling up capture values.
    """
    returns_aligned, factor_aligned = align_binary_metric_inputs(
        returns, factor_returns, alignment=alignment, normalize_tz=normalize_tz
    )

    is_series = isinstance(returns_aligned, pd.Series)

    if len(returns_aligned) < window:
        if isinstance(returns_aligned, pd.Series):
            if isinstance(returns_aligned.index, pd.DatetimeIndex):
                return pd.Series([], dtype=float, index=returns_aligned.index[:0])
            return pd.Series([], dtype=float)
        return np.array([], dtype=float)

    if not is_series:
        returns_aligned = pd.Series(returns_aligned)
        factor_aligned = pd.Series(factor_aligned)

    assert isinstance(returns_aligned, pd.Series) and isinstance(factor_aligned, pd.Series)
    n = len(returns_aligned) - window + 1
    out = np.empty(n, dtype=float)
    for i in range(n):
        out[i] = up_capture(returns_aligned.iloc[i : i + window], factor_aligned.iloc[i : i + window])

    if is_series:
        return pd.Series(out, index=returns_aligned.index[window - 1 :])
    return out


def roll_down_capture(
    returns: pd.Series | np.ndarray,
    factor_returns: pd.Series | np.ndarray,
    window: int = 252,
    *,
    alignment: AlignmentPolicy = "inner",
    normalize_tz: str | None = None,
) -> pd.Series | np.ndarray:
    """Calculate rolling down capture over a specified window.

    Parameters
    ----------
    returns : pd.Series or np.ndarray
        Non-cumulative returns of the strategy.
    factor_returns : pd.Series or np.ndarray
        Non-cumulative benchmark returns.
    window : int, optional
        Length of the rolling window (default 252).

    Returns
    -------
    pd.Series or np.ndarray
        Rolling down capture values.
    """
    returns_aligned, factor_aligned = align_binary_metric_inputs(
        returns, factor_returns, alignment=alignment, normalize_tz=normalize_tz
    )

    is_series = isinstance(returns_aligned, pd.Series)

    if len(returns_aligned) < window:
        if isinstance(returns_aligned, pd.Series):
            if isinstance(returns_aligned.index, pd.DatetimeIndex):
                return pd.Series([], dtype=float, index=returns_aligned.index[:0])
            return pd.Series([], dtype=float)
        return np.array([], dtype=float)

    if not is_series:
        returns_aligned = pd.Series(returns_aligned)
        factor_aligned = pd.Series(factor_aligned)

    assert isinstance(returns_aligned, pd.Series) and isinstance(factor_aligned, pd.Series)
    n = len(returns_aligned) - window + 1
    out = np.empty(n, dtype=float)
    for i in range(n):
        out[i] = down_capture(returns_aligned.iloc[i : i + window], factor_aligned.iloc[i : i + window])

    if is_series:
        return pd.Series(out, index=returns_aligned.index[window - 1 :])
    return out


def roll_up_down_capture(
    returns: pd.Series | np.ndarray,
    factor_returns: pd.Series | np.ndarray,
    window: int = 252,
    *,
    alignment: AlignmentPolicy = "inner",
    normalize_tz: str | None = None,
) -> pd.Series | np.ndarray:
    """Calculate rolling up/down capture ratio over a specified window.

    Parameters
    ----------
    returns : pd.Series or np.ndarray
        Non-cumulative returns of the strategy.
    factor_returns : pd.Series or np.ndarray
        Non-cumulative benchmark returns.
    window : int, optional
        Length of the rolling window (default 252).

    Returns
    -------
    pd.Series or np.ndarray
        Rolling up/down capture ratio values.
    """
    up_caps = roll_up_capture(
        returns,
        factor_returns,
        window,
        alignment=alignment,
        normalize_tz=normalize_tz,
    )
    down_caps = roll_down_capture(
        returns,
        factor_returns,
        window,
        alignment=alignment,
        normalize_tz=normalize_tz,
    )

    with np.errstate(divide="ignore", invalid="ignore"):
        return up_caps / down_caps


def rolling_volatility(
    returns: pd.Series,
    rolling_vol_window: int,
    period: str = DAILY,
    annualization: float | None = None,
) -> pd.Series:
    """Determine the rolling volatility of a strategy.

    Parameters
    ----------
    returns : pd.Series
        Non-cumulative returns of the strategy.
    rolling_vol_window : int
        Length of the rolling window.
    period : str, optional
        Frequency of the returns (default 'daily').
    annualization : float, optional
        Custom annualization factor.

    Returns
    -------
    pd.Series
        Rolling volatility, annualized.
    """
    ann_factor = annualization_factor(period, annualization)
    return returns.rolling(window=rolling_vol_window).std() * np.sqrt(ann_factor)


def rolling_sharpe(
    returns: pd.Series,
    rolling_sharpe_window: int,
    period: str = DAILY,
    annualization: float | None = None,
) -> pd.Series:
    """Determine the rolling Sharpe ratio of a strategy.

    Parameters
    ----------
    returns : pd.Series
        Non-cumulative returns of the strategy.
    rolling_sharpe_window : int
        Length of the rolling window.
    period : str, optional
        Frequency of the returns (default 'daily').
    annualization : float, optional
        Custom annualization factor.

    Returns
    -------
    pd.Series
        Rolling Sharpe ratio, annualized.
    """
    ann_factor = annualization_factor(period, annualization)
    rolling_mean = returns.rolling(window=rolling_sharpe_window).mean()
    rolling_std = returns.rolling(window=rolling_sharpe_window).std()

    with np.errstate(divide="ignore", invalid="ignore"):
        return rolling_mean / rolling_std * np.sqrt(ann_factor)


def rolling_beta(
    returns: pd.Series,
    factor_returns: pd.Series | pd.DataFrame,
    rolling_window: int = 126,
) -> pd.Series | pd.DataFrame:
    """Calculate rolling beta.

    Parameters
    ----------
    returns : pd.Series
        Daily returns of the strategy, noncumulative.
    factor_returns : pd.Series or pd.DataFrame
        Daily noncumulative returns of the benchmark factor.
    rolling_window : int, optional
        The size of the rolling window, in days (default 126).

    Returns
    -------
    pd.Series
        Rolling beta.
    """
    from functools import partial

    if factor_returns.ndim > 1:
        return factor_returns.apply(partial(rolling_beta, returns), rolling_window=rolling_window)
    returns_aligned, factor_aligned = returns.align(factor_returns, join="inner")
    rolling_cov = returns_aligned.rolling(rolling_window).cov(factor_aligned)
    rolling_var = factor_aligned.rolling(rolling_window).var()
    with np.errstate(divide="ignore", invalid="ignore"):
        return rolling_cov / rolling_var


def rolling_regression(
    returns: pd.Series | np.ndarray,
    factor_returns: pd.Series | np.ndarray,
    rolling_window: int = 126,
    *,
    alignment: AlignmentPolicy = "inner",
    normalize_tz: str | None = None,
) -> pd.DataFrame:
    """Calculate rolling regression alpha and beta.

    Note: The alpha returned here is the **non-annualized** (daily-frequency)
    regression intercept, unlike ``roll_alpha`` which returns annualized alpha.
    To annualize, multiply by the appropriate annualization factor
    (e.g. 252 for daily data).

    Parameters
    ----------
    returns : pd.Series
        Daily returns of the strategy, noncumulative.
    factor_returns : pd.Series
        Daily returns of the benchmark factor.
    rolling_window : int, optional
        Length of the rolling window (default 126).

    Returns
    -------
    pd.DataFrame
        Rolling alpha (non-annualized) and beta values with columns
        ['alpha', 'beta'].
    """
    returns_aligned, factor_aligned = align_binary_metric_inputs(
        returns, factor_returns, alignment=alignment, normalize_tz=normalize_tz
    )

    if len(returns_aligned) < rolling_window:
        return pd.DataFrame(columns=["alpha", "beta"])

    if not isinstance(returns_aligned, pd.Series):
        returns_aligned = pd.Series(returns_aligned)
        factor_aligned = pd.Series(factor_aligned)

    assert isinstance(returns_aligned, pd.Series) and isinstance(factor_aligned, pd.Series)
    rolling_cov = returns_aligned.rolling(rolling_window).cov(factor_aligned)
    rolling_var = factor_aligned.rolling(rolling_window).var()
    with np.errstate(divide="ignore", invalid="ignore"):
        rolling_beta_vals = rolling_cov / rolling_var
    rolling_mean_ret = returns_aligned.rolling(rolling_window).mean()
    rolling_mean_fac = factor_aligned.rolling(rolling_window).mean()
    rolling_alpha_vals = rolling_mean_ret - rolling_beta_vals * rolling_mean_fac

    result = pd.DataFrame({"alpha": rolling_alpha_vals, "beta": rolling_beta_vals})
    return result.dropna()


from fincore._dispatch import install_metric_module_surface as _install_metric_module_surface

_install_metric_module_surface(__name__)
del _install_metric_module_surface
