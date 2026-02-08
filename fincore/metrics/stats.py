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

"""统计指标函数模块."""

import numpy as np
import pandas as pd
from scipy import stats
from fincore.metrics.basic import aligned_series
from fincore.metrics.ratios import stability_of_timeseries

__all__ = [
    'skewness',
    'kurtosis',
    'hurst_exponent',
    'stutzer_index',
    'serial_correlation',
    'stock_market_correlation',
    'bond_market_correlation',
    'futures_market_correlation',
    'win_rate',
    'loss_rate',
    'r_cubed',
    'tracking_difference',
    'common_sense_ratio',
    'var_cov_var_normal',
    'normalize',
    'stability_of_timeseries',
]


def skewness(returns):
    """Calculate the skewness of a return series.

    This is a thin wrapper around :func:`scipy.stats.skew` with
    ``nan_policy="omit"``.

    Parameters
    ----------
    returns : array-like or pd.Series
        Non-cumulative returns.

    Returns
    -------
    float
        Sample skewness of the return distribution, or ``NaN`` if there
        are fewer than three observations.
    """
    if len(returns) < 3:
        return np.nan
    return stats.skew(returns, nan_policy="omit")


def kurtosis(returns):
    """Calculate the kurtosis of a return series.

    This is a thin wrapper around :func:`scipy.stats.kurtosis` with
    ``nan_policy="omit"``.

    Parameters
    ----------
    returns : array-like or pd.Series
        Non-cumulative returns.

    Returns
    -------
    float
        Sample excess kurtosis of the return distribution, or ``NaN`` if
        there are fewer than four observations.
    """
    if len(returns) < 4:
        return np.nan
    return stats.kurtosis(returns, nan_policy="omit")


def hurst_exponent(returns):
    """Estimate the Hurst exponent of a return series.

    The Hurst exponent is estimated via a rescaled range (R/S) analysis
    with safeguards for short and noisy series. Values near 0.5 indicate
    a random walk; values above 0.5 suggest persistence; values below 0.5
    suggest mean reversion.

    Parameters
    ----------
    returns : array-like or pd.Series
        Non-cumulative returns.

    Returns
    -------
    float
        Estimated Hurst exponent in the interval [0, 1], or ``NaN`` when
        there is insufficient data.
    """
    min_length = 8
    if len(returns) < min_length:
        return np.nan

    if not isinstance(returns, pd.Series):
        returns = pd.Series(returns)

    returns_array = returns.values
    returns_clean = returns_array[~np.isnan(returns_array)]

    if len(returns_clean) < min_length:
        return np.nan

    try:
        mean_return = np.mean(returns_clean)
        y_cumsum = np.cumsum(returns_clean - mean_return)
        r_range = np.max(y_cumsum) - np.min(y_cumsum)
        s_std = np.std(returns_clean, ddof=1)

        if s_std == 0 or r_range == 0:
            return np.nan

        n = len(returns_clean)
        max_lag = max(n // 3, 3)
        min_lag = 2

        lags = range(min_lag, max_lag + 1)
        rs_values = []

        for lag in lags:
            n_subseries = n // lag
            if n_subseries < 1:
                continue

            rs_list = []
            for i in range(n_subseries):
                sub_series = returns_clean[i * lag: (i + 1) * lag]
                if len(sub_series) < 2:
                    continue

                mean_sub = np.mean(sub_series)
                y_sub = np.cumsum(sub_series - mean_sub)
                r_sub = np.max(y_sub) - np.min(y_sub)
                s_sub = np.std(sub_series, ddof=1)

                if s_sub > 0 and r_sub > 0:
                    rs_list.append(r_sub / s_sub)

            if rs_list:
                rs_values.append((lag, np.mean(rs_list)))

        if len(rs_values) < 2:
            if s_std > 0 and r_range > 0:
                hurst = 0.5 + np.log(r_range / s_std) / np.log(2.0 * n)
                hurst = max(0.0, min(1.0, hurst))
                return float(hurst)
            return np.nan

        lags_array = np.array([item[0] for item in rs_values])
        rs_array = np.array([item[1] for item in rs_values])

        valid_mask = (lags_array > 0) & (rs_array > 0)
        lags_array = lags_array[valid_mask]
        rs_array = rs_array[valid_mask]

        if len(lags_array) < 2:
            return np.nan

        log_lags = np.log(lags_array)
        log_rs = np.log(rs_array)

        poly = np.polyfit(log_lags, log_rs, 1)
        hurst = poly[0]

        hurst = max(0.0, min(1.0, hurst))

        return float(hurst)

    except Exception:
        return np.nan


def stutzer_index(returns, target_return=0.0):
    """Calculate the Stutzer index.

    The Stutzer index is a downside-risk-adjusted performance measure
    based on an exponential tilting of the return distribution.

    Parameters
    ----------
    returns : array-like or pd.Series
        Non-cumulative strategy returns.
    target_return : float, optional
        Target or benchmark return used to compute excess returns.

    Returns
    -------
    float
        Stutzer index estimate, or ``NaN`` if the optimization fails or
        there is insufficient data.
    """
    if len(returns) < 2:
        return np.nan

    excess_returns = returns - target_return

    if len(excess_returns) == 0:
        return np.nan

    try:
        from scipy.optimize import minimize_scalar

        def neg_log_likelihood(theta):
            if theta <= 0:
                return np.inf
            exp_theta_r = np.exp(theta * excess_returns)
            if np.any(np.isinf(exp_theta_r)) or np.any(np.isnan(exp_theta_r)):
                return np.inf
            return -np.log(exp_theta_r.mean()) / theta

        result = minimize_scalar(neg_log_likelihood, bounds=(1e-10, 10), method="bounded")

        if result.success:
            return -result.fun
        else:
            return np.nan
    except Exception:
        return np.nan


def serial_correlation(returns, lag=1):
    """Determine the serial correlation of returns.

    This computes the correlation between ``r[t]`` and ``r[t-lag]``.

    Parameters
    ----------
    returns : array-like or pd.Series
        Non-cumulative strategy returns.
    lag : int, optional
        Number of periods to lag. Default is 1.

    Returns
    -------
    float
        Pearson correlation coefficient between returns and lagged returns,
        or ``NaN`` if there are fewer than ``lag + 1`` valid observations.
    """
    if returns is None:
        return np.nan

    arr = np.asarray(returns)
    if arr.size == 0:
        return np.nan

    mask = ~np.isnan(arr)
    arr = arr[mask]

    if arr.size <= lag:
        return np.nan

    arr_lagged = arr[lag:]
    arr_early = arr[:-lag]

    if arr_lagged.size < 2:
        return np.nan

    corr = np.corrcoef(arr_lagged, arr_early)[0, 1]
    return float(corr) if not np.isnan(corr) else np.nan


def stock_market_correlation(returns, market_returns):
    """Determine the correlation between strategy and stock market returns.

    Parameters
    ----------
    returns : array-like or pd.Series
        Non-cumulative strategy returns.
    market_returns : array-like or pd.Series
        Non-cumulative stock market (benchmark) returns.

    Returns
    -------
    float
        Pearson correlation coefficient in ``[-1, 1]`` between strategy
        and market returns, or ``NaN`` if there are fewer than two valid
        observations.
    """
    if len(returns) < 2 or len(market_returns) < 2:
        return np.nan

    if isinstance(returns, pd.Series) and isinstance(market_returns, pd.Series):
        returns, market_returns = returns.align(market_returns, join="inner")

    if len(returns) < 2:
        return np.nan

    returns_array = np.asanyarray(returns)
    market_array = np.asanyarray(market_returns)

    valid_mask = ~(np.isnan(returns_array) | np.isnan(market_array))
    returns_clean = returns_array[valid_mask]
    market_clean = market_array[valid_mask]

    if len(returns_clean) < 2:
        return np.nan

    correlation = np.corrcoef(returns_clean, market_clean)[0, 1]

    return float(correlation) if not np.isnan(correlation) else np.nan


def bond_market_correlation(returns, bond_returns):
    """Determine the correlation between strategy and bond market returns.

    Parameters
    ----------
    returns : array-like or pd.Series
        Non-cumulative strategy returns.
    bond_returns : array-like or pd.Series
        Non-cumulative bond market benchmark returns.

    Returns
    -------
    float
        Pearson correlation coefficient in ``[-1, 1]`` between strategy
        and bond market returns, or ``NaN`` if there are fewer than two
        valid observations.
    """
    if len(returns) < 2 or len(bond_returns) < 2:
        return np.nan

    if isinstance(returns, pd.Series) and isinstance(bond_returns, pd.Series):
        returns, bond_returns = returns.align(bond_returns, join="inner")

    if len(returns) < 2:
        return np.nan

    returns_array = np.asanyarray(returns)
    bond_array = np.asanyarray(bond_returns)

    valid_mask = ~(np.isnan(returns_array) | np.isnan(bond_array))
    returns_clean = returns_array[valid_mask]
    bond_clean = bond_array[valid_mask]

    if len(returns_clean) < 2:
        return np.nan

    correlation = np.corrcoef(returns_clean, bond_clean)[0, 1]

    return float(correlation) if not np.isnan(correlation) else np.nan


def futures_market_correlation(returns, futures_returns):
    """Determine the correlation between strategy and futures market returns.

    Parameters
    ----------
    returns : array-like or pd.Series
        Non-cumulative strategy returns.
    futures_returns : array-like or pd.Series
        Non-cumulative futures market benchmark returns.

    Returns
    -------
    float
        Pearson correlation coefficient in ``[-1, 1]`` between strategy
        and futures market returns, or ``NaN`` if there are fewer than two
        valid observations.
    """
    if returns is None or futures_returns is None:
        return np.nan

    if isinstance(returns, pd.Series) and isinstance(futures_returns, pd.Series):
        returns, futures_returns = returns.align(futures_returns, join="inner")

    if len(returns) < 2 or len(futures_returns) < 2:
        return np.nan

    returns_array = np.asanyarray(returns)
    futures_array = np.asanyarray(futures_returns)

    valid_mask = ~(np.isnan(returns_array) | np.isnan(futures_array))
    returns_clean = returns_array[valid_mask]
    futures_clean = futures_array[valid_mask]

    if len(returns_clean) < 2:
        return np.nan

    corr = np.corrcoef(returns_clean, futures_clean)[0, 1]
    return float(corr) if not np.isnan(corr) else np.nan


def win_rate(returns):
    """Calculate the percentage of positive return observations.

    Parameters
    ----------
    returns : array-like or pd.Series
        Non-cumulative strategy returns.

    Returns
    -------
    float
        Fraction of observations with strictly positive returns in
        ``[0, 1]``, or ``NaN`` if there are no valid observations.
    """
    if len(returns) < 1:
        return np.nan

    returns_array = np.asanyarray(returns)

    positive_count = np.sum(returns_array > 0)
    total_count = np.sum(~np.isnan(returns_array))

    if total_count == 0:
        return np.nan

    win_rate_value = positive_count / total_count

    if returns_array.ndim == 1:
        return win_rate_value.item() if isinstance(win_rate_value, np.ndarray) else win_rate_value
    else:
        return win_rate_value


def loss_rate(returns):
    """Calculate the percentage of negative return observations.

    Parameters
    ----------
    returns : array-like or pd.Series
        Non-cumulative strategy returns.

    Returns
    -------
    float
        Fraction of observations with strictly negative returns in
        ``[0, 1]``, or ``NaN`` if there are no valid observations.
    """
    if len(returns) < 1:
        return np.nan

    returns_array = np.asanyarray(returns)

    negative_count = np.sum(returns_array < 0)
    total_count = np.sum(~np.isnan(returns_array))

    if total_count == 0:
        return np.nan

    loss_rate_value = negative_count / total_count

    if returns_array.ndim == 1:
        return loss_rate_value.item() if isinstance(loss_rate_value, np.ndarray) else loss_rate_value
    else:
        return loss_rate_value


def r_cubed(returns, factor_returns):
    """Calculate the R-cubed (R³) measure.

    R³ is defined as the cube of the correlation between strategy returns
    and benchmark returns.

    Parameters
    ----------
    returns : array-like or pd.Series
        Non-cumulative strategy returns.
    factor_returns : array-like or pd.Series
        Non-cumulative benchmark or factor returns.

    Returns
    -------
    float
        Cube of the Pearson correlation coefficient, or ``NaN`` if there
        is insufficient data.
    """
    returns_aligned, factor_aligned = aligned_series(returns, factor_returns)

    if len(returns_aligned) < 2:
        return np.nan

    correlation = np.corrcoef(returns_aligned, factor_aligned)[0, 1]
    return correlation ** 3


def tracking_difference(returns, factor_returns):
    """Calculate tracking difference in cumulative returns.

    Tracking difference is defined as the cumulative strategy return minus
    the cumulative benchmark return over the full sample.

    Parameters
    ----------
    returns : array-like or pd.Series
        Non-cumulative strategy returns.
    factor_returns : array-like or pd.Series
        Non-cumulative benchmark or factor returns.

    Returns
    -------
    float
        Difference between cumulative strategy and benchmark returns.
    """
    from fincore.metrics.returns import cum_returns_final

    if len(returns) < 1:
        return np.nan

    returns_aligned, factor_aligned = aligned_series(returns, factor_returns)

    cum_strategy = cum_returns_final(returns_aligned, starting_value=0)
    cum_benchmark = cum_returns_final(factor_aligned, starting_value=0)

    result = cum_strategy - cum_benchmark
    if not isinstance(result, (float, np.floating)):
        result = result.item()
    return result


def common_sense_ratio(returns):
    """Calculate the common sense ratio.

    Delegates to :func:`fincore.metrics.ratios.common_sense_ratio`.

    Parameters
    ----------
    returns : array-like or pd.Series
        Non-cumulative strategy returns.

    Returns
    -------
    float
        Common sense ratio, or ``NaN`` if there is insufficient data.
    """
    from fincore.metrics.ratios import common_sense_ratio as _csr
    return _csr(returns)


def var_cov_var_normal(p, c, mu=0, sigma=1):
    """Calculate variance-covariance of daily Value-at-Risk in a portfolio.

    Delegates to :func:`fincore.metrics.risk.var_cov_var_normal`.

    Parameters
    ----------
    p : float
        Portfolio value.
    c : float
        Confidence level.
    mu : float, optional
        Mean. Default is 0.
    sigma : float, optional
        Standard deviation. Default is 1.

    Returns
    -------
    float
        Value-at-Risk.
    """
    from fincore.metrics.risk import var_cov_var_normal as _vcv
    return _vcv(p, c, mu, sigma)


def normalize(returns, starting_value=1):
    """Normalize a returns series to start at a given value.

    Delegates to :func:`fincore.metrics.returns.normalize`.

    Parameters
    ----------
    returns : array-like or pd.Series
        Non-cumulative strategy returns.
    starting_value : float, optional
        Starting value for the normalized series. Default is 1.

    Returns
    -------
    pd.Series or np.ndarray
        Normalized cumulative returns starting at ``starting_value``.
    """
    from fincore.metrics.returns import normalize as _normalize
    return _normalize(returns, starting_value=starting_value)
