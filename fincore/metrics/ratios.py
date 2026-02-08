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

"""比率指标函数模块."""

import numpy as np
import pandas as pd
from scipy import stats
from fincore.utils import nanmean, nanstd
from fincore.constants import DAILY, APPROX_BDAYS_PER_YEAR
from fincore.metrics.basic import (
    annualization_factor, adjust_returns, aligned_series
)
from fincore.metrics.risk import tail_ratio

__all__ = [
    'sharpe_ratio',
    'sortino_ratio',
    'excess_sharpe',
    'adjusted_sharpe_ratio',
    'conditional_sharpe_ratio',
    'calmar_ratio',
    'omega_ratio',
    'information_ratio',
    'treynor_ratio',
    'cal_treynor_ratio',
    'm_squared',
    'sterling_ratio',
    'burke_ratio',
    'kappa_three_ratio',
    'common_sense_ratio',
    'stability_of_timeseries',
    'capture',
    'up_capture',
    'down_capture',
    'up_down_capture',
]


def sharpe_ratio(returns, risk_free=0, period=DAILY, annualization=None, out=None):
    """Determine the annualized Sharpe ratio of a strategy.

    The Sharpe ratio is defined as the annualized mean excess return
    divided by the annualized standard deviation of returns.

    Parameters
    ----------
    returns : array-like or pd.Series or pd.DataFrame
        Non-cumulative simple returns.
    risk_free : float, optional
        Risk-free rate used when computing excess returns. Default is 0.
    period : str, optional
        Frequency of the input data (for example ``DAILY``). Used to
        infer the annualization factor when ``annualization`` is ``None``.
    annualization : float, optional
        Custom annualization factor. If provided, this value is used
        directly instead of inferring it from ``period``.
    out : np.ndarray, optional
        pre-allocated output array. If provided, the result is
        written in-place into this array.

    Returns
    -------
    float or np.ndarray or pd.Series
        Annualized Sharpe ratio. For 1D input a scalar is returned; for 2D
        input one value is returned per column.
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

    returns_risk_adj = np.asanyarray(adjust_returns(returns, risk_free))
    ann_factor = annualization_factor(period, annualization)

    std_returns = nanstd(returns_risk_adj, ddof=1, axis=0)
    mean_returns = nanmean(returns_risk_adj, axis=0)

    with np.errstate(divide="ignore", invalid="ignore"):
        np.multiply(np.divide(mean_returns, std_returns, out=out), np.sqrt(ann_factor), out=out)
    if return_1d:
        out = out.item()

    return out


def sortino_ratio(returns, required_return=0, period=DAILY, annualization=None, out=None, _downside_risk=None):
    """Determine the Sortino ratio of a strategy.

    The Sortino ratio is defined as the annualized excess return divided
    by the annualized downside risk. Only returns falling below
    ``required_return`` contribute to the downside risk term.

    Parameters
    ----------
    returns : array-like or pd.Series or pd.DataFrame
        Non-cumulative strategy returns.
    required_return : float, optional
        Minimum acceptable return used as the threshold when computing
        downside risk. Default is 0.
    period : str, optional
        Frequency of the input data (for example ``DAILY``). Used to
        infer the annualization factor when ``annualization`` is ``None``.
    annualization : float, optional
        Custom annualization factor. If provided, this value is used
        directly instead of inferring it from ``period``.
    out : np.ndarray, optional
        Optional pre-allocated output array. If given, the result is
        written in-place into this array.
    _downside_risk : float or np.ndarray, optional
        Optional pre-computed annualized downside risk. If provided, this
        value is reused instead of recomputing downside risk.

    Returns
    -------
    float or np.ndarray or pd.Series
        Sortino ratio of the strategy. For 1D input a scalar is
        returned; for 2D input one value is returned per column.
    """
    from fincore.metrics.risk import downside_risk as calc_downside_risk

    allocated_output = out is None
    if allocated_output:
        out = np.empty(returns.shape[1:])

    return_1d = returns.ndim == 1

    if len(returns) < 2:
        out[()] = np.nan
        if return_1d:
            out = out.item()
        return out

    adj_returns = np.asanyarray(adjust_returns(returns, required_return))
    ann_factor = annualization_factor(period, annualization)

    average_annual_return = nanmean(adj_returns, axis=0) * ann_factor
    if _downside_risk is None:
        annualized_downside_risk = calc_downside_risk(returns, required_return, period, annualization)
    else:
        annualized_downside_risk = _downside_risk

    with np.errstate(divide="ignore", invalid="ignore"):
        np.divide(average_annual_return, annualized_downside_risk, out=out)
    if return_1d:
        out = out.item()
    elif allocated_output and isinstance(returns, pd.DataFrame):
        out = pd.Series(out)
    return out


def excess_sharpe(returns, factor_returns, out=None):
    """Determine the excess Sharpe ratio of a strategy.

    The excess Sharpe ratio is defined as the mean active return divided
    by the tracking error, where active return is the difference between
    the strategy returns and the benchmark (factor) returns.

    Parameters
    ----------
    returns : array-like or pd.Series or pd.DataFrame
        Non-cumulative strategy returns.
    factor_returns : array-like or pd.Series or pd.DataFrame
        Non-cumulative benchmark or factor returns, aligned to
        ``returns``.
    out : np.ndarray, optional
        Optional pre-allocated output array. If given, the result is
        written in-place into this array.

    Returns
    -------
    float or np.ndarray or pd.Series
        Excess Sharpe ratio. For 1D input a scalar is returned; for 2D
        input one value is returned per column.
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

    active_return = adjust_returns(returns, factor_returns)
    tracking_err = np.nan_to_num(nanstd(active_return, ddof=1, axis=0))

    with np.errstate(divide="ignore", invalid="ignore"):
        out = np.divide(nanmean(active_return, axis=0, out=out), tracking_err, out=out)
    if returns_1d:
        out = out.item()
    return out


def adjusted_sharpe_ratio(returns, risk_free=0.0):
    """Calculate the adjusted Sharpe ratio.

    This version of the Sharpe ratio applies a correction for skewness
    and kurtosis of the return distribution, following the standard
    approximation with additional dampening for small sample sizes.

    Parameters
    ----------
    returns : array-like or pd.Series
        Non-cumulative strategy returns.
    risk_free : float, optional
        Risk-free rate used when computing excess returns. Default is 0.0.

    Returns
    -------
    float
        Adjusted Sharpe ratio that accounts for non-normality. Returns
        ``NaN`` when there is insufficient data.
    """
    from fincore.metrics.stats import skewness, kurtosis

    if len(returns) < 4:
        return np.nan

    sharpe = sharpe_ratio(returns, risk_free)

    if np.isnan(sharpe):
        return np.nan

    skew = skewness(returns)
    if np.isnan(skew):
        skew = 0

    kurt = kurtosis(returns)
    if np.isnan(kurt):
        kurt = 0

    n = len(returns)
    dampening = min(1.0, n / 50.0)

    skew_adj = (skew / 6) * sharpe * dampening
    kurt_adj = (kurt / 24) * (sharpe**2) * dampening
    adjustment = 1 + skew_adj - kurt_adj

    if n < 20:
        adjustment = max(0.9, min(1.1, adjustment))
    else:
        adjustment = max(0.8, min(1.3, adjustment))

    return sharpe * adjustment


def conditional_sharpe_ratio(returns, cutoff=0.05):
    """Calculate the Sharpe ratio conditional on the left tail.

    The conditional Sharpe ratio is computed on the subset of returns
    that fall below a given quantile of the full return distribution.

    Parameters
    ----------
    returns : array-like or pd.Series
        Non-cumulative strategy returns.
    cutoff : float, optional
        Left-tail probability level in (0, 1). For example ``0.05``
        selects the worst 5% of returns. Default is 0.05.

    Returns
    -------
    float
        Sharpe ratio computed on the conditional (left-tail) subsample,
        annualized using a 252 trading-day convention. Returns ``NaN`` if
        there are fewer than two observations.
    """
    if len(returns) < 2:
        return np.nan

    cutoff_value = np.percentile(returns, cutoff * 100)
    conditional_returns = returns[returns <= cutoff_value]

    if len(conditional_returns) < 2:
        return np.nan

    mean_ret = np.mean(conditional_returns)
    std_ret = np.std(conditional_returns, ddof=1)

    if std_ret == 0:
        return np.nan

    return mean_ret / std_ret * np.sqrt(252)


def calmar_ratio(returns, period=DAILY, annualization=None):
    """Determine the Calmar ratio (return-to-drawdown ratio).

    The Calmar ratio is defined as the annualized return divided by the
    absolute value of the maximum drawdown.

    Parameters
    ----------
    returns : array-like or pd.Series or pd.DataFrame
        Non-cumulative simple returns.
    period : str, optional
        Frequency of the input data (for example ``DAILY``). Used to
        annualize returns when ``annualization`` is ``None``.
    annualization : float, optional
        Custom annualization factor. If provided, this value is used
        directly instead of inferring it from ``period``.

    Returns
    -------
    float
        Calmar ratio of the return series. Returns ``NaN`` if maximum
        drawdown is non-negative or infinite.
    """
    from fincore.metrics.drawdown import max_drawdown
    from fincore.metrics.returns import cum_returns_final

    max_dd = max_drawdown(returns=returns)
    if max_dd < 0:
        ann_factor = annualization_factor(period, annualization)
        num_years = len(returns) / ann_factor
        ending_value = cum_returns_final(returns, starting_value=1)
        ann_return = ending_value ** (1 / num_years) - 1
        temp = ann_return / abs(max_dd)
    else:
        return np.nan

    if np.isinf(temp):
        return np.nan

    return temp


def omega_ratio(returns, risk_free=0.0, required_return=0.0, annualization=APPROX_BDAYS_PER_YEAR):
    """Determine the Omega ratio of a strategy.

    The Omega ratio is the probability-weighted ratio of gains over
    losses above/below a required return threshold.

    Parameters
    ----------
    returns : array-like or pd.Series
        Non-cumulative simple returns.
    risk_free : float, optional
        Risk-free rate used when computing excess returns. Default is 0.0.
    required_return : float, optional
        Minimum acceptable return threshold. Default is 0.0.
    annualization : float, optional
        Number of periods per year used when interpreting
        ``required_return`` as an annual rate (for example trading days
        per year). Default is ``APPROX_BDAYS_PER_YEAR``.

    Returns
    -------
    float
        Omega ratio of the strategy. Returns ``NaN`` if there are fewer
        than two observations or if the downside component is zero.
    """
    if len(returns) < 2:
        return np.nan

    if annualization == 1:
        return_threshold = required_return
    elif required_return <= -1:
        return np.nan
    else:
        return_threshold = (1 + required_return) ** (1.0 / annualization) - 1

    returns_less_thresh = returns - risk_free - return_threshold

    numer = sum(returns_less_thresh[returns_less_thresh > 0.0])
    denom = -1.0 * sum(returns_less_thresh[returns_less_thresh < 0.0])

    if denom > 0.0:
        return numer / denom
    else:
        return np.nan


def information_ratio(returns, factor_returns, period=DAILY, annualization=None):
    """Determine the information ratio versus a benchmark.

    The information ratio is defined as the annualized mean active return
    divided by the annualized tracking error of the active returns.

    Parameters
    ----------
    returns : pd.Series or pd.DataFrame
        Non-cumulative strategy returns.
    factor_returns : pd.Series or pd.DataFrame
        Non-cumulative benchmark or factor returns, aligned to
        ``returns``.
    period : str, optional
        Frequency of the input data (for example ``DAILY``). Used to
        infer the annualization factor when ``annualization`` is ``None``.
    annualization : float, optional
        Custom annualization factor. If provided, this value is used
        directly instead of inferring it from ``period``.

    Returns
    -------
    float or pd.Series
        Information ratio of the strategy versus the benchmark.
    """
    returns, factor_returns = aligned_series(returns, factor_returns)
    super_returns = returns - factor_returns

    ann_factor = annualization_factor(period, annualization)
    mean_excess_return = super_returns.mean()
    std_excess_return = super_returns.std(ddof=1)
    ir = (mean_excess_return * ann_factor) / (std_excess_return * np.sqrt(ann_factor))
    return ir


def cal_treynor_ratio(returns, factor_returns, risk_free=0.0, period=DAILY, annualization=None):
    r"""Calculate the Treynor ratio of a strategy.

    The Treynor ratio is defined as the strategy's annualized excess
    return divided by its beta with respect to a benchmark:

    .. math::

        T = \frac{R_p - R_f}{\beta_p}

    where :math:`R_p` is the annualized portfolio return, :math:`R_f` is
    the risk-free rate, and :math:`\beta_p` is the portfolio beta.

    Parameters
    ----------
    returns : pd.Series or np.ndarray
        Non-cumulative strategy returns.
    factor_returns : pd.Series or np.ndarray
        Non-cumulative benchmark or factor returns used to estimate beta.
    risk_free : float, optional
        Risk-free rate used when computing excess returns. Default is 0.0.
    period : str, optional
        Frequency of the input data (for example ``DAILY``). Used to
        annualize returns when ``annualization`` is ``None``.
    annualization : float, optional
        Custom annualization factor. If provided, this value is used
        directly instead of inferring it from ``period``.

    Returns
    -------
    float or np.ndarray or pd.Series
        Treynor ratio. For 1D input a scalar is returned; for 2D input one
        value is returned per column. Returns ``NaN`` when beta is zero,
        negative, or ``NaN``.
    """
    from fincore.metrics.alpha_beta import beta_aligned
    from fincore.metrics.returns import cum_returns_final

    allocated_output = True
    out = np.empty(returns.shape[1:])

    returns_1d = returns.ndim == 1

    if len(returns) < 2:
        out[()] = np.nan
        if returns_1d:
            out = out.item()
        return out

    returns, factor_returns = aligned_series(returns, factor_returns)

    ann_factor = annualization_factor(period, annualization)
    num_years = len(returns) / ann_factor
    ending_value = cum_returns_final(returns, starting_value=1)
    ann_return = ending_value ** (1 / num_years) - 1
    ann_excess_return = ann_return - risk_free

    b = beta_aligned(returns, factor_returns, risk_free)

    if returns_1d:
        if b == 0 or b < 0 or np.isnan(b):
            out = np.nan
        else:
            with np.errstate(divide="ignore", invalid="ignore"):
                out[()] = ann_excess_return / b
            out = out.item()
    else:
        if isinstance(b, (pd.Series, np.ndarray)):
            mask = (b == 0) | (b < 0) | np.isnan(b)
            with np.errstate(divide="ignore", invalid="ignore"):
                if isinstance(ann_excess_return, (pd.Series, pd.DataFrame)):
                    out = (ann_excess_return / b).values
                else:
                    out = ann_excess_return / b
            out[mask] = np.nan
        else:
            with np.errstate(divide="ignore", invalid="ignore"):
                out[()] = ann_excess_return / b

        if allocated_output and isinstance(returns, pd.DataFrame):
            out = pd.Series(out, index=returns.columns)

    return out


def treynor_ratio(returns, factor_returns, risk_free=0.0, period=DAILY, annualization=None):
    """Compute the Treynor ratio.

    This is a thin wrapper around :func:`cal_treynor_ratio`.

    Parameters
    ----------
    returns : pd.Series or np.ndarray
        Non-cumulative strategy returns.
    factor_returns : pd.Series or np.ndarray
        Non-cumulative benchmark or factor returns.
    risk_free : float, optional
        Risk-free rate. Default is 0.0.
    period : str, optional
        Frequency of the input data. Default is ``DAILY``.
    annualization : float, optional
        Custom annualization factor.

    Returns
    -------
    float or np.ndarray or pd.Series
        Treynor ratio.
    """
    return cal_treynor_ratio(returns, factor_returns, risk_free, period, annualization)


def m_squared(returns, factor_returns, risk_free=0.0, period=DAILY, annualization=None):
    r"""Calculate the Modigliani (M²) measure.

    The M² measure scales the portfolio's risk-adjusted performance to
    the benchmark's volatility. It is defined as:

    .. math::

        M^2 = (R_p - R_f) \frac{\sigma_b}{\sigma_p} + R_f

    where :math:`R_p` and :math:`R_f` are the annualized portfolio and
    risk-free returns, and :math:`\sigma_p` and :math:`\sigma_b` are the
    annualized volatilities of the portfolio and benchmark, respectively.

    Parameters
    ----------
    returns : pd.Series or np.ndarray
        Non-cumulative strategy returns.
    factor_returns : pd.Series or np.ndarray
        Non-cumulative benchmark or factor returns.
    risk_free : float, optional
        Risk-free rate used when computing excess returns. Default is 0.0.
    period : str, optional
        Frequency of the input data (for example ``DAILY``). Used to
        annualize returns and volatilities when ``annualization`` is
        ``None``.
    annualization : float, optional
        Custom annualization factor. If provided, this value is used
        directly instead of inferring it from ``period``.

    Returns
    -------
    float
        M² value of the strategy relative to the benchmark. Returns
        ``NaN`` if the portfolio volatility is zero, negative, or
        ``NaN``.
    """
    from fincore.metrics.risk import annual_volatility
    from fincore.metrics.returns import cum_returns_final

    if len(returns) < 2:
        return np.nan

    returns_aligned, factor_aligned = aligned_series(returns, factor_returns)

    ann_factor = annualization_factor(period, annualization)
    num_years = len(returns_aligned) / ann_factor
    ending_value = cum_returns_final(returns_aligned, starting_value=1)
    ann_return = ending_value ** (1 / num_years) - 1

    ann_vol = annual_volatility(returns_aligned, period=period, annualization=annualization)
    ann_factor_vol = annual_volatility(factor_aligned, period=period, annualization=annualization)

    if ann_vol == 0 or ann_vol < 0 or np.isnan(ann_vol):
        return np.nan

    excess_return = ann_return - risk_free
    risk_ratio = ann_factor_vol / ann_vol
    return excess_return * risk_ratio + risk_free


def sterling_ratio(returns, risk_free=0.0, period=DAILY, annualization=None):
    """Calculate the Sterling ratio of a strategy.

    The Sterling ratio is defined as the annualized excess return divided
    by the average drawdown. When no explicit drawdowns are detected,
    this implementation falls back to using downside returns as a proxy
    for risk.

    Parameters
    ----------
    returns : array-like or pd.Series
        Non-cumulative strategy returns.
    risk_free : float, optional
        Risk-free rate used when computing excess returns. Default is 0.0.
    period : str, optional
        Frequency of the input data (for example ``DAILY``). Used to
        annualize returns when ``annualization`` is ``None``.
    annualization : float, optional
        Custom annualization factor. If provided, this value is used
        directly instead of inferring it from ``period``.

    Returns
    -------
    float
        Sterling ratio of the strategy. Returns ``NaN`` when there is
        effectively no drawdown risk.
    """
    from fincore.metrics.drawdown import get_all_drawdowns
    from fincore.metrics.returns import cum_returns_final

    if len(returns) < 2:
        return np.nan

    drawdown_periods = get_all_drawdowns(returns)

    if len(drawdown_periods) == 0 or all(dd == 0 for dd in drawdown_periods):
        returns_array = np.asanyarray(returns)
        returns_clean = returns_array[~np.isnan(returns_array)]
        downside_returns = returns_clean[returns_clean < 0]

        if len(downside_returns) == 0:
            avg_drawdown = max(abs(np.std(returns_clean)), 1e-10)
        else:
            avg_drawdown = abs(np.mean(downside_returns))
    else:
        avg_drawdown = abs(np.mean(drawdown_periods))

    if avg_drawdown == 0 or avg_drawdown < 1e-10:
        ann_factor = annualization_factor(period, annualization)
        num_years = len(returns) / ann_factor
        ending_value = cum_returns_final(returns, starting_value=1)
        ann_ret = ending_value ** (1 / num_years) - 1
        return np.inf if ann_ret - risk_free > 0 else np.nan

    ann_factor = annualization_factor(period, annualization)
    num_years = len(returns) / ann_factor
    ending_value = cum_returns_final(returns, starting_value=1)
    ann_ret = ending_value ** (1 / num_years) - 1

    return (ann_ret - risk_free) / avg_drawdown


def burke_ratio(returns, risk_free=0.0, period=DAILY, annualization=None):
    """Calculate the Burke ratio of a strategy.

    The Burke ratio is defined as the annualized excess return divided by
    the square root of the sum of squared drawdowns.

    Parameters
    ----------
    returns : array-like or pd.Series
        Non-cumulative strategy returns.
    risk_free : float, optional
        Risk-free rate used when computing excess returns. Default is 0.0.
    period : str, optional
        Frequency of the input data (for example ``DAILY``). Used to
        annualize returns when ``annualization`` is ``None``.
    annualization : float, optional
        Custom annualization factor. If provided, this value is used
        directly instead of inferring it from ``period``.

    Returns
    -------
    float
        Burke ratio of the strategy. Returns ``NaN`` when there is
        effectively no drawdown risk.
    """
    from fincore.metrics.drawdown import get_all_drawdowns
    from fincore.metrics.returns import cum_returns_final

    if len(returns) < 2:
        return np.nan

    drawdown_periods = get_all_drawdowns(returns)

    if len(drawdown_periods) == 0 or all(dd == 0 for dd in drawdown_periods):
        returns_array = np.asanyarray(returns)
        returns_clean = returns_array[~np.isnan(returns_array)]
        downside_returns = returns_clean[returns_clean < 0]

        if len(downside_returns) == 0:
            burke_risk = float(max(np.std(returns_clean), 1e-10))
        else:
            burke_risk = float(np.std(downside_returns))
    else:
        squared_drawdowns = [dd**2 for dd in drawdown_periods]
        burke_risk = float(np.sqrt(np.sum(squared_drawdowns)))

    if burke_risk == 0 or burke_risk < 1e-10:
        ann_factor = annualization_factor(period, annualization)
        num_years = len(returns) / ann_factor
        ending_value = cum_returns_final(returns, starting_value=1)
        ann_ret = ending_value ** (1 / num_years) - 1
        return np.inf if ann_ret - risk_free > 0 else np.nan

    ann_factor = annualization_factor(period, annualization)
    num_years = len(returns) / ann_factor
    ending_value = cum_returns_final(returns, starting_value=1)
    ann_ret = ending_value ** (1 / num_years) - 1

    return (ann_ret - risk_free) / burke_risk


def kappa_three_ratio(returns, risk_free=0.0, period=DAILY, annualization=None, mar=0.0):
    """Calculate the Kappa 3 ratio based on third-order downside deviation.

    This variant of the Kappa ratio uses the third-order lower partial
    moment (LPM3) of returns below a minimum acceptable return (MAR) as
    the risk measure.

    Parameters
    ----------
    returns : array-like or pd.Series
        Non-cumulative strategy returns.
    risk_free : float, optional
        Risk-free rate used when computing excess returns. Default is 0.0.
    period : str, optional
        Frequency of the input data (for example ``DAILY``). Used to
        annualize LPM3 when ``annualization`` is ``None``.
    annualization : float, optional
        Custom annualization factor. If provided, this value is used
        directly instead of inferring it from ``period``.
    mar : float, optional
        Minimum acceptable return (MAR) used when computing the lower
        partial moment. Default is 0.0.

    Returns
    -------
    float
        Kappa 3 ratio of the strategy. Returns ``NaN`` when downside
        risk is effectively zero.
    """
    from fincore.metrics.returns import cum_returns_final

    if len(returns) < 2:
        return np.nan

    returns_array = np.asanyarray(returns)
    returns_clean = returns_array[~np.isnan(returns_array)]

    if len(returns_clean) < 2:
        return np.nan

    downside_deviations = np.maximum(0, mar - returns_clean)
    lpm3 = np.mean(downside_deviations**3)

    ann_factor = annualization_factor(period, annualization)

    if lpm3 == 0 or lpm3 < 1e-30:
        std_dev = np.std(returns_clean)
        if std_dev < 1e-10:
            num_years = len(returns) / ann_factor
            ending_value = cum_returns_final(returns, starting_value=1)
            ann_ret = ending_value ** (1 / num_years) - 1
            return np.inf if ann_ret - risk_free > 0 else np.nan
        lpm3_risk = std_dev * np.sqrt(ann_factor)
    else:
        lpm3_annualized = lpm3 * ann_factor
        lpm3_risk = lpm3_annualized ** (1.0 / 3.0)

    if lpm3_risk == 0 or lpm3_risk < 1e-10:
        num_years = len(returns) / ann_factor
        ending_value = cum_returns_final(returns, starting_value=1)
        ann_ret = ending_value ** (1 / num_years) - 1
        return np.inf if ann_ret - risk_free > 0 else np.nan

    num_years = len(returns) / ann_factor
    ending_value = cum_returns_final(returns, starting_value=1)
    ann_ret = ending_value ** (1 / num_years) - 1

    return (ann_ret - risk_free) / lpm3_risk


def common_sense_ratio(returns):
    """Calculate the common sense ratio.

    The common sense ratio combines the tail ratio with the win rate to
    provide a measure of the risk-reward profile of a strategy.

    Parameters
    ----------
    returns : array-like or pd.Series
        Non-cumulative strategy returns.

    Returns
    -------
    float
        Common sense ratio. Returns ``NaN`` if there are insufficient
        observations or if win rate is zero.
    """
    from fincore.metrics.risk import tail_ratio
    from fincore.metrics.stats import win_rate

    if len(returns) < 2:
        return np.nan

    tr = tail_ratio(returns)
    wr = win_rate(returns)

    if np.isnan(tr) or np.isnan(wr) or wr == 0:
        return np.nan

    return tr * wr / (1 - wr) if wr != 1 else np.inf


def stability_of_timeseries(returns):
    """Determine the R-squared of a linear fit to cumulative log returns.

    This computes the coefficient of determination (R²) from a linear
    regression of the cumulative log returns against time. Higher values
    indicate a more stable (less noisy) growth profile.

    Parameters
    ----------
    returns : array-like or pd.Series
        Non-cumulative returns.

    Returns
    -------
    float
        R-squared of the linear fit to cumulative log returns. Returns
        ``NaN`` if there are fewer than two valid observations.
    """
    if len(returns) < 2:
        return np.nan

    returns = np.asanyarray(returns)
    returns = returns[~np.isnan(returns)]

    if len(returns) < 2:
        return np.nan

    cum_log_returns = np.log1p(returns).cumsum()
    rhat = stats.linregress(np.arange(len(cum_log_returns)), cum_log_returns)[2]

    return rhat**2


def capture(returns, factor_returns, period=DAILY):
    """Calculate the capture ratio versus a benchmark.

    The capture ratio is defined as the strategy's annualized return
    divided by the benchmark's annualized return over the same period.

    Parameters
    ----------
    returns : pd.Series
        Non-cumulative strategy returns.
    factor_returns : pd.Series
        Non-cumulative benchmark or factor returns.
    period : str, optional
        Frequency of the input data (for example ``DAILY``). Used to
        annualize both strategy and benchmark returns.

    Returns
    -------
    float
        Capture ratio (strategy annualized return divided by benchmark
        annualized return). Returns ``NaN`` if there are insufficient
        observations or the benchmark annualized return is zero.
    """
    from fincore.metrics.returns import cum_returns_final

    if len(returns) < 1 or len(factor_returns) < 1:
        return np.nan

    ann_factor = annualization_factor(period, None)
    num_years = len(returns) / ann_factor
    strategy_ending = cum_returns_final(returns, starting_value=1)
    strategy_ann_return = strategy_ending ** (1 / num_years) - 1

    benchmark_num_years = len(factor_returns) / ann_factor
    benchmark_ending = cum_returns_final(factor_returns, starting_value=1)
    benchmark_ann_return = benchmark_ending ** (1 / benchmark_num_years) - 1

    if benchmark_ann_return == 0:
        return np.nan

    return strategy_ann_return / benchmark_ann_return


def up_capture(returns, factor_returns, period=DAILY):
    """Calculate the capture ratio for up-market periods.

    The up-capture ratio is defined as the capture ratio computed using
    only those periods where the benchmark (factor) return is positive.

    Parameters
    ----------
    returns : pd.Series
        Non-cumulative strategy returns.
    factor_returns : pd.Series
        Non-cumulative benchmark or factor returns.
    period : str, optional
        Frequency of the input data (for example ``DAILY``).

    Returns
    -------
    float
        Up-capture ratio of the strategy relative to the benchmark.
        Returns ``NaN`` if there are no positive benchmark periods.
    """
    returns, factor_returns = aligned_series(returns, factor_returns)

    returns = pd.Series(returns) if not isinstance(returns, pd.Series) else returns
    factor_returns = pd.Series(factor_returns) if not isinstance(factor_returns, pd.Series) else factor_returns

    up_returns = returns[factor_returns > 0]
    up_factor_returns = factor_returns[factor_returns > 0]

    if len(up_returns) < 1:
        return np.nan

    return capture(up_returns, up_factor_returns, period=period)


def down_capture(returns, factor_returns, period=DAILY):
    """Calculate the capture ratio for down-market periods.

    The down-capture ratio is defined as the capture ratio computed using
    only those periods where the benchmark (factor) return is negative.

    Parameters
    ----------
    returns : pd.Series
        Non-cumulative strategy returns.
    factor_returns : pd.Series
        Non-cumulative benchmark or factor returns.
    period : str, optional
        Frequency of the input data (for example ``DAILY``).

    Returns
    -------
    float
        Down-capture ratio of the strategy relative to the benchmark.
        Returns ``NaN`` if there are no negative benchmark periods.
    """
    returns, factor_returns = aligned_series(returns, factor_returns)

    returns = pd.Series(returns) if not isinstance(returns, pd.Series) else returns
    factor_returns = pd.Series(factor_returns) if not isinstance(factor_returns, pd.Series) else factor_returns

    down_returns = returns[factor_returns < 0]
    down_factor_returns = factor_returns[factor_returns < 0]

    if len(down_returns) < 1:
        return np.nan

    return capture(down_returns, down_factor_returns, period=period)


def up_down_capture(returns, factor_returns, period=DAILY):
    """Calculate the ratio of up-capture to down-capture.

    This computes ``up_capture / down_capture`` using
    :func:`up_capture` and :func:`down_capture`.

    Parameters
    ----------
    returns : pd.Series
        Non-cumulative strategy returns.
    factor_returns : pd.Series
        Non-cumulative benchmark or factor returns.
    period : str, optional
        Frequency of the input data (for example ``DAILY``).

    Returns
    -------
    float
        Ratio of up-capture to down-capture. Returns ``NaN`` if either
        capture is ``NaN`` or if the down-capture is zero.
    """
    up_cap = up_capture(returns, factor_returns, period=period)
    down_cap = down_capture(returns, factor_returns, period=period)

    if np.isnan(up_cap) or np.isnan(down_cap) or down_cap == 0:
        return np.nan

    return up_cap / down_cap
