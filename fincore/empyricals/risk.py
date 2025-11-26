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

"""风险指标函数模块."""

import numpy as np
import pandas as pd
from scipy import stats
from sys import float_info
from fincore.utils import nanmean, nanstd
from fincore.constants import DAILY
from fincore.empyricals.basic import (
    annualization_factor, adjust_returns, aligned_series
)

__all__ = [
    'annual_volatility',
    'downside_risk',
    'value_at_risk',
    'conditional_value_at_risk',
    'tail_ratio',
    'tracking_error',
    'residual_risk',
    'var_excess_return',
    'var_cov_var_normal',
    'trading_value_at_risk',
    'gpd_risk_estimates',
    'gpd_risk_estimates_aligned',
    'beta_fragility_heuristic',
    'beta_fragility_heuristic_aligned',
]


def annual_volatility(returns, period=DAILY, alpha_=2.0, annualization=None, out=None):
    """Determine the annualized volatility of a return series.

    Volatility is computed as the standard deviation of returns scaled by
    an annualization factor based on ``period`` or ``annualization``.

    Parameters
    ----------
    returns : array-like or pd.Series or pd.DataFrame
        Non-cumulative simple returns.
    period : str, optional
        Frequency of the input data (for example ``DAILY``). Used to
        infer the annualization factor when ``annualization`` is ``None``.
    alpha_ : float, optional
        Power used when scaling volatility (for example 2.0 for variance,
        1.0 for standard deviation). Defaults to 2.0 to match the
        original empyrical implementation.
    annualization : float, optional
        Custom annualization factor. If provided, this value is used
        directly instead of inferring it from ``period``.
    out : np.ndarray, optional
        pre-allocated output array. If provided, the result is
        written in-place into this array.

    Returns
    -------
    float or np.ndarray or pd.Series
        Annualized volatility. For 1D input a scalar is returned; for 2D
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

    ann_factor = annualization_factor(period, annualization)
    nanstd(returns, ddof=1, axis=0, out=out)
    out = np.multiply(out, ann_factor ** (1.0 / alpha_), out=out)
    if returns_1d:
        out = out.item()
    return out


def downside_risk(returns, required_return=0, period=DAILY, annualization=None, out=None):
    """Determine the annualized downside deviation below a threshold."""
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
        adjust_returns(np.asanyarray(returns), np.asanyarray(required_return)),
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


def value_at_risk(returns, cutoff=0.05):
    """Calculate the (historical) value at risk (VaR) of returns."""
    if len(returns) < 1:
        return np.nan
    return np.percentile(returns, cutoff * 100)


def conditional_value_at_risk(returns, cutoff=0.05):
    """Calculate the conditional value at risk (CVaR) of returns."""
    if len(returns) < 1:
        return np.nan
    cutoff_index = value_at_risk(returns, cutoff=cutoff)
    return np.mean(returns[returns <= cutoff_index])


def tail_ratio(returns):
    """Determine the ratio of right- to left-tail percentiles of returns."""
    if len(returns) < 1:
        return np.nan

    returns = np.asanyarray(returns)
    returns = returns[~np.isnan(returns)]
    if len(returns) < 1:
        return np.nan

    return np.abs(np.percentile(returns, 95)) / np.abs(np.percentile(returns, 5))


def tracking_error(returns, factor_returns, period=DAILY, annualization=None, out=None):
    """Determine the annualized tracking error versus a benchmark."""
    allocated_output = out is None
    if allocated_output:
        out = np.empty(returns.shape[1:])

    returns_1d = returns.ndim == 1

    if len(returns) < 2:
        out[()] = np.nan
        if returns_1d:
            out = out.item()
        return out

    returns, factor_returns = aligned_series(returns, factor_returns)
    active_return = adjust_returns(returns, factor_returns)
    ann_factor = annualization_factor(period, annualization)

    nanstd(active_return, ddof=1, axis=0, out=out)
    np.multiply(out, np.sqrt(ann_factor), out=out)

    if returns_1d:
        out = out.item()

    return out


def residual_risk(returns, factor_returns, risk_free=0.0):
    """Calculate residual risk (tracking error of alpha)."""
    returns_aligned, factor_aligned = aligned_series(returns, factor_returns)

    if len(returns_aligned) < 2:
        return np.nan

    excess_returns = returns_aligned - risk_free
    excess_factor = factor_aligned - risk_free

    beta_val = np.cov(excess_returns, excess_factor)[0, 1] / np.var(excess_factor, ddof=1)
    predicted_returns = beta_val * excess_factor
    residuals = excess_returns - predicted_returns

    return np.std(residuals, ddof=1) * np.sqrt(252)


def var_excess_return(returns, cutoff=0.05):
    """Calculate the mean excess return in the VaR tail."""
    if len(returns) < 2:
        return np.nan

    var_value = value_at_risk(returns, cutoff)
    excess_returns = returns[returns <= var_value]

    if len(excess_returns) == 0:
        return np.nan

    return np.mean(excess_returns)


def var_cov_var_normal(p, c, mu=0, sigma=1):
    """Calculate parametric Value at Risk using the normal distribution."""
    alpha = stats.norm.ppf(1 - c, mu, sigma)
    return p - p * (alpha + 1)


def trading_value_at_risk(returns, period=None, sigma=2.0):
    """Calculate trading Value at Risk."""
    if len(returns) < 2:
        return np.nan

    mean_ret = np.mean(returns)
    std_ret = np.std(returns, ddof=1)

    return mean_ret - sigma * std_ret


def gpd_risk_estimates(returns, var_p=0.01):
    """Estimate VaR and ES using the Generalized Pareto Distribution (GPD)."""
    from scipy import optimize

    if len(returns) < 3:
        result = np.zeros(5)
        if isinstance(returns, pd.Series):
            result = pd.Series(result)
        return result

    result = np.zeros(5)
    default_threshold = 0.2
    minimum_threshold = 0.000000001

    try:
        returns_array = pd.Series(returns).to_numpy()
    except AttributeError:
        returns_array = pd.Series(returns).values

    flipped_returns = -1 * returns_array
    losses = flipped_returns[flipped_returns > 0]
    threshold = default_threshold
    finished = False
    scale_param = 0
    shape_param = 0
    var_estimate = 0

    def gpd_loglikelihood(params, price_data):
        if params[1] != 0:
            return -_gpd_loglikelihood_scale_and_shape(params[0], params[1], price_data)
        else:
            return -_gpd_loglikelihood_scale_only(params[0], price_data)

    def _gpd_loglikelihood_scale_and_shape(scale, shape, price_data):
        n = len(price_data)
        result = -1 * float_info.max
        if scale != 0:
            param_factor = shape / scale
            if shape != 0 and param_factor >= 0 and scale >= 0:
                result = (-n * np.log(scale)) - (
                    ((1 / shape) + 1) * (np.log((shape / scale * price_data) + 1)).sum()
                )
        return result

    def _gpd_loglikelihood_scale_only(scale, price_data):
        n = len(price_data)
        data_sum = price_data.sum()
        result = -1 * float_info.max
        if scale >= 0:
            result = (-n * np.log(scale)) - (data_sum / scale)
        return result

    def _gpd_var_calculator(threshold, scale_param, shape_param, probability, total_n, exceedance_n):
        result = 0
        if exceedance_n > 0 and shape_param > 0:
            param_ratio = scale_param / shape_param
            prob_ratio = (total_n / exceedance_n) * probability
            result = threshold + (param_ratio * (pow(prob_ratio, -shape_param) - 1))
        return result

    def _gpd_es_calculator(var_estimate, threshold, scale_param, shape_param):
        result = 0
        if (1 - shape_param) != 0:
            var_ratio = var_estimate / (1 - shape_param)
            param_ratio = (scale_param - (shape_param * threshold)) / (1 - shape_param)
            result = var_ratio + param_ratio
        return result

    while not finished and threshold > minimum_threshold:
        losses_beyond_threshold = losses[losses >= threshold]
        if len(losses_beyond_threshold) > 0:
            try:
                optimization_results = optimize.minimize(
                    lambda params: gpd_loglikelihood(params, losses_beyond_threshold),
                    np.array([1, 1]),
                    method="Nelder-Mead",
                )
                if optimization_results.success:
                    resulting_params = optimization_results.x
                    if len(resulting_params) == 2:
                        scale_param = resulting_params[0]
                        shape_param = resulting_params[1]
                        var_estimate = _gpd_var_calculator(
                            threshold, scale_param, shape_param, var_p,
                            len(losses), len(losses_beyond_threshold)
                        )
                        if shape_param > 0 and var_estimate > 0:
                            finished = True
            except Exception:
                pass
        if not finished:
            threshold = threshold / 2

    if finished:
        es_estimate = _gpd_es_calculator(var_estimate, threshold, scale_param, shape_param)
        result = np.array([threshold, scale_param, shape_param, var_estimate, es_estimate])

    if isinstance(returns, pd.Series):
        result = pd.Series(result)
    return result


def gpd_risk_estimates_aligned(returns, var_p=0.01):
    """Calculate GPD risk estimates (aligned version for compatibility)."""
    return gpd_risk_estimates(returns, var_p)


def beta_fragility_heuristic(returns, factor_returns):
    """Estimate fragility to a drop in beta."""
    if len(returns) < 3 or len(factor_returns) < 3:
        return np.nan

    returns_aligned, factor_aligned = aligned_series(returns, factor_returns)

    returns_aligned = list(returns_aligned)
    factor_aligned = list(factor_aligned)

    if len(returns_aligned) < 3 or len(factor_aligned) < 3:
        return np.nan

    returns_series = pd.Series(returns_aligned)
    factor_returns_series = pd.Series(factor_aligned)
    pairs = pd.concat([returns_series, factor_returns_series], axis=1)
    pairs.columns = ["returns", "factor_returns"]
    pairs = pairs.dropna()
    pairs = pairs.sort_values(by=["factor_returns"], kind="mergesort")

    start_index = 0
    mid_index = int(np.around(len(pairs) / 2, 0))
    end_index = len(pairs) - 1

    (start_returns, start_factor_returns) = pairs.iloc[start_index]
    (mid_returns, mid_factor_returns) = pairs.iloc[mid_index]
    (end_returns, end_factor_returns) = pairs.iloc[end_index]

    factor_returns_range = end_factor_returns - start_factor_returns
    start_returns_weight = 0.5
    end_returns_weight = 0.5

    if not factor_returns_range == 0:
        start_returns_weight = (mid_factor_returns - start_factor_returns) / factor_returns_range
        end_returns_weight = (end_factor_returns - mid_factor_returns) / factor_returns_range

    heuristic = (
        (start_returns_weight * start_returns)
        + (end_returns_weight * end_returns)
        - mid_returns
    )

    return heuristic


def beta_fragility_heuristic_aligned(returns, factor_returns):
    """Calculate the beta fragility heuristic with aligned series."""
    return beta_fragility_heuristic(returns, factor_returns)
