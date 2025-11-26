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

"""绩效统计函数模块."""

import numpy as np
import pandas as pd
from collections import OrderedDict
from fincore.constants import DAILY
from fincore.empyricals.yearly import annual_return
from fincore.empyricals.returns import cum_returns_final
from fincore.empyricals.risk import annual_volatility
from fincore.empyricals.ratios import sharpe_ratio, sortino_ratio, calmar_ratio
from fincore.empyricals.drawdown import max_drawdown
from fincore.empyricals.stats import skewness, kurtosis

__all__ = [
    'perf_stats',
    'perf_stats_bootstrap',
    'calc_bootstrap',
    'calc_distribution_stats',
]


def perf_stats(returns, factor_returns=None, positions=None, transactions=None,
               turnover_denom='AGB', period=DAILY):
    """Calculate various performance metrics of a strategy.

    Parameters
    ----------
    returns : pd.Series
        Daily returns of the strategy, noncumulative.
    factor_returns : pd.Series, optional
        Daily noncumulative returns of the benchmark factor to which betas are
        computed. Usually a benchmark such as market returns.
    positions : pd.DataFrame, optional
        Daily net position values.
    transactions : pd.DataFrame, optional
        Prices and amounts of executed trades. One row per trade.
    turnover_denom : str, optional
        Either 'AGB' or 'portfolio_value', default 'AGB'.
    period : str, optional
        Frequency of the input data (for example ``DAILY``).

    Returns
    -------
    pd.Series
        Performance metrics.
    """
    stats = OrderedDict()

    stats['Annual return'] = annual_return(returns, period=period)
    stats['Cumulative returns'] = cum_returns_final(returns, starting_value=0)
    stats['Annual volatility'] = annual_volatility(returns, period=period)
    stats['Sharpe ratio'] = sharpe_ratio(returns, period=period)
    stats['Calmar ratio'] = calmar_ratio(returns, period=period)
    stats['Stability'] = None  # Placeholder
    stats['Max drawdown'] = max_drawdown(returns)
    stats['Omega ratio'] = None  # Placeholder
    stats['Sortino ratio'] = sortino_ratio(returns, period=period)
    stats['Skew'] = skewness(returns)
    stats['Kurtosis'] = kurtosis(returns)
    stats['Tail ratio'] = None  # Placeholder
    stats['Daily value at risk'] = None  # Placeholder

    if factor_returns is not None:
        from fincore.empyricals.alpha_beta import alpha, beta
        stats['Alpha'] = alpha(returns, factor_returns)
        stats['Beta'] = beta(returns, factor_returns)

    return pd.Series(stats)


def perf_stats_bootstrap(returns, factor_returns=None, return_stats=True, num_samples=1000):
    """Calculate various bootstrapped performance metrics of a strategy.

    Parameters
    ----------
    returns : pd.Series
        Daily returns of the strategy, noncumulative.
    factor_returns : pd.Series, optional
        Daily noncumulative returns of the benchmark factor.
    return_stats : bool, optional
        If True, returns summary statistics (mean, median, percentiles).
    num_samples : int, optional
        Number of bootstrap samples to draw.

    Returns
    -------
    dict
        Bootstrap statistics for each performance metric.
    """
    bootstrap_stats = {}

    def _resolve_stat_func(spec):
        if callable(spec):
            return spec
        stat_funcs = {
            'annual_return': annual_return,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'annual_volatility': annual_volatility,
        }
        return stat_funcs.get(spec, lambda x: np.nan)

    stat_specs = ['annual_return', 'sharpe_ratio', 'max_drawdown', 'annual_volatility']

    for spec in stat_specs:
        func = _resolve_stat_func(spec)
        bootstrap_result = calc_bootstrap(func, returns, num_samples=num_samples)
        bootstrap_stats[spec] = bootstrap_result

    return bootstrap_stats


def calc_bootstrap(func, returns, *args, num_samples=1000, **kwargs):
    """Calculate bootstrap distribution for a statistic.

    Parameters
    ----------
    func : callable
        Function to compute the statistic.
    returns : pd.Series or array-like
        Return series to bootstrap.
    num_samples : int, optional
        Number of bootstrap samples.

    Returns
    -------
    dict
        Dictionary with 'mean', 'std', 'ci_low', 'ci_high'.
    """
    n = len(returns)
    bootstrap_values = []

    for _ in range(num_samples):
        indices = np.random.choice(n, size=n, replace=True)
        sample_returns = returns.iloc[indices] if hasattr(returns, 'iloc') else returns[indices]
        try:
            value = func(sample_returns, *args, **kwargs)
            bootstrap_values.append(value)
        except Exception:
            bootstrap_values.append(np.nan)

    bootstrap_values = np.array(bootstrap_values)
    valid_values = bootstrap_values[~np.isnan(bootstrap_values)]

    if len(valid_values) == 0:
        return {
            'mean': np.nan,
            'std': np.nan,
            'ci_low': np.nan,
            'ci_high': np.nan,
        }

    return {
        'mean': np.mean(valid_values),
        'std': np.std(valid_values),
        'ci_low': np.percentile(valid_values, 2.5),
        'ci_high': np.percentile(valid_values, 97.5),
    }


def calc_distribution_stats(x):
    """Calculate distribution statistics.

    Parameters
    ----------
    x : array-like
        Data array.

    Returns
    -------
    dict
        Dictionary with 'mean', 'median', 'std', 'min', 'max', 'count'.
    """
    x = np.asarray(x)
    x = x[~np.isnan(x)]

    if len(x) == 0:
        return {
            'mean': np.nan,
            'median': np.nan,
            'std': np.nan,
            'min': np.nan,
            'max': np.nan,
            'count': 0,
        }

    return {
        'mean': np.mean(x),
        'median': np.median(x),
        'std': np.std(x),
        'min': np.min(x),
        'max': np.max(x),
        'count': len(x),
    }
