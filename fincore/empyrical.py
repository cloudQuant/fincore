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

"""
Empyrical - 金融性能分析库

包含原有的所有empyrical函数，以及新的面向对象Empyrical类。
"""
from __future__ import division
import warnings
import pandas as pd
import numpy as np
import math
from scipy import stats
import scipy as sp
from six import iteritems
from sys import float_info
import pymc as pm
from fincore.utils import nanmean, nanstd, nanmin, nanmax, nanargmax, nanargmin
from fincore.utils import up, down, rolling_window, roll
from fincore.constants import *
from sklearn import linear_model
from collections import OrderedDict, deque
from functools import partial

try:
    from zipline.assets import Equity, Future

    ZIPLINE = True
except ImportError:
    ZIPLINE = False
    warnings.warn(
        'Module "zipline.assets" not found; mutltipliers will not be applied' +
        ' to position notionals.'
    )


class Empyrical:
    """
    面向对象的性能指标计算类
    
    这个类将所有empyrical模块的函数封装为类方法，提供统一的数据管理和计算接口。
    初始化参数与pyfolio的create_full_tear_sheet函数参数保持一致。
    
    通过直接调用原有函数确保100%的计算一致性。
    """

    @classmethod
    def _get_returns(cls, returns):
        return returns

    @classmethod
    def _get_factor_returns(cls, factor_return):
        return factor_return

    # ================================
    # 计算方法（包装原有函数）
    # ================================
    @staticmethod
    def _ensure_datetime_index_series(data, period=DAILY):
        """Return a Series indexed by dates regardless of the input type."""
        if isinstance(data, pd.Series) and isinstance(data.index, pd.DatetimeIndex):
            return data

        values = data.values if isinstance(data, pd.Series) else np.asarray(data)

        if values.size == 0:
            return pd.Series(values)

        freq = PERIOD_TO_FREQ.get(period, "D")
        index = pd.date_range("1970-01-01", periods=values.size, freq=freq)
        return pd.Series(values, index=index)

    @staticmethod
    def _flatten(arr):
        return arr if not isinstance(arr, pd.Series) else arr.values

    @staticmethod
    def _adjust_returns(returns, adjustment_factor):
        """Returns the `returns` series adjusted by adjustment_factor."""
        if isinstance(adjustment_factor, (float, int)) and adjustment_factor == 0:
            return returns
        return returns - adjustment_factor

    @staticmethod
    def annualization_factor(period, annualization):
        """Return annualization factor from the period entered or if a custom value is passed in."""
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

    @staticmethod
    def _to_pandas(ob):
        """Convert an array-like to a `pandas` object."""
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

    @staticmethod
    def _aligned_series(*many_series):
        """Return a new list of series with their indices aligned."""
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
            for _, v in iteritems(pd.concat(map(Empyrical._to_pandas, many_series), axis=1))
        )

    @classmethod
    def simple_returns(cls, prices):
        """Compute simple returns from a timeseries of prices."""
        if isinstance(prices, (pd.DataFrame, pd.Series)):
            out = prices.pct_change().iloc[1:]
        else:
            # Assume np.ndarray
            out = np.diff(prices, axis=0)
            # Avoid division by zero warning
            with np.errstate(divide='ignore', invalid='ignore'):
                np.divide(out, prices[:-1], out=out)

        return out

    @classmethod
    def cum_returns(cls, returns, starting_value=0, out=None):
        """Compute cumulative returns from simple returns."""
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

    @classmethod
    def cum_returns_final(cls, returns, starting_value=0):
        """Compute total returns from simple returns."""
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

    @classmethod
    def aggregate_returns(cls, returns, convert_to='monthly'):
        """Aggregates returns by week, month, or year."""

        def cumulate_returns(x):
            return cls.cum_returns(x).iloc[-1]

        if convert_to == WEEKLY:
            grouping = [lambda x: x.year, lambda x: x.isocalendar()[1]]
        elif convert_to == MONTHLY:
            grouping = [lambda x: x.year, lambda x: x.month]
        elif convert_to == QUARTERLY:
            grouping = [lambda x: x.year, lambda x: int(math.ceil(x.month / 3.))]
        elif convert_to == YEARLY:
            grouping = [lambda x: x.year]
        else:
            raise ValueError(
                'convert_to must be {}, {} or {}'.format(WEEKLY, MONTHLY, YEARLY)
            )

        return returns.groupby(grouping).apply(cumulate_returns)

    @classmethod
    def max_drawdown(cls, returns, out=None):
        """Determines the maximum drawdown of a strategy."""
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
        cls.cum_returns(returns_array, starting_value=start, out=cumulative[1:])

        max_return = np.fmax.accumulate(cumulative, axis=0)

        nanmin((cumulative - max_return) / max_return, axis=0, out=out)
        if returns_1d:
            out = out.item()
        elif allocated_output and isinstance(returns, pd.DataFrame):
            out = pd.Series(out)

        return out

    @classmethod
    def annual_return(cls, returns, period=DAILY, annualization=None):
        """Determines the mean annual growth rate of returns."""
        if len(returns) < 1:
            return np.nan

        ann_factor = cls.annualization_factor(period, annualization)
        num_years = len(returns) / ann_factor
        # Pass array to ensure index -1 looks up successfully.
        ending_value = cls.cum_returns_final(returns, starting_value=1)

        return ending_value ** (1 / num_years) - 1

    @classmethod
    def cagr(cls, returns, period=DAILY, annualization=None):
        """Compute compound annual growth rate."""
        return cls.annual_return(returns, period, annualization)

    @classmethod
    def annual_volatility(cls, returns, period=DAILY, alpha_=2.0, annualization=None, out=None):
        """Determines the annual volatility of a strategy."""
        allocated_output = out is None
        if allocated_output:
            out = np.empty(returns.shape[1:])

        returns_1d = returns.ndim == 1

        if len(returns) < 2:
            out[()] = np.nan
            if returns_1d:
                out = out.item()
            return out

        ann_factor = cls.annualization_factor(period, annualization)
        nanstd(returns, ddof=1, axis=0, out=out)
        out = np.multiply(out, ann_factor ** (1.0 / alpha_), out=out)
        if returns_1d:
            out = out.item()
        return out

    @classmethod
    def calmar_ratio(cls, returns, period=DAILY, annualization=None):
        """Determines the Calmar ratio, or drawdown ratio, of a strategy."""
        max_dd = cls.max_drawdown(returns=returns)
        if max_dd < 0:
            temp = cls.annual_return(
                returns=returns,
                period=period,
                annualization=annualization
            ) / abs(max_dd)
        else:
            return np.nan

        if np.isinf(temp):
            return np.nan

        return temp

    @classmethod
    def omega_ratio(cls, returns, risk_free=0.0, required_return=0.0, annualization=APPROX_BDAYS_PER_YEAR):
        """Determines the Omega ratio of a strategy."""
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

    @classmethod
    def sharpe_ratio(cls, returns, risk_free=0, period=DAILY, annualization=None, out=None):
        """Determines the Sharpe ratio of a strategy."""
        allocated_output = out is None
        if allocated_output:
            out = np.empty(returns.shape[1:])

        return_1d = returns.ndim == 1

        if len(returns) < 2:
            out[()] = np.nan
            if return_1d:
                out = out.item()
            return out

        returns_risk_adj = np.asanyarray(cls._adjust_returns(returns, risk_free))
        ann_factor = cls.annualization_factor(period, annualization)

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

    # 实例方法，调用对应的类方法（保持向后兼容）
    @classmethod
    def alpha_beta(cls, returns, factor_returns, risk_free=0.0, period=DAILY, annualization=None, out=None):
        """Calculates annualized alpha and beta."""
        # Match original empyrical.stats.alpha_beta behaviour: align series
        # first, then delegate to the aligned implementation.
        if not (isinstance(returns, np.ndarray) and isinstance(factor_returns, np.ndarray)):
            returns, factor_returns = cls._aligned_series(returns, factor_returns)

        return cls.alpha_beta_aligned(
            returns,
            factor_returns,
            risk_free=risk_free,
            period=period,
            annualization=annualization,
            out=out,
        )

    @classmethod
    def alpha(cls, returns, factor_returns, risk_free=0.0, period=DAILY, annualization=None, out=None, _beta=None):
        """Calculates annualized alpha.

        This mirrors empyrical.stats.alpha, which internally calls
        alpha_aligned after aligning non-ndarray inputs.
        """
        if not (isinstance(returns, np.ndarray) and isinstance(factor_returns, np.ndarray)):
            returns, factor_returns = cls._aligned_series(returns, factor_returns)

        return cls.alpha_aligned(
            returns,
            factor_returns,
            risk_free=risk_free,
            period=period,
            annualization=annualization,
            out=out,
            _beta=_beta,
        )

    @classmethod
    def beta(cls, returns, factor_returns, risk_free=0.0, period=DAILY, annualization=None, out=None):
        """Calculates beta.

        This mirrors empyrical.stats.beta, which forwards to beta_aligned
        after aligning non-ndarray inputs.
        """
        if not (isinstance(returns, np.ndarray) and isinstance(factor_returns, np.ndarray)):
            returns, factor_returns = cls._aligned_series(returns, factor_returns)

        return cls.beta_aligned(
            returns,
            factor_returns,
            risk_free=risk_free,
            out=out,
        )

    @classmethod
    def alpha_beta_aligned(cls, returns, factor_returns, risk_free=0.0, period=DAILY, annualization=None, out=None):
        """Calculates annualized alpha and beta for already-aligned series."""
        if out is None:
            out = np.empty(returns.shape[1:] + (2,), dtype="float64")

        b = cls.beta_aligned(returns, factor_returns, risk_free, out=out[..., 1])
        cls.alpha_aligned(
            returns,
            factor_returns,
            risk_free=risk_free,
            period=period,
            annualization=annualization,
            out=out[..., 0],
            _beta=b,
        )

        return out

    @classmethod
    def alpha_aligned(cls, returns, factor_returns, risk_free=0.0, period=DAILY, annualization=None, out=None,
                      _beta=None):
        """Calculates annualized alpha for already-aligned series."""
        allocated_output = out is None
        if allocated_output:
            out = np.empty(returns.shape[1:], dtype="float64")

        if len(returns) < 2:
            out[()] = np.nan
            if returns.ndim == 1:
                out = out.item()
            return out

        ann_factor = cls.annualization_factor(period, annualization)

        if _beta is None:
            _beta = cls.beta_aligned(returns, factor_returns, risk_free)

        adj_returns = cls._adjust_returns(returns, risk_free)
        adj_factor_returns = cls._adjust_returns(factor_returns, risk_free)
        alpha_series = adj_returns - (_beta * adj_factor_returns)

        # Match original alpha_aligned: compound the average excess return.
        out = np.subtract(
            np.power(
                np.add(
                    nanmean(alpha_series, axis=0, out=out),
                    1,
                    out=out,
                ),
                ann_factor,
                out=out,
            ),
            1,
            out=out,
        )

        if allocated_output and isinstance(returns, pd.DataFrame):
            out = pd.Series(out)

        if returns.ndim == 1:
            out = out.item()

        return out

    @classmethod
    def beta_aligned(cls, returns, factor_returns, risk_free=0.0, out=None):
        """Calculates beta for already-aligned data (equivalent to beta_aligned)."""
        nanmean_local = np.nanmean
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
        independent = np.where(
            isnan(returns),
            nan,
            factor_returns,
        )

        ind_residual = independent - nanmean_local(independent, axis=0)

        covariances = nanmean_local(ind_residual * returns, axis=0)

        # Calculate independent variances
        np.square(ind_residual, out=ind_residual)
        independent_variances = nanmean_local(ind_residual, axis=0)
        independent_variances[independent_variances < 1.0e-30] = np.nan

        np.divide(covariances, independent_variances, out=out)

        if returns_1d:
            out = out.item()

        return out

    @classmethod
    def sortino_ratio(cls, returns, required_return=0, period=DAILY, annualization=None, out=None,
                      _downside_risk=None):
        """Determines the Sortino ratio of a strategy."""
        allocated_output = out is None
        if allocated_output:
            out = np.empty(returns.shape[1:])

        return_1d = returns.ndim == 1

        if len(returns) < 2:
            out[()] = np.nan
            if return_1d:
                out = out.item()
            return out

        adj_returns = np.asanyarray(cls._adjust_returns(returns, required_return))

        ann_factor = cls.annualization_factor(period, annualization)

        average_annual_return = nanmean(adj_returns, axis=0) * ann_factor
        annualized_downside_risk = (
            _downside_risk
            if _downside_risk is not None else
            cls.downside_risk(returns, required_return, period, annualization)
        )
        # Avoid division by zero warning
        with np.errstate(divide='ignore', invalid='ignore'):
            np.divide(average_annual_return, annualized_downside_risk, out=out)
        if return_1d:
            out = out.item()
        elif isinstance(returns, pd.DataFrame):
            out = pd.Series(out)

        return out

    @classmethod
    def downside_risk(cls, returns, required_return=0, period=DAILY, annualization=None, out=None):
        """Determines the downside deviation below a threshold."""
        allocated_output = out is None
        if allocated_output:
            out = np.empty(returns.shape[1:])

        returns_1d = returns.ndim == 1

        if len(returns) < 1:
            out[()] = np.nan
            if returns_1d:
                out = out.item()
            return out

        ann_factor = cls.annualization_factor(period, annualization)

        downside_diff = np.clip(
            cls._adjust_returns(
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

    @classmethod
    def excess_sharpe(cls, returns, factor_returns, out=None):
        """Determines the Excess Sharpe of a strategy."""
        allocated_output = out is None
        if allocated_output:
            out = np.empty(returns.shape[1:])

        returns_1d = returns.ndim == 1

        if len(returns) < 2:
            out[()] = np.nan
            if returns_1d:
                out = out.item()
            return out

        active_return = cls._adjust_returns(returns, factor_returns)
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

    @classmethod
    def tracking_error(cls, returns, factor_returns, period=DAILY, annualization=None, out=None):
        """Determines the tracking error of returns relative to factor returns."""
        allocated_output = out is None
        if allocated_output:
            out = np.empty(returns.shape[1:])

        returns_1d = returns.ndim == 1

        if len(returns) < 2:
            out[()] = np.nan
            if returns_1d:
                out = out.item()
            return out

        returns, factor_returns = cls._aligned_series(returns, factor_returns)

        active_return = cls._adjust_returns(returns, factor_returns)
        ann_factor = cls.annualization_factor(period, annualization)

        nanstd(active_return, ddof=1, axis=0, out=out)
        np.multiply(out, np.sqrt(ann_factor), out=out)

        if returns_1d:
            out = out.item()

        return out

    @classmethod
    def information_ratio(cls, returns, factor_returns, period=DAILY, annualization=None):
        """Determines the information ratio of returns relative to factor returns."""
        returns, factor_returns = cls._aligned_series(returns, factor_returns)
        super_returns = returns - factor_returns

        ann_factor = cls.annualization_factor(period, annualization)
        mean_excess_return = super_returns.mean()
        std_excess_return = super_returns.std(ddof=1)
        ir = (mean_excess_return * ann_factor) / (std_excess_return * np.sqrt(ann_factor))
        return ir

    @classmethod
    def value_at_risk(cls, returns, cutoff=0.05):
        """Calculates the daily value at risk (VaR) of returns."""
        if len(returns) < 1:
            return np.nan

        return np.percentile(returns, cutoff * 100)

    @classmethod
    def conditional_value_at_risk(cls, returns, cutoff=0.05):
        """Calculates the conditional value at risk (CVaR) of returns."""
        if len(returns) < 1:
            return np.nan

        cutoff_index = cls.value_at_risk(returns, cutoff=cutoff)
        return np.mean(returns[returns <= cutoff_index])

    @classmethod
    def tail_ratio(cls, returns):
        """Determines the ratio between the right (95th) and left (5th) percentile of the returns."""
        if len(returns) < 1:
            return np.nan

        returns = np.asanyarray(returns)
        # Be tolerant of nan's
        returns = returns[~np.isnan(returns)]
        if len(returns) < 1:
            return np.nan

        return np.abs(np.percentile(returns, 95)) / np.abs(np.percentile(returns, 5))

    @classmethod
    def stability_of_timeseries(cls, returns):
        """Determines R-squared of a linear fit to the cumulative log returns."""
        if len(returns) < 2:
            return np.nan

        returns = np.asanyarray(returns)
        returns = returns[~np.isnan(returns)]

        if len(returns) < 2:
            return np.nan

        cum_log_returns = np.log1p(returns).cumsum()
        rhat = stats.linregress(np.arange(len(cum_log_returns)), cum_log_returns)[2]

        return rhat ** 2

    @classmethod
    def capture(cls, returns, factor_returns, period=DAILY):
        """Calculates the capture ratio."""
        if len(returns) < 1 or len(factor_returns) < 1:
            return np.nan

        strategy_ann_return = cls.annual_return(returns, period=period)
        benchmark_ann_return = cls.annual_return(factor_returns, period=period)

        if benchmark_ann_return == 0:
            return np.nan

        return strategy_ann_return / benchmark_ann_return

    @classmethod
    def up_capture(cls, returns, factor_returns, period=DAILY):
        """Calculates the capture ratio for periods when the benchmark return is positive."""
        returns, factor_returns = cls._aligned_series(returns, factor_returns)

        up_returns = returns[factor_returns > 0]
        up_factor_returns = factor_returns[factor_returns > 0]

        if len(up_returns) < 1:
            return np.nan

        return cls.capture(up_returns, up_factor_returns, period=period)

    @classmethod
    def down_capture(cls, returns, factor_returns, period=DAILY):
        """Calculates the capture ratio for periods when the benchmark return is negative."""
        returns, factor_returns = cls._aligned_series(returns, factor_returns)

        down_returns = returns[factor_returns < 0]
        down_factor_returns = factor_returns[factor_returns < 0]

        if len(down_returns) < 1:
            return np.nan

        return cls.capture(down_returns, down_factor_returns, period=period)

    @classmethod
    def up_down_capture(cls, returns, factor_returns, period=DAILY):
        """Calculates the up and down capture ratios."""
        up_cap = cls.up_capture(returns, factor_returns, period=period)
        down_cap = cls.down_capture(returns, factor_returns, period=period)
        return up_cap, down_cap

    @classmethod
    def perf_attrib(cls,
                    returns=None,
                    positions=None,
                    factor_returns=None,
                    factor_loadings=None):
        returns = cls._get_returns(returns)
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
        factor_loadings.index = factor_loadings.index.set_names(['dt', 'ticker'])
        positions = positions.copy()
        positions.index = positions.index.set_names(['dt', 'ticker'])

        risk_exposures_portfolio = cls.compute_exposures(
            positions=positions,
            factor_loadings=factor_loadings,
        )
        risk_exposures_portfolio.index = returns.index

        perf_attrib_by_factor = risk_exposures_portfolio.multiply(factor_returns)
        common_returns = perf_attrib_by_factor.sum(axis='columns')
        tilt_exposure = risk_exposures_portfolio.mean()
        tilt_returns = factor_returns.multiply(tilt_exposure).sum(axis='columns')
        timing_returns = common_returns - tilt_returns
        specific_returns = returns - common_returns

        returns_df = pd.DataFrame(OrderedDict([
            ('total_returns', returns),
            ('common_returns', common_returns),
            ('specific_returns', specific_returns),
            ('tilt_returns', tilt_returns),
            ('timing_returns', timing_returns),
        ]))

        perf_attribution = pd.concat([perf_attrib_by_factor, returns_df],
                                     axis='columns')
        perf_attribution.index = returns.index

        return risk_exposures_portfolio, perf_attribution

    @classmethod
    def _compute_exposures(cls, positions=None, factor_loadings=None):
        if positions is None:
            raise ValueError("Either provide positions or set positions data")
        if factor_loadings is None:
            raise ValueError("Either provide factor_loadings or set factor_loadings data")
        risk_exposures = factor_loadings.multiply(positions, axis='rows')
        return risk_exposures.groupby(level='dt').sum()

    @classmethod
    def cal_treynor_ratio(cls, returns, factor_returns, risk_free=0.0, period=DAILY, annualization=None):
        """Calculates the Treynor ratio.
        """
        allocated_output = True
        out = np.empty(returns.shape[1:])

        returns_1d = returns.ndim == 1

        if len(returns) < 2:
            out[()] = np.nan
            if returns_1d:
                out = out.item()
            return out

        returns, factor_returns = cls._aligned_series(returns, factor_returns)

        # Annualized excess return
        ann_return = cls.annual_return(returns, period=period, annualization=annualization)
        ann_excess_return = ann_return - risk_free

        # Beta
        b = cls.beta_aligned(returns, factor_returns, risk_free)

        if returns_1d:
            if b == 0 or b < 0 or np.isnan(b):
                out = np.nan
            else:
                with np.errstate(divide='ignore', invalid='ignore'):
                    out[()] = ann_excess_return / b
                out = out.item()
        else:
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

    @classmethod
    def treynor_ratio(cls, returns=None, factor_returns=None, risk_free=0.0, period=DAILY, annualization=None):
        """Calculates the Treynor ratio."""
        returns = cls._get_returns(returns)
        factor_returns = cls._get_factor_returns(factor_returns)
        return cls.cal_treynor_ratio(returns, factor_returns, risk_free, period, annualization)

    @classmethod
    def m_squared(cls, returns, factor_returns, risk_free=0.0, period=DAILY, annualization=None):
        """Calculates the M-squared (M²) measure."""
        if len(returns) < 2:
            return np.nan

        returns_aligned, factor_aligned = cls._aligned_series(returns, factor_returns)

        # Calculate annualized returns and volatilities  
        ann_return = cls.annual_return(returns_aligned, period=period, annualization=annualization)
        ann_vol = cls.annual_volatility(returns_aligned, period=period, annualization=annualization)
        ann_factor_return = cls.annual_return(factor_aligned, period=period, annualization=annualization)
        ann_factor_vol = cls.annual_volatility(factor_aligned, period=period, annualization=annualization)

        # Handle division by zero or negative volatility
        if ann_vol == 0 or ann_vol < 0 or np.isnan(ann_vol):
            return np.nan

        # M² = (Rp - Rf) * (σb / σp) + Rf
        excess_return = ann_return - risk_free
        risk_ratio = ann_factor_vol / ann_vol
        return excess_return * risk_ratio + risk_free

    @classmethod
    def annual_return_by_year(cls, returns, period=DAILY, annualization=None):
        """Determines the annual return for each year."""
        if len(returns) < 1:
            return_as_array = isinstance(returns, np.ndarray)
            return np.array([]) if return_as_array else pd.Series(dtype='float64')

        return_as_array = isinstance(returns, np.ndarray)

        # Ensure we have a datetime-indexed Series
        returns = cls._ensure_datetime_index_series(returns, period=period)

        annual_returns = returns.groupby(returns.index.year).apply(
            lambda x: cls.annual_return(x, period=period, annualization=annualization)
        )

        return annual_returns.values if return_as_array else annual_returns

    @classmethod
    def sharpe_ratio_by_year(cls, returns, risk_free=0, period=DAILY, annualization=None):
        """Determines the Sharpe ratio for each year."""
        if len(returns) < 1:
            return_as_array = isinstance(returns, np.ndarray)
            return np.array([]) if return_as_array else pd.Series(dtype='float64')

        return_as_array = isinstance(returns, np.ndarray)

        returns = cls._ensure_datetime_index_series(returns, period=period)

        sharpe_by_year = returns.groupby(returns.index.year).apply(
            lambda x: cls.sharpe_ratio(x, risk_free=risk_free, period=period, annualization=annualization)
        )

        return sharpe_by_year.values if return_as_array else sharpe_by_year

    @classmethod
    def max_drawdown_by_year(cls, returns):
        """Determines the maximum drawdown for each year."""
        if len(returns) < 1:
            return_as_array = isinstance(returns, np.ndarray)
            return np.array([]) if return_as_array else pd.Series(dtype='float64')

        return_as_array = isinstance(returns, np.ndarray)

        returns = cls._ensure_datetime_index_series(returns, period=DAILY)

        max_dd_by_year = returns.groupby(returns.index.year).apply(
            lambda x: cls.max_drawdown(x)
        )
        return max_dd_by_year.values if return_as_array else max_dd_by_year

    @classmethod
    def skewness(cls, returns):
        """Calculates the skewness of the returns."""
        if len(returns) < 3:
            return np.nan

        return stats.skew(returns, nan_policy='omit')

    @classmethod
    def kurtosis(cls, returns):
        """Calculates the kurtosis of the returns."""
        if len(returns) < 4:
            return np.nan

        return stats.kurtosis(returns, nan_policy='omit')

    @classmethod
    def hurst_exponent(cls, returns):
        """Calculates the Hurst exponent of the returns."""
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
                    sub_series = returns_clean[i * lag:(i + 1) * lag]
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

        except Exception as e:
            print(e)
            return np.nan

    @classmethod
    def sterling_ratio(cls, returns, risk_free=0.0, period=DAILY, annualization=None):
        """Calculates the Sterling ratio."""
        if len(returns) < 2:
            return np.nan

        # Create temporary instance to use instance methods
        temp_instance = cls()

        # Get all drawdowns
        drawdown_periods = temp_instance._get_all_drawdowns(returns)

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
            ann_ret = cls.annual_return(returns, period, annualization)
            return np.inf if ann_ret - risk_free > 0 else np.nan

        # Calculate annualized return
        ann_ret = cls.annual_return(returns, period, annualization)

        # Sterling ratio = (annualized return - risk free) / average drawdown
        return (ann_ret - risk_free) / avg_drawdown

    @classmethod
    def burke_ratio(cls, returns, risk_free=0.0, period=DAILY, annualization=None):
        """Calculates the Burke ratio."""
        if len(returns) < 2:
            return np.nan

        # Create temporary instance to use instance methods
        temp_instance = cls()

        # Get all drawdowns
        drawdown_periods = temp_instance._get_all_drawdowns(returns)

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
            squared_drawdowns = [dd ** 2 for dd in drawdown_periods]
            burke_risk = np.sqrt(np.sum(squared_drawdowns))

        if burke_risk == 0 or burke_risk < 1e-10:
            # Extremely small risk, return large positive ratio or NaN
            ann_ret = cls.annual_return(returns, period, annualization)
            return np.inf if ann_ret - risk_free > 0 else np.nan

        # Calculate annualized return
        ann_ret = cls.annual_return(returns, period, annualization)

        # Burke ratio = (annualized return - risk free) / burke risk
        return (ann_ret - risk_free) / burke_risk

    @classmethod
    def kappa_three_ratio(cls, returns, risk_free=0.0, period=DAILY, annualization=None, mar=0.0):
        """Calculates the Kappa 3 ratio (downside deviation cubed)."""
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
        ann_factor = cls.annualization_factor(period, annualization)

        if lpm3 == 0 or lpm3 < 1e-30:
            # No downside risk, use standard deviation as alternative
            std_dev = np.std(returns_clean)
            if std_dev < 1e-10:
                # Very low risk, return large positive ratio if returns are positive
                ann_ret = cls.annual_return(returns, period, annualization)
                return np.inf if ann_ret - risk_free > 0 else np.nan
            lpm3_risk = std_dev * np.sqrt(ann_factor)
        else:
            lpm3_annualized = lpm3 * ann_factor
            # Take cube root of LPM3
            lpm3_risk = lpm3_annualized ** (1.0 / 3.0)

        if lpm3_risk == 0 or lpm3_risk < 1e-10:
            # Extremely small risk
            ann_ret = cls.annual_return(returns, period, annualization)
            return np.inf if ann_ret - risk_free > 0 else np.nan

        # Calculate annualized return
        ann_ret = cls.annual_return(returns, period, annualization)

        # Kappa 3 ratio = (annualized return - risk free) / LPM3^(1/3)
        return (ann_ret - risk_free) / lpm3_risk

    @classmethod
    def adjusted_sharpe_ratio(cls, returns, risk_free=0.0):
        """Calculates the adjusted Sharpe ratio (accounts for skewness and kurtosis)."""
        if len(returns) < 4:
            return np.nan

        sharpe = cls.sharpe_ratio(returns, risk_free)

        if np.isnan(sharpe):
            return np.nan

        # Calculate skewness and kurtosis with NaN handling
        skew = cls.skewness(returns)
        if np.isnan(skew):
            skew = 0

        kurt = cls.kurtosis(returns)
        if np.isnan(kurt):
            kurt = 0

        # Apply adjustment formula with dampening factor for small samples
        n = len(returns)
        dampening = min(1.0, n / 50.0)  # Full adjustment only for n >= 50

        skew_adj = (skew / 6) * sharpe * dampening
        kurt_adj = (kurt / 24) * (sharpe ** 2) * dampening
        adjustment = 1 + skew_adj - kurt_adj

        # Bound the adjustment to prevent extreme values
        if n < 20:
            adjustment = max(0.9, min(1.1, adjustment))
        else:
            adjustment = max(0.8, min(1.3, adjustment))

        return sharpe * adjustment

    @classmethod
    def stutzer_index(cls, returns, target_return=0.0):
        """Calculates the Stutzer index."""
        if len(returns) < 2:
            return np.nan

        excess_returns = returns - target_return

        if len(excess_returns) == 0:
            return np.nan

        # Use optimization to find the parameter that maximizes log likelihood
        try:
            from scipy.optimize import minimize_scalar

            def neg_log_likelihood(theta):
                if theta <= 0:
                    return np.inf
                exp_theta_r = np.exp(theta * excess_returns)
                if np.any(np.isinf(exp_theta_r)) or np.any(np.isnan(exp_theta_r)):
                    return np.inf
                return -np.log(exp_theta_r.mean()) / theta

            result = minimize_scalar(neg_log_likelihood, bounds=(1e-10, 10), method='bounded')

            if result.success:
                return -result.fun
            else:
                return np.nan
        except Exception as e:
            print(e)
            return np.nan

    @classmethod
    def max_consecutive_up_days(cls, returns):
        """Determines the maximum number of consecutive days with positive returns."""
        if len(returns) < 1:
            return np.nan

        up_days = returns > 0

        if not up_days.any():
            return 0

        # Find consecutive True values
        groups = (up_days != up_days.shift(1)).cumsum()
        consecutive_counts = up_days.groupby(groups).sum()

        return consecutive_counts.max()

    @classmethod
    def max_consecutive_down_days(cls, returns):
        """Determines the maximum number of consecutive days with negative returns."""
        if len(returns) < 1:
            return np.nan

        down_days = returns < 0

        if not down_days.any():
            return 0

        # Find consecutive True values
        groups = (down_days != down_days.shift(1)).cumsum()
        consecutive_counts = down_days.groupby(groups).sum()

        return consecutive_counts.max()

    @classmethod
    def max_consecutive_gain(cls, returns):
        """Determines the maximum consecutive gain."""
        if len(returns) < 1:
            return np.nan

        up_days = returns > 0

        if not up_days.any():
            return np.nan

        # Calculate cumulative gains for consecutive positive periods
        groups = (up_days != up_days.shift(1)).cumsum()
        consecutive_gains = returns.where(up_days, 0).groupby(groups).sum()

        return consecutive_gains.max()

    @classmethod
    def max_consecutive_loss(cls, returns):
        """Determines the maximum consecutive loss."""
        if len(returns) < 1:
            return np.nan

        down_days = returns < 0

        if not down_days.any():
            return np.nan

        # Calculate cumulative losses for consecutive negative periods
        groups = (down_days != down_days.shift(1)).cumsum()
        consecutive_losses = returns.where(down_days, 0).groupby(groups).sum()

        return consecutive_losses.min()

    @classmethod
    def max_single_day_gain(cls, returns):
        """Determines the maximum single day gain."""
        if len(returns) < 1:
            return np.nan

        return returns.max()

    @classmethod
    def max_single_day_loss(cls, returns):
        """Determines the maximum single day loss."""
        if len(returns) < 1:
            return np.nan

        return returns.min()

    @classmethod
    def stock_market_correlation(cls, returns, market_returns):
        """Determines the correlation with the stock market."""
        if len(returns) < 2 or len(market_returns) < 2:
            return np.nan

        # Align series if they are pandas objects
        if isinstance(returns, pd.Series) and isinstance(market_returns, pd.Series):
            returns, market_returns = returns.align(market_returns, join='inner')

        if len(returns) < 2:
            return np.nan

        returns_array = np.asanyarray(returns)
        market_array = np.asanyarray(market_returns)

        # Remove NaN values
        valid_mask = ~(np.isnan(returns_array) | np.isnan(market_array))
        returns_clean = returns_array[valid_mask]
        market_clean = market_array[valid_mask]

        if len(returns_clean) < 2:
            return np.nan

        # Calculate Pearson correlation
        correlation = np.corrcoef(returns_clean, market_clean)[0, 1]

        return float(correlation) if not np.isnan(correlation) else np.nan

    @classmethod
    def bond_market_correlation(cls, returns, bond_returns):
        """Determines the correlation with the bond market."""
        if len(returns) < 2 or len(bond_returns) < 2:
            return np.nan

        # Align series if they are pandas objects
        if isinstance(returns, pd.Series) and isinstance(bond_returns, pd.Series):
            returns, bond_returns = returns.align(bond_returns, join='inner')

        if len(returns) < 2:
            return np.nan

        returns_array = np.asanyarray(returns)
        bond_array = np.asanyarray(bond_returns)

        # Remove NaN values
        valid_mask = ~(np.isnan(returns_array) | np.isnan(bond_array))
        returns_clean = returns_array[valid_mask]
        bond_clean = bond_array[valid_mask]

        if len(returns_clean) < 2:
            return np.nan

        # Calculate Pearson correlation
        correlation = np.corrcoef(returns_clean, bond_clean)[0, 1]

        return float(correlation) if not np.isnan(correlation) else np.nan

    @classmethod
    def futures_market_correlation(cls, returns, futures_returns):
        """Determines the correlation with the futures market."""
        if len(returns) < 2 or len(futures_returns) < 2:
            return np.nan

        # Align series if they are pandas objects
        if isinstance(returns, pd.Series) and isinstance(futures_returns, pd.Series):
            returns, futures_returns = returns.align(futures_returns, join='inner')

        if len(returns) < 2:
            return np.nan

        returns_array = np.asanyarray(returns)
        futures_array = np.asanyarray(futures_returns)

        # Remove NaN values
        valid_mask = ~(np.isnan(returns_array) | np.isnan(futures_array))
        returns_clean = returns_array[valid_mask]
        futures_clean = futures_array[valid_mask]

        if len(returns_clean) < 2:
            return np.nan

        # Calculate Pearson correlation
        correlation = np.corrcoef(returns_clean, futures_clean)[0, 1]

        return float(correlation) if not np.isnan(correlation) else np.nan

    @classmethod
    def serial_correlation(cls, returns=None, lag=1):
        """Determines the serial correlation of returns."""
        returns = cls._get_returns(returns)
        return cls.serial_correlation(returns, lag)

    @classmethod
    def treynor_mazuy_timing(cls, returns, factor_returns, risk_free=0.0):
        """Calculates the Treynor-Mazuy market timing coefficient (gamma)."""
        returns_aligned, factor_aligned = cls._aligned_series(returns, factor_returns)

        if len(returns_aligned) < 10:
            return np.nan

        # Adjust for risk-free rate
        excess_returns = returns_aligned - risk_free
        excess_factor = factor_aligned - risk_free

        # Create the quadratic term
        factor_squared = excess_factor ** 2

        # Multiple regression: excess_returns = alpha + beta*excess_factor + gamma*factor_squared
        try:
            X = np.column_stack([np.ones(len(excess_factor)), excess_factor, factor_squared])
            coeffs = np.linalg.lstsq(X, excess_returns, rcond=None)[0]
            return coeffs[2]  # gamma coefficient
        except Exception as e:
            print(e)
            return np.nan

    @classmethod
    def henriksson_merton_timing(cls, returns, factor_returns, risk_free=0.0):
        """Calculates the Henriksson-Merton market timing coefficient."""
        returns_aligned, factor_aligned = cls._aligned_series(returns, factor_returns)

        if len(returns_aligned) < 10:
            return np.nan

        # Adjust for risk-free rate
        excess_returns = returns_aligned - risk_free
        excess_factor = factor_aligned - risk_free

        # Create the down-market dummy variable
        down_market = (excess_factor < 0).astype(float)

        # Multiple regression: excess_returns = alpha + beta*excess_factor + gamma*down_market
        try:
            X = np.column_stack([np.ones(len(excess_factor)), excess_factor, down_market])
            coeffs = np.linalg.lstsq(X, excess_returns, rcond=None)[0]
            return coeffs[2]  # gamma coefficient
        except Exception as e:
            print(e)
            return np.nan

    @classmethod
    def market_timing_return(cls, returns, factor_returns, risk_free=0.0):
        """Calculates market timing return component."""
        gamma = cls.treynor_mazuy_timing(returns, factor_returns, risk_free)

        if np.isnan(gamma):
            return np.nan

        returns_aligned, factor_aligned = cls._aligned_series(returns, factor_returns)
        excess_factor = factor_aligned - risk_free

        # Market timing return is gamma * factor_squared
        return gamma * np.mean(excess_factor ** 2)

    @classmethod
    def annual_alpha(cls, returns=None, factor_returns=None, risk_free=0.0, period=DAILY, annualization=None):
        """Determines the annual alpha for each year."""
        returns = cls._get_returns(returns)
        factor_returns = cls._get_factor_returns(factor_returns)

        if len(returns) < 1:
            return pd.Series([], dtype=float)

        # Ensure returns has a datetime index
        if not isinstance(returns.index, pd.DatetimeIndex):
            return pd.Series([], dtype=float)

        def alpha_for_year(group_data):
            year_returns = group_data[0]
            year_factor = group_data[1]
            return cls.alpha(year_returns, year_factor, risk_free, period, annualization)

        # Group by year and calculate alpha for each year
        grouped = returns.groupby(returns.index.year)
        factor_grouped = factor_returns.groupby(factor_returns.index.year)

        annual_alphas = []
        for year in grouped.groups.keys():
            if year in factor_grouped.groups.keys():
                year_returns = grouped.get_group(year)
                year_factor = factor_grouped.get_group(year)
                alpha_val = cls.alpha(year_returns, year_factor, risk_free, period, annualization)
                annual_alphas.append((year, alpha_val))

        if not annual_alphas:
            return pd.Series([], dtype=float)

        years, alphas = zip(*annual_alphas)
        return pd.Series(alphas, index=years)

    @classmethod
    def annual_beta(cls, returns=None, factor_returns=None, risk_free=0.0, period=DAILY, annualization=None):
        """Determines the annual beta for each year."""
        returns = cls._get_returns(returns)
        factor_returns = cls._get_factor_returns(factor_returns)

        if len(returns) < 1:
            return pd.Series([], dtype=float)

        # Ensure returns has a datetime index
        if not isinstance(returns.index, pd.DatetimeIndex):
            return pd.Series([], dtype=float)

        # Group by year and calculate beta for each year
        grouped = returns.groupby(returns.index.year)
        factor_grouped = factor_returns.groupby(factor_returns.index.year)

        annual_betas = []
        for year in grouped.groups.keys():
            if year in factor_grouped.groups.keys():
                year_returns = grouped.get_group(year)
                year_factor = factor_grouped.get_group(year)
                beta_val = cls.beta(year_returns, year_factor, risk_free, period, annualization)
                annual_betas.append((year, beta_val))

        if not annual_betas:
            return pd.Series([], dtype=float)

        years, betas = zip(*annual_betas)
        return pd.Series(betas, index=years)

    @classmethod
    def residual_risk(cls, returns=None, factor_returns=None, risk_free=0.0):
        """Calculates the residual risk (tracking error of alpha)."""
        returns = cls._get_returns(returns)
        factor_returns = cls._get_factor_returns(factor_returns)

        returns_aligned, factor_aligned = cls._aligned_series(returns, factor_returns)

        if len(returns_aligned) < 2:
            return np.nan

        # Adjust for risk-free rate
        excess_returns = returns_aligned - risk_free
        excess_factor = factor_aligned - risk_free

        # Calculate beta
        beta_val = np.cov(excess_returns, excess_factor)[0, 1] / np.var(excess_factor, ddof=1)

        # Calculate residuals
        predicted_returns = beta_val * excess_factor
        residuals = excess_returns - predicted_returns

        # Return standard deviation of residuals (annualized)
        return np.std(residuals, ddof=1) * np.sqrt(252)

    @classmethod
    def conditional_sharpe_ratio(cls, returns=None, cutoff=0.05):
        """Calculates the conditional Sharpe ratio."""
        returns = cls._get_returns(returns)

        if len(returns) < 2:
            return np.nan

        # Get returns below the cutoff percentile
        cutoff_value = np.percentile(returns, cutoff * 100)
        conditional_returns = returns[returns <= cutoff_value]

        if len(conditional_returns) < 2:
            return np.nan

        # Calculate Sharpe ratio for conditional returns
        mean_ret = np.mean(conditional_returns)
        std_ret = np.std(conditional_returns, ddof=1)

        if std_ret == 0:
            return np.nan

        return mean_ret / std_ret * np.sqrt(252)

    @classmethod
    def var_excess_return(cls, returns=None, cutoff=0.05):
        """Calculates the VaR excess return."""
        returns = cls._get_returns(returns)

        if len(returns) < 2:
            return np.nan

        var_value = cls.value_at_risk(returns, cutoff)
        excess_returns = returns[returns <= var_value]

        if len(excess_returns) == 0:
            return np.nan

        return np.mean(excess_returns)

    @classmethod
    def max_consecutive_up_weeks(cls, returns=None):
        """Determines the maximum number of consecutive weeks with positive returns."""
        returns = cls._get_returns(returns)

        if len(returns) < 1:
            return np.nan

        # Resample to weekly returns
        weekly_returns = returns.resample('W').apply(lambda x:
                                                     cls.cum_returns_final(x))

        up_weeks = weekly_returns > 0

        if not up_weeks.any():
            return 0

        # Find consecutive True values
        groups = (up_weeks != up_weeks.shift(1)).cumsum()
        consecutive_counts = up_weeks.groupby(groups).sum()

        return consecutive_counts.max()

    @classmethod
    def max_consecutive_down_weeks(cls, returns=None):
        """Determines the maximum number of consecutive weeks with negative returns."""
        returns = cls._get_returns(returns)

        if len(returns) < 1:
            return np.nan

        # Resample to weekly returns
        weekly_returns = returns.resample('W').apply(lambda x: cls.cum_returns_final(x))

        down_weeks = weekly_returns < 0

        if not down_weeks.any():
            return 0

        # Find consecutive True values
        groups = (down_weeks != down_weeks.shift(1)).cumsum()
        consecutive_counts = down_weeks.groupby(groups).sum()

        return consecutive_counts.max()

    @classmethod
    def max_consecutive_up_months(cls, returns=None):
        """Determines the maximum number of consecutive months with positive returns."""
        returns = cls._get_returns(returns)

        if len(returns) < 1:
            return np.nan

        # Resample to monthly returns
        monthly_returns = returns.resample('M').apply(lambda x:
                                                      cls.cum_returns_final(x))

        up_months = monthly_returns > 0

        if not up_months.any():
            return 0

        # Find consecutive True values
        groups = (up_months != up_months.shift(1)).cumsum()
        consecutive_counts = up_months.groupby(groups).sum()

        return consecutive_counts.max()

    @classmethod
    def max_consecutive_down_months(cls, returns=None):
        """Determines the maximum number of consecutive months with negative returns."""
        returns = cls._get_returns(returns)

        if len(returns) < 1:
            return np.nan

        # Resample to monthly returns
        monthly_returns = returns.resample('M').apply(lambda x:
                                                      cls.cum_returns_final(x))

        down_months = monthly_returns < 0

        if not down_months.any():
            return 0

        # Find consecutive True values
        groups = (down_months != down_months.shift(1)).cumsum()
        consecutive_counts = down_months.groupby(groups).sum()

        return consecutive_counts.max()

    @classmethod
    def win_rate(cls, returns=None):
        """Calculates the percentage of positive returns."""
        returns = cls._get_returns(returns)

        if len(returns) < 1:
            return np.nan

        returns_array = np.asanyarray(returns)

        # Count positive returns (excluding NaN)
        positive_count = np.sum(returns_array > 0)
        total_count = np.sum(~np.isnan(returns_array))

        if total_count == 0:
            return np.nan

        win_rate_value = positive_count / total_count

        if returns_array.ndim == 1:
            return win_rate_value.item() if isinstance(win_rate_value, np.ndarray) else win_rate_value
        else:
            return win_rate_value

    @classmethod
    def loss_rate(cls, returns=None):
        """Calculates the percentage of negative returns."""
        returns = cls._get_returns(returns)

        if len(returns) < 1:
            return np.nan

        returns_array = np.asanyarray(returns)

        # Count negative returns (excluding NaN)
        negative_count = np.sum(returns_array < 0)
        total_count = np.sum(~np.isnan(returns_array))

        if total_count == 0:
            return np.nan

        loss_rate_value = negative_count / total_count

        if returns_array.ndim == 1:
            return loss_rate_value.item() if isinstance(loss_rate_value, np.ndarray) else loss_rate_value
        else:
            return loss_rate_value

    @classmethod
    def get_max_drawdown_period(cls, returns=None):
        """Gets the start and end dates of the maximum drawdown period."""
        returns = cls._get_returns(returns)

        if len(returns) < 1:
            return None, None

        cum_returns = cls.cum_returns(returns, starting_value=1)

        if not isinstance(cum_returns, pd.Series):
            return None, None

        # Calculate rolling maximum
        rolling_max = cum_returns.expanding().max()

        # Calculate drawdown
        drawdown = cum_returns / rolling_max - 1

        # Find the end date of maximum drawdown
        end_date = drawdown.idxmin()

        # Find the start date of maximum drawdown (previous peak)
        start_date = cum_returns.loc[:end_date].idxmax()

        return start_date, end_date

    @classmethod
    def max_drawdown_days(cls, returns=None):
        """Calculates the duration of maximum drawdown in days."""
        returns = cls._get_returns(returns)

        if len(returns) < 1:
            return np.nan

        if not isinstance(returns, pd.Series):
            returns = pd.Series(returns)

        # Calculate cumulative returns
        cum_ret = cls.cum_returns(returns, starting_value=100)

        # Calculate rolling maximum
        rolling_max = cum_ret.expanding().max()

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

    @classmethod
    def futures_market_correlation(cls, returns=None, futures_returns=None):
        """Determines the correlation with the futures market."""
        returns = cls._get_returns(returns)
        if futures_returns is None:
            futures_returns = cls._get_factor_returns(futures_returns)
        return cls.cal_futures_market_correlation(returns, futures_returns)

    @classmethod
    def up_alpha_beta(cls, returns, factor_returns, risk_free=0.0, period=DAILY, annualization=None, out=None):
        """Calculates alpha and beta for up-market periods only."""
        # Handle output array
        if out is None:
            out = np.empty((2,), dtype='float64')

        if isinstance(returns, pd.Series) and isinstance(factor_returns, pd.Series):
            returns, factor_returns = returns.align(factor_returns, join='inner')

        returns_array = np.asanyarray(returns)
        factor_array = np.asanyarray(factor_returns)

        # Filter for up-market periods (factor returns > 0)
        up_mask = factor_array > 0
        up_returns = returns_array[up_mask]
        up_factor = factor_array[up_mask]

        if len(up_returns) < 2:
            out[0] = np.nan  # alpha
            out[1] = np.nan  # beta
            return out

        # Remove NaN values
        valid_mask = ~(np.isnan(up_returns) | np.isnan(up_factor))
        up_returns_clean = up_returns[valid_mask]
        up_factor_clean = up_factor[valid_mask]

        if len(up_returns_clean) < 2:
            out[0] = np.nan  # alpha
            out[1] = np.nan  # beta
            return out

        ann_factor = cls.annualization_factor(period, annualization)

        # Adjust returns for risk-free rate
        returns_adj = up_returns_clean - risk_free
        factor_returns_adj = up_factor_clean - risk_free

        # Calculate beta using covariance
        factor_var = np.var(factor_returns_adj, ddof=1)
        if factor_var == 0 or np.isnan(factor_var):
            out[0] = np.nan  # alpha
            out[1] = np.nan  # beta
            return out

        beta = np.cov(returns_adj, factor_returns_adj)[0, 1] / factor_var

        # Calculate alpha using compound growth like original implementation
        alpha_series = returns_adj - (beta * factor_returns_adj)
        mean_alpha = np.mean(alpha_series)
        alpha = (1 + mean_alpha) ** ann_factor - 1

        out[0] = alpha
        out[1] = beta
        return out

    @classmethod
    def down_alpha_beta(cls, returns, factor_returns, risk_free=0.0, period=DAILY, annualization=None, out=None):
        """Calculates alpha and beta for down-market periods only."""
        # Handle output array
        if out is None:
            out = np.empty((2,), dtype='float64')

        if isinstance(returns, pd.Series) and isinstance(factor_returns, pd.Series):
            returns, factor_returns = returns.align(factor_returns, join='inner')

        returns_array = np.asanyarray(returns)
        factor_array = np.asanyarray(factor_returns)

        # Filter for down-market periods (factor returns <= 0)
        down_mask = factor_array <= 0
        down_returns = returns_array[down_mask]
        down_factor = factor_array[down_mask]

        if len(down_returns) < 2:
            out[0] = np.nan  # alpha
            out[1] = np.nan  # beta
            return out

        # Remove NaN values
        valid_mask = ~(np.isnan(down_returns) | np.isnan(down_factor))
        down_returns_clean = down_returns[valid_mask]
        down_factor_clean = down_factor[valid_mask]

        if len(down_returns_clean) < 2:
            out[0] = np.nan  # alpha
            out[1] = np.nan  # beta
            return out

        ann_factor = cls.annualization_factor(period, annualization)

        # Adjust returns for risk-free rate
        returns_adj = down_returns_clean - risk_free
        factor_returns_adj = down_factor_clean - risk_free

        # Calculate beta using covariance
        factor_var = np.var(factor_returns_adj, ddof=1)
        if factor_var == 0 or np.isnan(factor_var):
            out[0] = np.nan  # alpha
            out[1] = np.nan  # beta
            return out

        beta = np.cov(returns_adj, factor_returns_adj)[0, 1] / factor_var

        # Calculate alpha using compound growth like original implementation
        alpha_series = returns_adj - (beta * factor_returns_adj)
        mean_alpha = np.mean(alpha_series)
        alpha = (1 + mean_alpha) ** ann_factor - 1

        out[0] = alpha
        out[1] = beta
        return out

    @classmethod
    def alpha_percentile_rank(cls, strategy_returns, all_strategies_returns, factor_returns, risk_free=0.0,
                              period=DAILY, annualization=None):
        """Calculates the percentile rank of alpha relative to a universe."""
        if len(strategy_returns) < 3:
            return np.nan

        # Calculate alpha for the target strategy
        strategy_alpha = cls.alpha(strategy_returns, factor_returns, risk_free, period, annualization)

        if np.isnan(strategy_alpha):
            return np.nan

        # Calculate alpha for all strategies
        all_alphas = []
        for other_returns in all_strategies_returns:
            if len(other_returns) < 3:
                continue
            other_alpha = cls.alpha(other_returns, factor_returns, risk_free, period, annualization)
            if not np.isnan(other_alpha):
                all_alphas.append(other_alpha)

        if len(all_alphas) == 0:
            return np.nan

        # Calculate percentile rank
        # Count how many strategies have alpha less than target strategy
        rank = sum(1 for a in all_alphas if a < strategy_alpha)
        percentile = rank / len(all_alphas)

        return float(percentile)

    @classmethod
    def cornell_timing(cls, returns, factor_returns, risk_free=0.0):
        """Calculates the Cornell timing model coefficient."""
        if len(returns) < 10 or len(factor_returns) < 10:
            return np.nan

        returns_aligned, factor_aligned = cls._aligned_series(returns, factor_returns)

        if len(returns_aligned) < 10:
            return np.nan

        returns_array = np.asanyarray(returns_aligned)
        factor_array = np.asanyarray(factor_aligned)

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

        except Exception as e:
            print(e)
            return np.nan

    @classmethod
    def r_cubed(cls, returns=None, factor_returns=None):
        """Calculates R-cubed (R³) measure."""
        returns = cls._get_returns(returns)
        factor_returns = cls._get_factor_returns(factor_returns)

        returns_aligned, factor_aligned = cls._aligned_series(returns, factor_returns)

        if len(returns_aligned) < 2:
            return np.nan

        # Calculate correlation coefficient
        correlation = np.corrcoef(returns_aligned, factor_aligned)[0, 1]

        # R-cubed is the cube of correlation
        return correlation ** 3

    @classmethod
    def regression_annual_return(cls, returns=None, factor_returns=None, risk_free=0.0, period=DAILY,
                                 annualization=None):
        """Calculates the annual return from regression (alpha + beta * benchmark_return)."""
        alpha_val = cls.alpha(returns, factor_returns, risk_free, period, annualization)
        beta_val = cls.beta(returns, factor_returns, risk_free, period, annualization)

        if np.isnan(alpha_val) or np.isnan(beta_val):
            return np.nan

        factor_returns = cls._get_factor_returns(factor_returns)
        benchmark_annual = cls.annual_return(factor_returns, period, annualization)

        if np.isnan(benchmark_annual):
            return np.nan

        return alpha_val + beta_val * benchmark_annual

    @classmethod
    def annualized_cumulative_return(cls, returns=None, period=DAILY, annualization=None):
        """Calculates the annualized cumulative return."""
        # This is essentially the same as annual_return
        return cls.annual_return(returns, period, annualization)

    @classmethod
    def annual_active_return(cls, returns=None, factor_returns=None, period=DAILY, annualization=None):
        """Calculates the annual active return (strategy - benchmark)."""
        returns = cls._get_returns(returns)
        factor_returns = cls._get_factor_returns(factor_returns)

        if len(returns) < 1:
            return np.nan

        # Align the series first
        returns_aligned, factor_aligned = cls._aligned_series(returns, factor_returns)

        # Calculate annualized returns on aligned data
        strategy_annual = cls.annual_return(returns_aligned, period, annualization)
        benchmark_annual = cls.annual_return(factor_aligned, period, annualization)

        if np.isnan(strategy_annual) or np.isnan(benchmark_annual):
            return np.nan

        return strategy_annual - benchmark_annual

    @classmethod
    def annual_active_risk(cls, returns=None, factor_returns=None, period=DAILY, annualization=None):
        """Calculates the annual active risk (tracking error)."""
        return cls.tracking_error(returns, factor_returns, period, annualization)

    @classmethod
    def tracking_difference(cls, returns=None, factor_returns=None):
        """Calculates the tracking difference (cumulative strategy return - cumulative benchmark return)."""
        returns = cls._get_returns(returns)
        factor_returns = cls._get_factor_returns(factor_returns)

        if len(returns) < 1:
            return np.nan

        returns_aligned, factor_aligned = cls._aligned_series(returns, factor_returns)

        # Calculate cumulative returns
        cum_strategy_return = cls.cum_returns_final(returns_aligned, starting_value=0)
        cum_benchmark_return = cls.cum_returns_final(factor_aligned, starting_value=0)

        # Tracking difference = cumulative strategy return - cumulative benchmark return
        result = cum_strategy_return - cum_benchmark_return
        if not isinstance(result, (float, np.floating)):
            result = result.item()
        return result

    @classmethod
    def annual_active_return_by_year(cls, returns=None, factor_returns=None, period=DAILY, annualization=None):
        """Determines the annual active return for each year."""
        returns = cls._get_returns(returns)
        factor_returns = cls._get_factor_returns(factor_returns)

        if len(returns) < 1:
            return pd.Series([], dtype=float)

        # Ensure returns has a datetime index
        if not isinstance(returns.index, pd.DatetimeIndex):
            return pd.Series([], dtype=float)

        # Group by year and calculate active return for each year
        grouped = returns.groupby(returns.index.year)
        factor_grouped = factor_returns.groupby(factor_returns.index.year)

        annual_active_returns = []
        for year in grouped.groups.keys():
            if year in factor_grouped.groups.keys():
                year_returns = grouped.get_group(year)
                year_factor = factor_grouped.get_group(year)
                active_return = cls.annual_active_return(year_returns, year_factor, period, annualization)
                annual_active_returns.append((year, active_return))

        if not annual_active_returns:
            return pd.Series([], dtype=float)

        years, active_returns = zip(*annual_active_returns)
        return pd.Series(active_returns, index=years)

    @classmethod
    def information_ratio_by_year(cls, returns=None, factor_returns=None, period=DAILY, annualization=None):
        """Determines the information ratio for each year."""
        returns = cls._get_returns(returns)
        factor_returns = cls._get_factor_returns(factor_returns)

        if len(returns) < 1:
            return_as_array = isinstance(returns, np.ndarray)
            return np.array([]) if return_as_array else pd.Series(dtype='float64')

        # Track whether input is array for return type
        return_as_array = isinstance(returns, np.ndarray)

        # Ensure we have datetime-indexed Series
        if return_as_array or not hasattr(returns, 'index') or not isinstance(returns.index, pd.DatetimeIndex):
            # For numpy arrays or non-datetime indexed data, convert to datetime-indexed series
            returns = cls._ensure_datetime_index_series(returns, period=period)
            factor_returns = cls._ensure_datetime_index_series(factor_returns, period=period)

        # Align the series
        returns_aligned, factor_aligned = cls._aligned_series(returns, factor_returns)

        # Group by year and calculate information ratio for each year
        information_ratios = returns_aligned.groupby(returns_aligned.index.year).apply(
            lambda x: cls._calculate_information_ratio_for_active_returns(
                x - factor_aligned.loc[x.index],
                period=period,
                annualization=annualization
            )
        )

        # Remove name attribute if it exists
        if hasattr(information_ratios, 'name'):
            information_ratios.name = None

        return information_ratios.values if return_as_array else information_ratios

    @classmethod
    def _calculate_information_ratio_for_active_returns(cls, active_returns, period=DAILY, annualization=None):
        """Calculate information ratio from active returns."""
        ann_factor = cls.annualization_factor(period, annualization)
        mean_excess_return = active_returns.mean()
        std_excess_return = active_returns.std(ddof=1)
        if std_excess_return == 0:
            return np.nan
        else:
            return (mean_excess_return * ann_factor) / (std_excess_return * np.sqrt(ann_factor))

    @classmethod
    def second_max_drawdown(cls, returns=None):
        """Determines the second maximum drawdown of a strategy."""
        returns = cls._get_returns(returns)

        drawdown_periods = cls._get_all_drawdowns(returns)

        if len(drawdown_periods) < 2:
            return np.nan

        # Sort drawdowns (most negative first)
        sorted_drawdowns = np.sort(drawdown_periods)

        # Get second largest (second most negative)
        return sorted_drawdowns[-2]

    @classmethod
    def third_max_drawdown(cls, returns=None):
        """Determines the third maximum drawdown of a strategy."""
        returns = cls._get_returns(returns)

        drawdown_periods = cls._get_all_drawdowns(returns)

        if len(drawdown_periods) < 3:
            return np.nan

        # Sort drawdowns (most negative first)
        sorted_drawdowns = np.sort(drawdown_periods)

        # Get third largest (third most negative)
        return sorted_drawdowns[-3]

    @classmethod
    def max_drawdown_weeks(cls, returns=None):
        """Calculates the duration of maximum drawdown in weeks."""
        returns = cls._get_returns(returns)

        if len(returns) < 1:
            return np.nan

        if not isinstance(returns, pd.Series):
            returns = pd.Series(returns)

        # Calculate cumulative returns
        cum_ret = cls.cum_returns(returns, starting_value=100)

        # Calculate rolling maximum
        rolling_max = cum_ret.expanding().max()

        # Calculate drawdown
        drawdown = (cum_ret - rolling_max) / rolling_max

        # Find the end date of maximum drawdown (lowest point)
        end_idx = drawdown.idxmin()

        # Find the start date (previous peak before the end)
        start_idx = cum_ret.loc[:end_idx].idxmax()

        # Calculate weeks difference
        if isinstance(returns.index, pd.DatetimeIndex):
            # Use index positions
            start_week = returns.index.get_loc(start_idx)
            end_week = returns.index.get_loc(end_idx)
            weeks_diff = end_week - start_week
        else:
            # For non-datetime index, count the number of periods
            start_pos = returns.index.get_loc(start_idx)
            end_pos = returns.index.get_loc(end_idx)
            weeks_diff = end_pos - start_pos

        return weeks_diff

    @classmethod
    def max_drawdown_months(cls, returns=None):
        """Calculates the duration of maximum drawdown in months."""
        returns = cls._get_returns(returns)

        if len(returns) < 1:
            return np.nan

        if not isinstance(returns, pd.Series):
            returns = pd.Series(returns)

        # Calculate cumulative returns
        cum_ret = cls.cum_returns(returns, starting_value=100)

        # Calculate rolling maximum
        rolling_max = cum_ret.expanding().max()

        # Calculate drawdown
        drawdown = (cum_ret - rolling_max) / rolling_max

        # Find the end date of maximum drawdown (lowest point)
        end_idx = drawdown.idxmin()

        # Find the start date (previous peak before the end)
        start_idx = cum_ret.loc[:end_idx].idxmax()

        # Calculate months difference using index positions
        start_pos = returns.index.get_loc(start_idx)
        end_pos = returns.index.get_loc(end_idx)
        months_diff = end_pos - start_pos

        return months_diff

    @classmethod
    def max_drawdown_recovery_days(cls, returns=None):
        """Calculates the recovery time from maximum drawdown in days."""
        returns = cls._get_returns(returns)

        if len(returns) < 1:
            return np.nan

        cum_returns = cls.cum_returns(returns, starting_value=1)

        if not isinstance(cum_returns, pd.Series):
            return np.nan

        # Calculate rolling maximum
        rolling_max = cum_returns.expanding().max()

        # Calculate drawdown
        drawdown = cum_returns / rolling_max - 1

        # Find the end date of maximum drawdown
        max_dd_date = drawdown.idxmin()
        max_dd_value = drawdown.min()

        # Find when it recovers to the previous high
        post_dd_data = cum_returns.loc[max_dd_date:]
        recovery_level = rolling_max.loc[max_dd_date]

        recovery_mask = post_dd_data >= recovery_level
        if recovery_mask.any():
            recovery_date = post_dd_data[recovery_mask].index[0]
            # Handle different index types
            if hasattr(recovery_date - max_dd_date, 'days'):
                # DatetimeIndex case
                return (recovery_date - max_dd_date).days
            else:
                # IntIndex or other numeric index case
                return int(recovery_date - max_dd_date)
        else:
            return np.nan

    @classmethod
    def max_drawdown_recovery_weeks(cls, returns=None):
        """Calculates the recovery time from maximum drawdown in weeks."""
        returns = cls._get_returns(returns)

        if len(returns) < 1:
            return np.nan

        if not isinstance(returns, pd.Series):
            returns = pd.Series(returns)

        # Calculate cumulative returns
        cum_ret = cls.cum_returns(returns, starting_value=100)

        # Calculate rolling maximum
        rolling_max = cum_ret.expanding().max()

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

            # Calculate weeks difference using index positions
            end_pos = returns.index.get_loc(end_idx)
            recovery_pos = returns.index.get_loc(recovery_idx)
            weeks_diff = recovery_pos - end_pos

            return weeks_diff
        else:
            # Never recovers
            return np.nan

    @classmethod
    def max_drawdown_recovery_months(cls, returns=None):
        """Calculates the recovery time from maximum drawdown in months."""
        returns = cls._get_returns(returns)

        if len(returns) < 1:
            return np.nan

        if not isinstance(returns, pd.Series):
            returns = pd.Series(returns)

        # Calculate cumulative returns
        cum_ret = cls.cum_returns(returns, starting_value=100)

        # Calculate rolling maximum
        rolling_max = cum_ret.expanding().max()

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

            # Calculate months difference using index positions
            end_pos = returns.index.get_loc(end_idx)
            recovery_pos = returns.index.get_loc(recovery_idx)
            months_diff = recovery_pos - end_pos

            return months_diff
        else:
            # Never recovers
            return np.nan

    @classmethod
    def annual_volatility_by_year(cls, returns=None, period=DAILY, annualization=None):
        """Determines the annual volatility for each year."""
        returns = cls._get_returns(returns)
        if len(returns) < 1:
            return_as_array = isinstance(returns, np.ndarray)
            return np.array([]) if return_as_array else pd.Series(dtype='float64')

        return_as_array = isinstance(returns, np.ndarray)

        returns = cls._ensure_datetime_index_series(returns, period=period)

        annual_vol_by_year = returns.groupby(returns.index.year).apply(
            lambda x: cls.annual_volatility(x, period=period, annualization=annualization)
        )

        return annual_vol_by_year.values if return_as_array else annual_vol_by_year

    @classmethod
    def max_single_day_gain_date(cls, returns=None):
        """Determines the date of maximum single day gain."""
        returns = cls._get_returns(returns)

        if len(returns) < 1:
            return None

        return returns.idxmax()

    @classmethod
    def max_single_day_loss_date(cls, returns=None):
        """Determines the date of maximum single day loss."""
        returns = cls._get_returns(returns)

        if len(returns) < 1:
            return None

        return returns.idxmin()

    @classmethod
    def max_consecutive_up_start_date(cls, returns=None):
        """Determines the start date of maximum consecutive up period."""
        returns = cls._get_returns(returns)

        if len(returns) < 1:
            return None

        up_days = returns > 0

        if not up_days.any():
            return None

        # Find consecutive True values
        groups = (up_days != up_days.shift(1)).cumsum()
        consecutive_counts = up_days.groupby(groups).sum()

        # Find the group with maximum consecutive days
        max_group = consecutive_counts.idxmax()

        # Find the start date of this group
        group_mask = groups == max_group
        return returns[group_mask & up_days].index[0]

    @classmethod
    def max_consecutive_up_end_date(cls, returns=None):
        """Determines the end date of maximum consecutive up period."""
        returns = cls._get_returns(returns)

        if len(returns) < 1:
            return None

        up_days = returns > 0

        if not up_days.any():
            return None

        # Find consecutive True values
        groups = (up_days != up_days.shift(1)).cumsum()
        consecutive_counts = up_days.groupby(groups).sum()

        # Find the group with maximum consecutive days
        max_group = consecutive_counts.idxmax()

        # Find the end date of this group
        group_mask = groups == max_group
        return returns[group_mask & up_days].index[-1]

    @classmethod
    def max_consecutive_down_start_date(cls, returns=None):
        """Determines the start date of maximum consecutive down period."""
        returns = cls._get_returns(returns)

        if len(returns) < 1:
            return None

        down_days = returns < 0

        if not down_days.any():
            return None

        # Find consecutive True values
        groups = (down_days != down_days.shift(1)).cumsum()
        consecutive_counts = down_days.groupby(groups).sum()

        # Find the group with maximum consecutive days
        max_group = consecutive_counts.idxmax()

        # Find the start date of this group
        group_mask = groups == max_group
        return returns[group_mask & down_days].index[0]

    @classmethod
    def max_consecutive_down_end_date(cls, returns=None):
        """Determines the end date of maximum consecutive down period."""
        returns = cls._get_returns(returns)

        if len(returns) < 1:
            return None

        down_days = returns < 0

        if not down_days.any():
            return None

        # Find consecutive True values
        groups = (down_days != down_days.shift(1)).cumsum()
        consecutive_counts = down_days.groupby(groups).sum()

        # Find the group with maximum consecutive days
        max_group = consecutive_counts.idxmax()

        # Find the end date of this group
        group_mask = groups == max_group
        return returns[group_mask & down_days].index[-1]

    @classmethod
    def beta_fragility_heuristic(cls, returns=None, factor_returns=None):
        """Estimate fragility to drop in beta."""
        returns = cls._get_returns(returns)
        factor_returns = cls._get_factor_returns(factor_returns)

        if len(returns) < 3 or len(factor_returns) < 3:
            return np.nan

        returns_aligned, factor_aligned = cls._aligned_series(returns, factor_returns)

        if len(returns_aligned) < 3 or len(factor_aligned) < 3:
            return np.nan

        # Combine returns and factor returns into pairs
        returns_series = pd.Series(returns_aligned)
        factor_returns_series = pd.Series(factor_aligned)
        pairs = pd.concat([returns_series, factor_returns_series], axis=1)
        pairs.columns = ['returns', 'factor_returns']

        # Exclude any rows where returns are nan
        pairs = pairs.dropna()

        # Sort by factor returns
        pairs = pairs.sort_values(by=['factor_returns'], kind='mergesort')

        # Find the three vectors, using median of 3
        start_index = 0
        mid_index = int(np.around(len(pairs) / 2, 0))
        end_index = len(pairs) - 1

        (start_returns, start_factor_returns) = pairs.iloc[start_index]
        (mid_returns, mid_factor_returns) = pairs.iloc[mid_index]
        (end_returns, end_factor_returns) = pairs.iloc[end_index]

        factor_returns_range = (end_factor_returns - start_factor_returns)
        start_returns_weight = 0.5
        end_returns_weight = 0.5

        # Find weights for the start and end returns using a convex combination
        if not factor_returns_range == 0:
            start_returns_weight = \
                (mid_factor_returns - start_factor_returns) / \
                factor_returns_range
            end_returns_weight = \
                (end_factor_returns - mid_factor_returns) / \
                factor_returns_range

        # Calculate fragility heuristic
        heuristic = (start_returns_weight * start_returns) + \
                    (end_returns_weight * end_returns) - mid_returns

        return heuristic

    @classmethod
    def beta_fragility_heuristic_aligned(cls, returns=None, factor_returns=None):
        """Calculates the beta fragility heuristic with aligned series."""
        # This is the same as beta_fragility_heuristic since we already align series
        return cls.beta_fragility_heuristic(returns, factor_returns)

    @classmethod
    def gpd_risk_estimates(cls, returns=None, var_p=0.01):
        """Estimate VaR and ES using the Generalized Pareto Distribution (GPD).
        
        Returns
        -------
        [threshold, scale_param, shape_param, var_estimate, es_estimate] : list[float]
        """
        returns = cls._get_returns(returns)

        if len(returns) < 3:
            result = np.zeros(5)
            if isinstance(returns, pd.Series):
                result = pd.Series(result)
            return result

        result = np.zeros(5)

        # Default parameters
        default_threshold = 0.2
        minimum_threshold = 0.000000001

        try:
            returns_array = pd.Series(returns).to_numpy()
        except AttributeError:
            # while zipline requires support for pandas < 0.25
            returns_array = pd.Series(returns).values

        flipped_returns = -1 * returns_array
        losses = flipped_returns[flipped_returns > 0]
        threshold = default_threshold
        finished = False
        scale_param = 0
        shape_param = 0
        var_estimate = 0

        while not finished and threshold > minimum_threshold:
            losses_beyond_threshold = losses[losses >= threshold]
            param_result = cls._gpd_loglikelihood_minimizer_aligned(losses_beyond_threshold)
            if (param_result[0] is not False and param_result[1] is not False):
                scale_param = param_result[0]
                shape_param = param_result[1]
                var_estimate = cls._gpd_var_calculator(threshold, scale_param,
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
            es_estimate = cls._gpd_es_calculator(var_estimate, threshold,
                                                 scale_param, shape_param)
            result = np.array([threshold, scale_param, shape_param,
                               var_estimate, es_estimate])

        if isinstance(returns, pd.Series):
            result = pd.Series(result)
        return result

    @classmethod
    def _gpd_es_calculator(cls, var_estimate, threshold, scale_param, shape_param):
        result = 0
        if (1 - shape_param) != 0:
            # this formula is from Gilli and Kellezi pg. 8
            var_ratio = (var_estimate / (1 - shape_param))
            param_ratio = ((scale_param - (shape_param * threshold)) /
                           (1 - shape_param))
            result = var_ratio + param_ratio
        return result

    @classmethod
    def _gpd_var_calculator(cls, threshold, scale_param, shape_param, probability, total_n, exceedance_n):
        result = 0
        if exceedance_n > 0 and shape_param > 0:
            # this formula is from Gilli and Kellezi pg. 12
            param_ratio = scale_param / shape_param
            prob_ratio = (total_n / exceedance_n) * probability
            result = threshold + (param_ratio * (pow(prob_ratio, -shape_param) - 1))
        return result

    @classmethod
    def _gpd_loglikelihood_minimizer_aligned(cls, price_data):
        from scipy import optimize
        result = [False, False]
        default_scale_param = 1
        default_shape_param = 1
        if len(price_data) > 0:
            gpd_loglikelihood_lambda = lambda params: cls._gpd_loglikelihood(params, price_data)
            try:
                optimization_results = optimize.minimize(gpd_loglikelihood_lambda,
                                                         [default_scale_param, default_shape_param],
                                                         method='Nelder-Mead')
                if optimization_results.success:
                    resulting_params = optimization_results.x
                    if len(resulting_params) == 2:
                        result[0] = resulting_params[0]
                        result[1] = resulting_params[1]
            except Exception as e:
                print(e)
        return result

    @classmethod
    def _gpd_loglikelihood(cls, params, price_data):
        if params[1] != 0:
            return -cls._gpd_loglikelihood_scale_and_shape(params[0], params[1], price_data)
        else:
            return -cls._gpd_loglikelihood_scale_only(params[0], price_data)

    @classmethod
    def _gpd_loglikelihood_scale_and_shape(cls, scale, shape, price_data):
        n = len(price_data)
        result = -1 * float_info.max
        if scale != 0:
            param_factor = shape / scale
            if shape != 0 and param_factor >= 0 and scale >= 0:
                result = ((-n * np.log(scale)) -
                          (((1 / shape) + 1) *
                           (np.log((shape / scale * price_data) + 1)).sum()))
        return result

    @classmethod
    def _gpd_loglikelihood_scale_only(cls, scale, price_data):
        n = len(price_data)
        data_sum = price_data.sum()
        result = -1 * float_info.max
        if scale >= 0:
            result = ((-n * np.log(scale)) - (data_sum / scale))
        return result

    @classmethod
    def gpd_risk_estimates_aligned(cls, returns=None, var_p=0.01):
        """Calculates GPD risk estimates (aligned version for compatibility)."""
        returns = cls._get_returns(returns)

        # For compatibility with original API, this is the same as gpd_risk_estimates
        return cls.gpd_risk_estimates(returns, var_p)

    @classmethod
    def second_max_drawdown_days(cls, returns=None):
        """Calculates the duration of second maximum drawdown in days."""
        returns = cls._get_returns(returns)

        drawdown_periods = cls._get_all_drawdowns_detailed(returns)

        if len(drawdown_periods) < 2:
            return np.nan

        # Sort by drawdown value (most negative first)
        sorted_drawdowns = sorted(drawdown_periods, key=lambda x: x['value'])

        # Get second largest drawdown duration
        return sorted_drawdowns[1]['duration']

    @classmethod
    def second_max_drawdown_recovery_days(cls, returns=None):
        """Calculates the recovery time from second maximum drawdown in days."""
        returns = cls._get_returns(returns)

        drawdown_periods = cls._get_all_drawdowns_detailed(returns)

        if len(drawdown_periods) < 2:
            return np.nan

        # Sort by drawdown value (most negative first)
        sorted_drawdowns = sorted(drawdown_periods, key=lambda x: x['value'])

        # Get second-largest drawdown recovery duration
        recovery_duration = sorted_drawdowns[1]['recovery_duration']
        return recovery_duration if recovery_duration is not None else np.nan

    @classmethod
    def third_max_drawdown_days(cls, returns=None):
        """Calculates the duration of third maximum drawdown in days."""
        returns = cls._get_returns(returns)

        drawdown_periods = cls._get_all_drawdowns_detailed(returns)

        if len(drawdown_periods) < 3:
            return np.nan

        # Sort by drawdown value (most negative first)
        sorted_drawdowns = sorted(drawdown_periods, key=lambda x: x['value'])

        # Get third largest drawdown duration
        return sorted_drawdowns[2]['duration']

    @classmethod
    def third_max_drawdown_recovery_days(cls, returns=None):
        """Calculates the recovery time from third maximum drawdown in days."""
        returns = cls._get_returns(returns)

        drawdown_periods = cls._get_all_drawdowns_detailed(returns)

        if len(drawdown_periods) < 3:
            return np.nan

        # Sort by drawdown value (most negative first)
        sorted_drawdowns = sorted(drawdown_periods, key=lambda x: x['value'])

        # Get third-largest drawdown recovery duration
        recovery_duration = sorted_drawdowns[2]['recovery_duration']
        return recovery_duration if recovery_duration is not None else np.nan

    @classmethod
    def roll_alpha(cls, returns=None, factor_returns=None, window=252, risk_free=0.0, period=DAILY,
                   annualization=None):
        """Calculates rolling alpha over a specified window."""
        returns = cls._get_returns(returns)
        factor_returns = cls._get_factor_returns(factor_returns)

        returns_aligned, factor_aligned = cls._aligned_series(returns, factor_returns)

        if len(returns_aligned) < window:
            return pd.Series([], dtype=float)

        rolling_alphas = []
        for i in range(window, len(returns_aligned) + 1):
            window_returns = returns_aligned.iloc[i - window:i]
            window_factor = factor_aligned.iloc[i - window:i]

            alpha_val = cls.alpha(window_returns, window_factor, risk_free, period, annualization)
            rolling_alphas.append(alpha_val)

        if isinstance(returns_aligned, pd.Series):
            return pd.Series(rolling_alphas, index=returns_aligned.index[window - 1:])
        else:
            return pd.Series(rolling_alphas)

    @classmethod
    def roll_beta(cls, returns=None, factor_returns=None, window=252, risk_free=0.0, period=DAILY, annualization=None):
        """Calculates rolling beta over a specified window."""
        returns = cls._get_returns(returns)
        factor_returns = cls._get_factor_returns(factor_returns)

        returns_aligned, factor_aligned = cls._aligned_series(returns, factor_returns)

        if len(returns_aligned) < window:
            return pd.Series([], dtype=float)

        rolling_betas = []
        for i in range(window, len(returns_aligned) + 1):
            window_returns = returns_aligned.iloc[i - window:i]
            window_factor = factor_aligned.iloc[i - window:i]

            beta_val = cls.beta(window_returns, window_factor, risk_free, period, annualization)
            rolling_betas.append(beta_val)

        if isinstance(returns_aligned, pd.Series):
            return pd.Series(rolling_betas, index=returns_aligned.index[window - 1:])
        else:
            return pd.Series(rolling_betas)

    @classmethod
    def roll_alpha_beta(cls, returns=None, factor_returns=None, window=252, risk_free=0.0, period=DAILY,
                        annualization=None):
        """Calculates rolling alpha and beta over a specified window."""
        returns = cls._get_returns(returns)
        factor_returns = cls._get_factor_returns(factor_returns)

        # Align series
        returns_aligned, factor_aligned = cls._aligned_series(returns, factor_returns)

        if len(returns_aligned) < window:
            # Return empty DataFrame for consistent behavior
            if isinstance(returns, np.ndarray) and isinstance(factor_returns, np.ndarray):
                return np.empty((0, 2), dtype=float)
            else:
                return pd.DataFrame(columns=['alpha', 'beta'], dtype=float)

        rolling_results = []
        for i in range(window - 1, len(returns_aligned)):
            if hasattr(returns_aligned, 'iloc'):
                window_returns = returns_aligned.iloc[i - window + 1:i + 1]
                window_factor = factor_aligned.iloc[i - window + 1:i + 1]
            else:
                window_returns = returns_aligned[i - window + 1:i + 1]
                window_factor = factor_aligned[i - window + 1:i + 1]

            try:
                alpha_beta_result = cls.alpha_beta(window_returns, window_factor, risk_free, period, annualization)
                rolling_results.append(alpha_beta_result)
            except Exception as e:
                print(e)
                rolling_results.append([np.nan, np.nan])

        # Convert to DataFrame or numpy array
        if isinstance(returns, np.ndarray) and isinstance(factor_returns, np.ndarray):
            return np.array(rolling_results)
        else:
            if hasattr(returns_aligned, 'index'):
                result_df = pd.DataFrame(rolling_results, columns=['alpha', 'beta'],
                                         index=returns_aligned.index[window - 1:])
            else:
                result_df = pd.DataFrame(rolling_results, columns=['alpha', 'beta'])
            return result_df

    @classmethod
    def _get_all_drawdowns_detailed(cls, returns):
        """Helper function to find all distinct drawdown periods with detailed information."""
        if len(returns) < 1:
            return []

        if not isinstance(returns, pd.Series):
            returns = pd.Series(returns)

        # Calculate cumulative returns
        cum_ret = cls.cum_returns(returns, starting_value=100)

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

    @classmethod
    def roll_sharpe_ratio(cls, returns=None, window=252, risk_free=0.0, period=DAILY, annualization=None):
        """Calculates rolling Sharpe ratio over a specified window."""
        returns = cls._get_returns(returns)

        if len(returns) < window:
            if isinstance(returns, np.ndarray):
                return np.array([], dtype=float)
            else:
                return pd.Series([], dtype=float, index=returns.index[:0])

        rolling_sharpes = []
        for i in range(window - 1, len(returns)):
            if hasattr(returns, 'iloc'):
                window_returns = returns.iloc[i - window + 1:i + 1]
            else:
                window_returns = returns[i - window + 1:i + 1]
            try:
                sharpe = cls.sharpe_ratio(window_returns, risk_free, period, annualization)
            except Exception as e:
                print(e)
                sharpe = np.nan
            rolling_sharpes.append(sharpe)

        if isinstance(returns, np.ndarray):
            return np.array(rolling_sharpes)
        else:
            return pd.Series(rolling_sharpes, index=returns.index[window - 1:])

    @classmethod
    def roll_max_drawdown(cls, returns=None, window=252):
        """Calculates rolling maximum drawdown over a specified window."""
        returns = cls._get_returns(returns)

        # Use the common roll helper so that ndarray/Series behaviour and
        # window semantics match other rolling helpers and the original
        # empyrical implementation.
        return roll(
            returns,
            window=window,
            function=cls.cal_max_drawdown,
        )

    @classmethod
    def roll_up_capture(cls, returns=None, factor_returns=None, window=252):
        """Calculates rolling up capture ratio over a specified window."""
        returns = cls._get_returns(returns)
        factor_returns = cls._get_factor_returns(factor_returns)

        # Align series
        returns_aligned, factor_aligned = cls._aligned_series(returns, factor_returns)

        if len(returns_aligned) < window:
            empty_series = pd.Series([], dtype=float)
            if isinstance(returns, np.ndarray) and isinstance(factor_returns, np.ndarray):
                return empty_series.values  # Return numpy array for numpy inputs
            elif hasattr(returns_aligned, 'index'):
                return pd.Series([], dtype=float, index=returns_aligned.index[:0])
            else:
                return empty_series

        rolling_up_capture = []
        for i in range(window - 1, len(returns_aligned)):
            if hasattr(returns_aligned, 'iloc'):
                window_returns = returns_aligned.iloc[i - window + 1:i + 1]
                window_factor = factor_aligned.iloc[i - window + 1:i + 1]
            else:
                window_returns = returns_aligned[i - window + 1:i + 1]
                window_factor = factor_aligned[i - window + 1:i + 1]

            try:
                up_cap = cls.up_capture(window_returns, window_factor)
            except Exception as e:
                print(e)
                up_cap = np.nan
            rolling_up_capture.append(up_cap)

        if hasattr(returns_aligned, 'index'):
            result = pd.Series(rolling_up_capture, index=returns_aligned.index[window - 1:])
        else:
            result = pd.Series(rolling_up_capture)

        # Convert to numpy array if input was numpy array to match expected return type
        if isinstance(returns, np.ndarray) and isinstance(factor_returns, np.ndarray):
            return result.values
        else:
            return result

    @classmethod
    def roll_down_capture(cls, returns=None, factor_returns=None, window=252):
        """Calculates rolling down capture ratio over a specified window."""
        returns = cls._get_returns(returns)
        factor_returns = cls._get_factor_returns(factor_returns)

        # Align series
        returns_aligned, factor_aligned = cls._aligned_series(returns, factor_returns)

        if len(returns_aligned) < window:
            empty_series = pd.Series([], dtype=float)
            if isinstance(returns, np.ndarray) and isinstance(factor_returns, np.ndarray):
                return empty_series.values  # Return numpy array for numpy inputs
            elif hasattr(returns_aligned, 'index'):
                return pd.Series([], dtype=float, index=returns_aligned.index[:0])
            else:
                return empty_series

        rolling_down_capture = []
        for i in range(window - 1, len(returns_aligned)):
            if hasattr(returns_aligned, 'iloc'):
                window_returns = returns_aligned.iloc[i - window + 1:i + 1]
                window_factor = factor_aligned.iloc[i - window + 1:i + 1]
            else:
                window_returns = returns_aligned[i - window + 1:i + 1]
                window_factor = factor_aligned[i - window + 1:i + 1]

            try:
                down_cap = cls.down_capture(window_returns, window_factor)
            except Exception as e:
                print(e)
                down_cap = np.nan
            rolling_down_capture.append(down_cap)

        if hasattr(returns_aligned, 'index'):
            result = pd.Series(rolling_down_capture, index=returns_aligned.index[window - 1:])
        else:
            result = pd.Series(rolling_down_capture)

        # Convert to numpy array if input was numpy array to match expected return type
        if isinstance(returns, np.ndarray) and isinstance(factor_returns, np.ndarray):
            return result.values
        else:
            return result

    @classmethod
    def roll_up_down_capture(cls, returns=None, factor_returns=None, window=252):
        """Calculates rolling up/down capture ratio over a specified window."""
        returns = cls._get_returns(returns)
        factor_returns = cls._get_factor_returns(factor_returns)

        # Align series
        returns_aligned, factor_aligned = cls._aligned_series(returns, factor_returns)

        if len(returns_aligned) < window:
            empty_series = pd.Series([], dtype=float)
            if isinstance(returns, np.ndarray) and isinstance(factor_returns, np.ndarray):
                return empty_series.values  # Return numpy array for numpy inputs
            elif hasattr(returns_aligned, 'index'):
                return pd.Series([], dtype=float, index=returns_aligned.index[:0])
            else:
                return empty_series

        rolling_up_down_capture = []
        for i in range(window - 1, len(returns_aligned)):
            if hasattr(returns_aligned, 'iloc'):
                window_returns = returns_aligned.iloc[i - window + 1:i + 1]
                window_factor = factor_aligned.iloc[i - window + 1:i + 1]
            else:
                window_returns = returns_aligned[i - window + 1:i + 1]
                window_factor = factor_aligned[i - window + 1:i + 1]

            try:
                up_down_cap = cls.up_down_capture(window_returns, window_factor)
            except Exception as e:
                print(e)
                up_down_cap = np.nan
            rolling_up_down_capture.append(up_down_cap)

        if hasattr(returns_aligned, 'index'):
            result = pd.Series(rolling_up_down_capture, index=returns_aligned.index[window - 1:])
        else:
            result = pd.Series(rolling_up_down_capture)

        # Convert to numpy array if input was numpy array to match expected return type
        if isinstance(returns, np.ndarray) and isinstance(factor_returns, np.ndarray):
            return result.values
        else:
            return result

    def _get_all_drawdowns(cls, returns):
        """Helper function to find all distinct drawdown periods and their values."""
        detailed = cls._get_all_drawdowns_detailed(returns)
        return [dd['value'] for dd in detailed]

    @classmethod
    def model_returns_t_alpha_beta(cls, data, bmark, samples=2000, progressbar=True):
        """
        Run Bayesian alpha-beta-model with T distributed returns.

        This model estimates intercept (alpha) and slope (beta) of two
        return sets. Usually, these will be algorithm returns and
        benchmark returns (e.g. S&P500). The data is assumed to be T-distributed and thus is robust to outliers and takes tail events
        into account.  If a pandas.DataFrame is passed as a benchmark, then
        multiple linear regression is used to estimate alpha and beta.

        Parameters
        ----------
        :param data : pandas.Series:
            Series of simple returns of an algorithm or stock.
        :param bmark : pandas.DataFrame:
            DataFrame of benchmark returns (e.g., S&P500) or risk factors (e.g.,
            Fama-French SMB, HML, and UMD).
            If bmark has more recent returns than returns_train, these dates
            will be treated as missing values and predictions will be
            generated for them taking market correlations into account.
        :param samples : Int (optional)
            Number of posterior samples to draw.
        :param progressbar : Bool (optional), default True

        Returns
        -------
        model : pymc.Model object
            PyMC3 model containing all random variables.
        trace : pymc3.sampling.BaseTrace object
            A PyMC3 trace object that contains samples for each parameter
            of the posterior.
        """

        data_bmark = pd.concat([data, bmark], axis=1).dropna()

        with pm.Model() as model:
            sigma = pm.HalfCauchy(
                'sigma',
                beta=1)
            nu = pm.Exponential('nu_minus_two', 1. / 10.)

            # alpha and beta
            X = data_bmark.iloc[:, 1]
            y = data_bmark.iloc[:, 0]

            alpha_reg = pm.Normal('alpha', mu=0, sd=.1)
            beta_reg = pm.Normal('beta', mu=0, sd=1)

            mu_reg = alpha_reg + beta_reg * X
            pm.StudentT('returns',
                        nu=nu + 2,
                        mu=mu_reg,
                        sd=sigma,
                        observed=y)
            trace = pm.sample(samples, progressbar=progressbar)

        return model, trace

    @classmethod
    def model_returns_normal(cls, data, samples=500, progressbar=True):
        """
        Run a Bayesian model assuming returns are normally distributed.

        Parameters
        ----------
        :param data : pandas.Series:
            Series of simple returns of an algorithm or stock.
        :param samples : Int (optional)
            Number of posterior samples to draw.
        :param progressbar : Bool (optional), default True

        Returns
        -------
        model : pymc.Model object
            PyMC3 model containing all random variables.
        trace : pymc3.sampling.BaseTrace object
            A PyMC3 trace object that contains samples for each parameter
            of the posterior.
        """

        with pm.Model() as model:
            mu = pm.Normal('mean returns', mu=0, sd=.01, testval=data.mean())
            sigma = pm.HalfCauchy('volatility', beta=1, testval=data.std())
            returns = pm.Normal('returns', mu=mu, sd=sigma, observed=data)
            pm.Deterministic(
                'annual volatility',
                returns.distribution.variance ** .5 *
                np.sqrt(252))
            pm.Deterministic(
                'sharpe',
                returns.distribution.mean /
                returns.distribution.variance ** .5 *
                np.sqrt(252))

            trace = pm.sample(samples, progressbar=progressbar)
        return model, trace

    @classmethod
    def model_returns_t(cls, data, samples=500, progressbar=True):
        """
        Run Bayesian model assuming returns are Student-T distributed.

        Compared with the normal model, this model assumes returns are
        T-distributed and thus have a 3rd parameter (nu) that controls the
        mass in the tails.

        Parameters
        ----------
        :param data : pandas.Series:
            Series of simple returns of an algorithm or stock.
        :param samples : int, optional
            Number of posterior samples to draw.
        :param progressbar : bool, optional, default: True

        Returns
        -------
        model : pymc.Model object
            PyMC3 model containing all random variables.
        trace : pymc3.sampling.BaseTrace object
            A PyMC3 trace object that contains samples for each parameter
            of the posterior.
        """

        with pm.Model() as model:
            mu = pm.Normal('mean returns', mu=0, sd=.01, testval=data.mean())
            sigma = pm.HalfCauchy('volatility', beta=1, testval=data.std())
            nu = pm.Exponential('nu_minus_two', 1. / 10., testval=3.)

            returns = pm.StudentT('returns', nu=nu + 2, mu=mu, sd=sigma,
                                  observed=data)
            pm.Deterministic('annual volatility',
                             returns.distribution.variance ** .5 * np.sqrt(252))

            pm.Deterministic('sharpe', returns.distribution.mean /
                             returns.distribution.variance ** .5 *
                             np.sqrt(252))

            trace = pm.sample(samples, progressbar=progressbar)
        return model, trace

    @classmethod
    def model_best(cls, y1, y2, samples=1000, progressbar=True):
        """
        Bayesian Estimation Supersedes the T-Test

        This model runs a Bayesian hypothesis comparing if y1 and y2 come
        from the same distribution. Returns are assumed to be T-distributed.

        In addition, it computes annual volatility and Sharpe of in and
        out-of-sample periods.

        This model replicates the example used in:
        Kruschke, John. (2012) Bayesian estimation supersedes the t
        test. Journal of Experimental Psychology: General.

        Parameters
        ----------
        :param y1 : array-like
            Array of returns (e.g., in-sample)
        :param y2 : array-like
            Array of returns (e.g., out-of-sample)
        :param samples : int, optional
            Number of posterior samples to draw.
        :param progressbar: bool, optional, default True

        Returns
        -------
        model : pymc.Model object
            PyMC3 model containing all random variables.
        trace : pymc3.sampling.BaseTrace object
            A PyMC3 trace object that contains samples for each parameter
            of the posterior.

        See Also
        --------
        plot_stoch_vol : plotting of the stochastic volatility model
        """

        y = np.concatenate((y1, y2))

        mu_m = np.mean(y)
        mu_p = 0.000001 * 1 / np.std(y) ** 2

        sigma_low = np.std(y) / 1000
        sigma_high = np.std(y) * 1000
        with pm.Model() as model:
            group1_mean = pm.Normal('group1_mean', mu=mu_m, tau=mu_p,
                                    testval=y1.mean())
            group2_mean = pm.Normal('group2_mean', mu=mu_m, tau=mu_p,
                                    testval=y2.mean())
            group1_std = pm.Uniform('group1_std', lower=sigma_low,
                                    upper=sigma_high, testval=y1.std())
            group2_std = pm.Uniform('group2_std', lower=sigma_low,
                                    upper=sigma_high, testval=y2.std())
            nu = pm.Exponential('nu_minus_two', 1 / 29., testval=4.) + 2.

            returns_group1 = pm.StudentT('group1', nu=nu, mu=group1_mean,
                                         lam=group1_std ** -2, observed=y1)
            returns_group2 = pm.StudentT('group2', nu=nu, mu=group2_mean,
                                         lam=group2_std ** -2, observed=y2)

            diff_of_means = pm.Deterministic('difference of means',
                                             group2_mean - group1_mean)
            pm.Deterministic('difference of stds',
                             group2_std - group1_std)
            pm.Deterministic('effect size', diff_of_means /
                             pm.math.sqrt((group1_std ** 2 +
                                           group2_std ** 2) / 2))

            pm.Deterministic('group1_annual_volatility',
                             returns_group1.distribution.variance ** .5 *
                             np.sqrt(252))
            pm.Deterministic('group2_annual_volatility',
                             returns_group2.distribution.variance ** .5 *
                             np.sqrt(252))

            pm.Deterministic('group1_sharpe', returns_group1.distribution.mean /
                             returns_group1.distribution.variance ** .5 *
                             np.sqrt(252))
            pm.Deterministic('group2_sharpe', returns_group2.distribution.mean /
                             returns_group2.distribution.variance ** .5 *
                             np.sqrt(252))

            trace = pm.sample(samples, progressbar=progressbar)
        return model, trace

    @classmethod
    def model_stoch_vol(cls, data, samples=2000, progressbar=True):
        """
        Run a stochastic volatility model.

        This model estimates the volatility of a `returns` series over time.
        Returns are assumed to be T-distributed. lambda (width of
        T-distributed) is assumed to follow a random-walk.

        Parameters
        ----------
        :param data : pandas.Series
            Return series to model.
        :param samples : int, optional
            Posterior samples to draw.
        :param progressbar : bool, optional, default: True

        Returns
        -------
        model : pymc.Model object
            PyMC3 model containing all random variables.
        trace : pymc3.sampling.BaseTrace object
            A PyMC3 trace object that contains samples for each parameter
            of the posterior.

        See Also
        --------
        plot_stoch_vol : plotting of a stochastic volatility model
        """

        from pymc.distributions.timeseries import GaussianRandomWalk

        with pm.Model() as model:
            nu = pm.Exponential('nu', 1. / 10, testval=5.)
            sigma = pm.Exponential('sigma', 1. / .02, testval=.1)
            s = GaussianRandomWalk('s', sigma ** -2, shape=len(data))
            volatility_process = pm.Deterministic('volatility_process',
                                                  pm.math.exp(-2 * s))
            pm.StudentT('r', nu, lam=volatility_process, observed=data)

            trace = pm.sample(samples, progressbar=progressbar)

        return model, trace

    @classmethod
    def compute_bayes_cone(cls, preds, starting_value=1.):
        """
        Compute 5, 25, 75 and 95 percentiles of cumulative returns, used
        for the Bayesian cone.

        Parameters
        ----------
        preds : numpy.array
            Multiple (simulated) cumulative returns.
        starting_value : int (optional)
            Have cumulative returns start around this value.
            Default = 1.

        Returns
        -------
        dict of percentiles over time
            Dictionary mapping percentiles (5, 25, 75, 95) to a
            timeseries.
        """

        def scoreatpercentile(cum_preds, p):
            return [stats.scoreatpercentile(
                c, p) for c in cum_preds.T]

        cum_preds = np.cumprod(preds + 1, 1) * starting_value
        perc = {p: scoreatpercentile(cum_preds, p) for p in (5, 25, 75, 95)}

        return perc

    @classmethod
    def compute_consistency_score(cls, returns_test, preds):
        """
        Compute Bayesian consistency score.

        Parameters
        ----------
        returns_test : pd.Series
            Observed cumulative returns.
        preds : numpy.array
            Multiple (simulated) cumulative returns.

        Returns
        -------
        Consistency score:
            Score from 100 (returns_test perfectly on the median line of the
            Bayesian cone spanned by preds) to 0 (returns_test completely
            outside of Bayesian cone.)
        """

        returns_test_cum = Empyrical.cal_cum_returns(returns_test, starting_value=1.)
        cum_preds = np.cumprod(preds + 1, 1)

        q = [sp.stats.percentileofscore(cum_preds[:, i],
                                        returns_test_cum.iloc[i],
                                        kind='weak')
             for i in range(len(returns_test_cum))]
        # normalize to be from 100 (perfect median line) to 0 (completely outside
        # of cone)
        return 100 - np.abs(50 - np.mean(q)) / .5

    @classmethod
    def run_model(cls, model, returns_train, returns_test=None,
                  bmark=None, samples=500, ppc=False, progressbar=True):
        """
        Run one of the Bayesian models.

        Parameters
        ----------
        :param model : {'alpha_beta', 't', 'normal', 'best'}
            Which model to run
        :param returns_train : pd.Series
            Timeseries of simple returns
        :param returns_test : pd.Series (optional)
            Out-of-sample returns. Datetimes in returns_test will be added to
            returns_train as missing values and predictions will be generated
            for them.
        :param bmark : pd.Series or pd.DataFrame (optional) is
            Only used for alpha_beta to estimate regression coefficients.
            If bmark has more recent returns than returns_train, these dates
            will be treated as missing values and predictions will be
            generated for them taking market correlations into account.
        :param samples : int (optional)
            Number of posterior samples to draw.
        :param ppc : boolean (optional)
            Whether to run a posterior predictive check. Will generate
            samples of length returns_test.  Returns a second argument
            that contains the PPC of shape samples x len(returns_test).
        :param progressbar: bool (optional), default True

        Returns
        -------
        trace : pymc3.sampling.BaseTrace object
            A PyMC3 trace object that contains samples for each parameter
            of the posterior.

        ppc : numpy.array (if ppc==True)
           PPC of shape samples x-len(returns_test).
        """

        if model == 'alpha_beta':
            model, trace = Empyrical.model_returns_t_alpha_beta(returns_train,
                                                                bmark, samples,
                                                                progressbar=progressbar)
        elif model == 't':
            model, trace = Empyrical.model_returns_t(returns_train, samples,
                                                     progressbar=progressbar)
        elif model == 'normal':
            model, trace = Empyrical.model_returns_normal(returns_train, samples,
                                                          progressbar=progressbar)
        elif model == 'best':
            model, trace = Empyrical.model_best(returns_train, returns_test,
                                                samples=samples,
                                                progressbar=progressbar)
        else:
            raise NotImplementedError(
                'Model {} not found.'
                'Use alpha_beta, t, normal, or best.'.format(model))

        if ppc:
            ppc_samples = pm.sample_ppc(trace, samples=samples,
                                        model=model, size=len(returns_test),
                                        progressbar=progressbar)
            return trace, ppc_samples['returns']

        return trace

    @classmethod
    def daily_txns_with_bar_data(cls, transactions, market_data):
        """
        Sums the absolute value of shares traded in each name on each day.
        Add columns containing the closing price and total daily volume for
        each day-ticker combination.

        Parameters
        ----------
        transactions : pd.DataFrame
            Prices and `amounts` of executed trades. One row per trade.
            - See full explanation in tears.create_full_tear_sheet
        market_data : pd.Panel, use dict replace
            Contains "volume" and "price" DataFrames for the tickers
            in the dict of (name, dataframe)

        Returns
        -------
        txn_daily : pd.DataFrame
            Daily totals for transacted shares in each traded name.
            Price and volume columns for close price and daily volume for
            the corresponding ticker, respectively.
        """

        transactions.index.name = 'date'
        txn_daily = pd.DataFrame(transactions.assign(
            amount=abs(transactions.amount)).groupby(
            ['symbol', pd.Grouper(freq='D')]).sum()['amount'])
        txn_daily['price'] = market_data['price'].unstack()
        txn_daily['volume'] = market_data['volume'].unstack()

        txn_daily = txn_daily.reset_index().set_index('date')

        return txn_daily

    @classmethod
    def days_to_liquidate_positions(cls, positions, market_data,
                                    max_bar_consumption=0.2,
                                    capital_base=1e6,
                                    mean_volume_window=5):
        """
        Compute the number of days that would have been required to fully liquidate each position
        on each day, based on the trailing n day mean daily bar volume,
        and a limit on the proportion of a daily bar that we are allowed to consume.

        This analysis uses portfolio allocations and a provided capital base
        rather than the dollar values in the positions DataFrame to remove the
        effect of compounding on days to liquidate. In other words, this function
        assumes that the net liquidation portfolio value will always remain
        constant at capital_base.

        Parameters
        ----------
        positions: pd.DataFrame
            Contains daily position values including cash
            - See full explanation in tears.create_full_tear_sheet
        market_data : pd.Panel, 因为pd不再使用面板数据，尝试使用dict代替
            Panel with items axis of 'price' and 'volume' DataFrames.
            The major and minor axes should match those of the
            passed positions DataFrame (same dates and symbols).
        max_bar_consumption : float
            Max proportion of a daily bar that can be consumed in the
            process of liquidating a position.
        capital_base : integer
            Capital base multiplied by portfolio allocation to compute
            position value that needs liquidating.
        mean_volume_window : float
            Trailing window to use in mean volume calculation.

        Returns
        -------
        days_to_liquidate : pd.DataFrame
            Number of days required to fully liquidate daily positions.
            Datetime index, symbols as columns.
        """
        # print(market_data['volume'].info())
        # print(market_data['price'].info())
        dv = market_data['volume'] * market_data['price']
        # DV = (market_data[market_data.index.get_level_values(1) == 'volume'] *
        #       market_data[market_data.index.get_level_values(1) == 'price'])
        roll_mean_dv = dv.rolling(window=mean_volume_window,
                                  center=False).mean().shift()
        roll_mean_dv = roll_mean_dv.replace(0, np.nan)

        positions_alloc = Empyrical.get_percent_alloc(positions)
        positions_alloc = positions_alloc.drop('cash', axis=1)

        days_to_liquidate = (positions_alloc * capital_base) / \
                            (max_bar_consumption * roll_mean_dv)

        return days_to_liquidate.iloc[mean_volume_window:]

    @classmethod
    def get_max_days_to_liquidate_by_ticker(cls, positions, market_data,
                                            max_bar_consumption=0.2,
                                            capital_base=1e6,
                                            mean_volume_window=5,
                                            last_n_days=None):
        """
        Finds the longest estimated liquidation time for each traded
        name over the course of backtest (or last n days of the backtest).

        Parameters
        ----------
        positions: pd.DataFrame
            Contains daily position values including cash
            - See full explanation in tears.create_full_tear_sheet
        market_data : pd.Panel:
            Panel with items axis of 'price' and 'volume' DataFrames.
            The major and minor axes should match those of the
            passed positions DataFrame (same dates and symbols).
        max_bar_consumption : float
            Max proportion of a daily bar that can be consumed in the
            process of liquidating a position.
        capital_base : integer
            Capital base multiplied by portfolio allocation to compute
            position value that needs liquidating.
        mean_volume_window : float
            Trailing window to use in mean volume calculation.
        last_n_days : integer
            Compute for only the last n days of the passed backtest data.

        Returns
        -------
        days_to_liquidate : pd.DataFrame
            Max Number of days required to fully liquidate each traded name.
            Index of symbols. Columns for days_to_liquidate and the corresponding
            date and position_alloc on that day.
        """

        dtlp = Empyrical.days_to_liquidate_positions(positions, market_data,
                                                     max_bar_consumption=max_bar_consumption,
                                                     capital_base=capital_base,
                                                     mean_volume_window=mean_volume_window)

        if last_n_days is not None:
            dtlp = dtlp.loc[dtlp.index.max() - pd.Timedelta(days=last_n_days):]

        pos_alloc = Empyrical.get_percent_alloc(positions)
        pos_alloc = pos_alloc.drop('cash', axis=1)

        liq_desc = pd.DataFrame()
        liq_desc['days_to_liquidate'] = dtlp.unstack()
        liq_desc['pos_alloc_pct'] = pos_alloc.unstack() * 100
        # liq_desc.index.levels[0].name = 'symbol'
        # liq_desc.index.levels[1].name = 'date'
        liq_desc.index = liq_desc.index.set_names(['symbol', 'date'])

        worst_liq = liq_desc.reset_index().sort_values(
            'days_to_liquidate', ascending=False).groupby('symbol').first()

        return worst_liq

    @classmethod
    def get_low_liquidity_transactions(cls, transactions, market_data,
                                       last_n_days=None):
        """
        For each traded name, find the daily transaction total that consumed
        the greatest proportion of available daily bar volume.

        Parameters
        ----------
        transactions : pd.DataFrame
            Prices and `amounts` of executed trades. One row per trade.
             - See full explanation in create_full_tear_sheet.
        market_data : pd.Panel, use dict replace
            Panel with `items` axis of 'price' and 'volume' DataFrames.
            The major and minor axes should match those of the
            passed positions DataFrame (same dates and symbols).
        last_n_days : integer
            Compute for only the last n days of the passed backtest data.
        """

        txn_daily_w_bar = Empyrical.daily_txns_with_bar_data(transactions, market_data)
        txn_daily_w_bar.index.name = 'date'
        txn_daily_w_bar = txn_daily_w_bar.reset_index()

        if last_n_days is not None:
            md = txn_daily_w_bar.date.max() - pd.Timedelta(days=last_n_days)
            txn_daily_w_bar = txn_daily_w_bar[txn_daily_w_bar.date > md]

        bar_consumption = txn_daily_w_bar.assign(
            max_pct_bar_consumed=(
                                         txn_daily_w_bar.amount / txn_daily_w_bar.volume) * 100
        ).sort_values('max_pct_bar_consumed', ascending=False)
        max_bar_consumption = bar_consumption.groupby('symbol').first()

        return max_bar_consumption[['date', 'max_pct_bar_consumed']]

    @classmethod
    def apply_slippage_penalty(cls, returns, txn_daily, simulate_starting_capital,
                               backtest_starting_capital, impact=0.1):
        """
        Applies a quadratic volume share slippage model to daily returns based
        on the proportion of the observed historical daily bar dollar volume
        consumed by the strategy's trades. Scales the size of trades based
        on the ratio of the starting capital we wish to test to the starting
        capital of the passed backtest data.

        Parameters
        ----------
        returns : pd.Series
            Time series of daily returns.
        txn_daily : pd.Series
            Daily transaction totals, closing price, and daily volume for
            each traded name. See price_volume_daily_txns for more details.
        simulate_starting_capital : integer
            capital at which we want to test
        backtest_starting_capital: capital base at which backtest was
            origionally run. impact: See Zipline volume share slippage model
        impact : float
            Scales the size of the slippage penalty.

        Returns
        -------
        adj_returns : pd.Series
            Slippage penalty adjusted daily returns.
        """

        mult = simulate_starting_capital / backtest_starting_capital
        simulate_traded_shares = abs(mult * txn_daily.amount)
        simulate_traded_dollars = txn_daily.price * simulate_traded_shares
        simulate_pct_volume_used = simulate_traded_shares / txn_daily.volume

        penalties = simulate_pct_volume_used ** 2 * impact * simulate_traded_dollars

        daily_penalty = penalties.resample('D').sum()
        daily_penalty = daily_penalty.reindex(returns.index)
        daily_penalty = pd.to_numeric(daily_penalty, errors='coerce').fillna(0)
        # daily_penalty = daily_penalty.reindex(returns.index).fillna(0)

        # Since we are scaling the numerator of the penalties linearly
        # by capital base, it makes the most sense to scale the denominator
        # similarly. In other words, since we aren't applying compounding to
        # simulate_traded_shares, we shouldn't apply compounding to pv.
        portfolio_value = Empyrical.cum_returns(
            returns, starting_value=backtest_starting_capital) * mult

        adj_returns = returns - (daily_penalty / portfolio_value)

        return adj_returns

    @classmethod
    def get_percent_alloc(cls, values):
        """
        Determines a portfolio's allocations.

        Parameters
        ----------
        values : pd.DataFrame
            Contains position values or amounts.

        Returns
        -------
        allocations : pd.DataFrame
            Positions and their allocations.
        """

        return values.divide(
            values.sum(axis='columns'),
            axis='rows'
        )

    @classmethod
    def get_top_long_short_abs(cls, positions, top=10):
        """
        Finds the top long, short, and absolute positions.

        Parameters
        ----------
        positions : pd.DataFrame
            The positions that the strategy takes over time.
        top : int, optional
            How many of each to find (default 10).

        Returns
        -------
        df_top_long : pd.DataFrame
            Top long positions.
        df_top_short : pd.DataFrame
            Top short positions.
        df_top_abs : pd.DataFrame
            Top absolute positions.
        """

        positions = positions.drop('cash', axis='columns')
        df_max = positions.max()
        df_min = positions.min()
        df_abs_max = positions.abs().max()
        df_top_long = df_max[df_max > 0].nlargest(top)
        df_top_short = df_min[df_min < 0].nsmallest(top)
        df_top_abs = df_abs_max.nlargest(top)
        return df_top_long, df_top_short, df_top_abs

    @classmethod
    def get_max_median_position_concentration(cls, positions):
        """
        Finds the max and median long and short position concentrations
        in each time period specified by the index of positions.

        Parameters
        ----------
        positions : pd.DataFrame
            The positions that the strategy takes over time.

        Returns
        -------
        pd.DataFrame
            Columns are the max long, max short, median long, and median short
            position concentrations.Rows are time periods.
        """

        expos = Empyrical.get_percent_alloc(positions)
        expos = expos.drop('cash', axis=1)

        longs = expos.where(expos.apply(lambda x: x > 0))
        shorts = expos.where(expos.apply(lambda x: x < 0))

        alloc_summary = pd.DataFrame()
        alloc_summary['max_long'] = longs.max(axis=1)
        alloc_summary['median_long'] = longs.median(axis=1)
        alloc_summary['median_short'] = shorts.median(axis=1)
        alloc_summary['max_short'] = shorts.min(axis=1)

        return alloc_summary

    @classmethod
    def extract_pos(cls, positions, cash):
        """
        Extract position values from the backtest object as returned by
        get_backtest() on the Quantopian research platform.

        Parameters
        ----------
        positions : pd.DataFrame
            timeseries containing one row per symbol (and potentially
            duplicate datetime indices) and columns for amount and
            last_sale_price.
        cash : pd.Series
            timeseries containing cash in the portfolio.

        Returns
        -------
        pd.DataFrame
            Daily net position values.
             - See full explanation in tears.create_full_tear_sheet.
        """

        positions = positions.copy()
        positions['values'] = positions.amount * positions.last_sale_price

        cash.name = 'cash'

        values = positions.reset_index().pivot_table(index='index',
                                                     columns='sid',
                                                     values='values')

        if ZIPLINE:
            for asset in values.columns:
                if type(asset) in [Equity, Future]:
                    values[asset] = values[asset] * asset.price_multiplier

        values = values.join(cash).fillna(0)

        # NOTE: Set the name of DataFrame.columns to sid, to match the behavior
        # of DataFrame.join in earlier versions of pandas.
        values.columns.name = 'sid'

        return values

    @classmethod
    def get_sector_exposures(cls, positions, symbol_sector_map):
        """
        Sum position exposures by sector.

        Parameters
        ----------
        positions : pd.DataFrame
            Contains position values or amounts.
            - Example
                index         'AAPL'         'MSFT'        'CHK'        cash
                2004-01-09    13939.380     -15012.993    -403.870      1477.483
                2004-01-12    14492.630     -18624.870    142.630       3989.610
                2004-01-13    -13853.280    13653.640     -100.980      100.000
        symbol_sector_map : dict or pd.Series
            Security identifier to sector mapping.
            Security ids as keys/index, sectors as values.
            - `Example`:
                {'AAPL': 'Technology'
                 'MSFT': 'Technology'
                 'CHK': 'Natural Resources'}

        Returns
        -------
        sector_exp : pd.DataFrame
            Sectors and their allocations.
            - Example:
                index         'Technology' 'Natural Resources' cash
                2004-01-09 -1073.613 -403.870 1477.4830
                2004-01-12 -4132.240 142.630 3989.6100
                2004-01-13 -199.640 -100.980 100.0000
        """

        cash = positions['cash']
        positions = positions.drop('cash', axis=1)

        unmapped_pos = np.setdiff1d(positions.columns.values,
                                    list(symbol_sector_map.keys()))
        if len(unmapped_pos) > 0:
            warn_message = """Warning: Symbols {} have no sector mapping.
            They will not be included in sector allocations""".format(
                ", ".join(map(str, unmapped_pos)))
            warnings.warn(warn_message, UserWarning)

        sector_exp = positions.groupby(
            by=symbol_sector_map, axis=1).sum()

        sector_exp['cash'] = cash

        return sector_exp

    @classmethod
    def get_long_short_pos(cls, positions):
        """
        Determines the long and short allocations in a portfolio.

        Parameters
        ----------
        positions : pd.DataFrame
            The positions that the strategy takes over time.

        Returns
        -------
        df_long_short : pd.DataFrame
            Long and short allocations as a decimal
            percentage of the total net liquidation
        """

        pos_wo_cash = positions.drop('cash', axis=1)
        longs = pos_wo_cash[pos_wo_cash > 0].sum(axis=1).fillna(0)
        shorts = pos_wo_cash[pos_wo_cash < 0].sum(axis=1).fillna(0)
        cash = positions.cash
        net_liquidation = longs + shorts + cash
        df_pos = pd.DataFrame({'long': longs.divide(net_liquidation, axis='index'),
                               'short': shorts.divide(net_liquidation,
                                                      axis='index')})
        df_pos['net exposure'] = df_pos['long'] + df_pos['short']
        return df_pos

    @classmethod
    def compute_style_factor_exposures(cls, positions, risk_factor):
        """
        Return style factor exposure of an algorithm's positions

        Parameters
        ----------
        positions : pd.DataFrame
            Daily equity positions of algorithm, in dollars.
            - See full explanation in create_risk_tear_sheet

        risk_factor : pd.DataFrame
            Daily risk factor per asset.
            - DataFrame with dates as index and equities as columns
            - Example:
                             Equity(24 Equity(62
                               [AAPL]) [ABT])
            2017-04-03	-0.51284 1.39173
            2017-04-04	-0.73381 0.98149
            2017-04-05	-0.90132 1.13981
        """

        positions_wo_cash = positions.drop('cash', axis='columns')
        gross_exposure = positions_wo_cash.abs().sum(axis='columns')

        style_factor_exposure = positions_wo_cash.multiply(risk_factor) \
            .divide(gross_exposure, axis='index')
        tot_style_factor_exposure = style_factor_exposure.sum(axis='columns',
                                                              skipna=True)

        return tot_style_factor_exposure

    @classmethod
    def compute_sector_exposures(cls, positions, sectors, sector_dict=SECTORS):
        """
        Returns arrays of long, short and gross sector exposures of an algorithm's
        positions

        Parameters
        ----------
        positions : pd.DataFrame
            Daily equity positions of algorithm, in dollars.
            - See full explanation in compute_style_factor_exposures.

        sectors : pd.DataFrame
            Daily Morningstar sector code per asset
            - See full explanation in create_risk_tear_sheet

        sector_dict : dict or OrderedDict
            Dictionary of all sectors
            - Keys are sector codes (e.g., ints or strings) and values are sector
              names (which must be strings)
            - Defaults to Morningstar sectors
        """

        sector_ids = sector_dict.keys()

        long_exposures = []
        short_exposures = []
        gross_exposures = []
        net_exposures = []

        positions_wo_cash = positions.drop('cash', axis='columns')
        long_exposure = positions_wo_cash[positions_wo_cash > 0] \
            .sum(axis='columns')
        short_exposure = positions_wo_cash[positions_wo_cash < 0] \
            .abs().sum(axis='columns')
        gross_exposure = positions_wo_cash.abs().sum(axis='columns')

        for sector_id in sector_ids:
            in_sector = positions_wo_cash[sectors == sector_id]

            long_sector = in_sector[in_sector > 0] \
                .sum(axis='columns').divide(long_exposure)
            short_sector = in_sector[in_sector < 0] \
                .sum(axis='columns').divide(short_exposure)
            gross_sector = in_sector.abs().sum(axis='columns') \
                .divide(gross_exposure)
            net_sector = long_sector.subtract(short_sector)

            long_exposures.append(long_sector)
            short_exposures.append(short_sector)
            gross_exposures.append(gross_sector)
            net_exposures.append(net_sector)

        return long_exposures, short_exposures, gross_exposures, net_exposures

    @classmethod
    def compute_cap_exposures(cls, positions, caps):
        """
        Returns arrays of long, short and gross market cap exposures of an
        algorithm's positions

        Parameters
        ----------
        positions : pd.DataFrame
            Daily equity positions of algorithm, in dollars.
            - See full explanation in compute_style_factor_exposures.

        caps : pd.DataFrame
            Daily Morningstar sector code per asset
            - See full explanation in create_risk_tear_sheet
        """

        long_exposures = []
        short_exposures = []
        gross_exposures = []
        net_exposures = []

        positions_wo_cash = positions.drop('cash', axis='columns')
        tot_gross_exposure = positions_wo_cash.abs().sum(axis='columns')
        tot_long_exposure = positions_wo_cash[positions_wo_cash > 0] \
            .sum(axis='columns')
        tot_short_exposure = positions_wo_cash[positions_wo_cash < 0] \
            .abs().sum(axis='columns')

        for bucket_name, boundaries in CAP_BUCKETS.items():
            in_bucket = positions_wo_cash[(caps >= boundaries[0]) &
                                          (caps <= boundaries[1])]

            gross_bucket = in_bucket.abs().sum(axis='columns') \
                .divide(tot_gross_exposure)
            long_bucket = in_bucket[in_bucket > 0] \
                .sum(axis='columns').divide(tot_long_exposure)
            short_bucket = in_bucket[in_bucket < 0] \
                .sum(axis='columns').divide(tot_short_exposure)
            net_bucket = long_bucket.subtract(short_bucket)

            gross_exposures.append(gross_bucket)
            long_exposures.append(long_bucket)
            short_exposures.append(short_bucket)
            net_exposures.append(net_bucket)

        return long_exposures, short_exposures, gross_exposures, net_exposures

    @classmethod
    def compute_volume_exposures(cls, shares_held, volumes, percentile):
        """
        Returns arrays of pth percentile of long, short and gross volume exposures
        of an algorithm's held shares

        Parameters
        ----------
        shares_held : pd.DataFrame
            Daily number of shares held by an algorithm.
            - See full explanation in create_risk_tear_sheet

        volumes : pd.DataFrame
            Daily volume per asset
            - See full explanation in create_risk_tear_sheet

        percentile : float
            Percentile to use when computing and plotting volume exposures
            - See full explanation in create_risk_tear_sheet
        """

        shares_held = shares_held.replace(0, np.nan)

        shares_longed = shares_held[shares_held > 0]
        shares_shorted = -1 * shares_held[shares_held < 0]
        shares_grossed = shares_held.abs()

        longed_frac = shares_longed.divide(volumes)
        shorted_frac = shares_shorted.divide(volumes)
        grossed_frac = shares_grossed.divide(volumes)

        # NOTE: To work around a bug in `quantile` with nan-handling in
        #       pandas 0.18, use np.nanpercentile by applying to each row of
        #       the dataframe. This is fixed in pandas 0.19.
        #
        # longed_threshold = 100*longed_frac.quantile(percentile, axis='columns')
        # shorted_threshold = 100*shorted_frac.quantile(percentile, axis='columns')
        # grossed_threshold = 100*grossed_frac.quantile(percentile, axis='columns')

        longed_threshold = 100 * longed_frac.apply(
            partial(np.nanpercentile, q=100 * percentile),
            axis='columns',
        )
        shorted_threshold = 100 * shorted_frac.apply(
            partial(np.nanpercentile, q=100 * percentile),
            axis='columns',
        )
        grossed_threshold = 100 * grossed_frac.apply(
            partial(np.nanpercentile, q=100 * percentile),
            axis='columns',
        )

        return longed_threshold, shorted_threshold, grossed_threshold

    @classmethod
    def map_transaction(cls, txn):
        """
        Maps a single transaction row to a dictionary.

        Parameters
        ----------
        txn : pd.DataFrame
            A single transaction object to convert to a dictionary.

        Returns
        -------
        dict
            Mapped transaction.
        """

        if isinstance(txn['sid'], dict):
            sid = txn['sid']['sid']
            symbol = txn['sid']['symbol']
        else:
            sid = txn['sid']
            symbol = txn['sid']

        return {'sid': sid,
                'symbol': symbol,
                'price': txn['price'],
                'order_id': txn['order_id'],
                'amount': txn['amount'],
                'commission': txn['commission'],
                'dt': txn['dt']}

    @classmethod
    def make_transaction_frame(cls, transactions):
        """
        Formats a transaction DataFrame.

        Parameters
        ----------
        transactions : pd.DataFrame
            Contains improperly formatted transactional data.

        Returns
        -------
        df : pd.DataFrame
            Daily transaction volume and dollar amount.
             - See full explanation in tears.create_full_tear_sheet.
        """

        transaction_list = []
        for dt in transactions.index:
            txns = transactions.loc[dt]
            if len(txns) == 0:
                continue

            for txn in txns:
                txn = Empyrical.map_transaction(txn)
                transaction_list.append(txn)
        df = pd.DataFrame(sorted(transaction_list, key=lambda x: x['dt']))
        df['txn_dollars'] = -df['amount'] * df['price']

        df.index = list(map(pd.Timestamp, df.dt.values))
        return df

    @classmethod
    def get_txn_vol(cls, transactions):
        """
        Extract daily transaction data from a set of transaction objects.

        Parameters
        ----------
        transactions : pd.DataFrame
            Time series containing one row per symbol (and potentially
            duplicate datetime indices) and columns for amount and
            price.

        Returns
        -------
        pd.DataFrame
            Daily transaction volume and number of shares.
             - See full explanation in tears.create_full_tear_sheet.
        """

        txn_norm = transactions.copy()
        txn_norm.index = txn_norm.index.normalize()
        amounts = txn_norm.amount.abs()
        prices = txn_norm.price
        values = amounts * prices
        daily_amounts = amounts.groupby(amounts.index).sum()
        daily_values = values.groupby(values.index).sum()
        daily_amounts.name = "txn_shares"
        daily_values.name = "txn_volume"
        return pd.concat([daily_values, daily_amounts], axis=1)

    @classmethod
    def adjust_returns_for_slippage(cls, returns, positions, transactions,
                                    slippage_bps):
        """
        Apply a slippage penalty for every dollar traded.

        Parameters
        ----------
        returns : pd.Series
            Daily returns of the strategy, noncumulative.
             - See full explanation in create_full_tear_sheet.
        positions : pd.DataFrame
            Daily net position values.
             - See full explanation in create_full_tear_sheet.
        transactions : pd.DataFrame
            Prices and `amounts` of executed trades.One row per trade.
             - See full explanation in create_full_tear_sheet.
        slippage_bps: int/float
            Basis points of slippage to apply.

        Returns
        -------
        pd.Series
            Time series of daily returns, adjusted for slippage.
        """

        slippage = 0.0001 * slippage_bps
        portfolio_value = positions.sum(axis=1)
        pnl = portfolio_value * returns
        traded_value = Empyrical.get_txn_vol(transactions).txn_volume
        slippage_dollars = traded_value * slippage
        adjusted_pnl = pnl.add(-slippage_dollars, fill_value=0)
        adjusted_returns = returns * adjusted_pnl / pnl

        return adjusted_returns

    @classmethod
    def get_turnover(cls, positions, transactions, denominator='AGB'):
        """
         Value of purchases and sales divided
        by either the actual gross book or the portfolio value
        for the time step.

        Parameters
        ----------
        positions : pd.DataFrame
            Contains daily position values including cash.
            - See full explanation in tears.create_full_tear_sheet
        transactions : pd.DataFrame
            Prices and `amounts` of executed trades.One row per trade.
            - See full explanation in tears.create_full_tear_sheet
        denominator : str, optional
            Either 'AGB' or 'portfolio_value', default AGB.
            - AGB (Actual gross book) is the gross market
            value (GMV) of the specific algo being analyzed.
            Swapping out an entire portfolio of stocks for
            another will yield 200% turnover, not 100%, since
            transactions are being made for both sides.
            - We use average of the previous and the current end-of-period
            AGB to avoid singularities when trading only into or
            out of an entire book in one trading period.
            - Portfolio_value is the total value of the algo's
            positions end-of-period, including cash.

        Returns
        -------
        turnover_rate : pd.Series
            timeseries of portfolio turnover rates.
        """

        txn_vol = Empyrical.get_txn_vol(transactions)
        traded_value = txn_vol.txn_volume

        if denominator == 'AGB':
            # Actual gross book is the same thing as the algo's GMV
            # We want our denom to be avg(AGB previous, AGB current)
            agb = positions.drop('cash', axis=1).abs().sum(axis=1)
            denom = agb.rolling(2).mean()

            # Since the first value of pd.rolling returns NaN, we
            # set our "day 0" AGB to 0.
            denom.iloc[0] = agb.iloc[0] / 2
        elif denominator == 'portfolio_value':
            denom = positions.sum(axis=1)
        else:
            raise ValueError(
                "Unexpected value for denominator '{}'. The "
                "denominator parameter must be either 'AGB'"
                " or 'portfolio_value'.".format(denominator)
            )

        denom.index = denom.index.normalize()
        turnover = traded_value.div(denom, axis='index')
        # 增加一行代码，处理inf的值，避免画图的时候出错
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)
            turnover = turnover.replace([np.inf, -np.inf], np.nan).infer_objects(copy=False)
        turnover = turnover.fillna(0)
        turnover = turnover.astype('float')
        return turnover

    @classmethod
    def perf_stats(cls, returns, factor_returns=None, positions=None,
                   transactions=None, turnover_denom='AGB'):
        """
        Calculates various performance metrics of a strategy, for use in
        plotting.show_perf_stats.

        Parameters
        ----------
        returns : pd.Series
            Daily returns of the strategy, noncumulative.
             - See full explanation in tears.create_full_tear_sheet.
        factor_returns : pd.Series, optional
            Daily noncumulative returns of the benchmark factor to which betas are
            computed. Usually a benchmark such as market returns.
             - This is in the same style as returns.
             - If `None`, do not compute the alpha, beta, and information ratio.
        positions : pd.DataFrame
            Daily net position values.
             - See full explanation in tears.create_full_tear_sheet.
        transactions : pd.DataFrame
            Prices and `amounts` of executed trades. One row per trade.
            - See full explanation in tears.create_full_tear_sheet.
        turnover_denom : str
            Either AGB or portfolio_value, default AGB.
            - See full explanation in txn.get_turnover.

        Returns
        -------
        pd.Series
            Performance metrics.
        """

        stats = pd.Series()
        for stat_func in SIMPLE_STAT_FUNCS:
            stats[STAT_FUNC_NAMES[stat_func.__name__]] = stat_func(returns)

        if positions is not None:
            stats['Gross leverage'] = Empyrical.gross_lev(positions).mean()
            if transactions is not None:
                stats['Daily turnover'] = Empyrical.get_turnover(positions,
                                                                 transactions,
                                                                 turnover_denom).mean()
        if factor_returns is not None:
            for stat_func in FACTOR_STAT_FUNCS:
                res = stat_func(returns, factor_returns)
                stats[STAT_FUNC_NAMES[stat_func.__name__]] = res

        return stats

    @classmethod
    def perf_stats_bootstrap(cls, returns, factor_returns=None, return_stats=True,
                             **_kwargs):
        """Calculates various bootstrapped performance metrics of a strategy.

        Parameters
        ----------
        returns : pd.Series
            Daily returns of the strategy, noncumulative.
             - See full explanation in tears.create_full_tear_sheet.
        factor_returns : pd.Series, optional
            Daily noncumulative returns of the benchmark factor to which betas are
            computed. Usually a benchmark such as market returns.
             - This is in the same style as returns.
             - If `None`, do not compute the alpha, beta, and information ratio.
        return_stats : boolean (optional)
            If True, returns a DataFrame of mean, median, 5 and 95 percentiles
            for each perf metric.
            If False, return a DataFrame with the bootstrap samples for
            each perf metric.

        Returns
        -------
        pd.DataFrame
            if return_stats is True:
            - Distributional statistics of bootstrapped sampling
            distribution of performance metrics.
            If return_stats is False:
            - Bootstrap samples for each performance metric.
        """

        bootstrap_values = OrderedDict()

        for stat_func in SIMPLE_STAT_FUNCS:
            stat_name = STAT_FUNC_NAMES[stat_func.__name__]
            bootstrap_values[stat_name] = Empyrical.calc_bootstrap(stat_func,
                                                                   returns)

        if factor_returns is not None:
            for stat_func in FACTOR_STAT_FUNCS:
                stat_name = STAT_FUNC_NAMES[stat_func.__name__]
                bootstrap_values[stat_name] = Empyrical.calc_bootstrap(
                    stat_func,
                    returns,
                    factor_returns=factor_returns)

        bootstrap_values = pd.DataFrame(bootstrap_values)

        if return_stats:
            stats = bootstrap_values.apply(Empyrical.calc_distribution_stats)
            return stats.T[['mean', 'median', '5%', '95%']]
        else:
            return bootstrap_values

    @classmethod
    def calc_bootstrap(cls, func, returns, *args, **kwargs):
        """Performs a bootstrap analysis on a user-defined function returning
        a summary statistic.

        Parameters
        ----------
        func : function
            either takes a single array (commonly returns)
            or two arrays (commonly returns and factor returns) and
            returns a single value (commonly a summary
            statistic). Additional args and kwargs are passed as well.
        returns : pd.Series
            Daily returns of the strategy, noncumulative.
             - See full explanation in tears.create_full_tear_sheet.
        factor_returns : pd.Series, optional
            Daily noncumulative returns of the benchmark factor to which betas are
            computed. Usually a benchmark such as market returns.
             - This is in the same style as returns.
        :n_samples : int, optional
            Number of bootstrap samples to draw. Default is 1000.
            Increasing this will lead to more stable / accurate estimates.

        Returns
        -------
        numpy.ndarray
            Bootstrapped sampling distribution of passed in func.
        """

        n_samples = kwargs.pop('n_samples', 1000)
        out = np.empty(n_samples)

        factor_returns = kwargs.pop('factor_returns', None)

        for i in range(n_samples):
            idx = np.random.randint(len(returns), size=len(returns))
            returns_i = returns.iloc[idx].reset_index(drop=True)
            if factor_returns is not None:
                factor_returns_i = factor_returns.iloc[idx].reset_index(drop=True)
                out[i] = func(returns_i, factor_returns_i,
                              *args, **kwargs)
            else:
                out[i] = func(returns_i,
                              *args, **kwargs)

        return out

    @classmethod
    def calc_distribution_stats(cls, x):
        """Calculate various summary statistics of data.

        Parameters
        ----------
        x : numpy.ndarray or pandas.Series
            Array to compute summary statistics for.

        Returns
        -------
        `pandas.Series` type
            Series containing mean, median, std, as well as 5, 25, 75 and
            95 percentiles of passed in values.
        """

        return pd.Series({'mean': np.mean(x),
                          'median': np.median(x),
                          'std': np.std(x),
                          '5%': np.percentile(x, 5),
                          '25%': np.percentile(x, 25),
                          '75%': np.percentile(x, 75),
                          '95%': np.percentile(x, 95),
                          'IQR': np.subtract.reduce(
                              np.percentile(x, [75, 25])),
                          })

    @classmethod
    def get_max_drawdown_underwater(cls, underwater):
        """
        Determines peak, valley, and recovery dates given an 'underwater'
        DataFrame.

        An underwater DataFrame is a DataFrame that has precomputed
        rolling drawdown.

        Parameters
        ----------
        underwater : pd.Series
           Underwater returns (rolling drawdown) of a strategy.

        Returns
        -------
        peak : datetime
            The maximum drawdown's peak.
        valley : datetime
            The maximum drawdown's valley.
        recovery : datetime
            The maximum drawdown's recovery.
        """

        # valley = np.argmin(underwater)  # end of the period
        # # print(valley)
        # # Find first 0
        # peak = underwater[:valley][underwater[:valley] == 0].index[-1]
        # # Find last 0
        # try:
        #     recovery = underwater[valley:][underwater[valley:] == 0].index[0]
        # except IndexError:
        #     recovery = np.nan  # drawdown not recovered
        # # print("get_max_drawdown_underwater",underwater)
        # # print("get_max_drawdown_underwater",underwater[:valley][underwater[:valley] == 0])
        # # print("get_max_drawdown_underwater",peak, valley, recovery)
        # # add a code,change index to datetime
        # valley = list(underwater.index)[valley]
        # return peak, valley, recovery
        # 原版
        valley = underwater.idxmin()  # end of the period
        # Find first 0
        peak = underwater[:valley][underwater[:valley] == 0].index[-1]
        # Find last 0
        try:
            recovery = underwater[valley:][underwater[valley:] == 0].index[0]
        except IndexError:
            recovery = np.nan  # drawdown isn't recovered
        # `print(peak, valley, recovery)`
        return peak, valley, recovery

    @classmethod
    def get_max_drawdown(cls, returns):
        """
        Determines the maximum drawdown of a strategy.

        Parameters
        ----------
        returns : pd.Series
            Daily returns of the strategy, noncumulative.
            - See full explanation in: func:`~pyfolio.timeseries.cum_returns`.

        Returns
        -------
        float
            Maximum drawdown.

        Note
        -----
        See https://en.wikipedia.org/wiki/Drawdown_(economics) for more details.
        """

        returns = returns.copy()
        df_cum = Empyrical.cum_returns(returns, 1.0)
        running_max = np.maximum.accumulate(df_cum)
        underwater = df_cum / running_max - 1
        return Empyrical.get_max_drawdown_underwater(underwater)

    @classmethod
    def get_top_drawdowns(cls, returns, top=10):
        """
        Finds top drawdowns, sorted by drawdown amount.

        Parameters
        ----------
        returns : pd.Series
            Daily returns of the strategy, noncumulative.
             - See full explanation in tears.create_full_tear_sheet.
        top : int, optional
            The `amount` of top drawdowns to find (default 10).

        Returns
        -------
        drawdowns : list tye
            List of drawdown peaks, valleys, and recoveries. See get_max_drawdown.
        """

        returns = returns.copy()
        df_cum = Empyrical.cum_returns(returns, 1.0)
        running_max = np.maximum.accumulate(df_cum)
        underwater = df_cum / running_max - 1
        # print("df_cum",df_cum)
        # print("running_max",running_max)
        # print("underwater",underwater)
        drawdowns = []
        for t in range(top):
            # print("len(underwater)",len(underwater))
            peak, valley, recovery = Empyrical.get_max_drawdown_underwater(underwater)
            # Slice out draw-down period
            if not pd.isnull(recovery):
                underwater.drop(underwater[peak: recovery].index[1:-1],
                                inplace=True)
            else:
                # the drawdown has not ended yet
                underwater = underwater.loc[:peak]
            # print("get_top_drawdowns",peak, valley, recovery)
            drawdowns.append((peak, valley, recovery))
            if (len(returns) == 0) or (len(underwater) == 0):
                break
        # print(drawdowns)
        return drawdowns

    @classmethod
    def gen_drawdown_table(cls, returns, top=10):
        """
        Places top drawdowns in a table.

        Parameters
        ----------
        returns : pd.Series
            Daily returns of the strategy, noncumulative.
             - See full explanation in tears.create_full_tear_sheet.
        top : int, optional
            The `amount` of top drawdowns to find (default 10).

        Returns
        -------
        df_drawdowns : pd.DataFrame
            Information about top drawdowns.
        """

        df_cum = Empyrical.cum_returns(returns, 1.0)
        drawdown_periods = Empyrical.get_top_drawdowns(returns, top=top)
        df_drawdowns = pd.DataFrame(index=list(range(top)),
                                    columns=['Net drawdown in %',
                                             'Peak date',
                                             'Valley date',
                                             'Recovery date',
                                             'Duration'])
        # print(df_drawdowns)
        # print(drawdown_periods)
        for i, (peak, valley, recovery) in enumerate(drawdown_periods):
            if pd.isnull(recovery):
                df_drawdowns.loc[i, 'Duration'] = np.nan
            else:
                df_drawdowns.loc[i, 'Duration'] = len(pd.date_range(peak,
                                                                    recovery,
                                                                    freq='B'))
            # to_pydatetime()疑似是老的API，使用pd.to_datetime替代
            # df_drawdowns.loc[i, 'Peak date'] = (peak.to_pydatetime()
            #                                     .strftime('%Y-%m-%d'))
            # df_drawdowns.loc[i, 'Valley date'] = (valley.to_pydatetime()
            #                                       .strftime('%Y-%m-%d'))
            # if isinstance(recovery, float):
            #     df_drawdowns.loc[i, 'Recovery date'] = recovery
            # else:
            #     df_drawdowns.loc[i, 'Recovery date'] = (recovery.to_pydatetime()
            #                                             .strftime('%Y-%m-%d'))

            df_drawdowns.loc[i, 'Peak date'] = (pd.to_datetime(peak).strftime('%Y-%m-%d'))
            df_drawdowns.loc[i, 'Valley date'] = (pd.to_datetime(valley).strftime('%Y-%m-%d'))

            if isinstance(recovery, float):
                df_drawdowns.loc[i, 'Recovery date'] = recovery
            else:
                df_drawdowns.loc[i, 'Recovery date'] = (pd.to_datetime(recovery).strftime('%Y-%m-%d'))

            df_drawdowns.loc[i, 'Net drawdown in %'] = ((df_cum.loc[peak] - df_cum.loc[valley]) / df_cum.loc[
                peak]) * 100

        df_drawdowns['Peak date'] = pd.to_datetime(df_drawdowns['Peak date'])
        df_drawdowns['Valley date'] = pd.to_datetime(df_drawdowns['Valley date'])
        df_drawdowns['Recovery date'] = pd.to_datetime(df_drawdowns['Recovery date'])
        # print(df_drawdowns)
        return df_drawdowns

    @classmethod
    def rolling_volatility(cls, returns, rolling_vol_window):
        """
        Determines the rolling volatility of a strategy.

        Parameters
        ----------
        returns : pd.Series
            Daily returns of the strategy, noncumulative.
             - See full explanation in tears.create_full_tear_sheet.
        rolling_vol_window : int
            Length of the rolling window, in days, over which to compute.

        Returns
        -------
        pd.Series
            Rolling volatility.
        """

        return returns.rolling(rolling_vol_window).std() \
            * np.sqrt(APPROX_BDAYS_PER_YEAR)

    @classmethod
    def rolling_sharpe(cls, returns, rolling_sharpe_window):
        """
        Determines the rolling Sharpe ratio of a strategy.

        Parameters
        ----------
        returns : pd.Series
            Daily returns of the strategy, noncumulative.
             - See full explanation in tears.create_full_tear_sheet.
        rolling_sharpe_window : int
            Length of the rolling window, in days, over which to compute.

        Returns
        -------
        pd.Series
            Rolling Sharpe ratio.

        Note
        -----
        See https://en.wikipedia.org/wiki/Sharpe_ratio for more details.
        """
        avg_returns = returns.rolling(rolling_sharpe_window).mean()
        std_returns = returns.rolling(rolling_sharpe_window).std()
        return avg_returns / std_returns * np.sqrt(APPROX_BDAYS_PER_YEAR)

    @classmethod
    def simulate_paths(cls, is_returns, num_days,
                       _starting_value=1, num_samples=1000, random_seed=None):
        """
        Generate alternate paths using available values from in-sample returns.

        Parameters
        ----------
        is_returns : pandas.core.frame.DataFrame
            Non-cumulative in-sample returns.
        num_days : int
            Number of days to project the probability cone forward.
        _starting_value : int or float
            Starting value of the out sample period.
        num_samples : int
            Number of samples to draw from the in-sample daily returns.
            Each sample will be an array with length num_days.
            A higher number of samples will generate a more accurate
            bootstrap cone.
        random_seed : int
            Seed for the pseudorandom number generator used by the pandas
            sample method.

        Returns
        -------
        samples : numpy.ndarray
        """

        samples = np.empty((num_samples, num_days))
        seed = np.random.RandomState(seed=random_seed)
        for i in range(num_samples):
            samples[i, :] = is_returns.sample(num_days, replace=True,
                                              random_state=seed)

        return samples

    @classmethod
    def summarize_paths(cls, samples, cone_std=(1., 1.5, 2.), starting_value=1.):
        """
        Generate the upper and lower bounds of an n standard deviation
        cone of forecasted cumulative returns.

        Parameters
        ----------
        :param samples : numpy.ndarray
            Alternative paths, or series of possible outcomes.
        :param cone_std : list of int/float
            Number of standard deviations to use in the boundaries of
            the cone. If multiple values are passed, cone bounds will
            be generated for each value.
        :param starting_value: default 1

        Returns
        -------
        samples : pandas.core.frame.DataFrame

        """

        cum_samples = Empyrical.cum_returns(samples.T,
                                            starting_value=starting_value).T

        cum_mean = cum_samples.mean(axis=0)
        cum_std = cum_samples.std(axis=0)

        if isinstance(cone_std, (float, int)):
            cone_std = [cone_std]

        # cone_bounds = pd.DataFrame(columns=pd.Float64Index([]))
        cone_bounds = pd.DataFrame(columns=pd.Index([], dtype='float64'))
        for num_std in cone_std:
            cone_bounds.loc[:, float(num_std)] = cum_mean + cum_std * num_std
            cone_bounds.loc[:, float(-num_std)] = cum_mean - cum_std * num_std

        return cone_bounds

    @classmethod
    def forecast_cone_bootstrap(cls, is_returns, num_days, cone_std=(1., 1.5, 2.),
                                starting_value=1, num_samples=1000,
                                random_seed=None):
        """
        Determines the upper and lower bounds of an n standard deviation
        cone of forecasted cumulative returns. Future cumulative mean and
        standard deviation are computed by repeatedly sampling from the
        in-sample daily returns (i.e., bootstrap). This cone is non-parametric,
        meaning it does not assume that returns are normally distributed.

        Parameters
        ----------
        is_returns : pd.Series
            In-sample daily returns of the strategy, noncumulative.
             - See full explanation in tears.create_full_tear_sheet.
        num_days : int
            Number of days to project the probability cone forward.
        cone_std : int, float, or list of int/float
            Number of standard deviations to use in the boundaries of
            the cone. If multiple values are passed, cone bounds will
            be generated for each value.
        starting_value : int or float
            Starting value of the out sample period.
        num_samples : int
            Number of samples to draw from the in-sample daily returns.
            Each sample will be an array with length num_days.
            A higher number of samples will generate a more accurate
            bootstrap cone.
        random_seed : int
            Seed for the pseudorandom number generator used by the pandas
            sample method.

        Returns
        -------
        pd.DataFrame
            Contains upper and lower cone boundaries. Column names are
            strings corresponding to the number of standard deviations
            above (positive) or below (negative) the projected mean
            cumulative returns.
        """

        samples = Empyrical.simulate_paths(
            is_returns=is_returns,
            num_days=num_days,
            _starting_value=starting_value,
            num_samples=num_samples,
            random_seed=random_seed
        )

        cone_bounds = Empyrical.summarize_paths(
            samples=samples,
            cone_std=cone_std,
            starting_value=starting_value
        )

        return cone_bounds

    @classmethod
    def extract_interesting_date_ranges(cls, returns):
        """
        Extracts returns based on interesting events. See
        gen_date_range_interesting.

        Parameters
        ----------
        returns : pd.Series
            Daily returns of the strategy, noncumulative.
             - See full explanation in tears.create_full_tear_sheet.

        Returns
        -------
        ranges : OrderedDict
            Date ranges, with returns, of all valid events.
        """

        returns_dupe = returns.copy()
        returns_dupe.index = returns_dupe.index.map(pd.Timestamp)
        ranges = OrderedDict()
        for name, (start, end) in PERIODS.items():
            try:
                period = returns_dupe.loc[start:end]
                if len(period) == 0:
                    continue
                ranges[name] = period
            except BaseException as e:
                print(e)
                continue

        return ranges

    @classmethod
    def var_cov_var_normal(cls, p, c, mu=0, sigma=1):
        """
        Variance-covariance calculation of daily Value-at-Risk in a
        portfolio.

        Parameters
        ----------
        :param p : float
            Portfolio value.
        :param c : float
            Confidence level.
        :param mu : float, optional
            Mean.
        :param sigma:

        Returns
        -------
        float
        """

        alpha = sp.stats.norm.ppf(1 - c, mu, sigma)
        return p - p * (alpha + 1)

    @classmethod
    def common_sense_ratio(cls, returns):
        """
        Common sense ratio is the multiplication of the tail ratio and the
        Gain-to-Pain-Ratio -- sum(profits) / sum(losses).

        See https://bit.ly/1ORzGBk for more information on the motivation of
        this metric.


        Parameters
        ----------
        returns : pd.Series
            Daily returns of the strategy, noncumulative.
             - See full explanation in tears.create_full_tear_sheet.

        Returns
        -------
        float
            common sense ratio
        """

        return Empyrical.tail_ratio(returns) * \
            (1 + Empyrical.annual_return(returns))

    @classmethod
    def normalize(cls, returns, starting_value=1):
        """
        Normalizes a returns timeseries based on the first value.

        Parameters
        ----------
        returns : pd.Series
            Daily returns of the strategy, noncumulative.
             - See full explanation in tears.create_full_tear_sheet.
        starting_value : float, optional
           The starting returns (default 1).

        Returns
        -------
        pd.Series
            Normalized returns.
        """

        return starting_value * (returns / returns.iloc[0])

    @classmethod
    def rolling_beta(cls, returns, factor_returns,
                     rolling_window=APPROX_BDAYS_PER_MONTH * 6):
        """
        Determines the rolling beta of a strategy.

        Parameters
        ----------
        returns : pd.Series
            Daily returns of the strategy, noncumulative.
             - See full explanation in tears.create_full_tear_sheet.
        factor_returns : pd.Series or pd.DataFrame
            Daily noncumulative returns of the benchmark factor to which betas are
            computed. Usually a benchmark such as market returns.
             - If DataFrame is passed, computes rolling beta for each column.
             - This is in the same style as returns.
        rolling_window : int, optional
            The size of the rolling window, in days, over which to compute
            beta (default 6 months).

        Returns
        -------
        pd.Series
            Rolling beta.

        Note
        -----
        See https://en.wikipedia.org/wiki/Beta_(finance) for more details.
        """

        if factor_returns.ndim > 1:
            # Apply column-wise
            return factor_returns.apply(partial(Empyrical.rolling_beta, returns),
                                        rolling_window=rolling_window)
        else:
            out = pd.Series(index=returns.index)
            for beg, end in zip(returns.index[0:-rolling_window],
                                returns.index[rolling_window:]):
                out.loc[end] = Empyrical.beta(
                    returns.loc[beg:end],
                    factor_returns.loc[beg:end])

            return out

    @classmethod
    def rolling_regression(cls, returns, factor_returns,
                           rolling_window=APPROX_BDAYS_PER_MONTH * 6,
                           nan_threshold=0.1):
        """
        Computes rolling factor betas using a multivariate linear regression
        (separate linear regressions are problematic because the factors may be
        confounded).

        Parameters
        ----------
        returns : pd.Series
            Daily returns of the strategy, noncumulative.
             - See full explanation in tears.create_full_tear_sheet.
        factor_returns : pd.DataFrame
            Daily noncumulative returns of the benchmark factor to which betas are
            computed. Usually a benchmark such as market returns.
             - Computes rolling beta for each column.
             - This is in the same style as returns.
        rolling_window : int, optional
            The `days` window over which to compute the beta. Defaults to 6 months.
        nan_threshold : float, optionally, If there is more than this fraction of NaNs,
            the rolling regression for the given date will be skipped.

        Returns
        -------
        pandas.DataFrame type
            DataFrame containing rolling beta coefficients to SMB, HML and UMD
        """

        # We need to drop NaNs to regress
        ret_no_na = returns.dropna()

        columns = ['alpha'] + factor_returns.columns.tolist()
        rolling_risk = pd.DataFrame(columns=columns,
                                    index=ret_no_na.index)

        rolling_risk.index.name = 'dt'

        for beg, end in zip(ret_no_na.index[:-rolling_window],
                            ret_no_na.index[rolling_window:]):
            returns_period = ret_no_na[beg:end]
            factor_returns_period = factor_returns.loc[returns_period.index]

            if np.all(factor_returns_period.isnull().mean()) < nan_threshold:
                factor_returns_period_dnan = factor_returns_period.dropna()
                reg = linear_model.LinearRegression(fit_intercept=True).fit(
                    factor_returns_period_dnan,
                    returns_period.loc[factor_returns_period_dnan.index])
                rolling_risk.loc[end, factor_returns.columns] = reg.coef_
                rolling_risk.loc[end, 'alpha'] = reg.intercept_

        return rolling_risk

    @classmethod
    def gross_lev(cls, positions):
        """
        Calculates the gross leverage of a strategy.

        Parameters
        ----------
        positions : pd.DataFrame
            Daily net position values.
             - See full explanation in tears.create_full_tear_sheet.

        Returns
        -------
        pd.Series
            Gross leverage.
        """

        exposure = positions.drop('cash', axis=1).abs().sum(axis=1)
        return exposure / positions.sum(axis=1)

    @classmethod
    def value_at_risk(cls, returns, period=None, sigma=2.0):
        """
        Get value at risk (VaR).

        Parameters
        ----------
        returns : pd.Series
            Daily returns of the strategy, noncumulative.
             - See full explanation in tears.create_full_tear_sheet.
        period : str, optional
            Period over which to calculate VaR. Set to 'weekly',
            'monthly', or 'yearly', otherwise defaults to a period of
            returns (typically daily).
        sigma : float, optional
            Standard deviations of VaR, default 2.
        """
        if period is not None:
            returns_agg = Empyrical.aggregate_returns(returns, period)
        else:
            returns_agg = returns.copy()

        value_at_risk = returns_agg.mean() - sigma * returns_agg.std()
        return value_at_risk

    @classmethod
    def agg_all_long_short(cls, round_trips, col, stats_dict):
        # Aggregating for all trades
        print("stats_dict = ", stats_dict)
        stats_all = (round_trips.assign(ones=1).groupby('ones')[col])
        stats_all = stats_all.agg(stats_dict)
        stats_all = stats_all.T.rename(columns={1.0: 'All trades'})

        # Aggregating for long and short trades
        # Use `rename(columns=...)` instead of `rename_axis`
        stats_long_short = (round_trips.groupby('long')[col])
        stats_long_short = stats_long_short.agg(stats_dict)
        stats_long_short = stats_long_short.T.rename(columns={False: 'Short trades', True: 'Long trades'})

        # Join the two results
        return stats_all.join(stats_long_short)

    @classmethod
    def agg_all_long_short(cls, round_trips, col, stats_dict):
        # Separate custom functions from built-in functions
        custom_funcs = {k: v for k, v in stats_dict.items() if callable(v)}
        built_in_funcs = [v for k, v in stats_dict.items() if not callable(v)]

        # Aggregating for all trades
        stats_all = (round_trips.assign(ones=1).groupby('ones')[col])

        # Apply custom functions manually
        stats_all_custom = {}
        for func_name, func in custom_funcs.items():
            stats_all_custom[func_name] = stats_all.apply(func)
        stats_all_custom = pd.DataFrame(stats_all_custom)

        # Apply built-in functions
        stats_all_built_in = stats_all.agg(built_in_funcs)

        # Combine results
        stats_all = pd.concat([stats_all_custom, stats_all_built_in], axis=1)
        stats_all = stats_all.T.rename(columns={1.0: 'All trades'})

        # Aggregating for long and short trades
        stats_long_short = (round_trips.groupby('long')[col])

        # Apply custom functions manually
        stats_long_short_custom = {}
        for func_name, func in custom_funcs.items():
            stats_long_short_custom[func_name] = stats_long_short.apply(func)
        stats_long_short_custom = pd.DataFrame(stats_long_short_custom)

        # Apply built-in functions
        stats_long_short_built_in = stats_long_short.agg(built_in_funcs)

        # Combine results
        stats_long_short = pd.concat([stats_long_short_custom, stats_long_short_built_in], axis=1)
        stats_long_short = stats_long_short.T.rename(columns={False: 'Short trades', True: 'Long trades'})

        # Join the two results
        return stats_all.join(stats_long_short)

    @classmethod
    def _groupby_consecutive(cls, txn, max_delta=pd.Timedelta('8h')):
        """Merge transactions of the same direction separated by less than
        max_delta time duration.

        Parameters
        ----------
        transactions : pd.DataFrame
            Prices and amounts of executed round_trips. One row per trade.
            - See full explanation in tears.create_full_tear_sheet

        max_delta : pandas.Timedelta (optional)
            Merge transactions in the same direction separated by less
            than max_delta time duration.


        Returns
        -------
        transactions : pd.DataFrame

        """

        def vwap(transaction):
            if transaction.amount.sum() == 0:
                warnings.warn('Zero transacted shares, setting vwap to nan.')
                return np.nan
            return (transaction.amount * transaction.price).sum() / \
                transaction.amount.sum()

        out = []
        for sym, t in txn.groupby('symbol'):
            t = t.sort_index()
            t.index.name = 'dt'
            t.index = pd.to_datetime(t.index)
            t = t.reset_index()

            t['order_sign'] = t.amount > 0
            t['block_dir'] = (t.order_sign.shift(
                1) != t.order_sign).astype(int).cumsum()
            t['block_time'] = ((t.dt - t.dt.shift(1)) > max_delta).astype(int).cumsum()
            # grouped_price = (t.groupby(('block_dir',
            #                            'block_time'))
            #                   .apply(vwap))
            # grouped_price = t.groupby(['block_dir', 'block_time']).apply(vwap)
            grouped_price = t.groupby(['block_dir', 'block_time'])[['amount', 'price']].apply(vwap)
            grouped_price.name = 'price'
            grouped_rest = t.groupby(['block_dir', 'block_time']).agg({
                'amount': 'sum',
                'symbol': 'first',
                'dt': 'first'})

            grouped = grouped_rest.join(grouped_price)

            out.append(grouped)

        out = pd.concat(out)
        out = out.set_index('dt')
        return out

    @classmethod
    def extract_round_trips(cls, transactions,
                            portfolio_value=None):
        """Group transactions into "round trips". First, transactions are
        grouped by day and directionality. Then, long and short
        transactions are matched to create round-trip round_trips for which
        PnL, duration and returns are computed. Crossings where a position
        changes from long to short and vice versa are handled correctly.

        Under the hood, we reconstruct the individual shares in a
        portfolio over time and match round_trips in a FIFO order.

        For example, the following transactions would constitute one round trip:
        index                  amount   price    symbol
        2004-01-09 12:18:01    10       50      'AAPL'
        2004-01-09 15:12:53    10       100      'AAPL'
        2004-01-13 14:41:23    -10      100      'AAPL'
        2004-01-13 15:23:34    -10      200       'AAPL'

        First, the first two and last two round_trips will be merged into two
        single transactions (computing the price via vwap). Then, during
        the portfolio reconstruction, the two resulting transactions will
        be merged and result in 1 round-trip trade with a PnL of
        (150 * 20) - (75 * 20) = 1500.

        Note that round trips do not have to close out positions
        completely. For example, we could have removed the last
        transaction in the example above and still generated a round-trip
        over 10 shares with 10 shares left in the portfolio to be matched
        with a later transaction.

        Parameters
        ----------
        transactions : pd.DataFrame
            Prices and amounts of executed round_trips. One row per trade.
            - See full explanation in tears.create_full_tear_sheet

        portfolio_value : pd.Series (optional)
            Portfolio value (all net assets including cash) over time.
            Note that portfolio_value needs to beginning of day, so either
            use .shift() or positions.sum(axis='columns') / (1+returns).

        Returns
        -------
        round_trips : pd.DataFrame:
            DataFrame with one row per round trip.  The `returns` column
            contains returns in respect to the portfolio value while
            rt_returns are the returns in regard to the invested capital
            into that particular round-trip.
        """

        transactions = Empyrical._groupby_consecutive(transactions)
        roundtrips = []

        for sym, trans_sym in transactions.groupby('symbol'):
            trans_sym = trans_sym.sort_index()
            price_stack = deque()
            dt_stack = deque()
            trans_sym['signed_price'] = trans_sym.price * np.sign(trans_sym.amount)
            trans_sym['abs_amount'] = trans_sym.amount.abs().astype(int)
            for dt, t in trans_sym.iterrows():
                if t.price < 0:
                    warnings.warn('Negative price detected, ignoring for'
                                  'round-trip.')
                    continue

                indiv_prices = [t.signed_price] * t.abs_amount
                if (len(price_stack) == 0) or \
                        (np.copysign(1, price_stack[-1]) == np.copysign(1, t.amount)):
                    price_stack.extend(indiv_prices)
                    dt_stack.extend([dt] * len(indiv_prices))
                else:
                    # Close round-trip
                    pnl = 0
                    invested = 0
                    cur_open_dts = []

                    for price in indiv_prices:
                        if len(price_stack) != 0 and \
                                (np.copysign(1, price_stack[-1]) != np.copysign(1, price)):
                            # Retrieve the first dt, stock-price pair from
                            # stack
                            prev_price = price_stack.popleft()
                            prev_dt = dt_stack.popleft()

                            pnl += -(price + prev_price)
                            cur_open_dts.append(prev_dt)
                            invested += abs(prev_price)

                        else:
                            # Push additional stock prices onto the stack
                            price_stack.append(price)
                            dt_stack.append(dt)

                    roundtrips.append({'pnl': pnl,
                                       'open_dt': cur_open_dts[0],
                                       'close_dt': dt,
                                       'long': price < 0,
                                       'rt_returns': pnl / invested,
                                       'symbol': sym,
                                       })

        roundtrips = pd.DataFrame(roundtrips)

        roundtrips['duration'] = roundtrips['close_dt'].sub(roundtrips['open_dt'])

        if portfolio_value is not None:
            # Need to normalize so that we can join
            pv = pd.DataFrame(portfolio_value,
                              columns=['portfolio_value']) \
                .assign(date=portfolio_value.index)

            roundtrips['date'] = roundtrips.close_dt.apply(lambda x:
                                                           x.replace(hour=0,
                                                                     minute=0,
                                                                     second=0))
            # Convert 'roundtrips.date' to UTC to match 'portfolio_value.index'
            if pv.index.tz is not None:  # portfolio_value.index has a timezone (e.g., UTC)
                roundtrips['date'] = roundtrips['date'].dt.tz_localize('UTC')

            tmp = roundtrips.join(pv, on='date', lsuffix='_')

            roundtrips['returns'] = tmp.pnl / tmp.portfolio_value
            roundtrips = roundtrips.drop('date', axis='columns')

        return roundtrips

    @classmethod
    def add_closing_transactions(cls, positions, transactions):
        """
        Appends transactions that close out all positions at the end of
        the timespan covered by positions data. Utilizes pricing information
        in the positions DataFrame to determine closing price.

        Parameters
        ----------
        positions : pd.DataFrame
            The positions that the strategy takes over time.
        transactions : pd.DataFrame
            Prices and amounts of executed round_trips. One row per trade.
            - See full explanation in tears.create_full_tear_sheet

        Returns
        -------
        closed_txns : pd.DataFrame
            Transactions with closing transactions appended.
        """

        closed_txns = transactions[['symbol', 'amount', 'price']]

        pos_at_end = positions.drop('cash', axis=1).iloc[-1]
        open_pos = pos_at_end.replace(0, np.nan).dropna()
        # Add closing round_trips one second after the close to be sure
        # they don't conflict with other round_trips executed at that time.
        end_dt = open_pos.name + pd.Timedelta(seconds=1)

        for sym, ending_val in open_pos.items():
            txn_sym = transactions[transactions.symbol == sym]

            ending_amount = txn_sym.amount.sum()

            ending_price = ending_val / ending_amount
            closing_txn = {'symbol': sym,
                           'amount': -ending_amount,
                           'price': ending_price}

            closing_txn = pd.DataFrame(closing_txn, index=[end_dt])
            # closed_txns = closed_txns.append(closing_txn)
            closed_txns = pd.concat([closed_txns, closing_txn], ignore_index=True)

        closed_txns = closed_txns[closed_txns.amount != 0]

        return closed_txns

    @classmethod
    def apply_sector_mappings_to_round_trips(cls, round_trips, sector_mappings):
        """
        Translates round trip symbols to sectors.

        Parameters
        ----------
        round_trips : pd.DataFrame:
            DataFrame with one row per-round-trip trade.
            - See full explanation in round_trips.extract_round_trips
        sector_mappings : dict or pd.Series, optional
            Security identifier to sector mapping.
            Security ids as keys, sectors as values.

        Returns
        -------
        sector_round_trips : pd.DataFrame
            Round trips with symbol names replaced by sector names.
        """

        sector_round_trips = round_trips.copy()
        sector_round_trips.symbol = sector_round_trips.symbol.apply(
            lambda x: sector_mappings.get(x, 'No Sector Mapping'))
        sector_round_trips = sector_round_trips.dropna(axis=0)

        return sector_round_trips

    # @classmethod
    # def gen_round_trip_stats(cls, round_trips):
    #     """Generate various round-trip statistics.
    #
    #     Parameters
    #     ----------
    #     round_trips : pd.DataFrame:
    #         DataFrame with one row per-round-trip trade.
    #         - See full explanation in round_trips.extract_round_trips
    #
    #     Returns
    #     -------
    #     stats : dict
    #        A dictionary where each value is a pandas DataFrame containing
    #        various round-trip statistics.
    #
    #     See also
    #     --------
    #     round_trips.print_round_trip_stats
    #     """
    #
    #     stats = {'pnl': agg_all_long_short(round_trips, 'pnl', PNL_STATS), 'summary': agg_all_long_short(round_trips, 'pnl',
    #                                                                                                      SUMMARY_STATS),
    #              'duration': agg_all_long_short(round_trips, 'duration',
    #                                             DURATION_STATS), 'returns': agg_all_long_short(round_trips, 'returns',
    #                                                                                            RETURN_STATS),
    #              'symbols': round_trips.groupby('symbol')['returns'].agg(RETURN_STATS).T}
    #
    #     return stats

    @classmethod
    def gen_round_trip_stats(cls, round_trips):
        """Generate various round-trip statistics.

        Parameters
        ----------
        round_trips : pd.DataFrame:
            DataFrame with one row per-round-trip trade.
            - See full explanation in round_trips.extract_round_trips

        Returns
        -------
        stats : dict
           A dictionary where each value is a pandas DataFrame containing
           various round-trip statistics.

        See also
        --------
        round_trips.print_round_trip_stats
        """

        # Helper function to apply custom and built-in functions
        def apply_custom_and_built_in_funcs(grouped, stats_dict):
            # Separate custom functions from built-in functions
            custom_funcs = {k: v for k, v in stats_dict.items() if callable(v)}
            built_in_funcs = [v for k, v in stats_dict.items() if not callable(v)]

            # Apply custom functions manually
            custom_results = {}
            for func_name, func in custom_funcs.items():
                custom_results[func_name] = grouped.apply(func)
            custom_results = pd.DataFrame(custom_results)

            # Apply built-in functions
            built_in_results = grouped.agg(built_in_funcs)

            # Combine results
            return pd.concat([custom_results, built_in_results], axis=1)

        # Generate statistics for pnl, summary, duration, and returns
        stats = {
            'pnl': Empyrical.agg_all_long_short(round_trips, 'pnl', PNL_STATS),
            'summary': Empyrical.agg_all_long_short(round_trips, 'pnl', SUMMARY_STATS),
            'duration': Empyrical.agg_all_long_short(round_trips, 'duration', DURATION_STATS),
            'returns': Empyrical.agg_all_long_short(round_trips, 'returns', RETURN_STATS),
            'symbols': apply_custom_and_built_in_funcs(round_trips.groupby('symbol')['returns'], RETURN_STATS).T
        }

        return stats

    @classmethod
    def perf_attrib(cls, returns,
                    positions,
                    factor_returns,
                    factor_loadings,
                    transactions=None,
                    pos_in_dollars=True):
        """
        Attributes the performance of a `returns` stream to a set of risk factors.

        Preprocesses inputs, and then calls empyrical.perf_attrib. See
        empyrical.perf_attrib for more info.

        Performance attribution determines how much each risk factor, e.g.,
        momentum, the technology sector, etc., contributed to total returns, as
        well as the daily exposure to each of the risk factors. The returns that
        can be attributed to one of the given risk factors are the
        `common_returns`, and the returns that _cannot_ be attributed to a risk
        factor are the `specific_returns`, or the alpha. The common_returns and
        specific_returns summed together will always equal the total returns.

        Parameters
        ----------
        returns : pd.Series
            Returns for each day in the date range.
            - Example:
                2017-01-01 -0.017098
                2017-01-02 0.002683
                2017-01-03 -0.008669

        positions: pd.DataFrame
            Daily holdings (in dollars or percentages), indexed by date.
            It Will be converted to percentages if positions are in dollars.
            Short positions show up as cash in the 'cash' column.
            - Examples:
                            AAPL  TLT XOM cash
                2017-01-01 34 58 10 0
                2017-01-02 22 77 18 0
                2017-01-03 -15 27 30 15

                                AAPL       TLT       XOM  cash
                2017-01-01  0.333333  0.568627  0.098039   0.0
                2017-01-02  0.188034  0.658120  0.153846   0.0
                2017-01-03  0.208333  0.375000  0.416667   0.0

        factor_returns : pd.DataFrame
            Returns by factor, with date as index and factors as columns
            - Example:
                            momentum  reversal
                2017-01-01  0.002779 -0.005453
                2017-01-02  0.001096  0.010290

        factor_loadings : pd.DataFrame
            Factor loadings for all days in the date range, with date and ticker as
            index, and factors as columns.
            - Example:
                                   momentum  reversal
                dt         ticker
                2017-01-01 AAPL   -1.592914  0.852830
                           TLT     0.184864  0.895534
                           XOM     0.993160  1.149353
                2017-01-02 AAPL   -0.140009 -0.524952
                           TLT    -1.066978  0.185435
                           XOM    -1.798401  0.761549


        transactions : pd.DataFrame, optional
            Executed trade volumes and fill prices. Used to check the turnover of
            the algorithm. Default is None, in which case the turnover check is
            skipped.

            - One row per trade.
            - Trades on different names that occur at the
              same time will have identical indices.
            - Example:
                index                  amount   price    symbol
                2004-01-09 12:18:01    483      324.12   'AAPL'
                2004-01-09 12:18:01    122      83.10    'MSFT'
                2004-01-13 14:12:23    -75      340.43   'AAPL'

        pos_in_dollars : bool
            Flag indicating whether `positions` are in dollars or percentages
            If True, positions are in dollars.

        Returns
        -------
        tuple of (risk_exposures_portfolio, perf_attribution)

        risk_exposures_portfolio : pd.DataFrame
            df indexed by datetime, with factors as columns
            - Example:
                            momentum  reversal
                dt
                2017-01-01 -0.238655 0.077123
                2017-01-02 0.821872 1.520515

        perf_attribution : pd.DataFrame
            df with factors, common returns, and specific returns as columns,
            and datetimes as index
            - Example:
                            momentum  reversal common_returns specific_returns
                dt
                2017-01-01 0.249087 0.935925 1.185012 1.185012
                2017-01-02 -0.003194 -0.400786 -0.403980 -0.403980
        """
        (returns,
         positions,
         factor_returns,
         factor_loadings) = Empyrical._align_and_warn(returns,
                                                      positions,
                                                      factor_returns,
                                                      factor_loadings,
                                                      transactions=transactions,
                                                      pos_in_dollars=pos_in_dollars)

        # Note that we convert positions to percentages *after* the checks
        # above, since get_turnover() expects positions in dollars.
        positions = Empyrical._stack_positions(positions, pos_in_dollars=pos_in_dollars)

        return Empyrical.perf_attrib(returns, positions, factor_returns, factor_loadings)

    @classmethod
    def compute_exposures(cls, positions, factor_loadings, stack_positions=True,
                          pos_in_dollars=True):
        """
        Compute daily risk factor exposures.

        Normalizes positions (if necessary) and calls ep.compute_exposures.
        See empyrical.compute_exposures for more info.

        Parameters
        ----------
        positions: pd.DataFrame or pd.Series
            Daily holdings (in dollars or percentages), indexed by date, OR
            a series of holdings indexed by date and ticker.
            - Examples:
                            AAPL  TLT XOM cash
                2017-01-01 34 58 10 0
                2017-01-02 22 77 18 0
                2017-01-03 -15 27 30 15

                                AAPL       TLT       XOM  cash
                2017-01-01  0.333333  0.568627  0.098039   0.0
                2017-01-02  0.188034  0.658120  0.153846   0.0
                2017-01-03  0.208333  0.375000  0.416667   0.0

                Dt ticker
                2017-01-01  AAPL      0.417582
                            TLT       0.010989
                            XOM       0.571429
                2017-01-02  AAPL      0.202381
                            TLT       0.535714
                            XOM       0.261905

        factor_loadings : pd.DataFrame
            Factor loadings for all days in the date range, with date and ticker as
            index, and factors as columns.
            - Example:
                                   momentum  reversal
                dt         ticker
                2017-01-01 AAPL   -1.592914  0.852830
                           TLT     0.184864  0.895534
                           XOM     0.993160  1.149353
                2017-01-02 AAPL   -0.140009 -0.524952
                           TLT    -1.066978  0.185435
                           XOM    -1.798401  0.761549

        stack_positions : bool
            Flag indicating whether `positions` should be converted to long format.

        pos_in_dollars : bool
            Flag indicating whether `positions` are in dollars or percentages
            If True, positions are in dollars.

        Returns
        -------
        risk_exposures_portfolio : pd.DataFrame
            df indexed by datetime, with factors as columns.
            - Example:
                            momentum  reversal
                dt
                2017-01-01 -0.238655 0.077123
                2017-01-02 0.821872 1.520515
        """
        if stack_positions:
            positions = Empyrical._stack_positions(positions, pos_in_dollars=pos_in_dollars)

        return Empyrical._compute_exposures(positions, factor_loadings)

    @classmethod
    def create_perf_attrib_stats(cls, perf_attrib_, risk_exposures):
        """
        Takes perf attribution data over a period of time and computes annualized the
        multifactor alpha, multifactor sharpe, risk exposures.
        """
        summary = OrderedDict()
        total_returns = perf_attrib_['total_returns']
        specific_returns = perf_attrib_['specific_returns']
        common_returns = perf_attrib_['common_returns']

        summary['Annualized Specific Return'] = \
            Empyrical.annual_return(specific_returns)
        summary['Annualized Common Return'] = \
            Empyrical.annual_return(common_returns)
        summary['Annualized Total Return'] = \
            Empyrical.annual_return(total_returns)

        summary['Specific Sharpe Ratio'] = \
            Empyrical.sharpe_ratio(specific_returns)

        summary['Cumulative Specific Return'] = \
            Empyrical.cum_returns_final(specific_returns)
        summary['Cumulative Common Return'] = \
            Empyrical.cum_returns_final(common_returns)
        summary['Total Returns'] = \
            Empyrical.cum_returns_final(total_returns)

        summary = pd.Series(summary, name='')

        annualized_returns_by_factor = [Empyrical.annual_return(perf_attrib_[c])
                                        for c in risk_exposures.columns]
        cumulative_returns_by_factor = [Empyrical.cum_returns_final(perf_attrib_[c])
                                        for c in risk_exposures.columns]

        risk_exposure_summary = pd.DataFrame(
            data=OrderedDict([
                (
                    'Average Risk Factor Exposure',
                    risk_exposures.mean(axis='rows')
                ),
                ('Annualized Return', annualized_returns_by_factor),
                ('Cumulative Return', cumulative_returns_by_factor),
            ]),
            index=risk_exposures.columns,
        )

        return summary, risk_exposure_summary

    @classmethod
    def _align_and_warn(cls, returns,
                        positions,
                        factor_returns,
                        factor_loadings,
                        transactions=None,
                        pos_in_dollars=True):
        """
        Make sure that all inputs have matching dates and tickers,
        and raise warnings if necessary.
        """
        missing_stocks = positions.columns.difference(
            factor_loadings.index.get_level_values(1).unique()
        )

        # cash will not be in factor_loadings
        num_stocks = len(positions.columns) - 1
        missing_stocks = missing_stocks.drop('cash')
        num_stocks_covered = num_stocks - len(missing_stocks)
        missing_ratio = round(len(missing_stocks) / num_stocks, ndigits=3)

        if num_stocks_covered == 0:
            raise ValueError("Could not perform performance attribution. "
                             "No factor loadings were available for this "
                             "algorithm's positions.")

        if len(missing_stocks) > 0:

            if len(missing_stocks) > 5:

                missing_stocks_displayed = (
                    " {} assets were missing factor loadings, including: {}..{}"
                ).format(len(missing_stocks),
                         ', '.join(missing_stocks[:5].map(str)),
                         missing_stocks[-1])
                avg_allocation_msg = "selected missing assets"

            else:
                missing_stocks_displayed = (
                    "The following assets were missing factor loadings: {}."
                ).format(list(missing_stocks))
                avg_allocation_msg = "missing assets"

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
                positions[missing_stocks[:5].union(missing_stocks[[-1]])].mean(),
            )

            warnings.warn(missing_stocks_warning_msg)

            positions = positions.drop(missing_stocks, axis='columns',
                                       errors='ignore')

        missing_factor_loadings_index = positions.index.difference(
            factor_loadings.index.get_level_values(0).unique()
        )

        if len(missing_factor_loadings_index) > 0:

            if len(missing_factor_loadings_index) > 5:
                missing_dates_displayed = (
                    "(first missing is {}, last missing is {})"
                ).format(
                    missing_factor_loadings_index[0],
                    missing_factor_loadings_index[-1]
                )
            else:
                missing_dates_displayed = list(missing_factor_loadings_index)

            warning_msg = (
                "Could not find factor loadings for {} dates: {}. "
                "Truncating date range for performance attribution. "
            ).format(len(missing_factor_loadings_index), missing_dates_displayed)

            warnings.warn(warning_msg)

            positions = positions.drop(missing_factor_loadings_index,
                                       errors='ignore')
            returns = returns.drop(missing_factor_loadings_index, errors='ignore')
            factor_returns = factor_returns.drop(missing_factor_loadings_index,
                                                 errors='ignore')

        if transactions is not None and pos_in_dollars:
            turnover = Empyrical.get_turnover(positions, transactions).mean()
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

    @classmethod
    def _stack_positions(cls, positions, pos_in_dollars=True):
        """
        Convert positions to percentages if necessary, and change them
        to long format.

        Parameters
        ----------
        positions: pd.DataFrame
            Daily holdings (in dollars or percentages), indexed by date.
            It Will be converted to percentages if positions are in dollars.
            Short positions show up as cash in the 'cash' column.

        pos_in_dollars : bool
            Flag indicating whether `positions` are in dollars or percentages
            If True, positions are in dollars.
        """
        if pos_in_dollars:
            # convert holdings to percentages
            positions = Empyrical.get_percent_alloc(positions)

        # remove cash after normalizing positions
        positions = positions.drop('cash', axis='columns')

        # convert positions to long format
        positions = positions.stack()
        positions.index = positions.index.set_names(['dt', 'ticker'])

        return positions

    @classmethod
    def _cumulative_returns_less_costs(cls, returns, costs):
        """
        Compute cumulative returns, less costs.
        """
        if costs is None:
            return Empyrical.cum_returns(returns)
        return Empyrical.cum_returns(returns - costs)


# 导出所有函数和类
__all__ = [
    # 新的类
    'Empyrical'
]
