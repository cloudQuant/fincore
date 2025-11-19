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

import pandas as pd
import numpy as np
from collections import OrderedDict
import math
from scipy import stats
from six import iteritems
from sys import float_info

from fincore.utils import nanmean, nanstd, nanmin, nanmax, nanargmax, nanargmin
from fincore.utils import up, down, roll, rolling_window
#     alpha_beta_aligned,
#     annual_return,
#     annual_volatility,
#     beta,
#     beta_aligned,
#     cagr,
#     beta_fragility_heuristic,
#     beta_fragility_heuristic_aligned,
#     gpd_risk_estimates,
#     gpd_risk_estimates_aligned,
#     calmar_ratio,
#     capture,
#     conditional_value_at_risk,
#     cum_returns,
#     cum_returns_final,
#     down_alpha_beta,
#     down_capture,
#     downside_risk,
#     excess_sharpe,
#     max_drawdown,
#     omega_ratio,
#     roll_alpha,
#     roll_alpha_aligned,
#     roll_alpha_beta,
#     roll_alpha_beta_aligned,
#     roll_annual_volatility,
#     roll_beta,
#     roll_beta_aligned,
#     roll_down_capture,
#     roll_max_drawdown,
#     roll_sharpe_ratio,
#     roll_sortino_ratio,
#     roll_up_capture,
#     roll_up_down_capture,
#     sharpe_ratio,
#     simple_returns,
#     sortino_ratio,
#     stability_of_timeseries,
#     tail_ratio,
#     up_alpha_beta,
#     up_capture,
#     up_down_capture,
#     value_at_risk,
#     information_ratio,
#     get_max_drawdown_period,
#     tracking_error,
#     roll_tracking_error,
#     treynor_ratio,
#     roll_treynor_ratio,
#     m_squared,
#     roll_m_squared,
#     annual_active_risk,
#     roll_annual_active_risk,
#     annual_active_return,
#     roll_annual_active_return,
#     tracking_difference,
#     annual_return_by_year,
#     sharpe_ratio_by_year,
#     information_ratio_by_year,
#     annual_volatility_by_year,
#     max_drawdown_by_year,
#     max_drawdown_days,
#     max_drawdown_weeks,
#     max_drawdown_months,
#     max_drawdown_recovery_days,
#     max_drawdown_recovery_weeks,
#     max_drawdown_recovery_months,
#     second_max_drawdown,
#     third_max_drawdown,
#     second_max_drawdown_days,
#     second_max_drawdown_recovery_days,
#     third_max_drawdown_days,
#     third_max_drawdown_recovery_days,
#     win_rate,
#     loss_rate,
#     max_consecutive_up_days,
#     max_consecutive_down_days,
#     max_consecutive_up_weeks,
#     max_consecutive_down_weeks,
#     max_consecutive_up_months,
#     max_consecutive_down_months,
#     max_consecutive_gain,
#     max_consecutive_loss,
#     max_single_day_gain,
#     max_single_day_loss,
#     max_consecutive_up_start_date,
#     max_consecutive_up_end_date,
#     max_consecutive_down_start_date,
#     max_consecutive_down_end_date,
#     max_single_day_gain_date,
#     max_single_day_loss_date,
#     skewness,
#     kurtosis,
#     hurst_exponent,
#     stock_market_correlation,
#     bond_market_correlation,
#     futures_market_correlation,
#     serial_correlation,
#     sterling_ratio,
#     burke_ratio,
#     kappa_three_ratio,
#     adjusted_sharpe_ratio,
#     stutzer_index,
#     annual_alpha,
#     annual_beta,
#     residual_risk,
#     conditional_sharpe_ratio,
#     var_excess_return,
#     regression_annual_return,
#     r_cubed,
#     annualized_cumulative_return,
#     annual_active_return_by_year,
#     treynor_mazuy_timing,
#     henriksson_merton_timing,
#     market_timing_return,
#     alpha_percentile_rank,
#     cornell_timing,
# )

from fincore.constants import (
    DAILY,
    WEEKLY,
    MONTHLY,
    QUARTERLY,
    YEARLY,
    APPROX_BDAYS_PER_YEAR,
    ANNUALIZATION_FACTORS,
)

# 重新导出utils以保持向后兼容性
from fincore import utils

__version__ = "0.6.0"

# Period to frequency mapping
_PERIOD_TO_FREQ = {
    DAILY: "D",
    WEEKLY: "W", 
    MONTHLY: "M",
    QUARTERLY: "Q",
    YEARLY: "A",
}


class Empyrical:
    """
    面向对象的性能指标计算类
    
    这个类将所有empyrical模块的函数封装为类方法，提供统一的数据管理和计算接口。
    初始化参数与pyfolio的create_full_tear_sheet函数参数保持一致。
    
    通过直接调用原有函数确保100%的计算一致性。
    """
    
    def __init__(self,
                 returns=None,
                 positions=None,
                 transactions=None,
                 market_data=None,
                 benchmark_rets=None,
                 slippage=None,
                 live_start_date=None,
                 sector_mappings=None,
                 bayesian=False,
                 round_trips=False,
                 estimate_intraday='infer',
                 hide_positions=False,
                 cone_std=(1.0, 1.5, 2.0),
                 bootstrap=False,
                 unadjusted_returns=None,
                 style_factor_panel=None,
                 sectors=None,
                 caps=None,
                 shares_held=None,
                 volumes=None,
                 percentile=None,
                 turnover_denom='AGB',
                 set_context=True,
                 factor_returns=None,
                 factor_loadings=None,
                 pos_in_dollars=True,
                 header_rows=None,
                 factor_partitions=None):
        """
        初始化Empyrical类实例
        
        Parameters
        ----------
        returns : pd.Series, optional
            Daily returns of the strategy, noncumulative.
        positions : pd.DataFrame, optional
            Daily net position values.
        transactions : pd.DataFrame, optional
            Executed trade volumes and fill prices.
        market_data : pd.Panel or dict, optional
            Panel/dict with items axis of 'price' and 'volume' DataFrames.
        benchmark_rets : pd.Series, optional
            Benchmark returns for comparison.
        factor_returns : pd.DataFrame, optional
            Returns by factor, with date as index and factors as columns.
        factor_loadings : pd.DataFrame, optional
            Factor loadings for all days in the date range.
        ... (其他参数与create_full_tear_sheet保持一致)
        """
        # 存储核心数据
        self.returns = returns
        self.positions = positions
        self.transactions = transactions
        self.market_data = market_data
        self.benchmark_rets = benchmark_rets
        self.factor_returns = factor_returns
        self.factor_loadings = factor_loadings
        
        # 存储配置参数
        self.slippage = slippage
        self.live_start_date = live_start_date
        self.sector_mappings = sector_mappings
        self.bayesian = bayesian
        self.round_trips = round_trips
        self.estimate_intraday = estimate_intraday
        self.hide_positions = hide_positions
        self.cone_std = cone_std
        self.bootstrap = bootstrap
        self.unadjusted_returns = unadjusted_returns
        self.style_factor_panel = style_factor_panel
        self.sectors = sectors
        self.caps = caps
        self.shares_held = shares_held
        self.volumes = volumes
        self.percentile = percentile
        self.turnover_denom = turnover_denom
        self.set_context = set_context
        self.pos_in_dollars = pos_in_dollars
        self.header_rows = header_rows
        self.factor_partitions = factor_partitions

    # ================================
    # 数据管理接口
    # ================================
    
    def set_returns(self, returns):
        """设置收益率数据"""
        self.returns = returns
        
    def get_returns(self):
        """获取收益率数据"""
        return self.returns
        
    def set_positions(self, positions):
        """设置持仓数据"""
        self.positions = positions
        
    def get_positions(self):
        """获取持仓数据"""
        return self.positions
        
    def set_transactions(self, transactions):
        """设置交易数据"""
        self.transactions = transactions
        
    def get_transactions(self):
        """获取交易数据"""
        return self.transactions
        
    def set_market_data(self, market_data):
        """设置市场数据"""
        self.market_data = market_data
        
    def get_market_data(self):
        """获取市场数据"""
        return self.market_data
        
    def set_benchmark_rets(self, benchmark_rets):
        """设置基准收益率"""
        self.benchmark_rets = benchmark_rets
        
    def get_benchmark_rets(self):
        """获取基准收益率"""
        return self.benchmark_rets
        
    def set_factor_returns(self, factor_returns):
        """设置因子收益率"""
        self.factor_returns = factor_returns
        
    def get_factor_returns(self):
        """获取因子收益率"""
        return self.factor_returns
        
    def set_factor_loadings(self, factor_loadings):
        """设置因子载荷"""
        self.factor_loadings = factor_loadings
        
    def get_factor_loadings(self):
        """获取因子载荷"""
        return self.factor_loadings

    # ================================
    # 计算方法（包装原有函数）
    # ================================
    
    def _get_returns(self, returns):
        """获取要使用的收益率数据"""
        if returns is None:
            if self.returns is None:
                raise ValueError("Either provide returns or set returns data")
            return self.returns
        return returns
    
    def _get_factor_returns(self, factor_returns):
        """获取要使用的因子收益率数据"""
        if factor_returns is None:
            if self.factor_returns is not None:
                return self.factor_returns
            elif self.benchmark_rets is not None:
                return self.benchmark_rets
            else:
                raise ValueError("Either provide factor_returns or set factor_returns/benchmark_rets")
        return factor_returns
    
    def _get_positions(self, positions):
        """获取要使用的持仓数据"""
        if positions is None:
            if self.positions is None:
                raise ValueError("Either provide positions or set positions data")
            return self.positions
        return positions
    
    def _get_factor_loadings(self, factor_loadings):
        """获取要使用的因子载荷数据"""
        if factor_loadings is None:
            if self.factor_loadings is None:
                raise ValueError("Either provide factor_loadings or set factor_loadings data")
            return self.factor_loadings
        return factor_loadings
    
    @staticmethod
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
    def _ensure_datetime_index_series(data, period=DAILY):
        """Return a Series indexed by dates regardless of the input type."""
        _PERIOD_TO_FREQ = {
            DAILY: "D",
            WEEKLY: "W", 
            MONTHLY: "M",
        }
        
        if isinstance(data, pd.Series) and isinstance(data.index, pd.DatetimeIndex):
            return data
            
        values = data.values if isinstance(data, pd.Series) else np.asarray(data)
        
        if values.size == 0:
            return pd.Series(values)
            
        freq = _PERIOD_TO_FREQ.get(period, "D")
        index = pd.date_range("1970-01-01", periods=values.size, freq=freq)
        return pd.Series(values, index=index)
    
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
    def cal_simple_returns(cls, prices):
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
    def cal_cum_returns(cls, returns, starting_value=0, out=None):
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
    def cal_cum_returns_final(cls, returns, starting_value=0):
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
    def cal_aggregate_returns(cls, returns, convert_to='monthly'):
        """Aggregates returns by week, month, or year."""
        def cumulate_returns(x):
            return cls.cal_cum_returns(x).iloc[-1]
        
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

    @classmethod
    def cal_max_drawdown(cls, returns, out=None):
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
        cls.cal_cum_returns(returns_array, starting_value=start, out=cumulative[1:])
        
        max_return = np.fmax.accumulate(cumulative, axis=0)
        
        nanmin((cumulative - max_return) / max_return, axis=0, out=out)
        if returns_1d:
            out = out.item()
        elif allocated_output and isinstance(returns, pd.DataFrame):
            out = pd.Series(out)
        
        return out

    @classmethod
    def cal_annual_return(cls, returns, period=DAILY, annualization=None):
        """Determines the mean annual growth rate of returns."""
        if len(returns) < 1:
            return np.nan
        
        ann_factor = cls.annualization_factor(period, annualization)
        num_years = len(returns) / ann_factor
        # Pass array to ensure index -1 looks up successfully.
        ending_value = cls.cal_cum_returns_final(returns, starting_value=1)
        
        return ending_value ** (1 / num_years) - 1

    @classmethod
    def cal_cagr(cls, returns, period=DAILY, annualization=None):
        """Compute compound annual growth rate."""
        return cls.cal_annual_return(returns, period, annualization)

    @classmethod
    def cal_annual_volatility(cls, returns, period=DAILY, alpha_=2.0, annualization=None, out=None):
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
    def cal_calmar_ratio(cls, returns, period=DAILY, annualization=None):
        """Determines the Calmar ratio, or drawdown ratio, of a strategy."""
        max_dd = cls.cal_max_drawdown(returns=returns)
        if max_dd < 0:
            temp = cls.cal_annual_return(
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
    def cal_omega_ratio(cls, returns, risk_free=0.0, required_return=0.0, annualization=APPROX_BDAYS_PER_YEAR):
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
    def cal_sharpe_ratio(cls, returns, risk_free=0, period=DAILY, annualization=None, out=None):
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
    def simple_returns(self, prices=None):
        """Compute simple returns from a timeseries of prices."""
        if prices is None:
            if self.returns is None:
                raise ValueError("Either provide prices or set returns data")
            return self.returns
        return self.cal_simple_returns(prices)
    
    def cum_returns(self, returns=None, starting_value=0, out=None):
        """Compute cumulative returns from simple returns."""
        returns = self._get_returns(returns)
        return self.cal_cum_returns(returns, starting_value, out)
    
    def cum_returns_final(self, returns=None, starting_value=0):
        """Compute total returns from simple returns."""
        returns = self._get_returns(returns)
        return self.cal_cum_returns_final(returns, starting_value)
    
    def aggregate_returns(self, returns=None, convert_to='monthly'):
        """Aggregates returns by week, month, or year."""
        returns = self._get_returns(returns)
        return self.cal_aggregate_returns(returns, convert_to)
    
    def max_drawdown(self, returns=None, out=None):
        """Determines the maximum drawdown of a strategy."""
        returns = self._get_returns(returns)
        return self.cal_max_drawdown(returns, out)
    
    def annual_return(self, returns=None, period=DAILY, annualization=None):
        """Determines the mean annual growth rate of returns."""
        returns = self._get_returns(returns)
        return self.cal_annual_return(returns, period, annualization)
    
    def cagr(self, returns=None, period=DAILY, annualization=None):
        """Compute compound annual growth rate."""
        returns = self._get_returns(returns)
        return self.cal_cagr(returns, period, annualization)
    
    def annual_volatility(self, returns=None, period=DAILY, alpha_=2.0, annualization=None, out=None):
        """Determines the annual volatility of a strategy."""
        returns = self._get_returns(returns)
        return self.cal_annual_volatility(returns, period, alpha_, annualization, out)
    
    def calmar_ratio(self, returns=None, period=DAILY, annualization=None):
        """Determines the Calmar ratio, or drawdown ratio, of a strategy."""
        returns = self._get_returns(returns)
        return self.cal_calmar_ratio(returns, period, annualization)
    
    def sharpe_ratio(self, returns=None, risk_free=0, period=DAILY, annualization=None, out=None):
        """Determines the Sharpe ratio of a strategy."""
        returns = self._get_returns(returns)
        return self.cal_sharpe_ratio(returns, risk_free, period, annualization, out)

    def omega_ratio(self, returns=None, risk_free=0.0, required_return=0.0, annualization=APPROX_BDAYS_PER_YEAR):
        """Determines the Omega ratio of a strategy."""
        returns = self._get_returns(returns)
        return self.cal_omega_ratio(returns, risk_free, required_return, annualization)

    @classmethod
    def cal_alpha_beta(cls, returns, factor_returns, risk_free=0.0, period=DAILY, annualization=None, out=None):
        """Calculates annualized alpha and beta."""
        # Match original empyrical.stats.alpha_beta behaviour: align series
        # first, then delegate to the aligned implementation.
        if not (isinstance(returns, np.ndarray) and isinstance(factor_returns, np.ndarray)):
            returns, factor_returns = cls._aligned_series(returns, factor_returns)

        return cls.cal_alpha_beta_aligned(
            returns,
            factor_returns,
            risk_free=risk_free,
            period=period,
            annualization=annualization,
            out=out,
        )
    
    @classmethod
    def cal_alpha(cls, returns, factor_returns, risk_free=0.0, period=DAILY, annualization=None, out=None, _beta=None):
        """Calculates annualized alpha.

        This mirrors empyrical.stats.alpha, which internally calls
        alpha_aligned after aligning non-ndarray inputs.
        """
        if not (isinstance(returns, np.ndarray) and isinstance(factor_returns, np.ndarray)):
            returns, factor_returns = cls._aligned_series(returns, factor_returns)

        return cls.cal_alpha_aligned(
            returns,
            factor_returns,
            risk_free=risk_free,
            period=period,
            annualization=annualization,
            out=out,
            _beta=_beta,
        )
    
    @classmethod
    def cal_beta(cls, returns, factor_returns, risk_free=0.0, period=DAILY, annualization=None, out=None):
        """Calculates beta.

        This mirrors empyrical.stats.beta, which forwards to beta_aligned
        after aligning non-ndarray inputs.
        """
        if not (isinstance(returns, np.ndarray) and isinstance(factor_returns, np.ndarray)):
            returns, factor_returns = cls._aligned_series(returns, factor_returns)

        return cls.cal_beta_aligned(
            returns,
            factor_returns,
            risk_free=risk_free,
            out=out,
        )

    def alpha_beta(self, returns=None, factor_returns=None, risk_free=0.0, period=DAILY, annualization=None, out=None):
        """Calculates annualized alpha and beta."""
        returns = self._get_returns(returns)
        factor_returns = self._get_factor_returns(factor_returns)
        return self.cal_alpha_beta(returns, factor_returns, risk_free, period, annualization, out)
    
    def alpha(self, returns=None, factor_returns=None, risk_free=0.0, period=DAILY, annualization=None):
        """Calculates annualized alpha."""
        returns = self._get_returns(returns)
        factor_returns = self._get_factor_returns(factor_returns)
        return self.cal_alpha(returns, factor_returns, risk_free, period, annualization)
    
    def beta(self, returns=None, factor_returns=None, risk_free=0.0, period=DAILY, annualization=None):
        """Calculates beta."""
        returns = self._get_returns(returns)
        factor_returns = self._get_factor_returns(factor_returns)
        return self.cal_beta(returns, factor_returns, risk_free, period, annualization)

    @classmethod
    def cal_alpha_beta_aligned(cls, returns, factor_returns, risk_free=0.0, period=DAILY, annualization=None, out=None):
        """Calculates annualized alpha and beta for already-aligned series."""
        if out is None:
            out = np.empty(returns.shape[1:] + (2,), dtype="float64")

        b = cls.cal_beta_aligned(returns, factor_returns, risk_free, out=out[..., 1])
        cls.cal_alpha_aligned(
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
    def cal_alpha_aligned(cls, returns, factor_returns, risk_free=0.0, period=DAILY, annualization=None, out=None, _beta=None):
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
            _beta = cls.cal_beta_aligned(returns, factor_returns, risk_free)

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
    def cal_beta_aligned(cls, returns, factor_returns, risk_free=0.0, out=None):
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
    def cal_sortino_ratio(cls, returns, required_return=0, period=DAILY, annualization=None, out=None, _downside_risk=None):
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
            cls.cal_downside_risk(returns, required_return, period, annualization)
        )
        # Avoid division by zero warning
        with np.errstate(divide='ignore', invalid='ignore'):
            np.divide(average_annual_return, annualized_downside_risk, out=out)
        if return_1d:
            out = out.item()
        elif isinstance(returns, pd.DataFrame):
            out = pd.Series(out)
        
        return out
    
    def sortino_ratio(self, returns=None, required_return=0, period=DAILY, annualization=None, out=None, _downside_risk=None):
        """Determines the Sortino ratio of a strategy."""
        returns = self._get_returns(returns)
        return self.cal_sortino_ratio(returns, required_return, period, annualization, out, _downside_risk)

    @classmethod
    def cal_downside_risk(cls, returns, required_return=0, period=DAILY, annualization=None, out=None):
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
    
    def downside_risk(self, returns=None, required_return=0, period=DAILY, annualization=None, out=None):
        """Determines the downside deviation below a threshold."""
        returns = self._get_returns(returns)
        return self.cal_downside_risk(returns, required_return, period, annualization, out)

    @classmethod
    def cal_excess_sharpe(cls, returns, factor_returns, out=None):
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
    
    def excess_sharpe(self, returns=None, factor_returns=None, out=None):
        """Determines the Excess Sharpe of a strategy."""
        returns = self._get_returns(returns)
        factor_returns = self._get_factor_returns(factor_returns)
        return self.cal_excess_sharpe(returns, factor_returns, out)

    @classmethod
    def cal_tracking_error(cls, returns, factor_returns, period=DAILY, annualization=None, out=None):
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
    
    def tracking_error(self, returns=None, factor_returns=None, period=DAILY, annualization=None, out=None):
        """Determines the tracking error of returns relative to factor returns."""
        returns = self._get_returns(returns)
        factor_returns = self._get_factor_returns(factor_returns)
        return self.cal_tracking_error(returns, factor_returns, period, annualization, out)

    @classmethod
    def cal_information_ratio(cls, returns, factor_returns, period=DAILY, annualization=None):
        """Determines the information ratio of returns relative to factor returns."""
        returns, factor_returns = cls._aligned_series(returns, factor_returns)
        super_returns = returns - factor_returns
        
        ann_factor = cls.annualization_factor(period, annualization)
        mean_excess_return = super_returns.mean()
        std_excess_return = super_returns.std(ddof=1)
        ir = (mean_excess_return * ann_factor) / (std_excess_return * np.sqrt(ann_factor))
        return ir
    
    def information_ratio(self, returns=None, factor_returns=None, period=DAILY, annualization=None):
        """Determines the information ratio of returns relative to factor returns."""
        returns = self._get_returns(returns)
        factor_returns = self._get_factor_returns(factor_returns)
        return self.cal_information_ratio(returns, factor_returns, period, annualization)

    @classmethod
    def cal_value_at_risk(cls, returns, cutoff=0.05):
        """Calculates the daily value at risk (VaR) of returns."""
        if len(returns) < 1:
            return np.nan
            
        return np.percentile(returns, cutoff * 100)
    
    def value_at_risk(self, returns=None, cutoff=0.05):
        """Calculates the daily value at risk (VaR) of returns."""
        returns = self._get_returns(returns)
        return self.cal_value_at_risk(returns, cutoff)
    
    @classmethod
    def cal_conditional_value_at_risk(cls, returns, cutoff=0.05):
        """Calculates the conditional value at risk (CVaR) of returns."""
        if len(returns) < 1:
            return np.nan
            
        cutoff_index = cls.cal_value_at_risk(returns, cutoff=cutoff)
        return np.mean(returns[returns <= cutoff_index])
    
    def conditional_value_at_risk(self, returns=None, cutoff=0.05):
        """Calculates the conditional value at risk (CVaR) of returns."""
        returns = self._get_returns(returns)
        return self.cal_conditional_value_at_risk(returns, cutoff)
    
    @classmethod
    def cal_tail_ratio(cls, returns):
        """Determines the ratio between the right (95th) and left (5th) percentile of the returns."""
        if len(returns) < 1:
            return np.nan
            
        returns = np.asanyarray(returns)
        # Be tolerant of nan's
        returns = returns[~np.isnan(returns)]
        if len(returns) < 1:
            return np.nan
            
        return np.abs(np.percentile(returns, 95)) / np.abs(np.percentile(returns, 5))
    
    def tail_ratio(self, returns=None):
        """Determines the ratio between the right (95th) and left (5th) percentile of the returns."""
        returns = self._get_returns(returns)
        return self.cal_tail_ratio(returns)
    
    @classmethod
    def cal_stability_of_timeseries(cls, returns):
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
    
    def stability_of_timeseries(self, returns=None):
        """Determines R-squared of a linear fit to the cumulative log returns."""
        returns = self._get_returns(returns)
        return self.cal_stability_of_timeseries(returns)
    
    @classmethod
    def cal_capture(cls, returns, factor_returns, period=DAILY):
        """Calculates the capture ratio."""
        if len(returns) < 1 or len(factor_returns) < 1:
            return np.nan
            
        strategy_ann_return = cls.cal_annual_return(returns, period=period)
        benchmark_ann_return = cls.cal_annual_return(factor_returns, period=period)
        
        if benchmark_ann_return == 0:
            return np.nan
            
        return strategy_ann_return / benchmark_ann_return
    
    def capture(self, returns=None, factor_returns=None, period=DAILY):
        """Calculates the capture ratio."""
        returns = self._get_returns(returns)
        factor_returns = self._get_factor_returns(factor_returns)
        return self.cal_capture(returns, factor_returns, period)
    
    @classmethod
    def cal_up_capture(cls, returns, factor_returns, period=DAILY):
        """Calculates the capture ratio for periods when the benchmark return is positive."""
        returns, factor_returns = cls._aligned_series(returns, factor_returns)
        
        up_returns = returns[factor_returns > 0]
        up_factor_returns = factor_returns[factor_returns > 0]
        
        if len(up_returns) < 1:
            return np.nan
            
        return cls.cal_capture(up_returns, up_factor_returns, period=period)
    
    def up_capture(self, returns=None, factor_returns=None, period=DAILY):
        """Calculates the capture ratio for periods when the benchmark return is positive."""
        returns = self._get_returns(returns)
        factor_returns = self._get_factor_returns(factor_returns)
        return self.cal_up_capture(returns, factor_returns, period)
    
    @classmethod
    def cal_down_capture(cls, returns, factor_returns, period=DAILY):
        """Calculates the capture ratio for periods when the benchmark return is negative."""
        returns, factor_returns = cls._aligned_series(returns, factor_returns)
        
        down_returns = returns[factor_returns < 0]
        down_factor_returns = factor_returns[factor_returns < 0]
        
        if len(down_returns) < 1:
            return np.nan
            
        return cls.cal_capture(down_returns, down_factor_returns, period=period)
    
    def down_capture(self, returns=None, factor_returns=None, period=DAILY):
        """Calculates the capture ratio for periods when the benchmark return is negative."""
        returns = self._get_returns(returns)
        factor_returns = self._get_factor_returns(factor_returns)
        return self.cal_down_capture(returns, factor_returns, period)
    
    @classmethod
    def cal_up_down_capture(cls, returns, factor_returns, period=DAILY):
        """Calculates the up and down capture ratios."""
        up_cap = cls.cal_up_capture(returns, factor_returns, period=period)
        down_cap = cls.cal_down_capture(returns, factor_returns, period=period)
        return up_cap, down_cap
    
    def up_down_capture(self, returns=None, factor_returns=None, period=DAILY):
        """Calculates the ratio of up_capture to down_capture."""
        returns = self._get_returns(returns)
        factor_returns = self._get_factor_returns(factor_returns)
        
        up_cap = self.cal_up_capture(returns, factor_returns, period)
        down_cap = self.cal_down_capture(returns, factor_returns, period)
        
        if down_cap == 0 or np.isnan(down_cap) or np.isnan(up_cap):
            return np.nan
        
        return up_cap / down_cap

    def perf_attrib(self,
                    returns=None,
                    positions=None,
                    factor_returns=None,
                    factor_loadings=None):
        returns = self._get_returns(returns)
        if positions is None:
            if self.positions is None:
                raise ValueError("Either provide positions or set positions data")
            positions = self.positions
        if factor_returns is None:
            if self.factor_returns is None and self.benchmark_rets is None:
                raise ValueError("Either provide factor_returns or set factor_returns/benchmark_rets")
            factor_returns = self.factor_returns if self.factor_returns is not None else self.benchmark_rets
        if factor_loadings is None:
            if self.factor_loadings is None:
                raise ValueError("Either provide factor_loadings or set factor_loadings data")
            factor_loadings = self.factor_loadings

        start = returns.index[0]
        end = returns.index[-1]
        factor_returns = factor_returns.loc[start:end]
        factor_loadings = factor_loadings.loc[start:end]
        factor_loadings = factor_loadings.copy()
        factor_loadings.index = factor_loadings.index.set_names(['dt', 'ticker'])
        positions = positions.copy()
        positions.index = positions.index.set_names(['dt', 'ticker'])

        risk_exposures_portfolio = self.compute_exposures(
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

    def compute_exposures(self, positions=None, factor_loadings=None):
        if positions is None:
            if self.positions is None:
                raise ValueError("Either provide positions or set positions data")
            positions = self.positions
        if factor_loadings is None:
            if self.factor_loadings is None:
                raise ValueError("Either provide factor_loadings or set factor_loadings data")
            factor_loadings = self.factor_loadings
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
        ann_return = cls.cal_annual_return(returns, period=period, annualization=annualization)
        ann_excess_return = ann_return - risk_free

        # Beta
        b = cls.cal_beta_aligned(returns, factor_returns, risk_free)

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
    
    def treynor_ratio(self, returns=None, factor_returns=None, risk_free=0.0, period=DAILY, annualization=None):
        """Calculates the Treynor ratio."""
        returns = self._get_returns(returns)
        factor_returns = self._get_factor_returns(factor_returns)
        return self.cal_treynor_ratio(returns, factor_returns, risk_free, period, annualization)
    
    @classmethod
    def cal_m_squared(cls, returns, factor_returns, risk_free=0.0, period=DAILY, annualization=None):
        """Calculates the M-squared (M²) measure."""
        if len(returns) < 2:
            return np.nan
            
        returns_aligned, factor_aligned = cls._aligned_series(returns, factor_returns)
        
        # Calculate annualized returns and volatilities  
        ann_return = cls.cal_annual_return(returns_aligned, period=period, annualization=annualization)
        ann_vol = cls.cal_annual_volatility(returns_aligned, period=period, annualization=annualization)
        ann_factor_return = cls.cal_annual_return(factor_aligned, period=period, annualization=annualization)
        ann_factor_vol = cls.cal_annual_volatility(factor_aligned, period=period, annualization=annualization)
        
        # Handle division by zero or negative volatility
        if ann_vol == 0 or ann_vol < 0 or np.isnan(ann_vol):
            return np.nan
        
        # M² = (Rp - Rf) * (σb / σp) + Rf
        excess_return = ann_return - risk_free
        risk_ratio = ann_factor_vol / ann_vol
        return excess_return * risk_ratio + risk_free
    
    def m_squared(self, returns=None, factor_returns=None, risk_free=0.0, period=DAILY, annualization=None):
        """Calculates the M-squared (M²) measure."""
        returns = self._get_returns(returns)
        factor_returns = self._get_factor_returns(factor_returns)
        return self.cal_m_squared(returns, factor_returns, risk_free, period, annualization)
    
    @classmethod
    def cal_annual_return_by_year(cls, returns, period=DAILY, annualization=None):
        """Determines the annual return for each year."""
        if len(returns) < 1:
            return_as_array = isinstance(returns, np.ndarray)
            return np.array([]) if return_as_array else pd.Series(dtype='float64')

        return_as_array = isinstance(returns, np.ndarray)

        # Ensure we have a datetime-indexed Series
        returns = cls._ensure_datetime_index_series(returns, period=period)

        annual_returns = returns.groupby(returns.index.year).apply(
            lambda x: cls.cal_annual_return(x, period=period, annualization=annualization)
        )

        return annual_returns.values if return_as_array else annual_returns
    
    def annual_return_by_year(self, returns=None, period=DAILY, annualization=None):
        """Determines the annual return for each year."""
        returns = self._get_returns(returns)
        return self.cal_annual_return_by_year(returns, period, annualization)
    
    @classmethod
    def cal_sharpe_ratio_by_year(cls, returns, risk_free=0, period=DAILY, annualization=None):
        """Determines the Sharpe ratio for each year."""
        if len(returns) < 1:
            return_as_array = isinstance(returns, np.ndarray)
            return np.array([]) if return_as_array else pd.Series(dtype='float64')

        return_as_array = isinstance(returns, np.ndarray)

        returns = cls._ensure_datetime_index_series(returns, period=period)

        sharpe_by_year = returns.groupby(returns.index.year).apply(
            lambda x: cls.cal_sharpe_ratio(x, risk_free=risk_free, period=period, annualization=annualization)
        )

        return sharpe_by_year.values if return_as_array else sharpe_by_year
    
    def sharpe_ratio_by_year(self, returns=None, risk_free=0, period=DAILY, annualization=None):
        """Determines the Sharpe ratio for each year."""
        returns = self._get_returns(returns)
        return self.cal_sharpe_ratio_by_year(returns, risk_free, period, annualization)

    @classmethod
    def cal_max_drawdown_by_year(cls, returns):
        """Determines the maximum drawdown for each year."""
        if len(returns) < 1:
            return_as_array = isinstance(returns, np.ndarray)
            return np.array([]) if return_as_array else pd.Series(dtype='float64')

        return_as_array = isinstance(returns, np.ndarray)

        returns = cls._ensure_datetime_index_series(returns, period=DAILY)

        max_dd_by_year = returns.groupby(returns.index.year).apply(
            lambda x: cls.cal_max_drawdown(x)
        )

        return max_dd_by_year.values if return_as_array else max_dd_by_year
    
    def max_drawdown_by_year(self, returns=None):
        """Determines the maximum drawdown for each year."""
        returns = self._get_returns(returns)
        return self.cal_max_drawdown_by_year(returns)

    @classmethod
    def cal_skewness(cls, returns):
        """Calculates the skewness of the returns."""
        if len(returns) < 3:
            return np.nan
            
        return stats.skew(returns, nan_policy='omit')
    
    def skewness(self, returns=None):
        """Calculates the skewness of the returns."""
        returns = self._get_returns(returns)
        return self.cal_skewness(returns)
    
    @classmethod
    def cal_kurtosis(cls, returns):
        """Calculates the kurtosis of the returns."""
        if len(returns) < 4:
            return np.nan
            
        return stats.kurtosis(returns, nan_policy='omit')
    
    def kurtosis(self, returns=None):
        """Calculates the kurtosis of the returns."""
        returns = self._get_returns(returns)
        return self.cal_kurtosis(returns)
    
    @classmethod
    def cal_hurst_exponent(cls, returns):
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
    
    def hurst_exponent(self, returns=None):
        """Calculates the Hurst exponent of the returns."""
        returns = self._get_returns(returns)
        return self.cal_hurst_exponent(returns)

    @classmethod
    def cal_sterling_ratio(cls, returns, risk_free=0.0, period=DAILY, annualization=None):
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
            ann_ret = cls.cal_annual_return(returns, period, annualization)
            return np.inf if ann_ret - risk_free > 0 else np.nan
        
        # Calculate annualized return
        ann_ret = cls.cal_annual_return(returns, period, annualization)
        
        # Sterling ratio = (annualized return - risk free) / average drawdown
        return (ann_ret - risk_free) / avg_drawdown
    
    def sterling_ratio(self, returns=None, risk_free=0.0):
        """Calculates the Sterling ratio."""
        returns = self._get_returns(returns)
        return self.cal_sterling_ratio(returns, risk_free)
    
    @classmethod
    def cal_burke_ratio(cls, returns, risk_free=0.0, period=DAILY, annualization=None):
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
            squared_drawdowns = [dd**2 for dd in drawdown_periods]
            burke_risk = np.sqrt(np.sum(squared_drawdowns))
        
        if burke_risk == 0 or burke_risk < 1e-10:
            # Extremely small risk, return large positive ratio or NaN
            ann_ret = cls.cal_annual_return(returns, period, annualization)
            return np.inf if ann_ret - risk_free > 0 else np.nan
        
        # Calculate annualized return
        ann_ret = cls.cal_annual_return(returns, period, annualization)
        
        # Burke ratio = (annualized return - risk free) / burke risk
        return (ann_ret - risk_free) / burke_risk
    
    def burke_ratio(self, returns=None, risk_free=0.0):
        """Calculates the Burke ratio."""
        returns = self._get_returns(returns)
        return self.cal_burke_ratio(returns, risk_free)
    
    @classmethod
    def cal_kappa_three_ratio(cls, returns, risk_free=0.0, period=DAILY, annualization=None, mar=0.0):
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
                ann_ret = cls.cal_annual_return(returns, period, annualization)
                return np.inf if ann_ret - risk_free > 0 else np.nan
            lpm3_risk = std_dev * np.sqrt(ann_factor)
        else:
            lpm3_annualized = lpm3 * ann_factor
            # Take cube root of LPM3
            lpm3_risk = lpm3_annualized ** (1.0 / 3.0)
        
        if lpm3_risk == 0 or lpm3_risk < 1e-10:
            # Extremely small risk
            ann_ret = cls.cal_annual_return(returns, period, annualization)
            return np.inf if ann_ret - risk_free > 0 else np.nan
        
        # Calculate annualized return
        ann_ret = cls.cal_annual_return(returns, period, annualization)
        
        # Kappa 3 ratio = (annualized return - risk free) / LPM3^(1/3)
        return (ann_ret - risk_free) / lpm3_risk
    
    def kappa_three_ratio(self, returns=None, risk_free=0.0, target_return=0.0):
        """Calculates the Kappa 3 ratio (downside deviation cubed)."""
        returns = self._get_returns(returns)
        return self.cal_kappa_three_ratio(returns, risk_free, mar=target_return)
    
    @classmethod
    def cal_adjusted_sharpe_ratio(cls, returns, risk_free=0.0):
        """Calculates the adjusted Sharpe ratio (accounts for skewness and kurtosis)."""
        if len(returns) < 4:
            return np.nan
            
        sharpe = cls.cal_sharpe_ratio(returns, risk_free)
        
        if np.isnan(sharpe):
            return np.nan
            
        # Calculate skewness and kurtosis with NaN handling
        skew = cls.cal_skewness(returns)
        if np.isnan(skew):
            skew = 0
            
        kurt = cls.cal_kurtosis(returns)
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
    
    def adjusted_sharpe_ratio(self, returns=None, risk_free=0.0):
        """Calculates the adjusted Sharpe ratio (accounts for skewness and kurtosis)."""
        returns = self._get_returns(returns)
        return self.cal_adjusted_sharpe_ratio(returns, risk_free)
    
    @classmethod
    def cal_stutzer_index(cls, returns, target_return=0.0):
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
        except:
            return np.nan
    
    def stutzer_index(self, returns=None, target_return=0.0):
        """Calculates the Stutzer index."""
        returns = self._get_returns(returns)
        return self.cal_stutzer_index(returns, target_return)

    @classmethod
    def cal_max_consecutive_up_days(cls, returns):
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
    
    def max_consecutive_up_days(self, returns=None):
        """Determines the maximum number of consecutive days with positive returns."""
        returns = self._get_returns(returns)
        return self.cal_max_consecutive_up_days(returns)
    
    @classmethod
    def cal_max_consecutive_down_days(cls, returns):
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
    
    def max_consecutive_down_days(self, returns=None):
        """Determines the maximum number of consecutive days with negative returns."""
        returns = self._get_returns(returns)
        return self.cal_max_consecutive_down_days(returns)
    
    @classmethod
    def cal_max_consecutive_gain(cls, returns):
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
    
    def max_consecutive_gain(self, returns=None):
        """Determines the maximum consecutive gain."""
        returns = self._get_returns(returns)
        return self.cal_max_consecutive_gain(returns)
    
    @classmethod
    def cal_max_consecutive_loss(cls, returns):
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
    
    def max_consecutive_loss(self, returns=None):
        """Determines the maximum consecutive loss."""
        returns = self._get_returns(returns)
        return self.cal_max_consecutive_loss(returns)
    
    @classmethod
    def cal_max_single_day_gain(cls, returns):
        """Determines the maximum single day gain."""
        if len(returns) < 1:
            return np.nan
            
        return returns.max()
    
    def max_single_day_gain(self, returns=None):
        """Determines the maximum single day gain."""
        returns = self._get_returns(returns)
        return self.cal_max_single_day_gain(returns)
    
    @classmethod
    def cal_max_single_day_loss(cls, returns):
        """Determines the maximum single day loss."""
        if len(returns) < 1:
            return np.nan
            
        return returns.min()
    
    def max_single_day_loss(self, returns=None):
        """Determines the maximum single day loss."""
        returns = self._get_returns(returns)
        return self.cal_max_single_day_loss(returns)
    
    @classmethod
    def cal_stock_market_correlation(cls, returns, market_returns):
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
    
    def stock_market_correlation(self, returns=None, market_returns=None):
        """Determines the correlation with the stock market."""
        returns = self._get_returns(returns)
        market_returns = self._get_factor_returns(market_returns)
        return self.cal_stock_market_correlation(returns, market_returns)
    
    @classmethod
    def cal_bond_market_correlation(cls, returns, bond_returns):
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
    
    def bond_market_correlation(self, returns=None, bond_returns=None):
        """Determines the correlation with the bond market."""
        returns = self._get_returns(returns)
        if bond_returns is None:
            bond_returns = self._get_factor_returns(bond_returns)
        return self.cal_bond_market_correlation(returns, bond_returns)
    
    @classmethod
    def cal_futures_market_correlation(cls, returns, futures_returns):
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
    def cal_serial_correlation(cls, returns, lag=1):
        """Determines the serial correlation of returns."""
        if len(returns) < lag + 1:
            return np.nan
            
        returns_lagged = returns.shift(lag)
        
        # Remove NaN values
        mask = ~(np.isnan(returns) | np.isnan(returns_lagged))
        returns_clean = returns[mask]
        returns_lagged_clean = returns_lagged[mask]
        
        if len(returns_clean) < 2:
            return np.nan
            
        return np.corrcoef(returns_clean, returns_lagged_clean)[0, 1]
    
    def serial_correlation(self, returns=None, lag=1):
        """Determines the serial correlation of returns."""
        returns = self._get_returns(returns)
        return self.cal_serial_correlation(returns, lag)
    
    @classmethod
    def cal_treynor_mazuy_timing(cls, returns, factor_returns, risk_free=0.0):
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
        except:
            return np.nan
    
    def treynor_mazuy_timing(self, returns=None, factor_returns=None, risk_free=0.0):
        """Calculates the Treynor-Mazuy market timing coefficient (gamma)."""
        returns = self._get_returns(returns)
        factor_returns = self._get_factor_returns(factor_returns)
        return self.cal_treynor_mazuy_timing(returns, factor_returns, risk_free)
    
    @classmethod
    def cal_henriksson_merton_timing(cls, returns, factor_returns, risk_free=0.0):
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
        except:
            return np.nan
    
    def henriksson_merton_timing(self, returns=None, factor_returns=None, risk_free=0.0):
        """Calculates the Henriksson-Merton market timing coefficient."""
        returns = self._get_returns(returns)
        factor_returns = self._get_factor_returns(factor_returns)
        return self.cal_henriksson_merton_timing(returns, factor_returns, risk_free)
    
    @classmethod
    def cal_market_timing_return(cls, returns, factor_returns, risk_free=0.0):
        """Calculates market timing return component."""
        gamma = cls.cal_treynor_mazuy_timing(returns, factor_returns, risk_free)
        
        if np.isnan(gamma):
            return np.nan
        
        returns_aligned, factor_aligned = cls._aligned_series(returns, factor_returns)
        excess_factor = factor_aligned - risk_free
        
        # Market timing return is gamma * factor_squared
        return gamma * np.mean(excess_factor ** 2)
    
    def market_timing_return(self, returns=None, factor_returns=None, risk_free=0.0):
        """Calculates market timing return component."""
        returns = self._get_returns(returns)
        factor_returns = self._get_factor_returns(factor_returns)
        return self.cal_market_timing_return(returns, factor_returns, risk_free)
    
    def annual_alpha(self, returns=None, factor_returns=None, risk_free=0.0, period=DAILY, annualization=None):
        """Determines the annual alpha for each year."""
        returns = self._get_returns(returns)
        factor_returns = self._get_factor_returns(factor_returns)
        
        if len(returns) < 1:
            return pd.Series([], dtype=float)
        
        # Ensure returns has a datetime index
        if not isinstance(returns.index, pd.DatetimeIndex):
            return pd.Series([], dtype=float)
        
        def alpha_for_year(group_data):
            year_returns = group_data[0]
            year_factor = group_data[1]
            return self.alpha(year_returns, year_factor, risk_free, period, annualization)
        
        # Group by year and calculate alpha for each year
        grouped = returns.groupby(returns.index.year)
        factor_grouped = factor_returns.groupby(factor_returns.index.year)
        
        annual_alphas = []
        for year in grouped.groups.keys():
            if year in factor_grouped.groups.keys():
                year_returns = grouped.get_group(year)
                year_factor = factor_grouped.get_group(year)
                alpha_val = self.alpha(year_returns, year_factor, risk_free, period, annualization)
                annual_alphas.append((year, alpha_val))
        
        if not annual_alphas:
            return pd.Series([], dtype=float)
            
        years, alphas = zip(*annual_alphas)
        return pd.Series(alphas, index=years)
    
    def annual_beta(self, returns=None, factor_returns=None, risk_free=0.0, period=DAILY, annualization=None):
        """Determines the annual beta for each year."""
        returns = self._get_returns(returns)
        factor_returns = self._get_factor_returns(factor_returns)
        
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
                beta_val = self.beta(year_returns, year_factor, risk_free, period, annualization)
                annual_betas.append((year, beta_val))
        
        if not annual_betas:
            return pd.Series([], dtype=float)
            
        years, betas = zip(*annual_betas)
        return pd.Series(betas, index=years)
    
    def residual_risk(self, returns=None, factor_returns=None, risk_free=0.0):
        """Calculates the residual risk (tracking error of alpha)."""
        returns = self._get_returns(returns)
        factor_returns = self._get_factor_returns(factor_returns)
        
        returns_aligned, factor_aligned = self._aligned_series(returns, factor_returns)
        
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
    
    def conditional_sharpe_ratio(self, returns=None, cutoff=0.05):
        """Calculates the conditional Sharpe ratio."""
        returns = self._get_returns(returns)
        
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
    
    def var_excess_return(self, returns=None, cutoff=0.05):
        """Calculates the VaR excess return."""
        returns = self._get_returns(returns)
        
        if len(returns) < 2:
            return np.nan
            
        var_value = self.value_at_risk(returns, cutoff)
        excess_returns = returns[returns <= var_value]
        
        if len(excess_returns) == 0:
            return np.nan
            
        return np.mean(excess_returns)
    
    def max_consecutive_up_weeks(self, returns=None):
        """Determines the maximum number of consecutive weeks with positive returns."""
        returns = self._get_returns(returns)
        
        if len(returns) < 1:
            return np.nan
            
        # Resample to weekly returns
        weekly_returns = returns.resample('W').apply(lambda x: self.cum_returns_final(x))
        
        up_weeks = weekly_returns > 0
        
        if not up_weeks.any():
            return 0
            
        # Find consecutive True values
        groups = (up_weeks != up_weeks.shift(1)).cumsum()
        consecutive_counts = up_weeks.groupby(groups).sum()
        
        return consecutive_counts.max()
    
    def max_consecutive_down_weeks(self, returns=None):
        """Determines the maximum number of consecutive weeks with negative returns."""
        returns = self._get_returns(returns)
        
        if len(returns) < 1:
            return np.nan
            
        # Resample to weekly returns
        weekly_returns = returns.resample('W').apply(lambda x: self.cum_returns_final(x))
        
        down_weeks = weekly_returns < 0
        
        if not down_weeks.any():
            return 0
            
        # Find consecutive True values
        groups = (down_weeks != down_weeks.shift(1)).cumsum()
        consecutive_counts = down_weeks.groupby(groups).sum()
        
        return consecutive_counts.max()
    
    def max_consecutive_up_months(self, returns=None):
        """Determines the maximum number of consecutive months with positive returns."""
        returns = self._get_returns(returns)
        
        if len(returns) < 1:
            return np.nan
            
        # Resample to monthly returns
        monthly_returns = returns.resample('M').apply(lambda x: self.cum_returns_final(x))
        
        up_months = monthly_returns > 0
        
        if not up_months.any():
            return 0
            
        # Find consecutive True values
        groups = (up_months != up_months.shift(1)).cumsum()
        consecutive_counts = up_months.groupby(groups).sum()
        
        return consecutive_counts.max()
    
    def max_consecutive_down_months(self, returns=None):
        """Determines the maximum number of consecutive months with negative returns."""
        returns = self._get_returns(returns)
        
        if len(returns) < 1:
            return np.nan
            
        # Resample to monthly returns
        monthly_returns = returns.resample('M').apply(lambda x: self.cum_returns_final(x))
        
        down_months = monthly_returns < 0
        
        if not down_months.any():
            return 0
            
        # Find consecutive True values
        groups = (down_months != down_months.shift(1)).cumsum()
        consecutive_counts = down_months.groupby(groups).sum()
        
        return consecutive_counts.max()
    
    def win_rate(self, returns=None):
        """Calculates the percentage of positive returns."""
        returns = self._get_returns(returns)
        
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
    
    def loss_rate(self, returns=None):
        """Calculates the percentage of negative returns."""
        returns = self._get_returns(returns)
        
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
    
    def get_max_drawdown_period(self, returns=None):
        """Gets the start and end dates of the maximum drawdown period."""
        returns = self._get_returns(returns)
        
        if len(returns) < 1:
            return None, None
            
        cum_returns = self.cum_returns(returns, starting_value=1)
        
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
    
    def max_drawdown_days(self, returns=None):
        """Calculates the duration of maximum drawdown in days."""
        returns = self._get_returns(returns)
        
        if len(returns) < 1:
            return np.nan

        if not isinstance(returns, pd.Series):
            returns = pd.Series(returns)

        # Calculate cumulative returns
        cum_ret = self.cum_returns(returns, starting_value=100)
        
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
    
    def futures_market_correlation(self, returns=None, futures_returns=None):
        """Determines the correlation with the futures market."""
        returns = self._get_returns(returns)
        if futures_returns is None:
            futures_returns = self._get_factor_returns(futures_returns)
        return self.cal_futures_market_correlation(returns, futures_returns)
    
    @classmethod
    def cal_up_alpha_beta(cls, returns, factor_returns, risk_free=0.0, period=DAILY, annualization=None, out=None):
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
    def cal_down_alpha_beta(cls, returns, factor_returns, risk_free=0.0, period=DAILY, annualization=None, out=None):
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
    def cal_alpha_percentile_rank(cls, strategy_returns, all_strategies_returns, factor_returns, risk_free=0.0, period=DAILY, annualization=None):
        """Calculates the percentile rank of alpha relative to a universe."""
        if len(strategy_returns) < 3:
            return np.nan
        
        # Calculate alpha for the target strategy
        strategy_alpha = cls.cal_alpha(strategy_returns, factor_returns, risk_free, period, annualization)
        
        if np.isnan(strategy_alpha):
            return np.nan
        
        # Calculate alpha for all strategies
        all_alphas = []
        for other_returns in all_strategies_returns:
            if len(other_returns) < 3:
                continue
            other_alpha = cls.cal_alpha(other_returns, factor_returns, risk_free, period, annualization)
            if not np.isnan(other_alpha):
                all_alphas.append(other_alpha)
        
        if len(all_alphas) == 0:
            return np.nan
        
        # Calculate percentile rank
        # Count how many strategies have alpha less than target strategy
        rank = sum(1 for a in all_alphas if a < strategy_alpha)
        percentile = rank / len(all_alphas)
        
        return float(percentile)
    
    def alpha_percentile_rank(self, strategy_returns, all_strategies_returns, factor_returns=None, risk_free=0.0, period=DAILY, annualization=None):
        """Calculates the percentile rank of alpha relative to a universe."""
        if factor_returns is None:
            factor_returns = self._get_factor_returns(factor_returns)
        return self.cal_alpha_percentile_rank(strategy_returns, all_strategies_returns, factor_returns, risk_free, period, annualization)
    
    @classmethod
    def cal_cornell_timing(cls, returns, factor_returns, risk_free=0.0):
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
            
        except Exception:
            return np.nan
    
    def cornell_timing(self, returns=None, factor_returns=None, risk_free=0.0):
        """Calculates the Cornell timing model coefficient."""
        returns = self._get_returns(returns)
        factor_returns = self._get_factor_returns(factor_returns)
        return self.cal_cornell_timing(returns, factor_returns, risk_free)
    
    def r_cubed(self, returns=None, factor_returns=None):
        """Calculates R-cubed (R³) measure."""
        returns = self._get_returns(returns)
        factor_returns = self._get_factor_returns(factor_returns)
        
        returns_aligned, factor_aligned = self._aligned_series(returns, factor_returns)
        
        if len(returns_aligned) < 2:
            return np.nan
            
        # Calculate correlation coefficient
        correlation = np.corrcoef(returns_aligned, factor_aligned)[0, 1]
        
        # R-cubed is the cube of correlation
        return correlation ** 3
    
    def regression_annual_return(self, returns=None, factor_returns=None, risk_free=0.0, period=DAILY, annualization=None):
        """Calculates the annual return from regression (alpha + beta * benchmark_return)."""
        alpha_val = self.alpha(returns, factor_returns, risk_free, period, annualization)
        beta_val = self.beta(returns, factor_returns, risk_free, period, annualization)
        
        if np.isnan(alpha_val) or np.isnan(beta_val):
            return np.nan
            
        factor_returns = self._get_factor_returns(factor_returns)
        benchmark_annual = self.annual_return(factor_returns, period, annualization)
        
        if np.isnan(benchmark_annual):
            return np.nan
            
        return alpha_val + beta_val * benchmark_annual
    
    def annualized_cumulative_return(self, returns=None, period=DAILY, annualization=None):
        """Calculates the annualized cumulative return."""
        # This is essentially the same as annual_return
        return self.annual_return(returns, period, annualization)
    
    def annual_active_return(self, returns=None, factor_returns=None, period=DAILY, annualization=None):
        """Calculates the annual active return (strategy - benchmark)."""
        returns = self._get_returns(returns)
        factor_returns = self._get_factor_returns(factor_returns)
        
        if len(returns) < 1:
            return np.nan
            
        # Align the series first
        returns_aligned, factor_aligned = self._aligned_series(returns, factor_returns)
        
        # Calculate annualized returns on aligned data
        strategy_annual = self.cal_annual_return(returns_aligned, period, annualization)
        benchmark_annual = self.cal_annual_return(factor_aligned, period, annualization)
        
        if np.isnan(strategy_annual) or np.isnan(benchmark_annual):
            return np.nan
            
        return strategy_annual - benchmark_annual
    
    def annual_active_risk(self, returns=None, factor_returns=None, period=DAILY, annualization=None):
        """Calculates the annual active risk (tracking error)."""
        return self.tracking_error(returns, factor_returns, period, annualization)
    
    def tracking_difference(self, returns=None, factor_returns=None):
        """Calculates the tracking difference (cumulative strategy return - cumulative benchmark return)."""
        returns = self._get_returns(returns)
        factor_returns = self._get_factor_returns(factor_returns)
        
        if len(returns) < 1:
            return np.nan
            
        returns_aligned, factor_aligned = self._aligned_series(returns, factor_returns)
        
        # Calculate cumulative returns
        cum_strategy_return = self.cal_cum_returns_final(returns_aligned, starting_value=0)
        cum_benchmark_return = self.cal_cum_returns_final(factor_aligned, starting_value=0)
        
        # Tracking difference = cumulative strategy return - cumulative benchmark return
        result = cum_strategy_return - cum_benchmark_return
        if not isinstance(result, (float, np.floating)):
            result = result.item()
        return result
    
    def annual_active_return_by_year(self, returns=None, factor_returns=None, period=DAILY, annualization=None):
        """Determines the annual active return for each year."""
        returns = self._get_returns(returns)
        factor_returns = self._get_factor_returns(factor_returns)
        
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
                active_return = self.annual_active_return(year_returns, year_factor, period, annualization)
                annual_active_returns.append((year, active_return))
        
        if not annual_active_returns:
            return pd.Series([], dtype=float)
            
        years, active_returns = zip(*annual_active_returns)
        return pd.Series(active_returns, index=years)
    
    def information_ratio_by_year(self, returns=None, factor_returns=None, period=DAILY, annualization=None):
        """Determines the information ratio for each year."""
        returns = self._get_returns(returns)
        factor_returns = self._get_factor_returns(factor_returns)
        
        if len(returns) < 1:
            return_as_array = isinstance(returns, np.ndarray)
            return np.array([]) if return_as_array else pd.Series(dtype='float64')
        
        # Track whether input is array for return type
        return_as_array = isinstance(returns, np.ndarray)
        
        # Ensure we have datetime-indexed Series
        if return_as_array or not hasattr(returns, 'index') or not isinstance(returns.index, pd.DatetimeIndex):
            # For numpy arrays or non-datetime indexed data, convert to datetime-indexed series
            returns = self._ensure_datetime_index_series(returns, period=period)
            factor_returns = self._ensure_datetime_index_series(factor_returns, period=period)
        
        # Align the series
        returns_aligned, factor_aligned = self._aligned_series(returns, factor_returns)
        
        # Group by year and calculate information ratio for each year
        information_ratios = returns_aligned.groupby(returns_aligned.index.year).apply(
            lambda x: self._calculate_information_ratio_for_active_returns(
                x - factor_aligned.loc[x.index],
                period=period,
                annualization=annualization
            )
        )
        
        # Remove name attribute if it exists
        if hasattr(information_ratios, 'name'):
            information_ratios.name = None
            
        return information_ratios.values if return_as_array else information_ratios
    
    def _calculate_information_ratio_for_active_returns(self, active_returns, period=DAILY, annualization=None):
        """Calculate information ratio from active returns."""
        ann_factor = self.annualization_factor(period, annualization)
        mean_excess_return = active_returns.mean()
        std_excess_return = active_returns.std(ddof=1)
        if std_excess_return == 0:
            return np.nan
        else:
            return (mean_excess_return * ann_factor) / (std_excess_return * np.sqrt(ann_factor))
    
    def alpha_aligned(self, returns=None, factor_returns=None, risk_free=0.0, period=DAILY, annualization=None):
        """Calculates alpha with aligned series.
        """
        returns = self._get_returns(returns)
        factor_returns = self._get_factor_returns(factor_returns)
        return self.cal_alpha_aligned(
            returns,
            factor_returns,
            risk_free=risk_free,
            period=period,
            annualization=annualization,
        )
    
    def beta_aligned(self, returns=None, factor_returns=None, risk_free=0.0, period=DAILY, annualization=None):
        """Calculates beta with aligned series.
        """
        returns = self._get_returns(returns)
        factor_returns = self._get_factor_returns(factor_returns)
        return self.cal_beta_aligned(
            returns,
            factor_returns,
            risk_free=risk_free,
        )
    
    def alpha_beta_aligned(self, returns=None, factor_returns=None, risk_free=0.0, period=DAILY, annualization=None, out=None):
        """Calculates both alpha and beta with aligned series.
        """
        returns = self._get_returns(returns)
        factor_returns = self._get_factor_returns(factor_returns)
        return self.cal_alpha_beta_aligned(
            returns,
            factor_returns,
            risk_free=risk_free,
            period=period,
            annualization=annualization,
            out=out,
        )
    
    def up_alpha_beta(self, returns=None, factor_returns=None, risk_free=0.0, period=DAILY, annualization=None):
        """Calculates alpha and beta for up-market periods only."""
        returns = self._get_returns(returns)
        factor_returns = self._get_factor_returns(factor_returns)
        return self.cal_up_alpha_beta(returns, factor_returns, risk_free, period, annualization)
    
    def down_alpha_beta(self, returns=None, factor_returns=None, risk_free=0.0, period=DAILY, annualization=None):
        """Calculates alpha and beta for down-market periods only."""
        returns = self._get_returns(returns)
        factor_returns = self._get_factor_returns(factor_returns)
        return self.cal_down_alpha_beta(returns, factor_returns, risk_free, period, annualization)
    
    def second_max_drawdown(self, returns=None):
        """Determines the second maximum drawdown of a strategy."""
        returns = self._get_returns(returns)
        
        drawdown_periods = self._get_all_drawdowns(returns)
        
        if len(drawdown_periods) < 2:
            return np.nan
        
        # Sort drawdowns (most negative first)
        sorted_drawdowns = np.sort(drawdown_periods)
        
        # Get second largest (second most negative)
        return sorted_drawdowns[-2]
    
    def third_max_drawdown(self, returns=None):
        """Determines the third maximum drawdown of a strategy."""
        returns = self._get_returns(returns)
        
        drawdown_periods = self._get_all_drawdowns(returns)
        
        if len(drawdown_periods) < 3:
            return np.nan
        
        # Sort drawdowns (most negative first)
        sorted_drawdowns = np.sort(drawdown_periods)
        
        # Get third largest (third most negative)
        return sorted_drawdowns[-3]
    
    def max_drawdown_weeks(self, returns=None):
        """Calculates the duration of maximum drawdown in weeks."""
        returns = self._get_returns(returns)
        
        if len(returns) < 1:
            return np.nan

        if not isinstance(returns, pd.Series):
            returns = pd.Series(returns)

        # Calculate cumulative returns
        cum_ret = self.cum_returns(returns, starting_value=100)
        
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
    
    def max_drawdown_months(self, returns=None):
        """Calculates the duration of maximum drawdown in months."""
        returns = self._get_returns(returns)
        
        if len(returns) < 1:
            return np.nan
            
        if not isinstance(returns, pd.Series):
            returns = pd.Series(returns)
            
        # Calculate cumulative returns
        cum_ret = self.cum_returns(returns, starting_value=100)
        
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
    
    def max_drawdown_recovery_days(self, returns=None):
        """Calculates the recovery time from maximum drawdown in days."""
        returns = self._get_returns(returns)
        
        if len(returns) < 1:
            return np.nan
            
        cum_returns = self.cum_returns(returns, starting_value=1)
        
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
    
    def max_drawdown_recovery_weeks(self, returns=None):
        """Calculates the recovery time from maximum drawdown in weeks."""
        returns = self._get_returns(returns)
        
        if len(returns) < 1:
            return np.nan
            
        if not isinstance(returns, pd.Series):
            returns = pd.Series(returns)
            
        # Calculate cumulative returns
        cum_ret = self.cum_returns(returns, starting_value=100)
        
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
    
    def max_drawdown_recovery_months(self, returns=None):
        """Calculates the recovery time from maximum drawdown in months."""
        returns = self._get_returns(returns)
        
        if len(returns) < 1:
            return np.nan
            
        if not isinstance(returns, pd.Series):
            returns = pd.Series(returns)
            
        # Calculate cumulative returns
        cum_ret = self.cum_returns(returns, starting_value=100)
        
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
    
    def annual_volatility_by_year(self, returns=None, period=DAILY, annualization=None):
        """Determines the annual volatility for each year."""
        returns = self._get_returns(returns)
        if len(returns) < 1:
            return_as_array = isinstance(returns, np.ndarray)
            return np.array([]) if return_as_array else pd.Series(dtype='float64')

        return_as_array = isinstance(returns, np.ndarray)

        returns = self._ensure_datetime_index_series(returns, period=period)

        annual_vol_by_year = returns.groupby(returns.index.year).apply(
            lambda x: self.annual_volatility(x, period=period, annualization=annualization)
        )

        return annual_vol_by_year.values if return_as_array else annual_vol_by_year
    
    def max_single_day_gain_date(self, returns=None):
        """Determines the date of maximum single day gain."""
        returns = self._get_returns(returns)
        
        if len(returns) < 1:
            return None
            
        return returns.idxmax()
    
    def max_single_day_loss_date(self, returns=None):
        """Determines the date of maximum single day loss."""
        returns = self._get_returns(returns)
        
        if len(returns) < 1:
            return None
            
        return returns.idxmin()
    
    def max_consecutive_up_start_date(self, returns=None):
        """Determines the start date of maximum consecutive up period."""
        returns = self._get_returns(returns)
        
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
    
    def max_consecutive_up_end_date(self, returns=None):
        """Determines the end date of maximum consecutive up period."""
        returns = self._get_returns(returns)
        
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
    
    def max_consecutive_down_start_date(self, returns=None):
        """Determines the start date of maximum consecutive down period."""
        returns = self._get_returns(returns)
        
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
    
    def max_consecutive_down_end_date(self, returns=None):
        """Determines the end date of maximum consecutive down period."""
        returns = self._get_returns(returns)
        
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
    
    def beta_fragility_heuristic(self, returns=None, factor_returns=None):
        """Estimate fragility to drop in beta."""
        returns = self._get_returns(returns)
        factor_returns = self._get_factor_returns(factor_returns)
        
        if len(returns) < 3 or len(factor_returns) < 3:
            return np.nan
            
        returns_aligned, factor_aligned = self._aligned_series(returns, factor_returns)
        
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
        heuristic = (start_returns_weight*start_returns) + \
            (end_returns_weight*end_returns) - mid_returns
            
        return heuristic
    
    def beta_fragility_heuristic_aligned(self, returns=None, factor_returns=None):
        """Calculates the beta fragility heuristic with aligned series."""
        # This is the same as beta_fragility_heuristic since we already align series
        return self.beta_fragility_heuristic(returns, factor_returns)
    
    def gpd_risk_estimates(self, returns=None, var_p=0.01):
        """Estimate VaR and ES using the Generalized Pareto Distribution (GPD).
        
        Returns
        -------
        [threshold, scale_param, shape_param, var_estimate, es_estimate] : list[float]
        """
        returns = self._get_returns(returns)
        
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
            param_result = self._gpd_loglikelihood_minimizer_aligned(losses_beyond_threshold)
            if (param_result[0] is not False and param_result[1] is not False):
                scale_param = param_result[0]
                shape_param = param_result[1]
                var_estimate = self._gpd_var_calculator(threshold, scale_param,
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
            es_estimate = self._gpd_es_calculator(var_estimate, threshold,
                                                scale_param, shape_param)
            result = np.array([threshold, scale_param, shape_param,
                             var_estimate, es_estimate])
                             
        if isinstance(returns, pd.Series):
            result = pd.Series(result)
        return result
    
    def _gpd_es_calculator(self, var_estimate, threshold, scale_param, shape_param):
        result = 0
        if (1 - shape_param) != 0:
            # this formula is from Gilli and Kellezi pg. 8
            var_ratio = (var_estimate/(1 - shape_param))
            param_ratio = ((scale_param - (shape_param * threshold)) /
                          (1 - shape_param))
            result = var_ratio + param_ratio
        return result
    
    def _gpd_var_calculator(self, threshold, scale_param, shape_param, probability, total_n, exceedance_n):
        result = 0
        if exceedance_n > 0 and shape_param > 0:
            # this formula is from Gilli and Kellezi pg. 12
            param_ratio = scale_param / shape_param
            prob_ratio = (total_n/exceedance_n) * probability
            result = threshold + (param_ratio * (pow(prob_ratio, -shape_param) - 1))
        return result
    
    def _gpd_loglikelihood_minimizer_aligned(self, price_data):
        from scipy import optimize
        result = [False, False]
        default_scale_param = 1
        default_shape_param = 1
        if len(price_data) > 0:
            gpd_loglikelihood_lambda = lambda params: self._gpd_loglikelihood(params, price_data)
            try:
                optimization_results = optimize.minimize(gpd_loglikelihood_lambda,
                                                        [default_scale_param, default_shape_param],
                                                        method='Nelder-Mead')
                if optimization_results.success:
                    resulting_params = optimization_results.x
                    if len(resulting_params) == 2:
                        result[0] = resulting_params[0]
                        result[1] = resulting_params[1]
            except:
                pass
        return result
    
    def _gpd_loglikelihood(self, params, price_data):
        if params[1] != 0:
            return -self._gpd_loglikelihood_scale_and_shape(params[0], params[1], price_data)
        else:
            return -self._gpd_loglikelihood_scale_only(params[0], price_data)
    
    def _gpd_loglikelihood_scale_and_shape(self, scale, shape, price_data):
        n = len(price_data)
        result = -1 * float_info.max
        if scale != 0:
            param_factor = shape / scale
            if shape != 0 and param_factor >= 0 and scale >= 0:
                result = ((-n * np.log(scale)) -
                         (((1 / shape) + 1) *
                          (np.log((shape / scale * price_data) + 1)).sum()))
        return result
    
    def _gpd_loglikelihood_scale_only(self, scale, price_data):
        n = len(price_data)
        data_sum = price_data.sum()
        result = -1 * float_info.max
        if scale >= 0:
            result = ((-n*np.log(scale)) - (data_sum/scale))
        return result
    
    def gpd_risk_estimates_aligned(self, returns=None, var_p=0.01):
        """Calculates GPD risk estimates (aligned version for compatibility)."""
        returns = self._get_returns(returns)
        
        # For compatibility with original API, this is the same as gpd_risk_estimates
        return self.gpd_risk_estimates(returns, var_p)
    
    def second_max_drawdown_days(self, returns=None):
        """Calculates the duration of second maximum drawdown in days."""
        returns = self._get_returns(returns)
        
        drawdown_periods = self._get_all_drawdowns_detailed(returns)
        
        if len(drawdown_periods) < 2:
            return np.nan
        
        # Sort by drawdown value (most negative first)
        sorted_drawdowns = sorted(drawdown_periods, key=lambda x: x['value'])
        
        # Get second largest drawdown duration
        return sorted_drawdowns[1]['duration']
    
    def second_max_drawdown_recovery_days(self, returns=None):
        """Calculates the recovery time from second maximum drawdown in days."""
        returns = self._get_returns(returns)
        
        drawdown_periods = self._get_all_drawdowns_detailed(returns)
        
        if len(drawdown_periods) < 2:
            return np.nan
        
        # Sort by drawdown value (most negative first)
        sorted_drawdowns = sorted(drawdown_periods, key=lambda x: x['value'])
        
        # Get second largest drawdown recovery duration
        recovery_duration = sorted_drawdowns[1]['recovery_duration']
        return recovery_duration if recovery_duration is not None else np.nan
    
    def third_max_drawdown_days(self, returns=None):
        """Calculates the duration of third maximum drawdown in days."""
        returns = self._get_returns(returns)
        
        drawdown_periods = self._get_all_drawdowns_detailed(returns)
        
        if len(drawdown_periods) < 3:
            return np.nan
        
        # Sort by drawdown value (most negative first)
        sorted_drawdowns = sorted(drawdown_periods, key=lambda x: x['value'])
        
        # Get third largest drawdown duration
        return sorted_drawdowns[2]['duration']
    
    def third_max_drawdown_recovery_days(self, returns=None):
        """Calculates the recovery time from third maximum drawdown in days."""
        returns = self._get_returns(returns)
        
        drawdown_periods = self._get_all_drawdowns_detailed(returns)
        
        if len(drawdown_periods) < 3:
            return np.nan
        
        # Sort by drawdown value (most negative first)
        sorted_drawdowns = sorted(drawdown_periods, key=lambda x: x['value'])
        
        # Get third largest drawdown recovery duration
        recovery_duration = sorted_drawdowns[2]['recovery_duration']
        return recovery_duration if recovery_duration is not None else np.nan
    
    def roll_alpha(self, returns=None, factor_returns=None, window=252, risk_free=0.0, period=DAILY, annualization=None):
        """Calculates rolling alpha over a specified window."""
        returns = self._get_returns(returns)
        factor_returns = self._get_factor_returns(factor_returns)
        
        returns_aligned, factor_aligned = self._aligned_series(returns, factor_returns)
        
        if len(returns_aligned) < window:
            return pd.Series([], dtype=float)
            
        rolling_alphas = []
        for i in range(window, len(returns_aligned) + 1):
            window_returns = returns_aligned.iloc[i-window:i]
            window_factor = factor_aligned.iloc[i-window:i]
            
            alpha_val = self.alpha(window_returns, window_factor, risk_free, period, annualization)
            rolling_alphas.append(alpha_val)
        
        if isinstance(returns_aligned, pd.Series):
            return pd.Series(rolling_alphas, index=returns_aligned.index[window-1:])
        else:
            return pd.Series(rolling_alphas)
    
    def roll_beta(self, returns=None, factor_returns=None, window=252, risk_free=0.0, period=DAILY, annualization=None):
        """Calculates rolling beta over a specified window."""
        returns = self._get_returns(returns)
        factor_returns = self._get_factor_returns(factor_returns)
        
        returns_aligned, factor_aligned = self._aligned_series(returns, factor_returns)
        
        if len(returns_aligned) < window:
            return pd.Series([], dtype=float)
            
        rolling_betas = []
        for i in range(window, len(returns_aligned) + 1):
            window_returns = returns_aligned.iloc[i-window:i]
            window_factor = factor_aligned.iloc[i-window:i]
            
            beta_val = self.beta(window_returns, window_factor, risk_free, period, annualization)
            rolling_betas.append(beta_val)
        
        if isinstance(returns_aligned, pd.Series):
            return pd.Series(rolling_betas, index=returns_aligned.index[window-1:])
        else:
            return pd.Series(rolling_betas)
    
    def roll_alpha_beta(self, returns=None, factor_returns=None, window=252, risk_free=0.0, period=DAILY, annualization=None):
        """Calculates rolling alpha and beta over a specified window."""
        returns = self._get_returns(returns)
        factor_returns = self._get_factor_returns(factor_returns)
        
        # Align series
        returns_aligned, factor_aligned = self._aligned_series(returns, factor_returns)
        
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
                alpha_beta_result = self.cal_alpha_beta(window_returns, window_factor, risk_free, period, annualization)
                rolling_results.append(alpha_beta_result)
            except:
                rolling_results.append([np.nan, np.nan])
        
        # Convert to DataFrame or numpy array
        if isinstance(returns, np.ndarray) and isinstance(factor_returns, np.ndarray):
            return np.array(rolling_results)
        else:
            if hasattr(returns_aligned, 'index'):
                result_df = pd.DataFrame(rolling_results, columns=['alpha', 'beta'], 
                                       index=returns_aligned.index[window-1:])
            else:
                result_df = pd.DataFrame(rolling_results, columns=['alpha', 'beta'])
            return result_df
    
    def _get_all_drawdowns_detailed(self, returns):
        """Helper function to find all distinct drawdown periods with detailed information."""
        if len(returns) < 1:
            return []
        
        if not isinstance(returns, pd.Series):
            returns = pd.Series(returns)
        
        # Calculate cumulative returns
        cum_ret = self.cum_returns(returns, starting_value=100)
        
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

    def roll_sharpe_ratio(self, returns=None, window=252, risk_free=0.0, period=DAILY, annualization=None):
        """Calculates rolling Sharpe ratio over a specified window."""
        returns = self._get_returns(returns)
        
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
                sharpe = self.cal_sharpe_ratio(window_returns, risk_free, period, annualization)
            except:
                sharpe = np.nan
            rolling_sharpes.append(sharpe)
        
        if isinstance(returns, np.ndarray):
            return np.array(rolling_sharpes)
        else:
            return pd.Series(rolling_sharpes, index=returns.index[window-1:])
    
    def roll_max_drawdown(self, returns=None, window=252):
        """Calculates rolling maximum drawdown over a specified window."""
        returns = self._get_returns(returns)

        # Use the common roll helper so that ndarray/Series behaviour and
        # window semantics match other rolling helpers and the original
        # empyrical implementation.
        return roll(
            returns,
            window=window,
            function=self.cal_max_drawdown,
        )
    
    def roll_up_capture(self, returns=None, factor_returns=None, window=252):
        """Calculates rolling up capture ratio over a specified window."""
        returns = self._get_returns(returns)
        factor_returns = self._get_factor_returns(factor_returns)
        
        # Align series
        returns_aligned, factor_aligned = self._aligned_series(returns, factor_returns)
        
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
                up_cap = self.cal_up_capture(window_returns, window_factor)
            except:
                up_cap = np.nan
            rolling_up_capture.append(up_cap)
        
        if hasattr(returns_aligned, 'index'):
            result = pd.Series(rolling_up_capture, index=returns_aligned.index[window-1:])
        else:
            result = pd.Series(rolling_up_capture)
        
        # Convert to numpy array if input was numpy array to match expected return type
        if isinstance(returns, np.ndarray) and isinstance(factor_returns, np.ndarray):
            return result.values
        else:
            return result
    
    def roll_down_capture(self, returns=None, factor_returns=None, window=252):
        """Calculates rolling down capture ratio over a specified window."""
        returns = self._get_returns(returns)
        factor_returns = self._get_factor_returns(factor_returns)
        
        # Align series
        returns_aligned, factor_aligned = self._aligned_series(returns, factor_returns)
        
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
                down_cap = self.cal_down_capture(window_returns, window_factor)
            except:
                down_cap = np.nan
            rolling_down_capture.append(down_cap)
        
        if hasattr(returns_aligned, 'index'):
            result = pd.Series(rolling_down_capture, index=returns_aligned.index[window-1:])
        else:
            result = pd.Series(rolling_down_capture)
        
        # Convert to numpy array if input was numpy array to match expected return type
        if isinstance(returns, np.ndarray) and isinstance(factor_returns, np.ndarray):
            return result.values
        else:
            return result
    
    def roll_up_down_capture(self, returns=None, factor_returns=None, window=252):
        """Calculates rolling up/down capture ratio over a specified window."""
        returns = self._get_returns(returns)
        factor_returns = self._get_factor_returns(factor_returns)
        
        # Align series
        returns_aligned, factor_aligned = self._aligned_series(returns, factor_returns)
        
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
                up_down_cap = self.up_down_capture(window_returns, window_factor)
            except:
                up_down_cap = np.nan
            rolling_up_down_capture.append(up_down_cap)
        
        if hasattr(returns_aligned, 'index'):
            result = pd.Series(rolling_up_down_capture, index=returns_aligned.index[window-1:])
        else:
            result = pd.Series(rolling_up_down_capture)
        
        # Convert to numpy array if input was numpy array to match expected return type
        if isinstance(returns, np.ndarray) and isinstance(factor_returns, np.ndarray):
            return result.values
        else:
            return result
    
    def _get_all_drawdowns(self, returns):
        """Helper function to find all distinct drawdown periods and their values."""
        detailed = self._get_all_drawdowns_detailed(returns)
        return [dd['value'] for dd in detailed]
    
    def __repr__(self):
        """返回对象的字符串表示"""
        info = f"Empyrical(returns={len(self.returns) if self.returns is not None else None} data points"
        if self.benchmark_rets is not None:
            info += f", benchmark={len(self.benchmark_rets)} data points"
        info += ")"
        return info


# 为了向后兼容，在模块级别提供函数接口
# 这样pyfolio等代码可以直接使用 empyrical.annual_return() 等函数

# 常用统计函数的模块级别接口
annual_return = Empyrical.cal_annual_return
cum_returns_final = Empyrical.cal_cum_returns_final
annual_volatility = Empyrical.cal_annual_volatility
sharpe_ratio = Empyrical.cal_sharpe_ratio
calmar_ratio = Empyrical.cal_calmar_ratio
stability_of_timeseries = Empyrical.cal_stability_of_timeseries
max_drawdown = Empyrical.cal_max_drawdown
omega_ratio = Empyrical.cal_omega_ratio
sortino_ratio = Empyrical.cal_sortino_ratio
skewness = Empyrical.cal_skewness
kurtosis = Empyrical.cal_kurtosis
down_capture = Empyrical.cal_down_capture
up_capture = Empyrical.cal_up_capture
cum_returns = Empyrical.cal_cum_returns
value_at_risk = Empyrical.cal_value_at_risk
alpha_beta = Empyrical.cal_alpha_beta
alpha = Empyrical.cal_alpha
beta = Empyrical.cal_beta
treynor_ratio = Empyrical.cal_treynor_ratio
tail_ratio = Empyrical.cal_tail_ratio

# 导出所有函数和类
__all__ = [
    # 新的类
    'Empyrical',
    # 兼容性函数接口
    'annual_return', 'cum_returns_final', 'annual_volatility', 'sharpe_ratio',
    'calmar_ratio', 'stability_of_timeseries', 'max_drawdown', 'omega_ratio',
    'sortino_ratio', 'skewness', 'kurtosis', 'down_capture', 'up_capture',
    'cum_returns', 'value_at_risk', 'alpha_beta', 'alpha', 'beta', 'treynor_ratio',
    'tail_ratio'
]