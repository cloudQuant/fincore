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
Empyrical - 金融性能分析库.

包含原有的所有empyrical函数，以及新的面向对象Empyrical类。
代码已重构，将具体实现拆分到metrics模块中。
"""
import warnings
import functools
import pandas as pd
import numpy as np
from fincore.constants import *


class _dual_method:
    """Descriptor that allows a method to work both as a class-level call and instance call.

    When accessed on the class (Empyrical.method), behaves like a classmethod -
    passes the class as the first argument.
    When accessed on an instance (emp.method), passes the instance as the first argument,
    allowing access to instance attributes like self.returns.
    """

    def __init__(self, func):
        self.func = func
        functools.update_wrapper(self, func)

    def __set_name__(self, owner, name):
        self.__name__ = name

    def __get__(self, obj, objtype=None):
        if obj is None:
            attr_name = '_cls_bound_' + self.__name__
            try:
                return objtype.__dict__[attr_name]
            except KeyError:
                @functools.wraps(self.func)
                def wrapper(*args, **kwargs):
                    return self.func(objtype, *args, **kwargs)
                setattr(objtype, attr_name, wrapper)
                return wrapper
        else:
            attr_name = '_bound_' + self.__name__
            try:
                return obj.__dict__[attr_name]
            except KeyError:
                @functools.wraps(self.func)
                def wrapper(*args, **kwargs):
                    return self.func(obj, *args, **kwargs)
                obj.__dict__[attr_name] = wrapper
                return wrapper

# 从metrics模块导入子模块（使用_module别名避免命名冲突）
from fincore.metrics import basic_module as _basic
from fincore.metrics import returns_module as _returns
from fincore.metrics import drawdown_module as _drawdown
from fincore.metrics import risk_module as _risk
from fincore.metrics import ratios_module as _ratios
from fincore.metrics import alpha_beta_module as _alpha_beta
from fincore.metrics import stats_module as _stats
from fincore.metrics import consecutive_module as _consecutive
from fincore.metrics import rolling_module as _rolling
from fincore.metrics import bayesian_module as _bayesian
from fincore.metrics import positions_module as _positions
from fincore.metrics import transactions_module as _transactions
from fincore.metrics import round_trips_module as _round_trips
from fincore.metrics import perf_attrib_module as _perf_attrib
from fincore.metrics import perf_stats_module as _perf_stats
from fincore.metrics import timing_module as _timing
from fincore.metrics import yearly_module as _yearly

try:
    from zipline.assets import Equity, Future
    ZIPLINE = True
except ImportError:
    ZIPLINE = False
    class _ZiplineAssetStub:
        price_multiplier = 1
    class Equity(_ZiplineAssetStub):
        pass
    class Future(_ZiplineAssetStub):
        pass
    _ZIPLINE_WARNING = 'Module "zipline.assets" not found; multipliers will not be applied to position notionals.'


class Empyrical:
    """
    面向对象的性能指标计算类.

    这个类将所有empyrical模块的函数封装为类方法，提供统一的数据管理和计算接口。
    初始化参数与pyfolio的create_full_tear_sheet函数参数保持一致。
    代码已重构，具体实现委托给metrics子模块中的函数。
    """

    def __init__(self, returns=None, positions=None, factor_returns=None, factor_loadings=None, **kwargs):
        """初始化Empyrical实例，存储收益、持仓、因子收益和因子载荷数据."""
        self.returns = returns
        self.positions = positions
        self.factor_returns = factor_returns
        self.factor_loadings = factor_loadings
        self._ctx = None
        if returns is not None:
            try:
                from fincore.core.context import AnalysisContext
                self._ctx = AnalysisContext(
                    returns,
                    factor_returns=factor_returns,
                    positions=positions,
                )
            except Exception:
                pass

    @_dual_method
    def _get_returns(self, returns):
        """获取收益数据. 当returns为None时，尝试使用实例的returns属性."""
        if returns is not None:
            return returns
        if not isinstance(self, type) and hasattr(self, 'returns') and self.returns is not None:
            return self.returns
        return None

    @_dual_method
    def _get_factor_returns(self, factor_return):
        """获取因子收益数据. 当factor_return为None时，尝试使用实例的factor_returns属性."""
        if factor_return is not None:
            return factor_return
        if not isinstance(self, type) and hasattr(self, 'factor_returns') and self.factor_returns is not None:
            return self.factor_returns
        return None

    # ================================
    # 基础工具方法
    # ================================

    @staticmethod
    def _ensure_datetime_index_series(data, period=DAILY):
        """将数据转换为带有DatetimeIndex的Series."""
        return _basic.ensure_datetime_index_series(data, period)

    @staticmethod
    def _flatten(arr):
        """将pandas Series展平为NumPy数组."""
        return _basic.flatten(arr)

    @staticmethod
    def _adjust_returns(returns, adjustment_factor):
        """通过减去调整因子来调整收益."""
        return _basic.adjust_returns(returns, adjustment_factor)

    @staticmethod
    def annualization_factor(period, annualization):
        """返回给定周期的年化因子."""
        return _basic.annualization_factor(period, annualization)

    @staticmethod
    def _to_pandas(ob):
        """将数组转换为pandas对象."""
        return _basic.to_pandas(ob)

    @staticmethod
    def _aligned_series(*many_series):
        """返回索引对齐的序列元组."""
        return _basic.aligned_series(*many_series)

    # ================================
    # 收益计算方法
    # ================================

    @classmethod
    def simple_returns(cls, prices):
        """从价格时间序列计算简单收益."""
        return _returns.simple_returns(prices)

    @classmethod
    def cum_returns(cls, returns, starting_value=0, out=None):
        """从简单收益计算累积收益."""
        return _returns.cum_returns(returns, starting_value, out)

    @classmethod
    def cum_returns_final(cls, returns, starting_value=0):
        """计算简单收益序列的最终累积收益."""
        return _returns.cum_returns_final(returns, starting_value)

    @classmethod
    def aggregate_returns(cls, returns, convert_to="monthly"):
        """按周/月/季/年频率聚合收益."""
        return _returns.aggregate_returns(returns, convert_to)

    @classmethod
    def normalize(cls, returns, starting_value=1):
        """将累积收益标准化为从给定值开始."""
        return _returns.normalize(returns, starting_value)

    # ================================
    # 回撤相关方法
    # ================================

    @classmethod
    def max_drawdown(cls, returns, out=None):
        """计算收益序列的最大回撤."""
        return _drawdown.max_drawdown(returns, out)

    @classmethod
    def _get_all_drawdowns(cls, returns):
        """提取所有不同的回撤值."""
        return _drawdown.get_all_drawdowns(returns)

    @classmethod
    def _get_all_drawdowns_detailed(cls, returns):
        """提取所有回撤的详细信息."""
        return _drawdown.get_all_drawdowns_detailed(returns)

    @classmethod
    def get_max_drawdown(cls, returns):
        """计算最大回撤（max_drawdown的别名）."""
        return _drawdown.get_max_drawdown(returns)

    @classmethod
    def get_max_drawdown_underwater(cls, underwater):
        """确定给定水下收益的峰值、谷值和恢复日期."""
        return _drawdown.get_max_drawdown_underwater(underwater)

    @classmethod
    def get_top_drawdowns(cls, returns, top=10):
        """找到按严重程度排序的前几个回撤."""
        return _drawdown.get_top_drawdowns(returns, top)

    @classmethod
    def gen_drawdown_table(cls, returns, top=10):
        """生成前几个回撤的表格."""
        return _drawdown.gen_drawdown_table(returns, top)

    @_dual_method
    def get_max_drawdown_period(self, returns=None):
        """获取最大回撤期间的开始和结束日期."""
        return _drawdown.get_max_drawdown_period(self._get_returns(returns))

    @_dual_method
    def max_drawdown_days(self, returns=None):
        """计算最大回撤持续的天数."""
        return _drawdown.max_drawdown_days(self._get_returns(returns))

    @_dual_method
    def second_max_drawdown(self, returns=None):
        """确定策略的第二大回撤."""
        return _drawdown.second_max_drawdown(self._get_returns(returns))

    @_dual_method
    def third_max_drawdown(self, returns=None):
        """确定策略的第三大回撤."""
        return _drawdown.third_max_drawdown(self._get_returns(returns))

    # ================================
    # 风险指标方法
    # ================================

    @classmethod
    def annual_volatility(cls, returns, period=DAILY, alpha_=2.0, annualization=None, out=None):
        """计算收益序列的年化波动率."""
        return _risk.annual_volatility(returns, period, alpha_, annualization, out)

    @classmethod
    def downside_risk(cls, returns, required_return=0, period=DAILY, annualization=None, out=None):
        """计算低于阈值的年化下行偏差."""
        return _risk.downside_risk(returns, required_return, period, annualization, out)

    @classmethod
    def value_at_risk(cls, returns, cutoff=0.05):
        """计算风险价值(VaR)."""
        return _risk.value_at_risk(returns, cutoff)

    @classmethod
    def conditional_value_at_risk(cls, returns, cutoff=0.05):
        """计算条件风险价值(CVaR/ES)."""
        return _risk.conditional_value_at_risk(returns, cutoff)

    @classmethod
    def tail_ratio(cls, returns):
        """计算尾部比率."""
        return _risk.tail_ratio(returns)

    @classmethod
    def tracking_error(cls, returns, factor_returns, period=DAILY, annualization=None, out=None):
        """计算相对于基准的年化跟踪误差."""
        return _risk.tracking_error(returns, factor_returns, period, annualization, out)

    # ================================
    # 比率指标方法
    # ================================

    @classmethod
    def sharpe_ratio(cls, returns, risk_free=0, period=DAILY, annualization=None, out=None):
        """计算策略的年化夏普比率."""
        return _ratios.sharpe_ratio(returns, risk_free, period, annualization, out)

    @classmethod
    def sortino_ratio(cls, returns, required_return=0, period=DAILY, annualization=None, out=None, _downside_risk=None):
        """计算策略的索提诺比率."""
        return _ratios.sortino_ratio(returns, required_return, period, annualization, out, _downside_risk)

    @classmethod
    def excess_sharpe(cls, returns, factor_returns, out=None):
        """计算策略的超额夏普比率."""
        return _ratios.excess_sharpe(returns, factor_returns, out)

    @classmethod
    def calmar_ratio(cls, returns, period=DAILY, annualization=None):
        """计算卡尔玛比率（收益回撤比）."""
        return _ratios.calmar_ratio(returns, period, annualization)

    @classmethod
    def omega_ratio(cls, returns, risk_free=0.0, required_return=0.0, annualization=APPROX_BDAYS_PER_YEAR):
        """计算策略的欧米伽比率."""
        return _ratios.omega_ratio(returns, risk_free, required_return, annualization)

    @classmethod
    def information_ratio(cls, returns, factor_returns, period=DAILY, annualization=None):
        """计算相对于基准的信息比率."""
        return _ratios.information_ratio(returns, factor_returns, period, annualization)

    @classmethod
    def stability_of_timeseries(cls, returns):
        """计算时间序列的稳定性（R平方）."""
        return _ratios.stability_of_timeseries(returns)

    @classmethod
    def capture(cls, returns, factor_returns, period=DAILY):
        """计算捕获比率."""
        return _ratios.capture(returns, factor_returns, period)

    @classmethod
    def up_capture(cls, returns, factor_returns, period=DAILY):
        """计算上行捕获比率."""
        return _ratios.up_capture(returns, factor_returns, period)

    @classmethod
    def down_capture(cls, returns, factor_returns, period=DAILY):
        """计算下行捕获比率."""
        return _ratios.down_capture(returns, factor_returns, period)

    @classmethod
    def up_down_capture(cls, returns, factor_returns, period=DAILY):
        """计算上行/下行捕获比率."""
        return _ratios.up_down_capture(returns, factor_returns, period)

    @classmethod
    def adjusted_sharpe_ratio(cls, returns, risk_free=0.0):
        """计算调整后的夏普比率（考虑偏度和峰度）."""
        return _ratios.adjusted_sharpe_ratio(returns, risk_free)

    @classmethod
    def conditional_sharpe_ratio(cls, returns, cutoff=0.05):
        """计算左尾条件下的夏普比率."""
        return _ratios.conditional_sharpe_ratio(returns, cutoff)

    # ================================
    # 阿尔法贝塔方法
    # ================================

    @classmethod
    def alpha(cls, returns, factor_returns, risk_free=0.0, period=DAILY, annualization=None, out=None, _beta=None):
        """计算相对于基准的年化阿尔法."""
        return _alpha_beta.alpha(returns, factor_returns, risk_free, period, annualization, out, _beta)

    @classmethod
    def alpha_aligned(cls, returns, factor_returns, risk_free=0.0, period=DAILY, annualization=None, out=None, _beta=None):
        """计算已对齐序列的年化阿尔法."""
        return _alpha_beta.alpha_aligned(returns, factor_returns, risk_free, period, annualization, out, _beta)

    @classmethod
    def beta(cls, returns, factor_returns, risk_free=0.0, _period=DAILY, _annualization=None, out=None):
        """计算相对于基准的贝塔."""
        return _alpha_beta.beta(returns, factor_returns, risk_free, _period, _annualization, out)

    @classmethod
    def beta_aligned(cls, returns, factor_returns, risk_free=0.0, out=None):
        """计算已对齐数据的贝塔."""
        return _alpha_beta.beta_aligned(returns, factor_returns, risk_free, out)

    @classmethod
    def alpha_beta(cls, returns, factor_returns, risk_free=0.0, period=DAILY, annualization=None, out=None):
        """计算相对于基准的年化阿尔法和贝塔."""
        return _alpha_beta.alpha_beta(returns, factor_returns, risk_free, period, annualization, out)

    @classmethod
    def alpha_beta_aligned(cls, returns, factor_returns, risk_free=0.0, period=DAILY, annualization=None, out=None):
        """计算已对齐序列的年化阿尔法和贝塔."""
        return _alpha_beta.alpha_beta_aligned(returns, factor_returns, risk_free, period, annualization, out)

    @classmethod
    def up_alpha_beta(cls, returns, factor_returns, risk_free=0.0, period=DAILY, annualization=None, out=None):
        """计算上行市场的阿尔法和贝塔."""
        return _alpha_beta.up_alpha_beta(returns, factor_returns, risk_free, period, annualization, out)

    @classmethod
    def down_alpha_beta(cls, returns, factor_returns, risk_free=0.0, period=DAILY, annualization=None, out=None):
        """计算下行市场的阿尔法和贝塔."""
        return _alpha_beta.down_alpha_beta(returns, factor_returns, risk_free, period, annualization, out)

    @classmethod
    def annual_alpha(cls, returns, factor_returns, risk_free=0.0, period=DAILY, annualization=None):
        """计算每个日历年的年化阿尔法."""
        return _alpha_beta.annual_alpha(returns, factor_returns, risk_free, period, annualization)

    @classmethod
    def annual_beta(cls, returns, factor_returns, risk_free=0.0, period=DAILY, annualization=None):
        """计算每个日历年的年化贝塔."""
        return _alpha_beta.annual_beta(returns, factor_returns, risk_free, period, annualization)

    # ================================
    # 统计方法
    # ================================

    @classmethod
    def skewness(cls, returns):
        """计算收益分布的偏度."""
        return _stats.skewness(returns)

    @classmethod
    def kurtosis(cls, returns):
        """计算收益分布的峰度."""
        return _stats.kurtosis(returns)

    @classmethod
    def hurst_exponent(cls, returns):
        """计算赫斯特指数."""
        return _stats.hurst_exponent(returns)

    @classmethod
    def stutzer_index(cls, returns, target_return=0.0):
        """计算斯图泽指数."""
        return _stats.stutzer_index(returns, target_return)

    @classmethod
    def stock_market_correlation(cls, returns, market_returns):
        """计算与股票市场的相关性."""
        return _stats.stock_market_correlation(returns, market_returns)

    @classmethod
    def bond_market_correlation(cls, returns, bond_returns):
        """计算与债券市场的相关性."""
        return _stats.bond_market_correlation(returns, bond_returns)

    @_dual_method
    def futures_market_correlation(self, returns=None, futures_returns=None):
        """计算与期货市场的相关性."""
        return _stats.futures_market_correlation(self._get_returns(returns), futures_returns)

    @_dual_method
    def serial_correlation(self, returns=None, lag=1):
        """计算序列自相关."""
        return _stats.serial_correlation(self._get_returns(returns), lag)

    @_dual_method
    def win_rate(self, returns=None):
        """计算胜率."""
        return _stats.win_rate(self._get_returns(returns))

    @_dual_method
    def loss_rate(self, returns=None):
        """计算亏损率."""
        return _stats.loss_rate(self._get_returns(returns))

    # ================================
    # 连续涨跌方法
    # ================================

    @classmethod
    def max_consecutive_up_days(cls, returns):
        """计算最大连续上涨天数."""
        return _consecutive.max_consecutive_up_days(returns)

    @classmethod
    def max_consecutive_down_days(cls, returns):
        """计算最大连续下跌天数."""
        return _consecutive.max_consecutive_down_days(returns)

    @classmethod
    def max_consecutive_gain(cls, returns):
        """计算最大连续上涨期间的累计收益."""
        return _consecutive.max_consecutive_gain(returns)

    @classmethod
    def max_consecutive_loss(cls, returns):
        """计算最大连续下跌期间的累计亏损."""
        return _consecutive.max_consecutive_loss(returns)

    @classmethod
    def max_single_day_gain(cls, returns):
        """计算单日最大收益."""
        return _consecutive.max_single_day_gain(returns)

    @classmethod
    def max_single_day_loss(cls, returns):
        """计算单日最大亏损."""
        return _consecutive.max_single_day_loss(returns)

    @_dual_method
    def max_consecutive_up_weeks(self, returns=None):
        """计算最大连续上涨周数."""
        return _consecutive.max_consecutive_up_weeks(self._get_returns(returns))

    @_dual_method
    def max_consecutive_down_weeks(self, returns=None):
        """计算最大连续下跌周数."""
        return _consecutive.max_consecutive_down_weeks(self._get_returns(returns))

    @_dual_method
    def max_consecutive_up_months(self, returns=None):
        """计算最大连续上涨月数."""
        return _consecutive.max_consecutive_up_months(self._get_returns(returns))

    @_dual_method
    def max_consecutive_down_months(self, returns=None):
        """计算最大连续下跌月数."""
        return _consecutive.max_consecutive_down_months(self._get_returns(returns))

    @_dual_method
    def max_single_day_gain_date(self, returns=None):
        """获取单日最大收益的日期."""
        return _consecutive.max_single_day_gain_date(self._get_returns(returns))

    @_dual_method
    def max_single_day_loss_date(self, returns=None):
        """获取单日最大亏损的日期."""
        return _consecutive.max_single_day_loss_date(self._get_returns(returns))

    @_dual_method
    def max_consecutive_up_start_date(self, returns=None):
        """获取最大连续上涨期的开始日期."""
        return _consecutive.max_consecutive_up_start_date(self._get_returns(returns))

    @_dual_method
    def max_consecutive_up_end_date(self, returns=None):
        """获取最大连续上涨期的结束日期."""
        return _consecutive.max_consecutive_up_end_date(self._get_returns(returns))

    @_dual_method
    def max_consecutive_down_start_date(self, returns=None):
        """获取最大连续下跌期的开始日期."""
        return _consecutive.max_consecutive_down_start_date(self._get_returns(returns))

    @_dual_method
    def max_consecutive_down_end_date(self, returns=None):
        """获取最大连续下跌期的结束日期."""
        return _consecutive.max_consecutive_down_end_date(self._get_returns(returns))

    # ================================
    # 市场时机方法
    # ================================

    @classmethod
    def treynor_mazuy_timing(cls, returns, factor_returns, risk_free=0.0):
        """计算特雷诺-马祖伊市场择时系数."""
        return _timing.treynor_mazuy_timing(returns, factor_returns, risk_free)

    @classmethod
    def henriksson_merton_timing(cls, returns, factor_returns, risk_free=0.0):
        """计算亨里克森-默顿市场择时系数."""
        return _timing.henriksson_merton_timing(returns, factor_returns, risk_free)

    @classmethod
    def market_timing_return(cls, returns, factor_returns, risk_free=0.0):
        """计算市场择时收益成分."""
        return _timing.market_timing_return(returns, factor_returns, risk_free)

    @classmethod
    def cornell_timing(cls, returns, factor_returns, risk_free=0.0):
        """计算康奈尔择时模型系数."""
        return _timing.cornell_timing(returns, factor_returns, risk_free)

    # ================================
    # 按年统计方法
    # ================================

    @classmethod
    def annual_return(cls, returns, period=DAILY, annualization=None):
        """计算年化收益率(CAGR)."""
        return _yearly.annual_return(returns, period, annualization)

    @classmethod
    def cagr(cls, returns, period=DAILY, annualization=None):
        """计算复合年增长率(annual_return的别名)."""
        return _yearly.annual_return(returns, period, annualization)

    @classmethod
    def annual_return_by_year(cls, returns, period=DAILY, annualization=None):
        """计算每个日历年的年化收益."""
        return _yearly.annual_return_by_year(returns, period, annualization)

    @classmethod
    def sharpe_ratio_by_year(cls, returns, risk_free=0, period=DAILY, annualization=None):
        """计算每个日历年的夏普比率."""
        return _yearly.sharpe_ratio_by_year(returns, risk_free, period, annualization)

    @classmethod
    def max_drawdown_by_year(cls, returns):
        """计算每个日历年的最大回撤."""
        return _yearly.max_drawdown_by_year(returns)

    # ================================
    # 滚动计算方法
    # ================================

    @classmethod
    def rolling_volatility(cls, returns, rolling_vol_window):
        """计算滚动波动率."""
        return _rolling.rolling_volatility(returns, rolling_vol_window)

    @classmethod
    def rolling_sharpe(cls, returns, rolling_sharpe_window):
        """计算滚动夏普比率."""
        return _rolling.rolling_sharpe(returns, rolling_sharpe_window)

    @classmethod
    def rolling_beta(cls, returns, factor_returns, rolling_window=126):
        """计算滚动贝塔."""
        return _rolling.rolling_beta(returns, factor_returns, rolling_window)

    # ================================
    # 持仓分析方法
    # ================================

    @classmethod
    def get_percent_alloc(cls, values):
        """获取值的百分比分配."""
        return _positions.get_percent_alloc(values)

    @classmethod
    def get_top_long_short_abs(cls, positions, top=10):
        """获取按绝对值排序的前几个多头和空头持仓."""
        return _positions.get_top_long_short_abs(positions, top)

    @classmethod
    def get_long_short_pos(cls, positions):
        """获取多头和空头持仓."""
        return _positions.get_long_short_pos(positions)

    @classmethod
    def gross_lev(cls, positions):
        """计算总杠杆."""
        return _positions.gross_lev(positions)

    # ================================
    # 交易分析方法
    # ================================

    @classmethod
    def get_txn_vol(cls, transactions):
        """获取交易量."""
        return _transactions.get_txn_vol(transactions)

    @classmethod
    def get_turnover(cls, positions, transactions, denominator="AGB"):
        """获取投资组合换手率."""
        return _transactions.get_turnover(positions, transactions, denominator)

    @classmethod
    def make_transaction_frame(cls, transactions):
        """创建交易DataFrame."""
        return _transactions.make_transaction_frame(transactions)

    # ================================
    # 往返交易方法
    # ================================

    @classmethod
    def extract_round_trips(cls, transactions, portfolio_value=None):
        """从交易中提取往返交易."""
        return _round_trips.extract_round_trips(transactions, portfolio_value)

    @classmethod
    def gen_round_trip_stats(cls, round_trips):
        """生成往返交易统计."""
        return _round_trips.gen_round_trip_stats(round_trips)

    # ================================
    # 绩效归因方法
    # ================================

    @_dual_method
    def perf_attrib(self, returns=None, positions=None, factor_returns=None, factor_loadings=None, transactions=None, pos_in_dollars=True, regression_style='OLS'):
        """计算绩效归因."""
        returns = self._get_returns(returns)
        if not isinstance(self, type):
            if positions is None and hasattr(self, 'positions') and self.positions is not None:
                positions = self.positions
            if factor_returns is None and hasattr(self, 'factor_returns') and self.factor_returns is not None:
                factor_returns = self.factor_returns
            if factor_loadings is None and hasattr(self, 'factor_loadings') and self.factor_loadings is not None:
                factor_loadings = self.factor_loadings
        return _perf_attrib.perf_attrib(returns, positions, factor_returns, factor_loadings, transactions, pos_in_dollars, regression_style)

    @classmethod
    def compute_exposures(cls, positions=None, factor_loadings=None):
        """从持仓计算因子敞口."""
        return _perf_attrib.compute_exposures(positions, factor_loadings)

    # ================================
    # 绩效统计方法
    # ================================

    @classmethod
    def perf_stats(cls, returns, factor_returns=None, positions=None, transactions=None, turnover_denom='AGB', period=DAILY):
        """计算绩效统计."""
        return _perf_stats.perf_stats(returns, factor_returns, positions, transactions, turnover_denom, period)

    @classmethod
    def calc_bootstrap(cls, func, returns, *args, **kwargs):
        """计算统计量的自助法分布."""
        return _perf_stats.calc_bootstrap(func, returns, *args, **kwargs)

    @classmethod
    def perf_stats_bootstrap(cls, returns, factor_returns=None, return_stats=True, **kwargs):
        """计算带自助法置信区间的绩效统计."""
        return _perf_stats.perf_stats_bootstrap(returns, factor_returns, return_stats, **kwargs)

    @classmethod
    def calc_distribution_stats(cls, x):
        """计算分布统计量."""
        return _perf_stats.calc_distribution_stats(x)

    # ================================
    # 贝叶斯模型方法
    # ================================

    @classmethod
    def model_returns_t_alpha_beta(cls, data, bmark, samples=2000, progressbar=True):
        """运行带阿尔法贝塔的t分布贝叶斯模型."""
        return _bayesian.model_returns_t_alpha_beta(data, bmark, samples, progressbar)

    @classmethod
    def model_returns_normal(cls, data, samples=500, progressbar=True):
        """运行正态分布贝叶斯模型."""
        return _bayesian.model_returns_normal(data, samples, progressbar)

    @classmethod
    def model_returns_t(cls, data, samples=500, progressbar=True):
        """运行t分布贝叶斯模型."""
        return _bayesian.model_returns_t(data, samples, progressbar)

    @classmethod
    def model_best(cls, y1, y2, samples=1000, progressbar=True):
        """贝叶斯估计替代t检验."""
        return _bayesian.model_best(y1, y2, samples, progressbar)

    @classmethod
    def model_stoch_vol(cls, data, samples=2000, progressbar=True):
        """运行随机波动率模型."""
        return _bayesian.model_stoch_vol(data, samples, progressbar)

    @classmethod
    def compute_bayes_cone(cls, preds, starting_value=1.0):
        """计算贝叶斯预测锥."""
        return _bayesian.compute_bayes_cone(preds, starting_value)

    @classmethod
    def compute_consistency_score(cls, returns_test, preds):
        """计算一致性得分."""
        return _bayesian.compute_consistency_score(returns_test, preds)

    @classmethod
    def run_model(cls, model, returns_train, returns_test=None, bmark=None, samples=500, ppc=False, progressbar=True):
        """运行贝叶斯模型."""
        return _bayesian.run_model(model, returns_train, returns_test, bmark, samples, ppc, progressbar)

    @classmethod
    def simulate_paths(cls, is_returns, num_days, starting_value=1, num_samples=1000, random_seed=None):
        """模拟蒙特卡洛路径."""
        return _bayesian.simulate_paths(is_returns, num_days, starting_value, num_samples, random_seed)

    @classmethod
    def summarize_paths(cls, samples, cone_std=(1.0, 1.5, 2.0), starting_value=1.0):
        """汇总路径统计."""
        return _bayesian.summarize_paths(samples, cone_std, starting_value)

    @classmethod
    def forecast_cone_bootstrap(cls, is_returns, num_days, cone_std=(1.0, 1.5, 2.0), starting_value=1, num_samples=1000, random_seed=None):
        """自助法预测锥."""
        return _bayesian.forecast_cone_bootstrap(is_returns, num_days, cone_std, starting_value, num_samples, random_seed)

    # ================================
    # 统计方法
    # ================================

    @_dual_method
    def r_cubed(self, returns=None, factor_returns=None):
        """计算R³指标."""
        returns = self._get_returns(returns)
        factor_returns = self._get_factor_returns(factor_returns)
        return _stats.r_cubed(returns, factor_returns)

    @_dual_method
    def tracking_difference(self, returns=None, factor_returns=None):
        """计算跟踪差异."""
        returns = self._get_returns(returns)
        factor_returns = self._get_factor_returns(factor_returns)
        return _stats.tracking_difference(returns, factor_returns)

    @_dual_method
    def common_sense_ratio(self, returns=None):
        """计算常识比率."""
        returns = self._get_returns(returns)
        return _ratios.common_sense_ratio(returns)

    @classmethod
    def var_cov_var_normal(cls, p, c, mu=0, sigma=1):
        """计算参数化VaR."""
        return _stats.var_cov_var_normal(p, c, mu, sigma)

    # ================================
    # 风险方法
    # ================================

    @_dual_method
    def gpd_risk_estimates(self, returns=None, var_p=0.01):
        """使用GPD估计VaR和ES."""
        returns = self._get_returns(returns)
        return _risk.gpd_risk_estimates(returns, var_p)

    @_dual_method
    def gpd_risk_estimates_aligned(self, returns=None, var_p=0.01):
        """使用GPD估计VaR和ES（对齐版本）."""
        returns = self._get_returns(returns)
        return _risk.gpd_risk_estimates_aligned(returns, var_p)

    @_dual_method
    def beta_fragility_heuristic(self, returns=None, factor_returns=None):
        """估计Beta脆弱性."""
        returns = self._get_returns(returns)
        factor_returns = self._get_factor_returns(factor_returns)
        return _risk.beta_fragility_heuristic(returns, factor_returns)

    @_dual_method
    def beta_fragility_heuristic_aligned(self, returns=None, factor_returns=None):
        """估计Beta脆弱性（对齐版本）."""
        returns = self._get_returns(returns)
        factor_returns = self._get_factor_returns(factor_returns)
        return _risk.beta_fragility_heuristic_aligned(returns, factor_returns)

    @classmethod
    def trading_value_at_risk(cls, returns, period=None, sigma=2.0):
        """计算交易VaR."""
        return _risk.trading_value_at_risk(returns, period, sigma)

    # ================================
    # 滚动方法
    # ================================

    @_dual_method
    def roll_alpha(self, returns=None, factor_returns=None, window=252, risk_free=0.0, period=DAILY, annualization=None):
        """计算滚动Alpha."""
        returns = self._get_returns(returns)
        factor_returns = self._get_factor_returns(factor_returns)
        return _rolling.roll_alpha(returns, factor_returns, window, risk_free, period, annualization)

    @_dual_method
    def roll_beta(self, returns=None, factor_returns=None, window=252, risk_free=0.0, period=DAILY, annualization=None):
        """计算滚动Beta."""
        returns = self._get_returns(returns)
        factor_returns = self._get_factor_returns(factor_returns)
        return _rolling.roll_beta(returns, factor_returns, window, risk_free, period, annualization)

    @_dual_method
    def roll_alpha_beta(self, returns=None, factor_returns=None, window=252, risk_free=0.0, period=DAILY, annualization=None):
        """计算滚动Alpha和Beta."""
        returns = self._get_returns(returns)
        factor_returns = self._get_factor_returns(factor_returns)
        return _rolling.roll_alpha_beta(returns, factor_returns, window, risk_free, period, annualization)

    @_dual_method
    def roll_sharpe_ratio(self, returns=None, window=252, risk_free=0.0, period=DAILY, annualization=None):
        """计算滚动夏普比率."""
        returns = self._get_returns(returns)
        return _rolling.roll_sharpe_ratio(returns, window, risk_free, period, annualization)

    @_dual_method
    def roll_max_drawdown(self, returns=None, window=252):
        """计算滚动最大回撤."""
        returns = self._get_returns(returns)
        return _rolling.roll_max_drawdown(returns, window)

    @_dual_method
    def roll_up_capture(self, returns=None, factor_returns=None, window=252):
        """计算滚动上行捕获率."""
        returns = self._get_returns(returns)
        factor_returns = self._get_factor_returns(factor_returns)
        return _rolling.roll_up_capture(returns, factor_returns, window)

    @_dual_method
    def roll_down_capture(self, returns=None, factor_returns=None, window=252):
        """计算滚动下行捕获率."""
        returns = self._get_returns(returns)
        factor_returns = self._get_factor_returns(factor_returns)
        return _rolling.roll_down_capture(returns, factor_returns, window)

    @_dual_method
    def roll_up_down_capture(self, returns=None, factor_returns=None, window=252):
        """计算滚动上下行捕获比率."""
        returns = self._get_returns(returns)
        factor_returns = self._get_factor_returns(factor_returns)
        return _rolling.roll_up_down_capture(returns, factor_returns, window)

    @classmethod
    def rolling_regression(cls, returns, factor_returns, rolling_window=126):
        """计算滚动回归."""
        return _rolling.rolling_regression(returns, factor_returns, rolling_window)

    # ================================
    # 年度方法
    # ================================

    @_dual_method
    def annual_active_return(self, returns=None, factor_returns=None, period=DAILY, annualization=None):
        """计算年度主动收益."""
        returns = self._get_returns(returns)
        factor_returns = self._get_factor_returns(factor_returns)
        return _yearly.annual_active_return(returns, factor_returns, period, annualization)

    @_dual_method
    def annual_active_risk(self, returns=None, factor_returns=None, period=DAILY, annualization=None):
        """计算年度主动风险（跟踪误差）."""
        returns = self._get_returns(returns)
        factor_returns = self._get_factor_returns(factor_returns)
        return _risk.tracking_error(returns, factor_returns, period, annualization)

    @_dual_method
    def annual_active_return_by_year(self, returns=None, factor_returns=None, period=DAILY, annualization=None):
        """计算每年的主动收益."""
        returns = self._get_returns(returns)
        factor_returns = self._get_factor_returns(factor_returns)
        return _yearly.annual_active_return_by_year(returns, factor_returns, period, annualization)

    @_dual_method
    def information_ratio_by_year(self, returns=None, factor_returns=None, period=DAILY, annualization=None):
        """计算每年的信息比率."""
        returns = self._get_returns(returns)
        factor_returns = self._get_factor_returns(factor_returns)
        return _yearly.information_ratio_by_year(returns, factor_returns, period, annualization)

    @_dual_method
    def annual_volatility_by_year(self, returns=None, period=DAILY, annualization=None):
        """计算每年的波动率."""
        returns = self._get_returns(returns)
        return _yearly.annual_volatility_by_year(returns, period, annualization)

    @_dual_method
    def regression_annual_return(self, returns=None, factor_returns=None, risk_free=0.0, period=DAILY, annualization=None):
        """计算回归年化收益."""
        returns = self._get_returns(returns)
        factor_returns = self._get_factor_returns(factor_returns)
        alpha_val = _alpha_beta.alpha(returns, factor_returns, risk_free, period, annualization)
        beta_val = _alpha_beta.beta(returns, factor_returns, risk_free, period, annualization)
        if np.isnan(alpha_val) or np.isnan(beta_val):
            return np.nan
        benchmark_annual = _yearly.annual_return(factor_returns, period, annualization)
        if np.isnan(benchmark_annual):
            return np.nan
        return alpha_val + beta_val * benchmark_annual

    @_dual_method
    def annualized_cumulative_return(self, returns=None, period=DAILY, annualization=None):
        """计算年化累计收益."""
        returns = self._get_returns(returns)
        return _yearly.annual_return(returns, period, annualization)

    # ================================
    # 回撤方法
    # ================================

    @_dual_method
    def second_max_drawdown_days(self, returns=None):
        """获取第二大回撤的天数."""
        returns = self._get_returns(returns)
        return _drawdown.second_max_drawdown_days(returns)

    @_dual_method
    def second_max_drawdown_recovery_days(self, returns=None):
        """获取第二大回撤的恢复天数."""
        returns = self._get_returns(returns)
        return _drawdown.second_max_drawdown_recovery_days(returns)

    @_dual_method
    def third_max_drawdown_days(self, returns=None):
        """获取第三大回撤的天数."""
        returns = self._get_returns(returns)
        return _drawdown.third_max_drawdown_days(returns)

    @_dual_method
    def third_max_drawdown_recovery_days(self, returns=None):
        """获取第三大回撤的恢复天数."""
        returns = self._get_returns(returns)
        return _drawdown.third_max_drawdown_recovery_days(returns)

    # ================================
    # 持仓方法
    # ================================

    @classmethod
    def get_max_median_position_concentration(cls, positions):
        """获取最大和中位持仓集中度."""
        return _positions.get_max_median_position_concentration(positions)

    @classmethod
    def extract_pos(cls, positions, cash):
        """提取持仓."""
        return _positions.extract_pos(positions, cash)

    @classmethod
    def get_sector_exposures(cls, positions, symbol_sector_map):
        """获取行业敞口."""
        return _positions.get_sector_exposures(positions, symbol_sector_map)

    @classmethod
    def compute_style_factor_exposures(cls, positions, risk_factor):
        """计算风格因子敞口."""
        return _positions.compute_style_factor_exposures(positions, risk_factor)

    @classmethod
    def compute_sector_exposures(cls, positions, sectors, sector_dict=None):
        """计算行业敞口."""
        return _positions.compute_sector_exposures(positions, sectors, sector_dict)

    @classmethod
    def compute_cap_exposures(cls, positions, caps):
        """计算市值敞口."""
        return _positions.compute_cap_exposures(positions, caps)

    @classmethod
    def compute_volume_exposures(cls, shares_held, volumes, percentile):
        """计算成交量敞口."""
        return _positions.compute_volume_exposures(shares_held, volumes, percentile)

    @classmethod
    def stack_positions(cls, positions, pos_in_dollars=True):
        """堆叠持仓."""
        return _positions.stack_positions(positions, pos_in_dollars)

    # ================================
    # 交易方法
    # ================================

    @classmethod
    def daily_txns_with_bar_data(cls, transactions, market_data):
        """增强交易数据."""
        return _transactions.daily_txns_with_bar_data(transactions, market_data)

    @classmethod
    def days_to_liquidate_positions(cls, positions, market_data, max_bar_consumption=0.2, capital_base=1e6, mean_volume_window=5):
        """计算清仓天数."""
        return _transactions.days_to_liquidate_positions(positions, market_data, max_bar_consumption, capital_base, mean_volume_window)

    @classmethod
    def get_max_days_to_liquidate_by_ticker(cls, positions, market_data, max_bar_consumption=0.2, capital_base=1e6, mean_volume_window=5, last_n_days=None):
        """按标的获取最大清仓天数."""
        return _transactions.get_max_days_to_liquidate_by_ticker(positions, market_data, max_bar_consumption, capital_base, mean_volume_window, last_n_days)

    @classmethod
    def get_low_liquidity_transactions(cls, transactions, market_data, last_n_days=None):
        """获取低流动性交易."""
        return _transactions.get_low_liquidity_transactions(transactions, market_data, last_n_days)

    @classmethod
    def apply_slippage_penalty(cls, returns, txn_daily, simulate_starting_capital, backtest_starting_capital, impact=0.1):
        """应用滑点惩罚."""
        return _transactions.apply_slippage_penalty(returns, txn_daily, simulate_starting_capital, backtest_starting_capital, impact)

    @classmethod
    def map_transaction(cls, txn):
        """映射交易."""
        return _transactions.map_transaction(txn)

    @classmethod
    def adjust_returns_for_slippage(cls, returns, positions, transactions, slippage_bps=10):
        """调整滑点后的收益."""
        return _transactions.adjust_returns_for_slippage(returns, positions, transactions, slippage_bps)

    # ================================
    # 往返交易方法
    # ================================

    @classmethod
    def agg_all_long_short(cls, round_trips, col, stats_dict):
        """聚合多空往返交易统计."""
        return _round_trips.agg_all_long_short(round_trips, col, stats_dict)

    @classmethod
    def add_closing_transactions(cls, positions, transactions):
        """添加平仓交易."""
        return _round_trips.add_closing_transactions(positions, transactions)

    @classmethod
    def apply_sector_mappings_to_round_trips(cls, round_trips, sector_mappings):
        """应用行业映射到往返交易."""
        return _round_trips.apply_sector_mappings_to_round_trips(round_trips, sector_mappings)

    @classmethod
    def create_perf_attrib_stats(cls, perf_attrib_, risk_exposures):
        """创建绩效归因统计."""
        return _perf_attrib.create_perf_attrib_stats(perf_attrib_, risk_exposures)

    # ================================
    # 其他方法
    # ================================

    @_dual_method
    def extract_interesting_date_ranges(self, returns=None):
        """提取有趣的日期范围."""
        returns = self._get_returns(returns)
        return _timing.extract_interesting_date_ranges(returns)

    # ================================
    # 额外比率方法
    # ================================

    @_dual_method
    def treynor_ratio(self, returns=None, factor_returns=None, risk_free=0.0, period=DAILY, annualization=None):
        """计算特雷诺比率."""
        returns = self._get_returns(returns)
        factor_returns = self._get_factor_returns(factor_returns)
        return _ratios.treynor_ratio(returns, factor_returns, risk_free, period, annualization)

    @_dual_method
    def m_squared(self, returns=None, factor_returns=None, risk_free=0.0, period=DAILY, annualization=None):
        """计算M²测度."""
        returns = self._get_returns(returns)
        factor_returns = self._get_factor_returns(factor_returns)
        return _ratios.m_squared(returns, factor_returns, risk_free, period, annualization)

    @_dual_method
    def sterling_ratio(self, returns=None, risk_free=0.0, period=DAILY, annualization=None):
        """计算斯特林比率."""
        returns = self._get_returns(returns)
        return _ratios.sterling_ratio(returns, risk_free, period, annualization)

    @_dual_method
    def burke_ratio(self, returns=None, risk_free=0.0, period=DAILY, annualization=None):
        """计算伯克比率."""
        returns = self._get_returns(returns)
        return _ratios.burke_ratio(returns, risk_free, period, annualization)

    @_dual_method
    def kappa_three_ratio(self, returns=None, risk_free=0.0, period=DAILY, annualization=None, mar=0.0):
        """计算Kappa3比率."""
        returns = self._get_returns(returns)
        return _ratios.kappa_three_ratio(returns, risk_free, period, annualization, mar)

    # ================================
    # 额外风险方法
    # ================================

    @_dual_method
    def residual_risk(self, returns=None, factor_returns=None, risk_free=0.0, period=DAILY, annualization=None):
        """计算残差风险."""
        returns = self._get_returns(returns)
        factor_returns = self._get_factor_returns(factor_returns)
        return _risk.residual_risk(returns, factor_returns, risk_free, period, annualization)

    @_dual_method
    def var_excess_return(self, returns=None, cutoff=0.05):
        """计算VaR超额收益."""
        returns = self._get_returns(returns)
        return _risk.var_excess_return(returns, cutoff)

    # ================================
    # 额外回撤方法
    # ================================

    @_dual_method
    def max_drawdown_weeks(self, returns=None):
        """计算最大回撤周数."""
        returns = self._get_returns(returns)
        return _drawdown.max_drawdown_weeks(returns)

    @_dual_method
    def max_drawdown_months(self, returns=None):
        """计算最大回撤月数."""
        returns = self._get_returns(returns)
        return _drawdown.max_drawdown_months(returns)

    @_dual_method
    def max_drawdown_recovery_days(self, returns=None):
        """计算最大回撤恢复天数."""
        returns = self._get_returns(returns)
        return _drawdown.max_drawdown_recovery_days(returns)

    @_dual_method
    def max_drawdown_recovery_weeks(self, returns=None):
        """计算最大回撤恢复周数."""
        returns = self._get_returns(returns)
        return _drawdown.max_drawdown_recovery_weeks(returns)

    @_dual_method
    def max_drawdown_recovery_months(self, returns=None):
        """计算最大回撤恢复月数."""
        returns = self._get_returns(returns)
        return _drawdown.max_drawdown_recovery_months(returns)

    # ================================
    # 额外阿尔法贝塔方法
    # ================================

    @classmethod
    def alpha_percentile_rank(cls, strategy_returns, all_strategies_returns, factor_returns,
                              risk_free=0.0, period=DAILY, annualization=None):
        """计算阿尔法百分位排名."""
        return _alpha_beta.alpha_percentile_rank(strategy_returns, all_strategies_returns, factor_returns,
                                                  risk_free, period, annualization)

    # ================================
    # 额外往返交易方法
    # ================================

    @classmethod
    def _groupby_consecutive(cls, txn, max_delta=None):
        """按连续交易分组."""
        if max_delta is None:
            max_delta = pd.Timedelta("8h")
        return _round_trips.groupby_consecutive(txn, max_delta)

    # ================================
    # 额外绩效归因方法
    # ================================

    @classmethod
    def _cumulative_returns_less_costs(cls, returns, costs):
        """计算扣除成本后的累积收益."""
        return _perf_attrib.cumulative_returns_less_costs(returns, costs)
