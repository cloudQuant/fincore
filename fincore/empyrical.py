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

代码已重构：

* 具体实现拆分到 ``fincore.metrics`` 子模块。
* ``@classmethod`` 门面方法通过 ``_LazyMethod`` 非数据描述符 +
  ``@_populate_from_registry`` 类装饰器从 ``fincore._registry``
  自动生成，避免手写 100+ 行委托代码。
* ``@_dual_method`` 门面方法（需要从实例自动填充 ``returns`` /
  ``factor_returns``）保留显式定义，以保持精确的函数签名。
"""
import importlib
import functools
import pandas as pd
import numpy as np
from fincore.constants import DAILY, APPROX_BDAYS_PER_YEAR, FACTOR_PARTITIONS
from fincore._registry import (
    CLASSMETHOD_REGISTRY,
    STATIC_METHODS,
    MODULE_PATHS,
)


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


# ---------------------------------------------------------------------------
# Lazy module resolver — replaces 17 top-level ``from fincore.metrics import …``
# ---------------------------------------------------------------------------
_MODULE_CACHE: dict = {}


def _resolve_module(alias: str):
    """Return the actual module for a registry alias like ``'_ratios'``."""
    try:
        return _MODULE_CACHE[alias]
    except KeyError:
        mod = importlib.import_module(MODULE_PATHS[alias])
        _MODULE_CACHE[alias] = mod
        return mod


# ---------------------------------------------------------------------------
# Lazy descriptor + class decorator — replaces metaclass for registry methods
# ---------------------------------------------------------------------------
class _LazyMethod:
    """Non-data descriptor that lazy-resolves a metric function on first access.

    On first attribute access (class-level or instance-level), the underlying
    function is resolved from the metric module, cached as a ``staticmethod``
    on the owner class, and returned.  Subsequent accesses hit the cached
    ``staticmethod`` directly — zero overhead.
    """

    __slots__ = ('mod_alias', 'func_name', 'attr_name', '_owner')

    def __init__(self, mod_alias, func_name, attr_name, owner):
        self.mod_alias = mod_alias
        self.func_name = func_name
        self.attr_name = attr_name
        self._owner = owner

    def __get__(self, obj, objtype=None):
        func = getattr(_resolve_module(self.mod_alias), self.func_name)
        setattr(self._owner, self.attr_name, staticmethod(func))
        return func


def _populate_from_registry(cls):
    """Class decorator: attach ``_LazyMethod`` descriptors for all registry entries."""
    for name, (mod_alias, func_name) in CLASSMETHOD_REGISTRY.items():
        setattr(cls, name, _LazyMethod(mod_alias, func_name, name, cls))
    for name, (mod_alias, func_name) in STATIC_METHODS.items():
        setattr(cls, name, _LazyMethod(mod_alias, func_name, name, cls))
    return cls


# ---------------------------------------------------------------------------
# Zipline compatibility stubs
# ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
# Empyrical class
# ---------------------------------------------------------------------------
@_populate_from_registry
class Empyrical:
    """面向对象的性能指标计算类.

    **Class-level access** (e.g. ``Empyrical.sharpe_ratio(returns)``)
    is auto-generated from the function registry in ``fincore._registry``
    via ``_LazyMethod`` descriptors attached by ``@_populate_from_registry``.

    **Instance-level access** with auto-fill of ``returns`` / ``factor_returns``
    is provided by explicitly defined ``@_dual_method`` methods below.
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

    def __getattr__(self, name):
        """Safety-net for registry-backed attributes on instance access.

        Normally ``_LazyMethod`` descriptors (set by ``@_populate_from_registry``)
        handle both class-level and instance-level lookups.  This method acts as
        a fallback for edge cases (e.g. subclass access before descriptor
        resolution) by delegating to the same registry and caching the result
        on the class so that subsequent accesses are zero-overhead.
        """
        entry = CLASSMETHOD_REGISTRY.get(name)
        if entry is not None:
            mod_alias, func_name = entry
            func = getattr(_resolve_module(mod_alias), func_name)
            setattr(type(self), name, staticmethod(func))
            return func

        entry = STATIC_METHODS.get(name)
        if entry is not None:
            mod_alias, func_name = entry
            func = getattr(_resolve_module(mod_alias), func_name)
            setattr(type(self), name, staticmethod(func))
            return func

        raise AttributeError(f"'{type(self).__name__}' object has no attribute {name!r}")

    # ------------------------------------------------------------------
    # Instance data helpers
    # ------------------------------------------------------------------

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

    # ------------------------------------------------------------------
    # CAGR alias (not in registry because it maps to a *different* name)
    # ------------------------------------------------------------------

    @classmethod
    def cagr(cls, returns, period=DAILY, annualization=None):
        """计算复合年增长率(annual_return的别名)."""
        return _resolve_module('_yearly').annual_return(returns, period, annualization)

    # ==================================================================
    # @_dual_method wrappers — auto-fill returns from instance state
    # ==================================================================

    # ---- drawdown ----

    @_dual_method
    def get_max_drawdown_period(self, returns=None):
        """获取最大回撤期间的开始和结束日期."""
        return _resolve_module('_drawdown').get_max_drawdown_period(self._get_returns(returns))

    @_dual_method
    def max_drawdown_days(self, returns=None):
        """计算最大回撤持续的天数."""
        return _resolve_module('_drawdown').max_drawdown_days(self._get_returns(returns))

    @_dual_method
    def second_max_drawdown(self, returns=None):
        """确定策略的第二大回撤."""
        return _resolve_module('_drawdown').second_max_drawdown(self._get_returns(returns))

    @_dual_method
    def third_max_drawdown(self, returns=None):
        """确定策略的第三大回撤."""
        return _resolve_module('_drawdown').third_max_drawdown(self._get_returns(returns))

    @_dual_method
    def second_max_drawdown_days(self, returns=None):
        """获取第二大回撤的天数."""
        return _resolve_module('_drawdown').second_max_drawdown_days(self._get_returns(returns))

    @_dual_method
    def second_max_drawdown_recovery_days(self, returns=None):
        """获取第二大回撤的恢复天数."""
        return _resolve_module('_drawdown').second_max_drawdown_recovery_days(self._get_returns(returns))

    @_dual_method
    def third_max_drawdown_days(self, returns=None):
        """获取第三大回撤的天数."""
        return _resolve_module('_drawdown').third_max_drawdown_days(self._get_returns(returns))

    @_dual_method
    def third_max_drawdown_recovery_days(self, returns=None):
        """获取第三大回撤的恢复天数."""
        return _resolve_module('_drawdown').third_max_drawdown_recovery_days(self._get_returns(returns))

    @_dual_method
    def max_drawdown_weeks(self, returns=None):
        """计算最大回撤周数."""
        return _resolve_module('_drawdown').max_drawdown_weeks(self._get_returns(returns))

    @_dual_method
    def max_drawdown_months(self, returns=None):
        """计算最大回撤月数."""
        return _resolve_module('_drawdown').max_drawdown_months(self._get_returns(returns))

    @_dual_method
    def max_drawdown_recovery_days(self, returns=None):
        """计算最大回撤恢复天数."""
        return _resolve_module('_drawdown').max_drawdown_recovery_days(self._get_returns(returns))

    @_dual_method
    def max_drawdown_recovery_weeks(self, returns=None):
        """计算最大回撤恢复周数."""
        return _resolve_module('_drawdown').max_drawdown_recovery_weeks(self._get_returns(returns))

    @_dual_method
    def max_drawdown_recovery_months(self, returns=None):
        """计算最大回撤恢复月数."""
        return _resolve_module('_drawdown').max_drawdown_recovery_months(self._get_returns(returns))

    # ---- stats (returns only) ----

    @_dual_method
    def futures_market_correlation(self, returns=None, futures_returns=None):
        """计算与期货市场的相关性."""
        return _resolve_module('_stats').futures_market_correlation(self._get_returns(returns), futures_returns)

    @_dual_method
    def serial_correlation(self, returns=None, lag=1):
        """计算序列自相关."""
        return _resolve_module('_stats').serial_correlation(self._get_returns(returns), lag)

    @_dual_method
    def win_rate(self, returns=None):
        """计算胜率."""
        return _resolve_module('_stats').win_rate(self._get_returns(returns))

    @_dual_method
    def loss_rate(self, returns=None):
        """计算亏损率."""
        return _resolve_module('_stats').loss_rate(self._get_returns(returns))

    # ---- consecutive (returns only) ----

    @_dual_method
    def max_consecutive_up_weeks(self, returns=None):
        """计算最大连续上涨周数."""
        return _resolve_module('_consecutive').max_consecutive_up_weeks(self._get_returns(returns))

    @_dual_method
    def max_consecutive_down_weeks(self, returns=None):
        """计算最大连续下跌周数."""
        return _resolve_module('_consecutive').max_consecutive_down_weeks(self._get_returns(returns))

    @_dual_method
    def max_consecutive_up_months(self, returns=None):
        """计算最大连续上涨月数."""
        return _resolve_module('_consecutive').max_consecutive_up_months(self._get_returns(returns))

    @_dual_method
    def max_consecutive_down_months(self, returns=None):
        """计算最大连续下跌月数."""
        return _resolve_module('_consecutive').max_consecutive_down_months(self._get_returns(returns))

    @_dual_method
    def max_single_day_gain_date(self, returns=None):
        """获取单日最大收益的日期."""
        return _resolve_module('_consecutive').max_single_day_gain_date(self._get_returns(returns))

    @_dual_method
    def max_single_day_loss_date(self, returns=None):
        """获取单日最大亏损的日期."""
        return _resolve_module('_consecutive').max_single_day_loss_date(self._get_returns(returns))

    @_dual_method
    def max_consecutive_up_start_date(self, returns=None):
        """获取最大连续上涨期的开始日期."""
        return _resolve_module('_consecutive').max_consecutive_up_start_date(self._get_returns(returns))

    @_dual_method
    def max_consecutive_up_end_date(self, returns=None):
        """获取最大连续上涨期的结束日期."""
        return _resolve_module('_consecutive').max_consecutive_up_end_date(self._get_returns(returns))

    @_dual_method
    def max_consecutive_down_start_date(self, returns=None):
        """获取最大连续下跌期的开始日期."""
        return _resolve_module('_consecutive').max_consecutive_down_start_date(self._get_returns(returns))

    @_dual_method
    def max_consecutive_down_end_date(self, returns=None):
        """获取最大连续下跌期的结束日期."""
        return _resolve_module('_consecutive').max_consecutive_down_end_date(self._get_returns(returns))

    # ---- ratios (returns only) ----

    @_dual_method
    def common_sense_ratio(self, returns=None):
        """计算常识比率."""
        return _resolve_module('_ratios').common_sense_ratio(self._get_returns(returns))

    @_dual_method
    def sterling_ratio(self, returns=None, risk_free=0.0, period=DAILY, annualization=None):
        """计算斯特林比率."""
        return _resolve_module('_ratios').sterling_ratio(self._get_returns(returns), risk_free, period, annualization)

    @_dual_method
    def burke_ratio(self, returns=None, risk_free=0.0, period=DAILY, annualization=None):
        """计算伯克比率."""
        return _resolve_module('_ratios').burke_ratio(self._get_returns(returns), risk_free, period, annualization)

    @_dual_method
    def kappa_three_ratio(self, returns=None, risk_free=0.0, period=DAILY, annualization=None, mar=0.0):
        """计算Kappa3比率."""
        return _resolve_module('_ratios').kappa_three_ratio(self._get_returns(returns), risk_free, period, annualization, mar)

    # ---- risk (returns only) ----

    @_dual_method
    def gpd_risk_estimates(self, returns=None, var_p=0.01):
        """使用GPD估计VaR和ES."""
        return _resolve_module('_risk').gpd_risk_estimates(self._get_returns(returns), var_p)

    @_dual_method
    def gpd_risk_estimates_aligned(self, returns=None, var_p=0.01):
        """使用GPD估计VaR和ES（对齐版本）."""
        return _resolve_module('_risk').gpd_risk_estimates_aligned(self._get_returns(returns), var_p)

    @_dual_method
    def var_excess_return(self, returns=None, cutoff=0.05):
        """计算VaR超额收益."""
        return _resolve_module('_risk').var_excess_return(self._get_returns(returns), cutoff)

    # ---- yearly (returns only) ----

    @_dual_method
    def annualized_cumulative_return(self, returns=None, period=DAILY, annualization=None):
        """计算年化累计收益."""
        return _resolve_module('_yearly').annual_return(self._get_returns(returns), period, annualization)

    @_dual_method
    def annual_volatility_by_year(self, returns=None, period=DAILY, annualization=None):
        """计算每年的波动率."""
        return _resolve_module('_yearly').annual_volatility_by_year(self._get_returns(returns), period, annualization)

    # ---- rolling (returns only) ----

    @_dual_method
    def roll_sharpe_ratio(self, returns=None, window=252, risk_free=0.0, period=DAILY, annualization=None):
        """计算滚动夏普比率."""
        return _resolve_module('_rolling').roll_sharpe_ratio(self._get_returns(returns), window, risk_free, period, annualization)

    @_dual_method
    def roll_max_drawdown(self, returns=None, window=252):
        """计算滚动最大回撤."""
        return _resolve_module('_rolling').roll_max_drawdown(self._get_returns(returns), window)

    # ---- timing (returns only) ----

    @_dual_method
    def extract_interesting_date_ranges(self, returns=None):
        """提取有趣的日期范围."""
        return _resolve_module('_timing').extract_interesting_date_ranges(self._get_returns(returns))

    # ==================================================================
    # @_dual_method wrappers — auto-fill returns AND factor_returns
    # ==================================================================

    @_dual_method
    def r_cubed(self, returns=None, factor_returns=None):
        """计算R³指标."""
        return _resolve_module('_stats').r_cubed(
            self._get_returns(returns), self._get_factor_returns(factor_returns))

    @_dual_method
    def tracking_difference(self, returns=None, factor_returns=None):
        """计算跟踪差异."""
        return _resolve_module('_stats').tracking_difference(
            self._get_returns(returns), self._get_factor_returns(factor_returns))

    @_dual_method
    def beta_fragility_heuristic(self, returns=None, factor_returns=None):
        """估计Beta脆弱性."""
        return _resolve_module('_risk').beta_fragility_heuristic(
            self._get_returns(returns), self._get_factor_returns(factor_returns))

    @_dual_method
    def beta_fragility_heuristic_aligned(self, returns=None, factor_returns=None):
        """估计Beta脆弱性（对齐版本）."""
        return _resolve_module('_risk').beta_fragility_heuristic_aligned(
            self._get_returns(returns), self._get_factor_returns(factor_returns))

    @_dual_method
    def treynor_ratio(self, returns=None, factor_returns=None, risk_free=0.0, period=DAILY, annualization=None):
        """计算特雷诺比率."""
        return _resolve_module('_ratios').treynor_ratio(
            self._get_returns(returns), self._get_factor_returns(factor_returns),
            risk_free, period, annualization)

    @_dual_method
    def m_squared(self, returns=None, factor_returns=None, risk_free=0.0, period=DAILY, annualization=None):
        """计算M²测度."""
        return _resolve_module('_ratios').m_squared(
            self._get_returns(returns), self._get_factor_returns(factor_returns),
            risk_free, period, annualization)

    @_dual_method
    def residual_risk(self, returns=None, factor_returns=None, risk_free=0.0, period=DAILY, annualization=None):
        """计算残差风险."""
        return _resolve_module('_risk').residual_risk(
            self._get_returns(returns), self._get_factor_returns(factor_returns),
            risk_free, period, annualization)

    # ---- rolling (returns + factor_returns) ----

    @_dual_method
    def roll_alpha(self, returns=None, factor_returns=None, window=252, risk_free=0.0, period=DAILY, annualization=None):
        """计算滚动Alpha."""
        return _resolve_module('_rolling').roll_alpha(
            self._get_returns(returns), self._get_factor_returns(factor_returns),
            window, risk_free, period, annualization)

    @_dual_method
    def roll_beta(self, returns=None, factor_returns=None, window=252, risk_free=0.0, period=DAILY, annualization=None):
        """计算滚动Beta."""
        return _resolve_module('_rolling').roll_beta(
            self._get_returns(returns), self._get_factor_returns(factor_returns),
            window, risk_free, period, annualization)

    @_dual_method
    def roll_alpha_beta(self, returns=None, factor_returns=None, window=252, risk_free=0.0, period=DAILY, annualization=None):
        """计算滚动Alpha和Beta."""
        return _resolve_module('_rolling').roll_alpha_beta(
            self._get_returns(returns), self._get_factor_returns(factor_returns),
            window, risk_free, period, annualization)

    @_dual_method
    def roll_up_capture(self, returns=None, factor_returns=None, window=252):
        """计算滚动上行捕获率."""
        return _resolve_module('_rolling').roll_up_capture(
            self._get_returns(returns), self._get_factor_returns(factor_returns), window)

    @_dual_method
    def roll_down_capture(self, returns=None, factor_returns=None, window=252):
        """计算滚动下行捕获率."""
        return _resolve_module('_rolling').roll_down_capture(
            self._get_returns(returns), self._get_factor_returns(factor_returns), window)

    @_dual_method
    def roll_up_down_capture(self, returns=None, factor_returns=None, window=252):
        """计算滚动上下行捕获比率."""
        return _resolve_module('_rolling').roll_up_down_capture(
            self._get_returns(returns), self._get_factor_returns(factor_returns), window)

    # ---- yearly (returns + factor_returns) ----

    @_dual_method
    def annual_active_return(self, returns=None, factor_returns=None, period=DAILY, annualization=None):
        """计算年度主动收益."""
        return _resolve_module('_yearly').annual_active_return(
            self._get_returns(returns), self._get_factor_returns(factor_returns),
            period, annualization)

    @_dual_method
    def annual_active_risk(self, returns=None, factor_returns=None, period=DAILY, annualization=None):
        """计算年度主动风险（跟踪误差）."""
        return _resolve_module('_risk').tracking_error(
            self._get_returns(returns), self._get_factor_returns(factor_returns),
            period, annualization)

    @_dual_method
    def annual_active_return_by_year(self, returns=None, factor_returns=None, period=DAILY, annualization=None):
        """计算每年的主动收益."""
        return _resolve_module('_yearly').annual_active_return_by_year(
            self._get_returns(returns), self._get_factor_returns(factor_returns),
            period, annualization)

    @_dual_method
    def information_ratio_by_year(self, returns=None, factor_returns=None, period=DAILY, annualization=None):
        """计算每年的信息比率."""
        return _resolve_module('_yearly').information_ratio_by_year(
            self._get_returns(returns), self._get_factor_returns(factor_returns),
            period, annualization)

    # ==================================================================
    # Special-case methods (custom logic, not simple forwarding)
    # ==================================================================

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
        return _resolve_module('_perf_attrib').perf_attrib(returns, positions, factor_returns, factor_loadings, transactions, pos_in_dollars, regression_style)

    @_dual_method
    def regression_annual_return(self, returns=None, factor_returns=None, risk_free=0.0, period=DAILY, annualization=None):
        """计算回归年化收益."""
        returns = self._get_returns(returns)
        factor_returns = self._get_factor_returns(factor_returns)
        _ab = _resolve_module('_alpha_beta')
        _yr = _resolve_module('_yearly')
        alpha_val = _ab.alpha(returns, factor_returns, risk_free, period, annualization)
        beta_val = _ab.beta(returns, factor_returns, risk_free, period, annualization)
        if np.isnan(alpha_val) or np.isnan(beta_val):
            return np.nan
        benchmark_annual = _yr.annual_return(factor_returns, period, annualization)
        if np.isnan(benchmark_annual):
            return np.nan
        return alpha_val + beta_val * benchmark_annual

    @classmethod
    def _groupby_consecutive(cls, txn, max_delta=None):
        """按连续交易分组."""
        if max_delta is None:
            max_delta = pd.Timedelta("8h")
        return _resolve_module('_round_trips').groupby_consecutive(txn, max_delta)


__all__ = ["Empyrical", "ZIPLINE"]
