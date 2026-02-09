# -*- coding: utf-8 -*-
"""
Tearsheets 模块

将 pyfolio.py 中的绘图和显示函数拆分到各个子模块中。
Pyfolio 类通过导入这些函数作为接口层。

所有子模块通过 ``__getattr__`` 懒加载，避免在 ``import fincore.tearsheets``
时立即加载 matplotlib / seaborn 等重量级依赖。
"""

import importlib as _importlib

# Maps exported name → (submodule, attribute)
_ATTR_MAP = {
    # utils
    'plotting_context': ('utils', 'plotting_context'),
    'axes_style': ('utils', 'axes_style'),
    # returns
    'plot_monthly_returns_heatmap': ('returns', 'plot_monthly_returns_heatmap'),
    'plot_annual_returns': ('returns', 'plot_annual_returns'),
    'plot_monthly_returns_dist': ('returns', 'plot_monthly_returns_dist'),
    'plot_returns': ('returns', 'plot_returns'),
    'plot_rolling_returns': ('returns', 'plot_rolling_returns'),
    'plot_rolling_beta': ('returns', 'plot_rolling_beta'),
    'plot_rolling_volatility': ('returns', 'plot_rolling_volatility'),
    'plot_rolling_sharpe': ('returns', 'plot_rolling_sharpe'),
    'plot_drawdown_periods': ('returns', 'plot_drawdown_periods'),
    'plot_drawdown_underwater': ('returns', 'plot_drawdown_underwater'),
    'plot_return_quantiles': ('returns', 'plot_return_quantiles'),
    'plot_monthly_returns_timeseries': ('returns', 'plot_monthly_returns_timeseries'),
    'plot_perf_stats': ('returns', 'plot_perf_stats'),
    'show_perf_stats': ('returns', 'show_perf_stats'),
    'show_worst_drawdown_periods': ('returns', 'show_worst_drawdown_periods'),
    # positions
    'plot_holdings': ('positions', 'plot_holdings'),
    'plot_long_short_holdings': ('positions', 'plot_long_short_holdings'),
    'plot_exposures': ('positions', 'plot_exposures'),
    'plot_gross_leverage': ('positions', 'plot_gross_leverage'),
    'plot_max_median_position_concentration': ('positions', 'plot_max_median_position_concentration'),
    'plot_sector_allocations': ('positions', 'plot_sector_allocations'),
    'show_and_plot_top_positions': ('positions', 'show_and_plot_top_positions'),
    # transactions
    'plot_turnover': ('transactions', 'plot_turnover'),
    'plot_daily_volume': ('transactions', 'plot_daily_volume'),
    'plot_daily_turnover_hist': ('transactions', 'plot_daily_turnover_hist'),
    'plot_txn_time_hist': ('transactions', 'plot_txn_time_hist'),
    'plot_slippage_sweep': ('transactions', 'plot_slippage_sweep'),
    'plot_slippage_sensitivity': ('transactions', 'plot_slippage_sensitivity'),
    # round_trips
    'plot_round_trip_lifetimes': ('round_trips', 'plot_round_trip_lifetimes'),
    'plot_prob_profit_trade': ('round_trips', 'plot_prob_profit_trade'),
    'print_round_trip_stats': ('round_trips', 'print_round_trip_stats'),
    'show_profit_attribution': ('round_trips', 'show_profit_attribution'),
    # bayesian
    'plot_best': ('bayesian', 'plot_best'),
    'plot_stoch_vol': ('bayesian', 'plot_stoch_vol'),
    'plot_bayes_cone': ('bayesian', 'plot_bayes_cone'),
    '_plot_bayes_cone': ('bayesian', '_plot_bayes_cone'),
    # risk
    'plot_style_factor_exposures': ('risk', 'plot_style_factor_exposures'),
    'plot_sector_exposures_longshort': ('risk', 'plot_sector_exposures_longshort'),
    'plot_sector_exposures_gross': ('risk', 'plot_sector_exposures_gross'),
    'plot_sector_exposures_net': ('risk', 'plot_sector_exposures_net'),
    'plot_cap_exposures_longshort': ('risk', 'plot_cap_exposures_longshort'),
    'plot_cap_exposures_gross': ('risk', 'plot_cap_exposures_gross'),
    'plot_cap_exposures_net': ('risk', 'plot_cap_exposures_net'),
    'plot_volume_exposures_longshort': ('risk', 'plot_volume_exposures_longshort'),
    'plot_volume_exposures_gross': ('risk', 'plot_volume_exposures_gross'),
    # perf_attrib
    'plot_perf_attrib_returns': ('perf_attrib', 'plot_perf_attrib_returns'),
    'plot_alpha_returns': ('perf_attrib', 'plot_alpha_returns'),
    'plot_factor_contribution_to_perf': ('perf_attrib', 'plot_factor_contribution_to_perf'),
    'plot_risk_exposures': ('perf_attrib', 'plot_risk_exposures'),
    'show_perf_attrib_stats': ('perf_attrib', 'show_perf_attrib_stats'),
    # capacity
    'plot_capacity_sweep': ('capacity', 'plot_capacity_sweep'),
    'plot_cones': ('capacity', 'plot_cones'),
    # sheets
    'create_full_tear_sheet': ('sheets', 'create_full_tear_sheet'),
    'create_simple_tear_sheet': ('sheets', 'create_simple_tear_sheet'),
    'create_returns_tear_sheet': ('sheets', 'create_returns_tear_sheet'),
    'create_position_tear_sheet': ('sheets', 'create_position_tear_sheet'),
    'create_txn_tear_sheet': ('sheets', 'create_txn_tear_sheet'),
    'create_round_trip_tear_sheet': ('sheets', 'create_round_trip_tear_sheet'),
    'create_interesting_times_tear_sheet': ('sheets', 'create_interesting_times_tear_sheet'),
    'create_capacity_tear_sheet': ('sheets', 'create_capacity_tear_sheet'),
    'create_bayesian_tear_sheet': ('sheets', 'create_bayesian_tear_sheet'),
    'create_risk_tear_sheet': ('sheets', 'create_risk_tear_sheet'),
    'create_perf_attrib_tear_sheet': ('sheets', 'create_perf_attrib_tear_sheet'),
}

_SUBMODULE_CACHE = {}


def __getattr__(name):
    entry = _ATTR_MAP.get(name)
    if entry is not None:
        submod_name, attr_name = entry
        if submod_name not in _SUBMODULE_CACHE:
            _SUBMODULE_CACHE[submod_name] = _importlib.import_module(
                f'fincore.tearsheets.{submod_name}'
            )
        attr = getattr(_SUBMODULE_CACHE[submod_name], attr_name)
        globals()[name] = attr
        return attr
    raise AttributeError(f"module 'fincore.tearsheets' has no attribute {name!r}")


__all__ = list(_ATTR_MAP.keys())
