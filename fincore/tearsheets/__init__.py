# -*- coding: utf-8 -*-
"""
Tearsheets 模块

将 pyfolio.py 中的绘图和显示函数拆分到各个子模块中。
Pyfolio 类通过导入这些函数作为接口层。
"""

# 从各子模块导入函数
from fincore.tearsheets.utils import (
    plotting_context,
    axes_style,
)

from fincore.tearsheets.returns import (
    plot_monthly_returns_heatmap,
    plot_annual_returns,
    plot_monthly_returns_dist,
    plot_returns,
    plot_rolling_returns,
    plot_rolling_beta,
    plot_rolling_volatility,
    plot_rolling_sharpe,
    plot_drawdown_periods,
    plot_drawdown_underwater,
    plot_return_quantiles,
    plot_monthly_returns_timeseries,
    plot_perf_stats,
    show_perf_stats,
    show_worst_drawdown_periods,
)

from fincore.tearsheets.positions import (
    plot_holdings,
    plot_long_short_holdings,
    plot_exposures,
    plot_gross_leverage,
    plot_max_median_position_concentration,
    plot_sector_allocations,
    show_and_plot_top_positions,
)

from fincore.tearsheets.transactions import (
    plot_turnover,
    plot_daily_volume,
    plot_daily_turnover_hist,
    plot_txn_time_hist,
    plot_slippage_sweep,
    plot_slippage_sensitivity,
)

from fincore.tearsheets.round_trips import (
    plot_round_trip_lifetimes,
    plot_prob_profit_trade,
    print_round_trip_stats,
    show_profit_attribution,
)

from fincore.tearsheets.bayesian import (
    plot_best,
    plot_stoch_vol,
    plot_bayes_cone,
    _plot_bayes_cone,
)

from fincore.tearsheets.risk import (
    plot_style_factor_exposures,
    plot_sector_exposures_longshort,
    plot_sector_exposures_gross,
    plot_sector_exposures_net,
    plot_cap_exposures_longshort,
    plot_cap_exposures_gross,
    plot_cap_exposures_net,
    plot_volume_exposures_longshort,
    plot_volume_exposures_gross,
)

from fincore.tearsheets.perf_attrib import (
    plot_perf_attrib_returns,
    plot_alpha_returns,
    plot_factor_contribution_to_perf,
    plot_risk_exposures,
    show_perf_attrib_stats,
)

from fincore.tearsheets.capacity import (
    plot_capacity_sweep,
    plot_cones,
)

from fincore.tearsheets.sheets import (
    create_full_tear_sheet,
    create_simple_tear_sheet,
    create_returns_tear_sheet,
    create_position_tear_sheet,
    create_txn_tear_sheet,
    create_round_trip_tear_sheet,
    create_interesting_times_tear_sheet,
    create_capacity_tear_sheet,
    create_bayesian_tear_sheet,
    create_risk_tear_sheet,
    create_perf_attrib_tear_sheet,
)

__all__ = [
    # utils
    'plotting_context',
    'axes_style',
    # returns
    'plot_monthly_returns_heatmap',
    'plot_annual_returns',
    'plot_monthly_returns_dist',
    'plot_returns',
    'plot_rolling_returns',
    'plot_rolling_beta',
    'plot_rolling_volatility',
    'plot_rolling_sharpe',
    'plot_drawdown_periods',
    'plot_drawdown_underwater',
    'plot_return_quantiles',
    'plot_monthly_returns_timeseries',
    'plot_perf_stats',
    'show_perf_stats',
    'show_worst_drawdown_periods',
    # positions
    'plot_holdings',
    'plot_long_short_holdings',
    'plot_exposures',
    'plot_gross_leverage',
    'plot_max_median_position_concentration',
    'plot_sector_allocations',
    'show_and_plot_top_positions',
    # transactions
    'plot_turnover',
    'plot_daily_volume',
    'plot_daily_turnover_hist',
    'plot_txn_time_hist',
    'plot_slippage_sweep',
    'plot_slippage_sensitivity',
    # round_trips
    'plot_round_trip_lifetimes',
    'plot_prob_profit_trade',
    'print_round_trip_stats',
    'show_profit_attribution',
    # bayesian
    'plot_best',
    'plot_stoch_vol',
    'plot_bayes_cone',
    '_plot_bayes_cone',
    # risk
    'plot_style_factor_exposures',
    'plot_sector_exposures_longshort',
    'plot_sector_exposures_gross',
    'plot_sector_exposures_net',
    'plot_cap_exposures_longshort',
    'plot_cap_exposures_gross',
    'plot_cap_exposures_net',
    'plot_volume_exposures_longshort',
    'plot_volume_exposures_gross',
    # perf_attrib
    'plot_perf_attrib_returns',
    'plot_alpha_returns',
    'plot_factor_contribution_to_perf',
    'plot_risk_exposures',
    'show_perf_attrib_stats',
    # capacity
    'plot_capacity_sweep',
    'plot_cones',
    # sheets
    'create_full_tear_sheet',
    'create_simple_tear_sheet',
    'create_returns_tear_sheet',
    'create_position_tear_sheet',
    'create_txn_tear_sheet',
    'create_round_trip_tear_sheet',
    'create_interesting_times_tear_sheet',
    'create_capacity_tear_sheet',
    'create_bayesian_tear_sheet',
    'create_risk_tear_sheet',
    'create_perf_attrib_tear_sheet',
]
