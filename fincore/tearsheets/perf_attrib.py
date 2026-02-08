# -*- coding: utf-8 -*-
"""
绩效归因相关的绘图和显示函数

包含绩效归因收益、因子贡献、风险暴露等绘图函数。
"""

import pandas as pd
import matplotlib.pyplot as plt

from fincore.utils import print_table, configure_legend


def plot_perf_attrib_returns(empyrical_instance, perf_attrib_data, cost=None, ax=None):
    """
    Plot total, specific, and common returns.

    Parameters
    ----------
    empyrical_instance : Empyrical
        Empyrical 实例，用于调用计算方法
    perf_attrib_data : pd.DataFrame
        df with factors, common returns, and specific returns as columns,
        and datetimes as index. Assumes the `total_returns` column is NOT
        cost adjusted.
    cost : pd.Series, optional
        if present, gets subtracted from `perf_attrib_data['total_returns']`,
        and gets plotted separately
    ax : matplotlib.axes.Axes, optional
        axes on which plots are made. If None, current axes will be used

    Returns
    -------
    ax : matplotlib.axes.Axes
    """
    if ax is None:
        ax = plt.gca()

    returns = perf_attrib_data['total_returns']
    total_returns_label = 'Total returns'

    cumulative_returns_less_costs = empyrical_instance._cumulative_returns_less_costs(
        returns,
        cost
    )
    if cost is not None:
        total_returns_label += ' (adjusted)'

    specific_returns = perf_attrib_data['specific_returns']
    common_returns = perf_attrib_data['common_returns']

    ax.plot(cumulative_returns_less_costs, color='b',
            label=total_returns_label)
    ax.plot(empyrical_instance.cum_returns(specific_returns), color='g',
            label='Cumulative specific returns')
    ax.plot(empyrical_instance.cum_returns(common_returns), color='r',
            label='Cumulative common returns')

    if cost is not None:
        ax.plot(-empyrical_instance.cum_returns(cost), color='k',
                label='Cumulative cost spent')

    ax.set_title('Time series of cumulative returns')
    ax.set_ylabel('Returns')

    configure_legend(ax)

    return ax


def plot_alpha_returns(alpha_returns, ax=None):
    """
    Plot histogram of daily multifactor alpha returns (specific returns).

    Parameters
    ----------
    alpha_returns : pd.Series
        series of daily alpha returns indexed by datetime
    ax : matplotlib.axes.Axes, optional
        axes on which plots are made. If None, current axes will be used

    Returns
    -------
    ax : matplotlib.axes.Axes
    """
    if ax is None:
        ax = plt.gca()

    ax.hist(alpha_returns, color='g', label='Multi-factor alpha')
    ax.set_title('Histogram of alphas')
    ax.axvline(0, color='k', linestyle='--', label='Zero')

    avg = alpha_returns.mean()
    ax.axvline(avg, color='b', label='Mean = {: 0.5f}'.format(avg))
    configure_legend(ax)

    return ax


def plot_factor_contribution_to_perf(empyrical_instance,
                                     perf_attrib_data,
                                     ax=None,
                                     title='Cumulative common returns attribution'):
    """
    Plot each factor's contribution to performance.

    Parameters
    ----------
    empyrical_instance : Empyrical
        Empyrical 实例，用于调用计算方法
    perf_attrib_data : pd.DataFrame
        df with factors, common returns, and specific returns as columns,
        and datetimes as index
    ax : matplotlib.axes.Axes, optional
        axes on which plots are made. If None, current axes will be used
    title : str, optional
        title of plot

    Returns
    -------
    ax : matplotlib.axes.Axes
    """
    if ax is None:
        ax = plt.gca()

    factors_to_plot = perf_attrib_data.drop(
        ['total_returns', 'common_returns'], axis='columns', errors='ignore'
    )

    factors_cumulative = pd.DataFrame()
    for factor in factors_to_plot:
        factors_cumulative[factor] = empyrical_instance.cum_returns(factors_to_plot[factor])

    for col in factors_cumulative:
        ax.plot(factors_cumulative[col])

    ax.axhline(0, color='k')
    configure_legend(ax, change_colors=True)

    ax.set_ylabel('Cumulative returns by factor')
    ax.set_title(title)

    return ax


def plot_risk_exposures(exposures, ax=None,
                        title='Daily risk factor exposures'):
    """
    Plot daily risk factor exposures.

    Parameters
    ----------
    exposures : pd.DataFrame
        df indexed by datetime, with factors as columns
    ax : matplotlib.axes.Axes, optional
        axes on which plots are made. If None, current axes will be used
    title: string, optional
        Title for the plot, default 'Daily risk factor exposures'

    Returns
    -------
    ax : matplotlib.axes.Axes
    """
    if ax is None:
        ax = plt.gca()

    for col in exposures:
        ax.plot(exposures[col])

    configure_legend(ax, change_colors=True)
    ax.set_ylabel('Factor exposures')
    ax.set_title(title)

    return ax


def show_perf_attrib_stats(empyrical_instance, returns,
                           positions,
                           factor_returns,
                           factor_loadings,
                           transactions=None,
                           pos_in_dollars=True):
    """
    Calls `perf_attrib` using inputs, and displays outputs using
    `print_table`.

    Parameters
    ----------
    empyrical_instance : Empyrical
        Empyrical 实例，用于调用计算方法
    returns : pd.Series
        Daily returns of the strategy.
    positions : pd.DataFrame
        Daily net position values.
    factor_returns : pd.DataFrame
        Returns by factor.
    factor_loadings : pd.DataFrame
        Factor loadings for all days.
    transactions : pd.DataFrame, optional
        Executed trade volumes and fill prices.
    pos_in_dollars : bool, optional
        Whether positions are in dollars.
    """
    risk_exposures, perf_attrib_data = empyrical_instance.perf_attrib(
        returns,
        positions,
        factor_returns,
        factor_loadings,
        transactions,
        pos_in_dollars=pos_in_dollars,
    )

    perf_attrib_stats, risk_exposure_stats = \
        empyrical_instance.create_perf_attrib_stats(perf_attrib_data, risk_exposures)

    percentage_formatter = '{:.2%}'.format
    float_formatter = '{:.2f}'.format

    summary_stats = perf_attrib_stats.loc[['Annualized Specific Return',
                                           'Annualized Common Return',
                                           'Annualized Total Return',
                                           'Specific Sharpe Ratio']]

    # Format return rows in summary stats table as percentages.
    for col_name in (
            'Annualized Specific Return',
            'Annualized Common Return',
            'Annualized Total Return',
    ):
        summary_stats[col_name] = percentage_formatter(summary_stats[col_name])

    # Display sharpe to two decimal places.
    summary_stats['Specific Sharpe Ratio'] = float_formatter(
        summary_stats['Specific Sharpe Ratio']
    )

    print_table(summary_stats, name='Summary Statistics')

    print_table(
        risk_exposure_stats,
        name='Exposures Summary',
        # In `exposures` table, format exposure column to 2 decimal places, and
        # return columns as percentages.
        formatters={
            'Average Risk Factor Exposure': float_formatter,
            'Annualized Return': percentage_formatter,
            'Cumulative Return': percentage_formatter,
        },
    )
