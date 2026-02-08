# -*- coding: utf-8 -*-
"""
贝叶斯分析相关的绘图函数

包含 BEST 分析、随机波动率、贝叶斯锥形图等绘图函数。
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

from fincore.empyrical import Empyrical


def plot_best(empyrical_instance, trace=None, data_train=None, data_test=None,
              samples=1000, burn=200, axs=None):
    """
    Plot the BEST significance analysis.

    Parameters
    ----------
    empyrical_instance : Empyrical
        Empyrical 实例，用于调用计算方法
    trace : pymc3.sampling.BaseTrace, optional
        trace object as returned by model_best()
        If not passed, will run model_best(), for which
        data_train and data_test are required.
    data_train : pandas.Series, optional
        Returns of an in-sample period.
        Required if trace=None.
    data_test : pandas.Series, optional
        Returns of an out-of-sample period.
        Required if trace=None.
    samples : int, optional
        Posterior samples to draw.
    burn : int
        Posterior samples to discard as burn-in.
    axs : array of matplotlib.axes objects, optional
        Plot into passed axes objects. Needs six axes.

    Returns
    -------
    None
    """
    if trace is None:
        if (data_train is not None) or (data_test is not None):
            raise ValueError('Either pass trace or data_train and data_test')
        trace = empyrical_instance.model_best(data_train, data_test, samples=samples)

    trace = trace[burn:]
    if axs is None:
        fig, axs = plt.subplots(ncols=2, nrows=3, figsize=(16, 4))

    def distplot_w_perc(trace_data, ax):
        sns.histplot(trace_data, ax=ax)
        ax.axvline(
            stats.scoreatpercentile(trace_data, 2.5),
            color='0.5', label='2.5 and 97.5 percentiles')
        ax.axvline(
            stats.scoreatpercentile(trace_data, 97.5),
            color='0.5')

    sns.histplot(trace['group1_mean'], ax=axs[0], label='Backtest')
    sns.histplot(trace['group2_mean'], ax=axs[0], label='Forward')
    axs[0].legend(loc=0, frameon=True, framealpha=0.5)
    axs[1].legend(loc=0, frameon=True, framealpha=0.5)

    distplot_w_perc(trace['difference of means'], axs[1])

    axs[0].set(xlabel='Mean', ylabel='Belief', yticklabels=[])
    axs[1].set(xlabel='Difference of means', yticklabels=[])

    sns.histplot(trace['group1_annual_volatility'], ax=axs[2],
                 label='Backtest')
    sns.histplot(trace['group2_annual_volatility'], ax=axs[2],
                 label='Forward')
    distplot_w_perc(trace['group2_annual_volatility'] -
                    trace['group1_annual_volatility'], axs[3])
    axs[2].set(xlabel='Annual volatility', ylabel='Belief',
               yticklabels=[])
    axs[2].legend(loc=0, frameon=True, framealpha=0.5)
    axs[3].set(xlabel='Difference of volatility', yticklabels=[])

    sns.histplot(trace['group1_sharpe'], ax=axs[4], label='Backtest')
    sns.histplot(trace['group2_sharpe'], ax=axs[4], label='Forward')
    distplot_w_perc(trace['group2_sharpe'] - trace['group1_sharpe'],
                    axs[5])
    axs[4].set(xlabel='Sharpe', ylabel='Belief', yticklabels=[])
    axs[4].legend(loc=0, frameon=True, framealpha=0.5)
    axs[5].set(xlabel='Difference of Sharpes', yticklabels=[])

    sns.histplot(trace['effect size'], ax=axs[6])
    axs[6].axvline(
        stats.scoreatpercentile(trace['effect size'], 2.5),
        color='0.5')
    axs[6].axvline(
        stats.scoreatpercentile(trace['effect size'], 97.5),
        color='0.5')
    axs[6].set(xlabel='Difference of means normalized by volatility',
               ylabel='Belief', yticklabels=[])


def plot_stoch_vol(empyrical_instance, data, trace=None, ax=None):
    """
    Generate plot for a stochastic volatility model.

    Parameters
    ----------
    empyrical_instance : Empyrical
        Empyrical 实例，用于调用计算方法
    data : pandas.Series
        Returns to model.
    trace : pymc3.sampling.BaseTrace object, optional
        trace as returned by model_stoch_vol
        If not passed, sample from the model.
    ax : matplotlib.axes object, optional
        Plot into an axe object

    Returns
    -------
    ax object
    """
    if trace is None:
        trace = empyrical_instance.model_stoch_vol(data)

    if ax is None:
        fig, ax = plt.subplots(figsize=(15, 8))

    data.abs().plot(ax=ax)
    ax.plot(data.index, np.exp(trace['s', ::30].T), 'r', alpha=.03)
    ax.set(title='Stochastic volatility', xlabel='Time', ylabel='Volatility')
    ax.legend(['Abs returns', 'Stochastic volatility process'],
              frameon=True, framealpha=0.5)

    return ax


def _plot_bayes_cone(empyrical_instance, returns_train, returns_test,
                     preds, plot_train_len=None, ax=None):
    """
    Internal function to plot Bayesian cone.

    Parameters
    ----------
    empyrical_instance : Empyrical
        Empyrical 实例，用于调用计算方法
    returns_train : pd.Series
        In-sample returns.
    returns_test : pd.Series
        Out-of-sample returns.
    preds : np.array
        Posterior predictive samples.
    plot_train_len : int, optional
        How many data points to plot of returns_train.
    ax : matplotlib.Axis, optional
        Axes upon which to plot.

    Returns
    -------
    ax : matplotlib.Axes
        The axes that were plotted on.
    """
    if ax is None:
        ax = plt.gca()

    returns_train_cum = Empyrical.cum_returns(returns_train, starting_value=1.0)
    returns_test_cum = Empyrical.cum_returns(
        returns_test,
        starting_value=returns_train_cum.iloc[-1],
    )

    perc = empyrical_instance.compute_bayes_cone(preds, starting_value=returns_train_cum.iloc[-1])
    # Add indices
    perc = {k: pd.Series(v, index=returns_test.index) for k, v in perc.items()}

    returns_test_cum_rel = returns_test_cum
    # Stitch together train and test
    returns_train_cum.loc[returns_test_cum_rel.index[0]] = \
        returns_test_cum_rel.iloc[0]

    # Plotting
    if plot_train_len is not None:
        returns_train_cum = returns_train_cum.iloc[-plot_train_len:]

    returns_train_cum.plot(ax=ax, color='g', label='In-sample')
    returns_test_cum_rel.plot(ax=ax, color='r', label='Out-of-sample')

    ax.fill_between(returns_test.index, perc[5], perc[95], alpha=.3)
    ax.fill_between(returns_test.index, perc[25], perc[75], alpha=.6)
    ax.legend(loc='best', frameon=True, framealpha=0.5)
    ax.set_title('Bayesian cone')
    ax.set_xlabel('')
    ax.set_ylabel('Cumulative returns')

    return ax


def plot_bayes_cone(empyrical_instance, returns_train, returns_test, ppc,
                    plot_train_len=50, ax=None):
    """
    Generate cumulative returns plot with Bayesian cone.

    Parameters
    ----------
    empyrical_instance : Empyrical
        Empyrical 实例，用于调用计算方法
    returns_train : pd.Series
        Timeseries of simple returns
    returns_test : pd.Series
        Out-of-sample returns.
    ppc : np.array
        Posterior predictive samples of shape samples x len(returns_test).
    plot_train_len : int, optional
        How many data points to plot of returns_train.
    ax : matplotlib.Axis, optional
        Axes upon which to plot.

    Returns
    -------
    score : float
        Consistency score (see compute_consistency_score)
    """
    score = empyrical_instance.compute_consistency_score(returns_test, ppc)

    ax = _plot_bayes_cone(
        empyrical_instance,
        returns_train,
        returns_test,
        ppc,
        plot_train_len=plot_train_len,
        ax=ax)
    ax.text(
        0.40,
        0.90,
        'Consistency score: %.1f' % score,
        verticalalignment='bottom',
        horizontalalignment='right',
        transform=ax.transAxes,
    )

    ax.set_ylabel('Cumulative returns')
    return score
