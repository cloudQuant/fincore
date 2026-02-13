"""Statistics computation engine for strategy reports.

Computes all metrics, time-series data, and summary text needed by the
HTML / PDF renderers.  This module has **no** rendering logic.
"""

from __future__ import annotations

from collections import OrderedDict

import numpy as np
import pandas as pd


def compute_sections(
    returns,
    benchmark_rets,
    positions,
    transactions,
    trades,
    rolling_window,
):
    """计算所有需要的统计数据，返回 sections dict。"""
    from fincore import Empyrical

    sections = {}

    # ------ 基础信息 ------
    sections["date_range"] = (
        returns.index[0].strftime("%Y-%m-%d"),
        returns.index[-1].strftime("%Y-%m-%d"),
    )
    sections["n_days"] = len(returns)
    sections["n_months"] = int(len(returns) / 21)

    # ------ 核心绩效（与 perf_stats 一致） ------
    perf = OrderedDict()
    perf["Annual Return"] = Empyrical.annual_return(returns)
    perf["Cumulative Returns"] = Empyrical.cum_returns_final(returns)
    perf["Annual Volatility"] = Empyrical.annual_volatility(returns)
    perf["Sharpe Ratio"] = Empyrical.sharpe_ratio(returns)
    perf["Calmar Ratio"] = Empyrical.calmar_ratio(returns)
    perf["Stability"] = Empyrical.stability_of_timeseries(returns)
    perf["Max Drawdown"] = Empyrical.max_drawdown(returns)
    perf["Omega Ratio"] = Empyrical.omega_ratio(returns)
    perf["Sortino Ratio"] = Empyrical.sortino_ratio(returns)
    perf["Skew"] = Empyrical.skewness(returns)
    perf["Kurtosis"] = Empyrical.kurtosis(returns)
    perf["Tail Ratio"] = Empyrical.tail_ratio(returns)
    perf["Daily Value at Risk"] = Empyrical.value_at_risk(returns)
    perf["Downside Risk"] = Empyrical.downside_risk(returns)

    # 日度统计
    perf["Daily Mean Return"] = float(np.nanmean(returns))
    perf["Daily Std Return"] = float(np.nanstd(returns, ddof=1))
    perf["Best Day"] = float(returns.max())
    perf["Worst Day"] = float(returns.min())

    # Benchmark 相关指标也放入核心绩效表
    if benchmark_rets is not None:
        a, b = Empyrical.alpha_beta(returns, benchmark_rets)
        perf["Alpha"] = a
        perf["Beta"] = b

    # 如果有 positions + transactions，计算 turnover
    if positions is not None and transactions is not None:
        try:
            turnover = Empyrical.get_turnover(positions, transactions)
            perf["Avg Daily Turnover"] = float(turnover.mean())
        except Exception:
            pass

    # Gross leverage 统计
    if positions is not None:
        try:
            gl = Empyrical.gross_lev(positions)
            perf["Avg Gross Leverage"] = float(gl.mean())
            perf["Max Gross Leverage"] = float(gl.max())
        except Exception:
            pass

    sections["perf_stats"] = perf

    # ------ 扩展风险指标 ------
    ext = OrderedDict()
    emp = Empyrical(returns=returns)
    ext["Win Rate (daily)"] = emp.win_rate()
    ext["Loss Rate (daily)"] = emp.loss_rate()
    ext["Serial Correlation"] = emp.serial_correlation()
    ext["Common Sense Ratio"] = emp.common_sense_ratio()
    ext["Sterling Ratio"] = emp.sterling_ratio()
    ext["Burke Ratio"] = emp.burke_ratio()
    ext["Kappa Three Ratio"] = emp.kappa_three_ratio()
    ext["Max Drawdown Days"] = emp.max_drawdown_days()
    ext["Max Drawdown Recovery Days"] = emp.max_drawdown_recovery_days()
    ext["2nd Max Drawdown"] = emp.second_max_drawdown()
    ext["3rd Max Drawdown"] = emp.third_max_drawdown()
    ext["Max Consecutive Up Days"] = Empyrical.max_consecutive_up_days(returns)
    ext["Max Consecutive Down Days"] = Empyrical.max_consecutive_down_days(returns)
    ext["Max Single Day Gain"] = Empyrical.max_single_day_gain(returns)
    ext["Max Single Day Loss"] = Empyrical.max_single_day_loss(returns)
    ext["Hurst Exponent"] = Empyrical.hurst_exponent(returns)
    sections["extended_stats"] = ext

    # ------ 时间序列数据 ------
    sections["returns"] = returns
    sections["cum_returns"] = Empyrical.cum_returns(returns, starting_value=1.0)
    cum_ret_0 = Empyrical.cum_returns(returns, starting_value=0)
    running_max = (1 + cum_ret_0).cummax()
    sections["drawdown"] = (1 + cum_ret_0) / running_max - 1
    sections["rolling_sharpe"] = Empyrical.rolling_sharpe(returns, rolling_sharpe_window=rolling_window)
    sections["rolling_volatility"] = Empyrical.rolling_volatility(returns, rolling_vol_window=rolling_window)
    sections["dd_table"] = Empyrical.gen_drawdown_table(returns, top=5)

    # ------ 按年统计 ------
    sections["yearly_stats"] = pd.DataFrame(
        {
            "Annual Return": Empyrical.annual_return_by_year(returns),
            "Sharpe Ratio": Empyrical.sharpe_ratio_by_year(returns),
            "Max Drawdown": Empyrical.max_drawdown_by_year(returns),
        }
    )
    # ------ 按月统计 ------
    sections["monthly_returns"] = Empyrical.aggregate_returns(returns, "monthly")
    # 按月收益统计
    monthly_rets = Empyrical.aggregate_returns(returns, "monthly")
    sections["best_month"] = float(monthly_rets.max())
    sections["worst_month"] = float(monthly_rets.min())
    sections["avg_month"] = float(monthly_rets.mean())
    # 按年收益
    yearly_rets = Empyrical.aggregate_returns(returns, "yearly")
    sections["best_year"] = float(yearly_rets.max())
    sections["worst_year"] = float(yearly_rets.min())

    # ------ 收益分位数 ------
    q = returns.quantile([0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99])
    sections["return_quantiles"] = q

    # ------ Benchmark 相关 ------
    if benchmark_rets is not None:
        bm = OrderedDict()
        bm["Alpha"] = perf["Alpha"]
        bm["Beta"] = perf["Beta"]
        bm["Information Ratio"] = Empyrical.information_ratio(returns, benchmark_rets)
        bm["Tracking Error"] = Empyrical.tracking_error(returns, benchmark_rets)
        bm["Up Capture"] = Empyrical.up_capture(returns, benchmark_rets)
        bm["Down Capture"] = Empyrical.down_capture(returns, benchmark_rets)
        bm["Capture Ratio"] = bm["Up Capture"] / bm["Down Capture"] if bm["Down Capture"] != 0 else np.nan
        bm["Correlation"] = float(returns.corr(benchmark_rets))
        sections["benchmark_stats"] = bm

        sections["benchmark_cum"] = Empyrical.cum_returns(benchmark_rets, starting_value=1.0)
        sections["rolling_beta"] = Empyrical.rolling_beta(
            returns,
            benchmark_rets,
            rolling_window=rolling_window,
        )

    # ------ Positions 相关 ------
    if positions is not None:
        sections["has_positions"] = True
        pos_no_cash = positions.drop("cash", axis=1, errors="ignore")
        sections["positions"] = positions
        sections["pos_no_cash"] = pos_no_cash
        sections["pos_long"] = pos_no_cash.where(pos_no_cash > 0, 0).sum(axis=1)
        sections["pos_short"] = pos_no_cash.where(pos_no_cash < 0, 0).sum(axis=1)
        total = positions.sum(axis=1).replace(0, np.nan)
        exposure = pos_no_cash.abs().sum(axis=1)
        sections["gross_leverage"] = (exposure / total).replace([np.inf, -np.inf], np.nan)

        # 持仓集中度
        pos_abs = pos_no_cash.abs()
        pos_total = pos_abs.sum(axis=1).replace(0, np.nan)
        pos_pct = pos_abs.div(pos_total, axis=0).fillna(0)
        sections["pos_max_concentration"] = pos_pct.max(axis=1)
        sections["pos_median_concentration"] = pos_pct.median(axis=1)

        # 持仓占比
        pos_alloc = pos_no_cash.div(total, axis=0).fillna(0)
        sections["pos_alloc"] = pos_alloc

        # 持仓汇总
        pos_summary = OrderedDict()
        pos_summary["Avg Gross Leverage"] = sections["gross_leverage"].mean()
        pos_summary["Max Gross Leverage"] = sections["gross_leverage"].max()
        pos_summary["Avg Long Exposure"] = sections["pos_long"].mean()
        pos_summary["Avg Short Exposure"] = sections["pos_short"].mean()
        pos_summary["Avg Max Position Concentration"] = sections["pos_max_concentration"].mean()
        pos_summary["Number of Assets"] = len(pos_no_cash.columns)
        sections["position_summary"] = pos_summary

    # ------ Transactions 相关 ------
    if transactions is not None:
        sections["has_transactions"] = True
        txn = transactions.copy()
        txn_norm = txn.copy()
        txn_norm.index = txn_norm.index.normalize()
        sections["daily_txn_count"] = txn_norm.groupby(txn_norm.index).size()
        sections["daily_txn_value"] = (txn_norm["amount"].abs() * txn_norm["price"]).groupby(txn_norm.index).sum()

        # 交易时间分布（小时）
        if hasattr(txn.index, "hour"):
            sections["txn_hours"] = txn.index.hour

        # Turnover（如果有 positions）
        if positions is not None:
            try:
                sections["turnover"] = Empyrical.get_turnover(positions, transactions)
            except Exception:
                pass

        # 交易汇总
        txn_summary = OrderedDict()
        txn_summary["Total Transactions"] = len(transactions)
        txn_summary["Total Transaction Days"] = len(sections["daily_txn_count"])
        txn_summary["Avg Daily Trades"] = float(sections["daily_txn_count"].mean())
        txn_summary["Max Daily Trades"] = int(sections["daily_txn_count"].max())
        txn_summary["Avg Daily Volume"] = float(sections["daily_txn_value"].mean())
        txn_summary["Max Daily Volume"] = float(sections["daily_txn_value"].max())
        if "symbol" in transactions.columns:
            txn_summary["Unique Symbols Traded"] = int(transactions["symbol"].nunique())
        sections["txn_summary"] = txn_summary

    # ------ Trades 相关 ------
    if trades is not None and len(trades) > 0:
        ts = OrderedDict()
        n_trades = len(trades)
        winners = trades[trades["pnlcomm"] > 0]
        losers = trades[trades["pnlcomm"] <= 0]
        n_win = len(winners)
        n_loss = len(losers)

        ts["Total Trades"] = n_trades
        ts["Winning Trades"] = n_win
        ts["Losing Trades"] = n_loss
        ts["Win Rate"] = n_win / n_trades if n_trades > 0 else 0
        ts["Total PnL"] = float(trades["pnlcomm"].sum())
        ts["Avg PnL per Trade"] = float(trades["pnlcomm"].mean())
        ts["Median PnL per Trade"] = float(trades["pnlcomm"].median())
        ts["PnL Std Dev"] = float(trades["pnlcomm"].std())
        ts["Avg Win"] = float(winners["pnlcomm"].mean()) if n_win > 0 else 0
        ts["Avg Loss"] = float(losers["pnlcomm"].mean()) if n_loss > 0 else 0
        ts["Max Win"] = float(winners["pnlcomm"].max()) if n_win > 0 else 0
        ts["Max Loss"] = float(losers["pnlcomm"].min()) if n_loss > 0 else 0
        avg_loss = ts["Avg Loss"]
        ts["Profit/Loss Ratio"] = abs(ts["Avg Win"] / avg_loss) if avg_loss != 0 else np.nan
        # Expectancy
        ts["Expectancy"] = ts["Win Rate"] * ts["Avg Win"] + (1 - ts["Win Rate"]) * ts["Avg Loss"]

        if "commission" in trades.columns:
            ts["Total Commission"] = float(trades["commission"].sum())
            ts["Avg Commission per Trade"] = float(trades["commission"].mean())

        if "long" in trades.columns:
            long_mask = trades["long"] == 1
            short_mask = ~long_mask
            ts["Long Trades"] = int(long_mask.sum())
            ts["Short Trades"] = int(short_mask.sum())
            if long_mask.sum() > 0:
                long_trades = trades[long_mask]
                ts["Long Win Rate"] = float((long_trades["pnlcomm"] > 0).sum() / len(long_trades))
                ts["Long Avg PnL"] = float(long_trades["pnlcomm"].mean())
                ts["Long Total PnL"] = float(long_trades["pnlcomm"].sum())
            if short_mask.sum() > 0:
                short_trades = trades[short_mask]
                ts["Short Win Rate"] = float((short_trades["pnlcomm"] > 0).sum() / len(short_trades))
                ts["Short Avg PnL"] = float(short_trades["pnlcomm"].mean())
                ts["Short Total PnL"] = float(short_trades["pnlcomm"].sum())

        if "barlen" in trades.columns:
            ts["Avg Holding Bars"] = float(trades["barlen"].mean())
            ts["Median Holding Bars"] = float(trades["barlen"].median())
            ts["Max Holding Bars"] = int(trades["barlen"].max())
            ts["Min Holding Bars"] = int(trades["barlen"].min())

        sections["trade_stats"] = ts
        sections["trade_pnl"] = trades["pnlcomm"].values
        if "long" in trades.columns:
            sections["trade_pnl_long"] = trades.loc[trades["long"] == 1, "pnlcomm"].values
            sections["trade_pnl_short"] = trades.loc[trades["long"] == 0, "pnlcomm"].values
        if "barlen" in trades.columns:
            sections["trade_barlen"] = trades["barlen"].values

    # ------ 区间收益 Period Returns ------
    end_date = returns.index[-1]
    _tz = getattr(end_date, "tzinfo", None)
    _ytd_ts = pd.Timestamp(end_date.year, 1, 1, tz=_tz)
    period_defs = [
        ("近一周", 5),
        ("近一月", 21),
        ("近三月", 63),
        ("近六月", 126),
        ("近一年", 252),
        ("近三年", 756),
        ("近五年", 1260),
    ]
    pr = OrderedDict()
    for label, days in period_defs:
        if len(returns) >= days:
            pr[label] = float(Empyrical.cum_returns_final(returns.iloc[-days:]))
        else:
            pr[label] = np.nan
    ytd_mask = returns.index >= _ytd_ts
    if ytd_mask.sum() > 0:
        pr["年初至今"] = float(Empyrical.cum_returns_final(returns[ytd_mask]))
    pr["成立以来"] = float(Empyrical.cum_returns_final(returns))
    sections["period_returns"] = pr

    if benchmark_rets is not None:
        _bm_tz = getattr(benchmark_rets.index[-1], "tzinfo", None)
        _bm_ytd_ts = pd.Timestamp(end_date.year, 1, 1, tz=_bm_tz)
        bpr = OrderedDict()
        for label, days in period_defs:
            if len(benchmark_rets) >= days:
                bpr[label] = float(Empyrical.cum_returns_final(benchmark_rets.iloc[-days:]))
            else:
                bpr[label] = np.nan
        bm_ytd = benchmark_rets[benchmark_rets.index >= _bm_ytd_ts]
        if len(bm_ytd) > 0:
            bpr["年初至今"] = float(Empyrical.cum_returns_final(bm_ytd))
        bpr["成立以来"] = float(Empyrical.cum_returns_final(benchmark_rets))
        sections["benchmark_period_returns"] = bpr

    # ------ 区间胜率 Period Win Rates ------
    wr = OrderedDict()
    for label, days in period_defs:
        if len(returns) >= days:
            r = returns.iloc[-days:]
            wr[label] = float((r > 0).sum() / len(r))
        else:
            wr[label] = np.nan
    ytd_r = returns[ytd_mask]
    if len(ytd_r) > 0:
        wr["年初至今"] = float((ytd_r > 0).sum() / len(ytd_r))
    wr["成立以来"] = float((returns > 0).sum() / len(returns))
    sections["period_win_rates"] = wr

    # ------ 总结文本 Summary Text ------
    _ann = perf.get("Annual Return", np.nan)
    _shp = perf.get("Sharpe Ratio", np.nan)
    _mdd = perf.get("Max Drawdown", np.nan)
    _vol = perf.get("Annual Volatility", np.nan)
    _sor = perf.get("Sortino Ratio", np.nan)
    _cal = perf.get("Calmar Ratio", np.nan)

    def _perf_tag(sh):
        if np.isnan(sh):
            return "N/A"
        return "优秀" if sh > 1.5 else ("良好" if sh > 1.0 else ("一般" if sh > 0.5 else "较差"))

    def _risk_tag(dd):
        if np.isnan(dd):
            return "N/A"
        a = abs(dd)
        return (
            "风险控制优秀"
            if a < 0.1
            else ("风险控制良好" if a < 0.2 else ("风险控制一般" if a < 0.3 else "风险控制较差"))
        )

    _txt = (
        f"报告区间内，产品年化收益率为{_ann * 100:.2f}%，表现{_perf_tag(_shp)}。"
        f"夏普比率为{_shp:.2f}，索提诺比率为{_sor:.2f}，卡尔玛比率为{_cal:.2f}。"
        f"最大回撤为{abs(_mdd) * 100:.2f}%，年化波动率为{_vol * 100:.2f}%，{_risk_tag(_mdd)}。"
    )
    if benchmark_rets is not None:
        _a = perf.get("Alpha", np.nan)
        _b = perf.get("Beta", np.nan)
        _txt += f" Alpha为{_a:.4f}，Beta为{_b:.4f}。"
    sections["summary_text"] = _txt

    return sections
