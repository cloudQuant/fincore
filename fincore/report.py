"""
策略报告生成器 — 根据传入数据动态生成 HTML 或 PDF 策略分析报告。

传入的数据越多，生成的报告越详细：

- **returns** (必需): 基础绩效指标 + 收益图表
- **+ benchmark_rets**: Alpha/Beta、信息比率、跟踪误差、滚动Beta
- **+ positions**: 持仓分析、多空暴露、杠杆率、持仓集中度
- **+ transactions**: 换手率、交易量分析、交易时间分布
- **+ trades**: 交易统计（胜率、盈亏比、多空分解、持仓时长分布）

用法::

    from fincore.report import create_strategy_report

    # 最简单：只传 returns
    create_strategy_report(returns, output="report.html")

    # 完整：传入所有数据
    create_strategy_report(
        returns,
        benchmark_rets=benchmark,
        positions=positions,
        transactions=transactions,
        trades=closed_trades_df,
        title="My Strategy",
        output="report.pdf",
    )
"""

from __future__ import annotations

import json
import warnings
from collections import OrderedDict
from typing import Any, Optional, Union

import numpy as np
import pandas as pd


def create_strategy_report(
    returns: pd.Series,
    *,
    benchmark_rets: pd.Series | None = None,
    positions: pd.DataFrame | None = None,
    transactions: pd.DataFrame | None = None,
    trades: pd.DataFrame | None = None,
    title: str = "Strategy Report",
    output: str = "report.html",
    rolling_window: int = 63,
) -> str:
    """根据传入数据动态生成策略分析报告。

    Parameters
    ----------
    returns : pd.Series
        日收益率序列（必需），DatetimeIndex。
    benchmark_rets : pd.Series, optional
        基准收益率。传入后增加 Alpha/Beta、跟踪误差、滚动 Beta 等分析。
    positions : pd.DataFrame, optional
        每日持仓 DataFrame（列 = 资产名 + 'cash'）。传入后增加持仓分析。
    transactions : pd.DataFrame, optional
        交易记录 DataFrame（需含 amount, price, symbol 列）。传入后增加交易分析。
    trades : pd.DataFrame, optional
        已平仓交易记录（需含 pnlcomm 列，可选 long, barlen, commission 列）。
        传入后增加交易统计（胜率、盈亏比等）。
    title : str
        报告标题。
    output : str
        输出文件路径。以 ``.html`` 结尾生成 HTML，以 ``.pdf`` 结尾生成 PDF。
    rolling_window : int
        滚动指标的窗口大小（交易日），默认 63（约 3 个月）。

    Returns
    -------
    str
        输出文件的路径。
    """
    if output.lower().endswith(".pdf"):
        return _generate_pdf(
            returns,
            benchmark_rets=benchmark_rets,
            positions=positions,
            transactions=transactions,
            trades=trades,
            title=title,
            output=output,
            rolling_window=rolling_window,
        )
    else:
        return _generate_html(
            returns,
            benchmark_rets=benchmark_rets,
            positions=positions,
            transactions=transactions,
            trades=trades,
            title=title,
            output=output,
            rolling_window=rolling_window,
        )


# =========================================================================
# 内部：计算引擎
# =========================================================================


def _compute_sections(
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


# =========================================================================
# HTML 报告生成
# =========================================================================

_HTML_CSS = """\
<style>
:root {
  --primary: #1a365d; --primary-lt: #2b6cb0; --accent: #3182ce;
  --green: #38a169; --red: #e53e3e; --orange: #dd6b20;
  --g50: #f7fafc; --g100: #edf2f7; --g200: #e2e8f0; --g300: #cbd5e0;
  --g500: #718096; --g600: #4a5568; --g700: #2d3748; --sidebar-w: 200px;
}
* { box-sizing: border-box; margin: 0; padding: 0; }
body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
       color: var(--g700); line-height: 1.6; background: var(--g50); }
.sidebar { position: fixed; left: 0; top: 0; bottom: 0; width: var(--sidebar-w);
           background: linear-gradient(180deg, var(--primary), #0d2137); color: #fff;
           padding: 16px 0; overflow-y: auto; z-index: 100; }
.sidebar h2 { font-size: 0.95em; padding: 0 16px 12px;
              border-bottom: 1px solid rgba(255,255,255,.15); margin-bottom: 6px; }
.sidebar a { display: block; padding: 7px 16px; color: rgba(255,255,255,.75);
             text-decoration: none; font-size: 0.82em; transition: .2s; }
.sidebar a:hover { background: rgba(255,255,255,.1); color: #fff;
                   border-left: 3px solid var(--accent); }
.content { margin-left: var(--sidebar-w); padding: 20px 28px; max-width: 1180px; }
.content h1 { font-size: 1.4em; color: var(--primary);
              border-bottom: 3px solid var(--primary); padding-bottom: 6px; margin-bottom: 4px; }
.meta { color: var(--g500); font-size: 0.88em; margin-bottom: 18px; }
.sec { background: #fff; border-radius: 8px; box-shadow: 0 1px 3px rgba(0,0,0,.07);
       padding: 18px 22px; margin-bottom: 18px; }
.sec-title { color: var(--primary); font-size: 1.1em; font-weight: 700; margin-bottom: 14px;
             padding-bottom: 6px; border-bottom: 2px solid var(--g200);
             display: flex; align-items: center; gap: 8px; }
.sec-title::before { content: ''; width: 4px; height: 18px; background: var(--accent);
                     border-radius: 2px; display: inline-block; }
.cards { display: grid; grid-template-columns: repeat(auto-fill, minmax(145px, 1fr));
         gap: 10px; margin-bottom: 16px; }
.card { background: var(--g50); border: 1px solid var(--g200); border-radius: 8px;
        padding: 12px 14px; text-align: center; }
.card .val { font-size: 1.25em; font-weight: 700; margin-bottom: 2px; }
.card .lbl { font-size: 0.76em; color: var(--g500); }
.pos { color: var(--green); } .neg { color: var(--red); }
table { border-collapse: collapse; width: 100%; font-size: 0.85em; margin: 10px 0; }
th, td { border: 1px solid var(--g200); padding: 7px 10px; }
th { background: var(--g100); font-weight: 600; text-align: left; }
td { text-align: right; }
tr:nth-child(even) { background: var(--g50); }
tr:hover { background: #ebf8ff; }
.grid-2 { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }
.chart-box { width: 100%; height: 340px; margin: 10px 0; }
.chart-sm { width: 100%; height: 260px; margin: 10px 0; }
h3.sub { margin: 12px 0 6px; color: var(--primary); font-size: 1em; }
footer { margin-top: 28px; padding: 14px 0; border-top: 1px solid var(--g200);
         font-size: 0.78em; color: var(--g500); text-align: center; }
@media (max-width: 860px) {
  .sidebar { display: none; } .content { margin-left: 0; }
  .grid-2 { grid-template-columns: 1fr; }
  .cards { grid-template-columns: repeat(2, 1fr); }
}
@media print {
  .sidebar { display: none !important; }
  .content { margin-left: 0 !important; max-width: 100% !important; padding: 8px 14px !important; }
  body { background: #fff !important; -webkit-print-color-adjust: exact !important; print-color-adjust: exact !important; }
  .sec-title { page-break-after: avoid; break-after: avoid; }
  .chart-box, .chart-sm { page-break-inside: avoid; break-inside: avoid; }
  .grid-2 { page-break-inside: avoid; break-inside: avoid; }
  table { page-break-inside: avoid; break-inside: avoid; }
  .cards { page-break-inside: avoid; break-inside: avoid; }
  .summary-box { page-break-inside: avoid; break-inside: avoid; }
  h1, .meta { page-break-after: avoid; break-after: avoid; }
  h3.sub { page-break-after: avoid; break-after: avoid; }
}
.summary-box { background: linear-gradient(135deg, #ebf8ff 0%, #f0fff4 100%);
               border-left: 4px solid var(--accent); border-radius: 6px;
               padding: 14px 18px; margin-bottom: 16px; font-size: 0.92em;
               line-height: 1.7; color: var(--g700); }
.ptbl { font-size: 0.83em; }
.ptbl th { background: var(--primary); color: #fff; text-align: center; padding: 8px 6px; }
.ptbl td { text-align: center; padding: 7px 6px; }
.ptbl tr:first-child td { font-weight: 700; }
.card-hl { border-top: 3px solid var(--accent); }
.card-hl.cg { border-top-color: var(--green); }
.card-hl.cr { border-top-color: var(--red); }
.card-hl.co { border-top-color: var(--orange); }
.grid-3 { display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 14px; }
@media (max-width: 860px) { .grid-3 { grid-template-columns: 1fr; } }
.tbl-left td { text-align: left; }
</style>
"""


def _fmt(v, pct=False):
    """格式化数值。"""
    if isinstance(v, (int, np.integer)):
        return f"{v:,}" if abs(v) >= 10000 else str(v)
    if isinstance(v, (float, np.floating)):
        if np.isnan(v):
            return "N/A"
        if pct:
            return f"{v * 100:.2f}%"
        a = abs(v)
        if a >= 1e6:
            return f"{v:,.0f}"
        if a >= 100:
            return f"{v:,.2f}"
        return f"{v:.4f}"
    return str(v)


def _css_cls(v):
    """Return CSS class for positive/negative coloring."""
    if isinstance(v, (int, float, np.integer, np.floating)):
        try:
            if np.isnan(v):
                return ""
        except (TypeError, ValueError):
            pass
        return "pos" if v > 0 else ("neg" if v < 0 else "")
    return ""


def _html_table(d, pct_keys=None):
    """OrderedDict → HTML table."""
    pct_keys = set(pct_keys or [])
    rows = []
    for k, v in d.items():
        css = _css_cls(v)
        rows.append(f'<tr><th>{k}</th><td class="{css}">{_fmt(v, pct=k in pct_keys)}</td></tr>')
    return "<table>" + "".join(rows) + "</table>"


def _html_df(df, float_format=".4f", table_class="", left_align=False):
    """DataFrame → HTML table."""
    cls_attr = f' class="{table_class}"' if table_class else ""
    _td_sty = ' style="text-align:left"' if left_align else ""
    hdr = "<tr><th></th>" + "".join(f"<th>{c}</th>" for c in df.columns) + "</tr>"
    rows = [hdr]
    for idx, row in df.iterrows():
        cells = f"<th>{idx}</th>"
        for v in row:
            if isinstance(v, (float, np.floating)):
                if np.isnan(v):
                    cells += f"<td{_td_sty}></td>"
                else:
                    cells += f'<td class="{_css_cls(v)}"{_td_sty}>{v:{float_format}}</td>'
            else:
                cells += f"<td{_td_sty}>{v}</td>"
        rows.append(f"<tr>{cells}</tr>")
    return f"<table{cls_attr}>" + "".join(rows) + "</table>"


def _html_cards(d, keys, pct_keys=None):
    """Render metric cards."""
    pct_keys = set(pct_keys or [])
    cards = []
    for k in keys:
        v = d.get(k, np.nan)
        css = _css_cls(v)
        cards.append(
            f'<div class="card"><div class="val {css}">{_fmt(v, pct=k in pct_keys)}</div>'
            f'<div class="lbl">{k}</div></div>'
        )
    return '<div class="cards">' + "".join(cards) + "</div>"


def _safe_list(arr, decimals=6, pct=False):
    """Convert numpy array to JSON-safe list."""
    factor = 100.0 if pct else 1.0
    out = []
    for v in np.asanyarray(arr, dtype=float):
        if np.isnan(v) or np.isinf(v):
            out.append(None)
        else:
            out.append(round(float(v) * factor, decimals))
    return out


def _date_list(index):
    """Convert DatetimeIndex to list of 'YYYY-MM-DD' strings."""
    return [d.strftime("%Y-%m-%d") for d in index]


def _generate_html(returns, benchmark_rets, positions, transactions, trades, title, output, rolling_window):
    """生成交互式 HTML 报告（使用 ECharts 图表 + 侧边栏导航）。"""
    s = _compute_sections(returns, benchmark_rets, positions, transactions, trades, rolling_window)

    # ---- chart data ----
    cd = {}
    dates = _date_list(s["cum_returns"].index)
    cd["dates"] = dates
    cd["cumRet"] = _safe_list(s["cum_returns"])
    cd["dd"] = _safe_list(s["drawdown"], decimals=4, pct=True)
    cd["dailyRet"] = _safe_list(s["returns"], decimals=4, pct=True)

    rs = s["rolling_sharpe"].dropna()
    cd["rsDates"], cd["rsVals"] = _date_list(rs.index), _safe_list(rs, decimals=4)
    rv = s["rolling_volatility"].dropna()
    cd["rvDates"], cd["rvVals"] = _date_list(rv.index), _safe_list(rv, decimals=4, pct=True)

    if "benchmark_cum" in s:
        cd["benchCum"] = _safe_list(s["benchmark_cum"])
    if "rolling_beta" in s:
        rb = s["rolling_beta"].dropna()
        cd["rbDates"], cd["rbVals"] = _date_list(rb.index), _safe_list(rb, decimals=4)

    monthly_tbl = s["monthly_returns"].unstack()
    hm_months = [f"{int(c):02d}" for c in monthly_tbl.columns]
    hm_years = [str(y) for y in monthly_tbl.index]
    hm_data = []
    for yi, y in enumerate(monthly_tbl.index):
        for mi, m in enumerate(monthly_tbl.columns):
            val = monthly_tbl.loc[y, m]
            if not (isinstance(val, (float, np.floating)) and np.isnan(val)):
                hm_data.append([mi, yi, round(float(val) * 100, 2)])
    cd["hmMonths"], cd["hmYears"], cd["hmData"] = hm_months, hm_years, hm_data

    ys = s["yearly_stats"]
    cd["yrLabels"] = [str(y) for y in ys.index]
    cd["yrRets"] = [round(float(v) * 100, 2) for v in ys["Annual Return"].values]

    ret_vals = s["returns"].values
    hist, edges = np.histogram(ret_vals * 100, bins=min(60, max(10, len(ret_vals) // 5)))
    cd["histBins"] = [round(float((edges[i] + edges[i + 1]) / 2), 3) for i in range(len(hist))]
    cd["histCnts"] = [int(v) for v in hist]

    if "has_positions" in s:
        cd["posDates"] = _date_list(s["pos_long"].index)
        cd["posLong"] = _safe_list(s["pos_long"], decimals=2)
        cd["posShort"] = _safe_list(s["pos_short"], decimals=2)
        gl = s["gross_leverage"].dropna()
        cd["glDates"], cd["glVals"] = _date_list(gl.index), _safe_list(gl, decimals=4)

    if "has_transactions" in s:
        dv = s["daily_txn_value"]
        cd["txnDates"], cd["txnVol"] = _date_list(dv.index), _safe_list(dv, decimals=2)
        dc = s["daily_txn_count"]
        cd["txnCntDates"], cd["txnCnts"] = _date_list(dc.index), [int(v) for v in dc.values]
        if "turnover" in s:
            to = s["turnover"].dropna()
            cd["toDates"], cd["toVals"] = _date_list(to.index), _safe_list(to, decimals=4)
        if "txn_hours" in s:
            hours = s["txn_hours"]
            hh, he = np.histogram(hours, bins=range(25))
            cd["txnHrLabels"] = [f"{int(he[i]):02d}" for i in range(len(hh))]
            cd["txnHrCnts"] = [int(v) for v in hh]

    if "trade_pnl" in s:
        pnl = s["trade_pnl"]
        ph, pe = np.histogram(pnl, bins=max(10, min(40, len(pnl) // 3)))
        cd["pnlBins"] = [round(float((pe[i] + pe[i + 1]) / 2), 2) for i in range(len(ph))]
        cd["pnlCnts"] = [int(v) for v in ph]
        if "trade_pnl_long" in s and len(s["trade_pnl_long"]) > 0:
            pl = s["trade_pnl_long"]
            plh, ple = np.histogram(pl, bins=max(5, min(30, len(pl) // 3)))
            cd["pnlLBins"] = [round(float((ple[i] + ple[i + 1]) / 2), 2) for i in range(len(plh))]
            cd["pnlLCnts"] = [int(v) for v in plh]
        if "trade_pnl_short" in s and len(s["trade_pnl_short"]) > 0:
            ps = s["trade_pnl_short"]
            psh, pse = np.histogram(ps, bins=max(5, min(30, len(ps) // 3)))
            cd["pnlSBins"] = [round(float((pse[i] + pse[i + 1]) / 2), 2) for i in range(len(psh))]
            cd["pnlSCnts"] = [int(v) for v in psh]
        if "trade_barlen" in s:
            bl = s["trade_barlen"]
            bh, be = np.histogram(bl, bins=max(10, min(50, len(bl) // 3)))
            cd["blBins"] = [round(float((be[i] + be[i + 1]) / 2), 1) for i in range(len(bh))]
            cd["blCnts"] = [int(v) for v in bh]

    # Position allocation data (for stacked area chart, max 20 assets)
    if "pos_alloc" in s:
        pa = s["pos_alloc"]
        if len(pa.columns) <= 20:
            cd["paDates"] = _date_list(pa.index)
            cd["paNames"] = list(pa.columns)
            cd["paData"] = {str(c): _safe_list(pa[c], decimals=4) for c in pa.columns}

    # Return quantiles data
    q = s["return_quantiles"]
    cd["quantLabels"] = [str(p) for p in q.index]
    cd["quantVals"] = [round(float(v) * 100, 4) for v in q.values]

    # Monthly returns distribution
    monthly_vals = s["monthly_returns"].dropna().values
    if len(monthly_vals) > 2:
        mh, me = np.histogram(monthly_vals * 100, bins=min(30, max(5, len(monthly_vals) // 2)))
        cd["mDistBins"] = [round(float((me[i] + me[i + 1]) / 2), 2) for i in range(len(mh))]
        cd["mDistCnts"] = [int(v) for v in mh]

    # Period returns data
    pr = s["period_returns"]
    cd["prLabels"] = list(pr.keys())
    cd["prVals"] = [
        round(float(v) * 100, 2) if not (isinstance(v, float) and np.isnan(v)) else None for v in pr.values()
    ]
    if "benchmark_period_returns" in s:
        bpr = s["benchmark_period_returns"]
        cd["bprVals"] = [
            round(float(bpr.get(k, np.nan)) * 100, 2) if not np.isnan(bpr.get(k, np.nan)) else None for k in pr
        ]

    # Position concentration data
    if "pos_max_concentration" in s:
        mc = s["pos_max_concentration"]
        cd["mcDates"] = _date_list(mc.index)
        cd["mcMax"] = _safe_list(mc, decimals=4, pct=True)
        cd["mcMed"] = _safe_list(s["pos_median_concentration"], decimals=4, pct=True)

    # ---- sidebar ----
    nav = [
        ("overview", "产品概览"),
        ("period", "区间收益"),
        ("performance", "绩效分析"),
        ("returns", "收益分析"),
        ("rolling", "滚动指标"),
        ("drawdown", "回撤分析"),
    ]
    if "benchmark_stats" in s:
        nav.append(("benchmark", "基准对比"))
    if "has_positions" in s:
        nav.append(("positions", "持仓分析"))
    if "has_transactions" in s:
        nav.append(("transactions", "交易分析"))
    if "trade_stats" in s:
        nav.append(("trades", "交易统计"))
    sidebar = '<nav class="sidebar"><h2>📊 Report</h2>'
    for aid, label in nav:
        sidebar += f'<a href="#{aid}">{label}</a>'
    sidebar += "</nav>"

    # ---- body sections ----
    pct_perf = {
        "Annual Return",
        "Cumulative Returns",
        "Annual Volatility",
        "Max Drawdown",
        "Downside Risk",
        "Daily Value at Risk",
        "Daily Mean Return",
        "Daily Std Return",
        "Best Day",
        "Worst Day",
        "Avg Daily Turnover",
    }
    b = []  # body parts

    b.append(f"<h1>{title}</h1>")
    b.append(
        f'<div class="meta">{s["date_range"][0]} → {s["date_range"][1]}'
        f" | {s['n_days']} 交易日 | ~{s['n_months']} 个月</div>"
    )

    # -- Summary --
    b.append(f'<div class="summary-box">{s["summary_text"]}</div>')

    # -- Overview --
    b.append('<div class="sec" id="overview"><div class="sec-title">产品概览 Overview</div>')
    b.append(
        _html_cards(
            s["perf_stats"],
            [
                "Sharpe Ratio",
                "Annual Return",
                "Max Drawdown",
                "Annual Volatility",
                "Sortino Ratio",
                "Calmar Ratio",
                "Omega Ratio",
                "Stability",
            ],
            pct_keys={"Annual Return", "Max Drawdown", "Annual Volatility"},
        )
    )
    b.append('<div class="chart-box" id="c-cum"></div>')
    b.append('<div class="chart-sm" id="c-cum-log"></div>')
    b.append('<div class="chart-sm" id="c-dd"></div>')
    b.append("</div>")

    # -- Period Returns --
    b.append('<div class="sec" id="period"><div class="sec-title">区间收益 Period Returns</div>')
    b.append('<div class="chart-sm" id="c-period"></div>')
    _pr = s["period_returns"]
    _has_bpr = "benchmark_period_returns" in s
    _wr = s["period_win_rates"]
    _phdr = "<tr><th>统计项</th>" + "".join(f"<th>{k}</th>" for k in _pr) + "</tr>"
    _prow1 = '<tr><td style="text-align:left;font-weight:600">本产品</td>'
    for _k, _v in _pr.items():
        _prow1 += f'<td class="{_css_cls(_v)}">{_fmt(_v, pct=True)}</td>'
    _prow1 += "</tr>"
    _prows = _phdr + _prow1
    if _has_bpr:
        _bpr = s["benchmark_period_returns"]
        _prow2 = '<tr><td style="text-align:left;font-weight:600">基准</td>'
        for _k in _pr:
            _bv = _bpr.get(_k, np.nan)
            _prow2 += f'<td class="{_css_cls(_bv)}">{_fmt(_bv, pct=True)}</td>'
        _prow2 += "</tr>"
        _prow3 = '<tr><td style="text-align:left;font-weight:600">超额收益</td>'
        for _k in _pr:
            _sv = _pr.get(_k, np.nan)
            _bv2 = _bpr.get(_k, np.nan)
            _exc = (_sv - _bv2) if not (np.isnan(_sv) or np.isnan(_bv2)) else np.nan
            _prow3 += f'<td class="{_css_cls(_exc)}">{_fmt(_exc, pct=True)}</td>'
        _prow3 += "</tr>"
        _prows += _prow2 + _prow3
    _prow_wr = '<tr><td style="text-align:left;font-weight:600">日胜率</td>'
    for _k in _pr:
        _wv = _wr.get(_k, np.nan)
        _prow_wr += f"<td>{_fmt(_wv, pct=True)}</td>"
    _prow_wr += "</tr>"
    _prows += _prow_wr
    b.append(f'<table class="ptbl">{_prows}</table>')
    b.append("</div>")

    # -- Performance --
    b.append('<div class="sec" id="performance"><div class="sec-title">绩效分析 Performance</div>')
    b.append('<div class="grid-2"><div>')
    b.append('<h3 class="sub">核心指标</h3>')
    b.append(_html_table(s["perf_stats"], pct_keys=pct_perf))
    b.append("</div><div>")
    b.append('<h3 class="sub">扩展风险指标</h3>')
    b.append(_html_table(s["extended_stats"]))
    b.append("</div></div></div>")

    # -- Returns --
    b.append('<div class="sec" id="returns"><div class="sec-title">收益分析 Returns</div>')
    b.append('<div class="grid-2">')
    b.append('<div class="chart-sm" id="c-daily"></div>')
    b.append('<div class="chart-sm" id="c-dist"></div>')
    b.append("</div>")
    b.append('<div class="grid-2">')
    b.append('<div class="chart-sm" id="c-quant"></div>')
    b.append('<div class="chart-sm" id="c-mdist"></div>')
    b.append("</div>")
    b.append('<div class="chart-box" id="c-hm" style="height:280px"></div>')
    b.append('<h3 class="sub">月度收益 (%)</h3>')
    mt = monthly_tbl.copy()
    mt.columns = hm_months
    b.append(_html_df(mt * 100, float_format=".2f"))
    b.append('<div class="chart-sm" id="c-yr"></div>')
    b.append('<h3 class="sub">年度统计</h3>')
    b.append(_html_df(s["yearly_stats"], left_align=True))
    extremes = OrderedDict()
    extremes["Best Month"] = s["best_month"]
    extremes["Worst Month"] = s["worst_month"]
    extremes["Avg Month"] = s["avg_month"]
    extremes["Best Year"] = s["best_year"]
    extremes["Worst Year"] = s["worst_year"]
    b.append('<h3 class="sub">月/年极值</h3>')
    b.append(_html_table(extremes, pct_keys=set(extremes.keys())))
    b.append("</div>")

    # -- Rolling --
    b.append(f'<div class="sec" id="rolling"><div class="sec-title">滚动指标 Rolling ({rolling_window}d)</div>')
    b.append('<div class="chart-box" id="c-rs"></div>')
    b.append('<div class="chart-sm" id="c-rv"></div>')
    b.append("</div>")

    # -- Drawdown --
    b.append('<div class="sec" id="drawdown"><div class="sec-title">回撤分析 Drawdown</div>')
    b.append('<h3 class="sub">最大回撤区间</h3>')
    b.append(_html_df(s["dd_table"], float_format=".2f"))
    b.append("</div>")

    # -- Benchmark --
    if "benchmark_stats" in s:
        b.append('<div class="sec" id="benchmark"><div class="sec-title">基准对比 Benchmark</div>')
        b.append(
            _html_cards(
                s["benchmark_stats"],
                [
                    "Alpha",
                    "Beta",
                    "Information Ratio",
                    "Tracking Error",
                    "Up Capture",
                    "Down Capture",
                    "Capture Ratio",
                    "Correlation",
                ],
            )
        )
        b.append(_html_table(s["benchmark_stats"]))
        if "rolling_beta" in s:
            b.append('<div class="chart-sm" id="c-rb"></div>')
        b.append("</div>")

    # -- Positions --
    if "has_positions" in s:
        b.append('<div class="sec" id="positions"><div class="sec-title">持仓分析 Positions</div>')
        b.append(_html_cards(s["position_summary"], list(s["position_summary"].keys())))
        b.append(_html_table(s["position_summary"]))
        b.append('<div class="chart-box" id="c-expo"></div>')
        b.append('<div class="grid-2">')
        b.append('<div class="chart-sm" id="c-lev"></div>')
        b.append('<div class="chart-sm" id="c-conc"></div>')
        b.append("</div>")
        if "pos_alloc" in s and len(s["pos_alloc"].columns) <= 20:
            b.append('<div class="chart-box" id="c-alloc"></div>')
        b.append("</div>")

    # -- Transactions --
    if "has_transactions" in s:
        b.append('<div class="sec" id="transactions"><div class="sec-title">交易分析 Transactions</div>')
        b.append(_html_cards(s["txn_summary"], list(s["txn_summary"].keys())))
        b.append(_html_table(s["txn_summary"]))
        b.append('<div class="grid-2">')
        b.append('<div class="chart-sm" id="c-txn"></div>')
        b.append('<div class="chart-sm" id="c-txn-cnt"></div>')
        b.append("</div>")
        if "turnover" in s:
            b.append('<div class="chart-sm" id="c-turnover"></div>')
        if "txn_hours" in s:
            b.append('<div class="chart-sm" id="c-txn-hours"></div>')
        b.append("</div>")

    # -- Trades --
    if "trade_stats" in s:
        ts = s["trade_stats"]
        pct_t = {"Win Rate", "Long Win Rate", "Short Win Rate"}
        b.append('<div class="sec" id="trades"><div class="sec-title">交易统计 Trades</div>')
        b.append(
            _html_cards(
                ts,
                ["Total Trades", "Win Rate", "Profit/Loss Ratio", "Total PnL", "Expectancy", "Avg PnL per Trade"],
                pct_keys=pct_t,
            )
        )
        b.append(_html_table(ts, pct_keys=pct_t))
        if "trade_pnl" in s:
            b.append('<div class="chart-sm" id="c-pnl"></div>')
            if "trade_pnl_long" in s or "trade_pnl_short" in s:
                b.append('<div class="grid-2">')
                b.append('<div class="chart-sm" id="c-pnl-long"></div>')
                b.append('<div class="chart-sm" id="c-pnl-short"></div>')
                b.append("</div>")
        if "trade_barlen" in s:
            b.append('<div class="chart-sm" id="c-barlen"></div>')
        b.append("</div>")

    b.append("<footer>Generated by <strong>fincore</strong> | create_strategy_report()</footer>")
    body_html = "\n".join(b)

    # ---- ECharts JavaScript ----
    js_parts = _build_echart_js(s, rolling_window)
    js = "\n".join(js_parts)

    # Deduplicate date arrays to reduce JSON size
    _dedup_aliases = []
    _base_dates = cd.get("dates")
    for _dk in ("posDates", "glDates", "mcDates"):
        if _dk in cd and cd[_dk] == _base_dates:
            del cd[_dk]
            _dedup_aliases.append(f"D.{_dk}=D.dates;")
    if "rvDates" in cd and "rsDates" in cd and cd["rvDates"] == cd["rsDates"]:
        del cd["rvDates"]
        _dedup_aliases.append("D.rvDates=D.rsDates;")
    chart_json = json.dumps(cd, ensure_ascii=False)
    _alias_js = "\n".join(_dedup_aliases)

    html = (
        f"<!DOCTYPE html>\n<html lang='zh'><head><meta charset='utf-8'>\n"
        f"<title>{title}</title>\n"
        f"{_HTML_CSS}\n</head>\n<body>\n"
        f"{sidebar}\n<main class='content'>\n{body_html}\n</main>\n"
        f'<script src="https://cdn.bootcdn.net/ajax/libs/echarts/5.5.0/echarts.min.js"></script>\n'
        f"<script>window.echarts||document.write('<script src=\"https://cdnjs.cloudflare.com/ajax/libs/echarts/5.5.0/echarts.min.js\"><\\/script>')</script>\n"
        f"<script>window.echarts||document.write('<script src=\"https://cdn.jsdelivr.net/npm/echarts@5.5.0/dist/echarts.min.js\"><\\/script>')</script>\n"
        f"<script>\nvar D={chart_json};\n{_alias_js}\n"
        f"var _charts=[];\n"
        f"function C(id,o){{\n"
        f"  var e=document.getElementById(id);if(!e)return;\n"
        f"  var c=echarts.init(e);c.setOption(o);_charts.push(c);\n"
        f"}}\n"
        f"window.addEventListener('resize',function(){{_charts.forEach(function(c){{c.resize()}})}});\n"
        f"var B='#3182ce',G='#38a169',R='#e53e3e',O='#dd6b20',P='#805ad5',GY='#a0aec0';\n"
        f"{js}\n</script>\n</body></html>"
    )

    with open(output, "w", encoding="utf-8") as f:
        f.write(html)

    return output


def _build_echart_js(s, rw):
    """Build ECharts initialization JavaScript statements."""
    js = []
    _grid = "grid:{left:60,right:30,bottom:30,top:50}"
    _grid_s = "grid:{left:55,right:15,bottom:25,top:45}"
    _zoom = "dataZoom:[{type:'inside'},{type:'slider',height:18,bottom:4}]"
    _zoom_s = "dataZoom:[{type:'inside'}]"

    # Period returns comparison
    bench_period = ""
    if "benchmark_period_returns" in s:
        bench_period = ",{name:'基准',type:'bar',data:D.bprVals,barWidth:'30%',itemStyle:{color:GY}}"
    js.append(
        f"C('c-period',{{"
        f"title:{{text:'区间收益对比 (%)',textStyle:{{fontSize:12}}}},"
        f"tooltip:{{trigger:'axis',valueFormatter:function(v){{return v==null?'N/A':v.toFixed(2)+'%'}}}},"
        f"legend:{{top:4,right:10}},{_grid_s},"
        f"xAxis:{{type:'category',data:D.prLabels,axisLabel:{{fontSize:10}}}},"
        f"yAxis:{{type:'value'}},"
        f"series:[{{name:'本产品',type:'bar',data:D.prVals,barWidth:'30%',"
        f"itemStyle:{{color:B}}}}"
        f"{bench_period}]"
        f"}});"
    )

    # Cumulative returns
    bench_series = ""
    if "benchmark_cum" in s:
        bench_series = (
            ",{name:'Benchmark',type:'line',data:D.benchCum,showSymbol:false,"
            "lineStyle:{width:1,color:GY,type:'dashed'}}"
        )
    js.append(
        f"C('c-cum',{{"
        f"title:{{text:'累计收益 Cumulative Returns',textStyle:{{fontSize:13}}}},"
        f"tooltip:{{trigger:'axis'}},legend:{{top:4,right:10}},{_grid},"
        f"xAxis:{{type:'category',data:D.dates,axisLabel:{{fontSize:10}}}},"
        f"yAxis:{{type:'value',axisLabel:{{fontSize:10}}}},"
        f"{_zoom},"
        f"series:[{{name:'Strategy',type:'line',data:D.cumRet,showSymbol:false,"
        f"lineStyle:{{width:1.5,color:B}},"
        f"areaStyle:{{color:{{type:'linear',x:0,y:0,x2:0,y2:1,"
        f"colorStops:[{{offset:0,color:'rgba(49,130,206,0.15)'}},"
        f"{{offset:1,color:'rgba(49,130,206,0.01)'}}]}}}}}}"
        f"{bench_series}]"
        f"}});"
    )

    # Cumulative returns (log scale)
    bench_series_log = ""
    if "benchmark_cum" in s:
        bench_series_log = (
            ",{name:'Benchmark',type:'line',data:D.benchCum,showSymbol:false,"
            "lineStyle:{width:1,color:GY,type:'dashed'}}"
        )
    js.append(
        f"C('c-cum-log',{{"
        f"title:{{text:'累计收益 (对数坐标)',textStyle:{{fontSize:12}}}},"
        f"tooltip:{{trigger:'axis'}},legend:{{top:4,right:10}},{_grid_s},"
        f"xAxis:{{type:'category',data:D.dates,axisLabel:{{fontSize:10}}}},"
        f"yAxis:{{type:'log',axisLabel:{{fontSize:10}}}},"
        f"{_zoom_s},"
        f"series:[{{name:'Strategy',type:'line',data:D.cumRet,showSymbol:false,"
        f"lineStyle:{{width:1.2,color:B}}}}"
        f"{bench_series_log}]"
        f"}});"
    )

    # Drawdown
    js.append(
        f"C('c-dd',{{"
        f"title:{{text:'回撤 Drawdown (%)',textStyle:{{fontSize:12}}}},"
        f"tooltip:{{trigger:'axis',valueFormatter:function(v){{return v.toFixed(2)+'%'}}}},"
        f"{_grid_s},"
        f"xAxis:{{type:'category',data:D.dates,axisLabel:{{fontSize:10}}}},"
        f"yAxis:{{type:'value',axisLabel:{{fontSize:10}}}},"
        f"{_zoom_s},"
        f"series:[{{type:'line',data:D.dd,showSymbol:false,"
        f"lineStyle:{{width:1,color:R}},areaStyle:{{color:'rgba(229,62,62,0.2)'}}}}]"
        f"}});"
    )

    # Daily returns
    js.append(
        f"C('c-daily',{{"
        f"title:{{text:'日收益率 (%)',textStyle:{{fontSize:12}}}},"
        f"tooltip:{{trigger:'axis',valueFormatter:function(v){{return v.toFixed(3)+'%'}}}},"
        f"{_grid_s},"
        f"xAxis:{{type:'category',data:D.dates,axisLabel:{{show:false}}}},"
        f"yAxis:{{type:'value',axisLabel:{{fontSize:10}}}},"
        f"series:[{{type:'bar',data:D.dailyRet,barWidth:'60%',"
        f"itemStyle:{{color:function(p){{return p.value>=0?G:R}}}}}}]"
        f"}});"
    )

    # Return distribution
    js.append(
        f"C('c-dist',{{"
        f"title:{{text:'收益分布 (%)',textStyle:{{fontSize:12}}}},"
        f"tooltip:{{trigger:'axis'}},"
        f"{_grid_s},"
        f"xAxis:{{type:'category',data:D.histBins,axisLabel:{{fontSize:9}}}},"
        f"yAxis:{{type:'value',axisLabel:{{fontSize:10}}}},"
        f"series:[{{type:'bar',data:D.histCnts,barWidth:'80%',"
        f"itemStyle:{{color:B,borderRadius:[2,2,0,0]}}}}]"
        f"}});"
    )

    # Return quantiles horizontal bar
    js.append(
        "C('c-quant',{"
        "title:{text:'收益分位数 (%)',textStyle:{fontSize:12}},"
        "tooltip:{trigger:'axis',valueFormatter:function(v){return v.toFixed(4)+'%'}},"
        "grid:{left:80,right:20,bottom:25,top:45},"
        "yAxis:{type:'category',data:D.quantLabels,axisLabel:{fontSize:9}},"
        "xAxis:{type:'value'},"
        "series:[{type:'bar',data:D.quantVals,barWidth:'60%',"
        "itemStyle:{color:function(p){return p.value>=0?G:R}}}]"
        "});"
    )

    # Monthly returns distribution
    monthly_vals = s["monthly_returns"].dropna().values
    if len(monthly_vals) > 2:
        js.append(
            f"C('c-mdist',{{"
            f"title:{{text:'月度收益分布 (%)',textStyle:{{fontSize:12}}}},"
            f"tooltip:{{trigger:'axis'}},"
            f"{_grid_s},"
            f"xAxis:{{type:'category',data:D.mDistBins,axisLabel:{{fontSize:9}}}},"
            f"yAxis:{{type:'value'}},"
            f"series:[{{type:'bar',data:D.mDistCnts,barWidth:'80%',"
            f"itemStyle:{{color:B,borderRadius:[2,2,0,0]}}}}]"
            f"}});"
        )

    # Monthly heatmap
    js.append(
        "C('c-hm',{"
        "title:{text:'月度收益热力图 (%)',textStyle:{fontSize:12}},"
        "tooltip:{formatter:function(p){return D.hmYears[p.value[1]]+'-'+D.hmMonths[p.value[0]]+': '+p.value[2]+'%'}},"
        "grid:{left:60,right:80,bottom:20,top:35},"
        "xAxis:{type:'category',data:D.hmMonths,splitArea:{show:true}},"
        "yAxis:{type:'category',data:D.hmYears,splitArea:{show:true}},"
        "visualMap:{min:-10,max:10,calculable:true,orient:'vertical',right:5,top:'center',"
        "inRange:{color:['#c53030','#fc8181','#fff5f5','#f0fff4','#68d391','#276749']},"
        "textStyle:{fontSize:10}},"
        "series:[{type:'heatmap',data:D.hmData,label:{show:true,fontSize:9,"
        "formatter:function(p){return p.value[2].toFixed(1)}}}]"
        "});"
    )

    # Yearly returns
    js.append(
        f"C('c-yr',{{"
        f"title:{{text:'年度收益 (%)',textStyle:{{fontSize:12}}}},"
        f"tooltip:{{trigger:'axis',valueFormatter:function(v){{return v.toFixed(2)+'%'}}}},"
        f"{_grid_s},"
        f"xAxis:{{type:'category',data:D.yrLabels}},"
        f"yAxis:{{type:'value'}},"
        f"series:[{{type:'bar',data:D.yrRets,barWidth:'50%',"
        f"itemStyle:{{color:function(p){{return p.value>=0?G:R}}}}}}]"
        f"}});"
    )

    # Rolling Sharpe
    js.append(
        f"C('c-rs',{{"
        f"title:{{text:'滚动夏普比率 ({rw}d)',textStyle:{{fontSize:12}}}},"
        f"tooltip:{{trigger:'axis'}},{_grid},{_zoom_s},"
        f"xAxis:{{type:'category',data:D.rsDates,axisLabel:{{fontSize:10}}}},"
        f"yAxis:{{type:'value'}},"
        f"series:[{{type:'line',data:D.rsVals,showSymbol:false,"
        f"lineStyle:{{width:1.2,color:B}},"
        f"areaStyle:{{color:{{type:'linear',x:0,y:0,x2:0,y2:1,"
        f"colorStops:[{{offset:0,color:'rgba(49,130,206,0.12)'}},"
        f"{{offset:1,color:'rgba(49,130,206,0.01)'}}]}}}},"
        f"markLine:{{data:[{{yAxis:0,lineStyle:{{color:GY,type:'dashed'}}}}]}}}}]"
        f"}});"
    )

    # Rolling Volatility
    js.append(
        f"C('c-rv',{{"
        f"title:{{text:'滚动波动率 ({rw}d, %)',textStyle:{{fontSize:12}}}},"
        f"tooltip:{{trigger:'axis',valueFormatter:function(v){{return v.toFixed(2)+'%'}}}},"
        f"{_grid_s},{_zoom_s},"
        f"xAxis:{{type:'category',data:D.rvDates,axisLabel:{{fontSize:10}}}},"
        f"yAxis:{{type:'value'}},"
        f"series:[{{type:'line',data:D.rvVals,showSymbol:false,"
        f"lineStyle:{{width:1.2,color:O}},areaStyle:{{color:'rgba(221,107,32,0.12)'}}}}]"
        f"}});"
    )

    # Rolling Beta
    if "rolling_beta" in s:
        js.append(
            f"C('c-rb',{{"
            f"title:{{text:'滚动 Beta ({rw}d)',textStyle:{{fontSize:12}}}},"
            f"tooltip:{{trigger:'axis'}},{_grid_s},{_zoom_s},"
            f"xAxis:{{type:'category',data:D.rbDates,axisLabel:{{fontSize:10}}}},"
            f"yAxis:{{type:'value'}},"
            f"series:[{{type:'line',data:D.rbVals,showSymbol:false,"
            f"lineStyle:{{width:1.2,color:P}},"
            f"markLine:{{data:[{{yAxis:1.0,lineStyle:{{color:GY,type:'dashed'}}}}]}}}}]"
            f"}});"
        )

    # Position exposure
    if "has_positions" in s:
        js.append(
            f"C('c-expo',{{"
            f"title:{{text:'多空暴露 Long/Short Exposure',textStyle:{{fontSize:12}}}},"
            f"tooltip:{{trigger:'axis'}},legend:{{top:4,right:10}},{_grid},{_zoom_s},"
            f"xAxis:{{type:'category',data:D.posDates,axisLabel:{{fontSize:10}}}},"
            f"yAxis:{{type:'value'}},"
            f"series:["
            f"{{name:'Long',type:'line',data:D.posLong,showSymbol:false,"
            f"lineStyle:{{width:1,color:G}},areaStyle:{{color:'rgba(56,161,105,0.2)'}}}},"
            f"{{name:'Short',type:'line',data:D.posShort,showSymbol:false,"
            f"lineStyle:{{width:1,color:R}},areaStyle:{{color:'rgba(229,62,62,0.2)'}}}}]"
            f"}});"
        )
        js.append(
            f"C('c-lev',{{"
            f"title:{{text:'总杠杆率 Gross Leverage',textStyle:{{fontSize:12}}}},"
            f"tooltip:{{trigger:'axis'}},{_grid_s},{_zoom_s},"
            f"xAxis:{{type:'category',data:D.glDates,axisLabel:{{fontSize:10}}}},"
            f"yAxis:{{type:'value'}},"
            f"series:[{{type:'line',data:D.glVals,showSymbol:false,"
            f"lineStyle:{{width:1.2,color:'#2b6cb0'}},"
            f"areaStyle:{{color:'rgba(43,108,176,0.1)'}}}}]"
            f"}});"
        )
        # Position concentration
        if "pos_max_concentration" in s:
            js.append(
                f"C('c-conc',{{"
                f"title:{{text:'持仓集中度 (%)',textStyle:{{fontSize:12}}}},"
                f"tooltip:{{trigger:'axis',valueFormatter:function(v){{return v.toFixed(2)+'%'}}}},"
                f"legend:{{top:4,right:10}},{_grid_s},{_zoom_s},"
                f"xAxis:{{type:'category',data:D.mcDates,axisLabel:{{fontSize:10}}}},"
                f"yAxis:{{type:'value'}},"
                f"series:["
                f"{{name:'Max',type:'line',data:D.mcMax,showSymbol:false,"
                f"lineStyle:{{width:1,color:'#c53030'}},areaStyle:{{color:'rgba(197,48,48,0.1)'}}}},"
                f"{{name:'Median',type:'line',data:D.mcMed,showSymbol:false,"
                f"lineStyle:{{width:1,color:O}},areaStyle:{{color:'rgba(221,107,32,0.08)'}}}}]"
                f"}});"
            )
        # Holdings allocation stacked area
        if "pos_alloc" in s and len(s["pos_alloc"].columns) <= 20:
            _pa_colors = "['#3182ce','#38a169','#e53e3e','#dd6b20','#805ad5','#d69e2e','#319795','#b83280','#2b6cb0','#276749','#c53030','#9c4221','#6b46c1','#975a16','#2c7a7b','#97266d','#2a4365','#22543d','#742a2a','#7b341e']"
            _pa_series = ",".join(
                f"{{name:D.paNames[{i}],type:'line',stack:'alloc',data:D.paData[D.paNames[{i}]],"
                f"showSymbol:false,areaStyle:{{}},lineStyle:{{width:0.5}}}}"
                for i in range(len(s["pos_alloc"].columns))
            )
            js.append(
                f"C('c-alloc',{{"
                f"title:{{text:'持仓配置 Holdings Allocation',textStyle:{{fontSize:12}}}},"
                f"tooltip:{{trigger:'axis'}},legend:{{top:4,right:10,textStyle:{{fontSize:9}}}},"
                f"color:{_pa_colors},"
                f"{_grid},{_zoom_s},"
                f"xAxis:{{type:'category',data:D.paDates,axisLabel:{{fontSize:10}}}},"
                f"yAxis:{{type:'value'}},"
                f"series:[{_pa_series}]"
                f"}});"
            )

    # Transaction volume + count
    if "has_transactions" in s:
        js.append(
            f"C('c-txn',{{"
            f"title:{{text:'日成交额 Daily Volume',textStyle:{{fontSize:12}}}},"
            f"tooltip:{{trigger:'axis'}},{_grid_s},"
            f"xAxis:{{type:'category',data:D.txnDates,axisLabel:{{fontSize:10}}}},"
            f"yAxis:{{type:'value'}},"
            f"series:[{{type:'bar',data:D.txnVol,barWidth:'60%',itemStyle:{{color:B}}}}]"
            f"}});"
        )
        js.append(
            f"C('c-txn-cnt',{{"
            f"title:{{text:'日成交笔数 Daily Count',textStyle:{{fontSize:12}}}},"
            f"tooltip:{{trigger:'axis'}},{_grid_s},"
            f"xAxis:{{type:'category',data:D.txnCntDates,axisLabel:{{fontSize:10}}}},"
            f"yAxis:{{type:'value'}},"
            f"series:[{{type:'bar',data:D.txnCnts,barWidth:'60%',itemStyle:{{color:O}}}}]"
            f"}});"
        )
        if "turnover" in s:
            js.append(
                f"C('c-turnover',{{"
                f"title:{{text:'日换手率 Daily Turnover',textStyle:{{fontSize:12}}}},"
                f"tooltip:{{trigger:'axis'}},{_grid_s},{_zoom_s},"
                f"xAxis:{{type:'category',data:D.toDates,axisLabel:{{fontSize:10}}}},"
                f"yAxis:{{type:'value'}},"
                f"series:[{{type:'line',data:D.toVals,showSymbol:false,"
                f"lineStyle:{{width:1.2,color:P}},areaStyle:{{color:'rgba(128,90,213,0.1)'}}}}]"
                f"}});"
            )
        if "txn_hours" in s:
            js.append(
                f"C('c-txn-hours',{{"
                f"title:{{text:'交易时间分布 (小时)',textStyle:{{fontSize:12}}}},"
                f"tooltip:{{trigger:'axis'}},{_grid_s},"
                f"xAxis:{{type:'category',data:D.txnHrLabels}},"
                f"yAxis:{{type:'value'}},"
                f"series:[{{type:'bar',data:D.txnHrCnts,barWidth:'60%',"
                f"itemStyle:{{color:'#319795',borderRadius:[2,2,0,0]}}}}]"
                f"}});"
            )

    # PnL distribution (all trades)
    if "trade_pnl" in s:
        js.append(
            f"C('c-pnl',{{"
            f"title:{{text:'交易盈亏分布 PnL Distribution',textStyle:{{fontSize:12}}}},"
            f"tooltip:{{trigger:'axis'}},{_grid_s},"
            f"xAxis:{{type:'category',data:D.pnlBins,axisLabel:{{fontSize:9}}}},"
            f"yAxis:{{type:'value'}},"
            f"series:[{{type:'bar',data:D.pnlCnts,barWidth:'80%',"
            f"itemStyle:{{color:function(p){{return D.pnlBins[p.dataIndex]>=0?G:R}}}}}}]"
            f"}});"
        )
        # Long trades PnL
        if "trade_pnl_long" in s and len(s["trade_pnl_long"]) > 0:
            js.append(
                f"C('c-pnl-long',{{"
                f"title:{{text:'多头交易盈亏 Long PnL',textStyle:{{fontSize:12}}}},"
                f"tooltip:{{trigger:'axis'}},{_grid_s},"
                f"xAxis:{{type:'category',data:D.pnlLBins,axisLabel:{{fontSize:9}}}},"
                f"yAxis:{{type:'value'}},"
                f"series:[{{type:'bar',data:D.pnlLCnts,barWidth:'80%',"
                f"itemStyle:{{color:function(p){{return D.pnlLBins[p.dataIndex]>=0?G:R}}}}}}]"
                f"}});"
            )
        # Short trades PnL
        if "trade_pnl_short" in s and len(s["trade_pnl_short"]) > 0:
            js.append(
                f"C('c-pnl-short',{{"
                f"title:{{text:'空头交易盈亏 Short PnL',textStyle:{{fontSize:12}}}},"
                f"tooltip:{{trigger:'axis'}},{_grid_s},"
                f"xAxis:{{type:'category',data:D.pnlSBins,axisLabel:{{fontSize:9}}}},"
                f"yAxis:{{type:'value'}},"
                f"series:[{{type:'bar',data:D.pnlSCnts,barWidth:'80%',"
                f"itemStyle:{{color:function(p){{return D.pnlSBins[p.dataIndex]>=0?G:R}}}}}}]"
                f"}});"
            )
        # Holding time distribution
        if "trade_barlen" in s:
            js.append(
                f"C('c-barlen',{{"
                f"title:{{text:'持仓时间分布 Holding Time',textStyle:{{fontSize:12}}}},"
                f"tooltip:{{trigger:'axis'}},{_grid_s},"
                f"xAxis:{{type:'category',data:D.blBins,axisLabel:{{fontSize:9}}}},"
                f"yAxis:{{type:'value'}},"
                f"series:[{{type:'bar',data:D.blCnts,barWidth:'80%',"
                f"itemStyle:{{color:'#319795',borderRadius:[2,2,0,0]}}}}]"
                f"}});"
            )

    return js


# =========================================================================
# PDF 报告生成 — 通过渲染 HTML 实现与 HTML 报告完全一致
# =========================================================================


def _generate_pdf(returns, benchmark_rets, positions, transactions, trades, title, output, rolling_window):
    """生成 PDF 报告：先生成 HTML，再用 Playwright 渲染为 PDF，确保与 HTML 完全一致。

    步骤：
    1. 调用 _generate_html 生成临时 HTML 文件
    2. 用 Playwright (headless Chromium) 渲染 HTML → PDF
    3. 用 PyPDF2 添加可点击的书签目录
    """
    import os
    import tempfile

    # 1) 生成临时 HTML 文件
    out_dir = os.path.dirname(os.path.abspath(output)) or "."
    with tempfile.NamedTemporaryFile(suffix=".html", delete=False, dir=out_dir) as tmp:
        tmp_html = tmp.name

    _generate_html(
        returns,
        benchmark_rets=benchmark_rets,
        positions=positions,
        transactions=transactions,
        trades=trades,
        title=title,
        output=tmp_html,
        rolling_window=rolling_window,
    )

    # 2) 用 Playwright 将 HTML 渲染为 PDF
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        raise ImportError(
            "生成 PDF 需要 playwright 库，请执行:\n  pip install playwright && python -m playwright install chromium"
        )

    # 临时 PDF 路径（后续添加书签后写入最终 output）
    tmp_pdf = output + ".tmp.pdf"

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page(viewport={"width": 1200, "height": 900})
        page.goto(f"file://{os.path.abspath(tmp_html)}", wait_until="networkidle", timeout=60000)

        # 智能等待：检测所有 ECharts 实例渲染完毕
        page.evaluate("""() => {
            return new Promise((resolve) => {
                let attempts = 0;
                const check = () => {
                    attempts++;
                    const containers = document.querySelectorAll('[id^="c-"]');
                    let allReady = true;
                    containers.forEach(el => {
                        const canvas = el.querySelector('canvas');
                        if (!canvas) allReady = false;
                    });
                    if (allReady || attempts > 30) resolve();
                    else setTimeout(check, 200);
                };
                setTimeout(check, 500);
            });
        }""")
        # 额外等待确保图表动画完成
        page.wait_for_timeout(1500)

        # 收集各节的标题和页面位置（用于生成书签）
        section_info = page.evaluate("""() => {
            const sections = document.querySelectorAll('.sec');
            const results = [];
            sections.forEach(sec => {
                const titleEl = sec.querySelector('.sec-title');
                if (titleEl) {
                    const rect = sec.getBoundingClientRect();
                    results.push({
                        id: sec.id,
                        title: titleEl.textContent.trim(),
                        top: rect.top + window.scrollY
                    });
                }
            });
            // 文档总高度
            const totalHeight = document.documentElement.scrollHeight;
            return { sections: results, totalHeight: totalHeight };
        }""")

        page.pdf(
            path=tmp_pdf,
            format="A4",
            print_background=True,
            margin={"top": "12mm", "bottom": "12mm", "left": "10mm", "right": "10mm"},
        )
        browser.close()

    # 3) 清理临时 HTML
    try:
        os.remove(tmp_html)
    except OSError:
        pass

    # 4) 添加 PDF 书签（可点击目录）
    _add_pdf_bookmarks(tmp_pdf, output, section_info, title)

    # 清理临时 PDF
    try:
        os.remove(tmp_pdf)
    except OSError:
        pass

    return output


def _add_pdf_bookmarks(input_pdf, output_pdf, section_info, report_title):
    """给 PDF 添加可点击的书签/大纲目录。"""
    try:
        from PyPDF2 import PdfReader, PdfWriter
    except ImportError:
        # 无 PyPDF2 时直接复制文件
        import shutil

        shutil.copy2(input_pdf, output_pdf)
        return

    reader = PdfReader(input_pdf)
    writer = PdfWriter()

    # 复制所有页面
    for page in reader.pages:
        writer.add_page(page)

    # 计算文档总高度到页面的映射
    total_pages = len(reader.pages)
    if total_pages == 0:
        with open(output_pdf, "wb") as f:
            writer.write(f)
        return

    total_height = section_info.get("totalHeight", 1)
    sections = section_info.get("sections", [])

    # 每个 A4 页面的 CSS 像素高度（约 1123px at 96dpi for A4 minus margins）
    # Playwright uses 96dpi; A4 = 297mm ≈ 1123px, minus margins (12mm*2 = ~91px)
    page_css_height = 1123 - 91  # ≈ 1032px per page content area

    # 添加根书签
    writer.add_outline_item(report_title, 0)

    for sec in sections:
        sec_top = sec["top"]
        sec_title = sec["title"]

        # 估算此节在第几页
        est_page = int(sec_top / page_css_height) if page_css_height > 0 else 0
        est_page = min(est_page, total_pages - 1)

        writer.add_outline_item(sec_title, est_page)

    with open(output_pdf, "wb") as f:
        writer.write(f)
