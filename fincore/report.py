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
</style>
"""


def _fmt(v, pct=False):
    """格式化数值。"""
    if isinstance(v, (int, np.integer)):
        return str(v)
    if isinstance(v, (float, np.floating)):
        if np.isnan(v):
            return "N/A"
        if pct:
            return f"{v * 100:.2f}%"
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


def _html_df(df, float_format=".4f"):
    """DataFrame → HTML table."""
    hdr = "<tr><th></th>" + "".join(f"<th>{c}</th>" for c in df.columns) + "</tr>"
    rows = [hdr]
    for idx, row in df.iterrows():
        cells = f"<th>{idx}</th>"
        for v in row:
            if isinstance(v, (float, np.floating)):
                cells += f'<td class="{_css_cls(v)}">{v:{float_format}}</td>'
            else:
                cells += f"<td>{v}</td>"
        rows.append(f"<tr>{cells}</tr>")
    return "<table>" + "".join(rows) + "</table>"


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

    monthly_tbl = s["monthly_returns"].unstack().fillna(0)
    hm_months = [f"{int(c):02d}" for c in monthly_tbl.columns]
    hm_years = [str(y) for y in monthly_tbl.index]
    hm_data = []
    for yi, y in enumerate(monthly_tbl.index):
        for mi, m in enumerate(monthly_tbl.columns):
            hm_data.append([mi, yi, round(float(monthly_tbl.loc[y, m]) * 100, 2)])
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

    if "trade_pnl" in s:
        pnl = s["trade_pnl"]
        ph, pe = np.histogram(pnl, bins=max(10, min(40, len(pnl) // 3)))
        cd["pnlBins"] = [round(float((pe[i] + pe[i + 1]) / 2), 2) for i in range(len(ph))]
        cd["pnlCnts"] = [int(v) for v in ph]

    chart_json = json.dumps(cd, ensure_ascii=False)

    # ---- sidebar ----
    nav = [
        ("overview", "产品概览"),
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
        "Annual Return", "Cumulative Returns", "Annual Volatility", "Max Drawdown",
        "Downside Risk", "Daily Value at Risk", "Daily Mean Return", "Daily Std Return",
        "Best Day", "Worst Day", "Avg Daily Turnover",
    }
    b = []  # body parts

    b.append(f"<h1>{title}</h1>")
    b.append(
        f'<div class="meta">{s["date_range"][0]} → {s["date_range"][1]}'
        f' | {s["n_days"]} 交易日 | ~{s["n_months"]} 个月</div>'
    )

    # -- Overview --
    b.append('<div class="sec" id="overview"><div class="sec-title">产品概览 Overview</div>')
    b.append(_html_cards(
        s["perf_stats"],
        ["Sharpe Ratio", "Annual Return", "Max Drawdown", "Annual Volatility",
         "Sortino Ratio", "Calmar Ratio", "Omega Ratio", "Stability"],
        pct_keys={"Annual Return", "Max Drawdown", "Annual Volatility"},
    ))
    b.append('<div class="chart-box" id="c-cum"></div>')
    b.append('<div class="chart-sm" id="c-dd"></div>')
    b.append("</div>")

    # -- Performance --
    b.append('<div class="sec" id="performance"><div class="sec-title">绩效分析 Performance</div>')
    b.append('<div class="grid-2"><div>')
    b.append('<h3 class="sub">核心指标</h3>')
    b.append(_html_table(s["perf_stats"], pct_keys=pct_perf))
    b.append('</div><div>')
    b.append('<h3 class="sub">扩展风险指标</h3>')
    b.append(_html_table(s["extended_stats"]))
    b.append("</div></div></div>")

    # -- Returns --
    b.append('<div class="sec" id="returns"><div class="sec-title">收益分析 Returns</div>')
    b.append('<div class="grid-2">')
    b.append('<div class="chart-sm" id="c-daily"></div>')
    b.append('<div class="chart-sm" id="c-dist"></div>')
    b.append("</div>")
    q = s["return_quantiles"]
    b.append('<h3 class="sub">收益分位数</h3>')
    q_rows = "".join(
        f'<tr><td style="text-align:left;font-weight:600">{p}</td>'
        f'<td class="{_css_cls(v)}">{v * 100:.4f}%</td></tr>'
        for p, v in q.items()
    )
    b.append(f'<table><tr><th>Percentile</th><th>Return</th></tr>{q_rows}</table>')
    b.append('<div class="chart-box" id="c-hm" style="height:280px"></div>')
    b.append('<h3 class="sub">月度收益 (%)</h3>')
    mt = monthly_tbl.copy()
    mt.columns = hm_months
    b.append(_html_df(mt * 100, float_format=".2f"))
    b.append('<div class="grid-2">')
    b.append('<div class="chart-sm" id="c-yr"></div>')
    b.append("<div>")
    b.append('<h3 class="sub">年度统计</h3>')
    b.append(_html_df(s["yearly_stats"]))
    b.append("</div></div>")
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
        b.append(_html_cards(
            s["benchmark_stats"],
            ["Alpha", "Beta", "Information Ratio", "Tracking Error",
             "Up Capture", "Down Capture", "Capture Ratio", "Correlation"],
        ))
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
        b.append('<div class="chart-sm" id="c-lev"></div>')
        b.append("</div>")

    # -- Transactions --
    if "has_transactions" in s:
        b.append('<div class="sec" id="transactions"><div class="sec-title">交易分析 Transactions</div>')
        b.append(_html_cards(s["txn_summary"], list(s["txn_summary"].keys())))
        b.append(_html_table(s["txn_summary"]))
        b.append('<div class="chart-sm" id="c-txn"></div>')
        b.append("</div>")

    # -- Trades --
    if "trade_stats" in s:
        ts = s["trade_stats"]
        pct_t = {"Win Rate", "Long Win Rate", "Short Win Rate"}
        b.append('<div class="sec" id="trades"><div class="sec-title">交易统计 Trades</div>')
        b.append(_html_cards(
            ts,
            ["Total Trades", "Win Rate", "Profit/Loss Ratio", "Total PnL", "Expectancy", "Avg PnL per Trade"],
            pct_keys=pct_t,
        ))
        b.append(_html_table(ts, pct_keys=pct_t))
        if "trade_pnl" in s:
            b.append('<div class="chart-sm" id="c-pnl"></div>')
        b.append("</div>")

    b.append('<footer>Generated by <strong>fincore</strong> | create_strategy_report()</footer>')
    body_html = "\n".join(b)

    # ---- ECharts JavaScript ----
    js_parts = _build_echart_js(s, rolling_window)
    js = "\n".join(js_parts)

    html = (
        f"<!DOCTYPE html>\n<html lang='zh'><head><meta charset='utf-8'>\n"
        f"<title>{title}</title>\n"
        f'<script src="https://cdn.jsdelivr.net/npm/echarts@5/dist/echarts.min.js"></script>\n'
        f"{_HTML_CSS}\n</head>\n<body>\n"
        f"{sidebar}\n<main class='content'>\n{body_html}\n</main>\n"
        f"<script>\nvar D={chart_json};\n"
        f"function C(id,o){{var e=document.getElementById(id);if(!e)return;"
        f"var c=echarts.init(e);c.setOption(o);"
        f"window.addEventListener('resize',function(){{c.resize()}});return c;}}\n"
        f"var B='#3182ce',G='#38a169',R='#e53e3e',O='#dd6b20',P='#805ad5',GY='#a0aec0';\n"
        f"{js}\n</script>\n</body></html>"
    )

    with open(output, "w", encoding="utf-8") as f:
        f.write(html)

    return output


def _build_echart_js(s, rw):
    """Build ECharts initialization JavaScript statements."""
    js = []
    _grid = "grid:{left:60,right:30,bottom:30,top:40}"
    _grid_s = "grid:{left:55,right:15,bottom:25,top:35}"
    _zoom = "dataZoom:[{type:'inside'},{type:'slider',height:18,bottom:4}]"
    _zoom_s = "dataZoom:[{type:'inside'}]"

    # Cumulative returns
    bench_series = ""
    if "benchmark_cum" in s:
        bench_series = (",{name:'Benchmark',type:'line',data:D.benchCum,showSymbol:false,"
                        "lineStyle:{width:1,color:GY,type:'dashed'}}")
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
        f"yAxis:{{type:'value',name:'频次',axisLabel:{{fontSize:10}}}},"
        f"series:[{{type:'bar',data:D.histCnts,barWidth:'80%',"
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

    # Transaction volume
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

    # PnL distribution
    if "trade_pnl" in s:
        js.append(
            f"C('c-pnl',{{"
            f"title:{{text:'交易盈亏分布 PnL Distribution',textStyle:{{fontSize:12}}}},"
            f"tooltip:{{trigger:'axis'}},{_grid_s},"
            f"xAxis:{{type:'category',data:D.pnlBins,axisLabel:{{fontSize:9}}}},"
            f"yAxis:{{type:'value',name:'频次'}},"
            f"series:[{{type:'bar',data:D.pnlCnts,barWidth:'80%',"
            f"itemStyle:{{color:function(p){{return D.pnlBins[p.dataIndex]>=0?G:R}}}}}}]"
            f"}});"
        )

    return js


# =========================================================================
# PDF 报告生成
# =========================================================================


def _generate_pdf(returns, benchmark_rets, positions, transactions, trades, title, output, rolling_window):
    """生成 PDF 报告（使用 matplotlib）。"""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.colors as mcolors
    import matplotlib.gridspec as gridspec
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    s = _compute_sections(returns, benchmark_rets, positions, transactions, trades, rolling_window)

    pdf = PdfPages(output)
    page_count = 0

    def save_page(fig):
        nonlocal page_count
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)
        page_count += 1

    def dict_to_fig(d, fig_title, pct_keys=None):
        pct_keys = set(pct_keys or [])
        n = len(d)
        fig, ax = plt.subplots(figsize=(10, max(3, 0.38 * n + 1.5)))
        ax.axis("off")
        ax.set_title(fig_title, fontsize=13, fontweight="bold", pad=12)
        cell_text = []
        for k, v in d.items():
            pct = k in pct_keys
            cell_text.append([k, _fmt(v, pct=pct)])
        tbl = ax.table(cellText=cell_text, colLabels=["Metric", "Value"], cellLoc="center", loc="center")
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(9)
        tbl.scale(1.2, 1.4)
        fig.tight_layout()
        return fig

    def df_to_fig(df, fig_title, float_fmt=".4f"):
        n = len(df)
        fig, ax = plt.subplots(figsize=(12, max(3, 0.38 * n + 1.5)))
        ax.axis("off")
        ax.set_title(fig_title, fontsize=13, fontweight="bold", pad=12)
        cell_text = []
        for _, row in df.iterrows():
            cells = []
            for v in row:
                if isinstance(v, (float, np.floating)):
                    cells.append(f"{v:{float_fmt}}")
                else:
                    cells.append(str(v))
            cell_text.append(cells)
        tbl = ax.table(
            cellText=cell_text,
            rowLabels=[str(i) for i in df.index],
            colLabels=[str(c) for c in df.columns],
            cellLoc="center",
            loc="center",
        )
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(9)
        tbl.scale(1.1, 1.4)
        fig.tight_layout()
        return fig

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

    # === P1: Performance Statistics ===
    save_page(dict_to_fig(s["perf_stats"], f"{title}\nPerformance Statistics", pct_keys=pct_perf))

    # === P2: Extended Metrics ===
    save_page(dict_to_fig(s["extended_stats"], "Extended Risk Metrics"))

    # === P3: Benchmark (if available) ===
    if "benchmark_stats" in s:
        save_page(dict_to_fig(s["benchmark_stats"], "Benchmark Comparison"))

    # === P4: Yearly Statistics ===
    save_page(df_to_fig(s["yearly_stats"], "Yearly Statistics"))

    # === P5: Worst Drawdown Periods ===
    save_page(df_to_fig(s["dd_table"], "Worst Drawdown Periods", float_fmt=".2f"))

    # === P6: Cumulative Returns + Drawdown ===
    fig = plt.figure(figsize=(14, 10))
    gs = gridspec.GridSpec(2, 1, hspace=0.35)

    ax1 = fig.add_subplot(gs[0])
    ax1.plot(s["cum_returns"].index, s["cum_returns"].values, color="steelblue", linewidth=1.2, label="Strategy")
    if "benchmark_cum" in s:
        ax1.plot(
            s["benchmark_cum"].index,
            s["benchmark_cum"].values,
            color="gray",
            linewidth=0.9,
            alpha=0.7,
            label="Benchmark",
        )
        ax1.legend()
    ax1.axhline(y=1.0, color="gray", linestyle="--", linewidth=0.6)
    ax1.set_title("Cumulative Returns", fontsize=12, fontweight="bold")
    ax1.set_ylabel("Growth of $1")
    ax1.fill_between(
        s["cum_returns"].index,
        1.0,
        s["cum_returns"].values,
        where=s["cum_returns"].values >= 1.0,
        alpha=0.1,
        color="green",
    )
    ax1.fill_between(
        s["cum_returns"].index,
        1.0,
        s["cum_returns"].values,
        where=s["cum_returns"].values < 1.0,
        alpha=0.1,
        color="red",
    )
    ax1.grid(True, alpha=0.3)

    ax2 = fig.add_subplot(gs[1], sharex=ax1)
    dd = s["drawdown"]
    ax2.fill_between(dd.index, dd.values, 0, color="red", alpha=0.3)
    ax2.plot(dd.index, dd.values, color="red", linewidth=0.8)
    ax2.set_title("Drawdown", fontsize=12, fontweight="bold")
    ax2.set_ylabel("Drawdown")
    ax2.grid(True, alpha=0.3)
    save_page(fig)

    # === P7: Cumulative Returns (Log Scale) ===
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.semilogy(s["cum_returns"].index, s["cum_returns"].values, color="steelblue", linewidth=1.2, label="Strategy")
    if "benchmark_cum" in s:
        ax.semilogy(
            s["benchmark_cum"].index,
            s["benchmark_cum"].values,
            color="gray",
            linewidth=0.9,
            alpha=0.7,
            label="Benchmark",
        )
        ax.legend()
    ax.set_title("Cumulative Returns (Log Scale)", fontsize=12, fontweight="bold")
    ax.set_ylabel("Growth of $1 (log)")
    ax.grid(True, alpha=0.3)
    save_page(fig)

    # === P8: Daily Returns + Distribution ===
    rets = s["returns"]
    fig = plt.figure(figsize=(14, 10))
    gs = gridspec.GridSpec(2, 2, hspace=0.35, wspace=0.3)

    ax1 = fig.add_subplot(gs[0, :])
    ax1.bar(rets.index, rets.values, color=["green" if v >= 0 else "red" for v in rets.values], alpha=0.5, width=1)
    ax1.axhline(y=0, color="gray", linewidth=0.5)
    ax1.set_title("Daily Returns", fontsize=12, fontweight="bold")
    ax1.set_ylabel("Return")
    ax1.grid(True, alpha=0.3)

    ax2 = fig.add_subplot(gs[1, 0])
    ax2.hist(rets.values, bins=min(80, len(rets) // 5 + 1), color="steelblue", alpha=0.7, edgecolor="white")
    ax2.axvline(x=0, color="gray", linestyle="--", linewidth=1)
    ax2.axvline(x=rets.mean(), color="blue", linestyle="-", linewidth=1, label=f"Mean: {rets.mean() * 100:.3f}%")
    ax2.legend(fontsize=9)
    ax2.set_title("Daily Returns Distribution", fontsize=11, fontweight="bold")
    ax2.set_xlabel("Return")
    ax2.grid(True, alpha=0.3)

    ax3 = fig.add_subplot(gs[1, 1])
    q = s["return_quantiles"]
    ax3.barh(range(len(q)), q.values * 100, color=["green" if v >= 0 else "red" for v in q.values], alpha=0.7)
    ax3.set_yticks(range(len(q)))
    ax3.set_yticklabels([str(p) for p in q.index])
    ax3.axvline(x=0, color="gray", linestyle="--", linewidth=0.6)
    ax3.set_title("Return Quantiles (%)", fontsize=11, fontweight="bold")
    ax3.set_xlabel("Return (%)")
    ax3.grid(True, alpha=0.3, axis="x")
    save_page(fig)

    # === P9: Rolling Sharpe + Volatility ===
    fig = plt.figure(figsize=(14, 10))
    gs = gridspec.GridSpec(2, 1, hspace=0.35)

    ax1 = fig.add_subplot(gs[0])
    rs = s["rolling_sharpe"].dropna()
    ax1.plot(rs.index, rs.values, color="steelblue", linewidth=0.9)
    ax1.fill_between(rs.index, 0, rs.values, where=rs.values >= 0, alpha=0.1, color="green")
    ax1.fill_between(rs.index, 0, rs.values, where=rs.values < 0, alpha=0.1, color="red")
    ax1.axhline(y=0, color="gray", linestyle="--", linewidth=0.6)
    ax1.set_title(f"Rolling Sharpe Ratio ({rolling_window}d)", fontsize=12, fontweight="bold")
    ax1.grid(True, alpha=0.3)

    ax2 = fig.add_subplot(gs[1], sharex=ax1)
    rv = s["rolling_volatility"].dropna()
    ax2.plot(rv.index, rv.values, color="orange", linewidth=0.9)
    ax2.fill_between(rv.index, 0, rv.values, alpha=0.15, color="orange")
    ax2.set_title(f"Rolling Volatility ({rolling_window}d)", fontsize=12, fontweight="bold")
    ax2.grid(True, alpha=0.3)
    save_page(fig)

    # === P10: Rolling Beta (if benchmark) ===
    if "rolling_beta" in s:
        fig, ax = plt.subplots(figsize=(14, 5))
        rb = s["rolling_beta"].dropna()
        ax.plot(rb.index, rb.values, color="purple", linewidth=0.9)
        ax.axhline(y=1.0, color="gray", linestyle="--", linewidth=0.6)
        ax.axhline(y=0, color="lightgray", linestyle=":", linewidth=0.5)
        ax.fill_between(rb.index, 1.0, rb.values, alpha=0.1, color="purple")
        ax.set_title(f"Rolling Beta ({rolling_window}d)", fontsize=12, fontweight="bold")
        ax.grid(True, alpha=0.3)
        save_page(fig)

    # === P11: Monthly Heatmap + Annual Returns + Monthly Distribution ===
    monthly = s["monthly_returns"]
    monthly_tbl = monthly.unstack().fillna(0)

    fig = plt.figure(figsize=(16, 6))
    gs = gridspec.GridSpec(1, 3, wspace=0.35)

    ax1 = fig.add_subplot(gs[0])
    vals = monthly_tbl.values * 100
    try:
        norm = mcolors.TwoSlopeNorm(vmin=vals.min(), vcenter=0, vmax=max(vals.max(), 0.01))
    except ValueError:
        norm = None
    im = ax1.imshow(vals, cmap="RdYlGn", norm=norm, aspect="auto")
    ax1.set_xticks(range(vals.shape[1]))
    ax1.set_xticklabels([f"{int(c):02d}" for c in monthly_tbl.columns], fontsize=7)
    ax1.set_yticks(range(vals.shape[0]))
    ax1.set_yticklabels(monthly_tbl.index, fontsize=8)
    ax1.set_title("Monthly Returns (%)", fontsize=11, fontweight="bold")
    plt.colorbar(im, ax=ax1, shrink=0.8)

    ax2 = fig.add_subplot(gs[1])
    ys = s["yearly_stats"]
    colors = ["green" if v >= 0 else "red" for v in ys["Annual Return"]]
    ax2.barh(range(len(ys)), ys["Annual Return"].values * 100, color=colors, alpha=0.7)
    ax2.set_yticks(range(len(ys)))
    ax2.set_yticklabels([str(y) for y in ys.index])
    ax2.set_xlabel("Return (%)")
    ax2.set_title("Annual Returns", fontsize=11, fontweight="bold")
    ax2.axvline(x=0, color="gray", linestyle="--", linewidth=0.6)
    ax2.grid(True, alpha=0.3, axis="x")

    ax3 = fig.add_subplot(gs[2])
    monthly_vals = monthly.values
    ax3.hist(
        monthly_vals * 100, bins=min(30, len(monthly_vals) // 2 + 1), color="steelblue", alpha=0.7, edgecolor="white"
    )
    ax3.axvline(x=0, color="gray", linestyle="--", linewidth=0.6)
    ax3.set_title("Monthly Returns Dist (%)", fontsize=11, fontweight="bold")
    ax3.set_xlabel("Return (%)")
    ax3.grid(True, alpha=0.3)
    save_page(fig)

    # === P12: Position Analysis (if available) ===
    if "has_positions" in s:
        # Position summary table
        save_page(dict_to_fig(s["position_summary"], "Position Summary"))

        # Position charts: exposure + leverage + concentration
        fig = plt.figure(figsize=(14, 15))
        gs = gridspec.GridSpec(3, 1, hspace=0.35)

        ax1 = fig.add_subplot(gs[0])
        ax1.fill_between(s["pos_long"].index, s["pos_long"].values, 0, color="green", alpha=0.3, label="Long")
        ax1.fill_between(s["pos_short"].index, s["pos_short"].values, 0, color="red", alpha=0.3, label="Short")
        ax1.legend()
        ax1.set_title("Long / Short Exposure", fontsize=12, fontweight="bold")
        ax1.grid(True, alpha=0.3)

        ax2 = fig.add_subplot(gs[1], sharex=ax1)
        gl = s["gross_leverage"].dropna()
        ax2.plot(gl.index, gl.values, color="navy", linewidth=0.9)
        ax2.fill_between(gl.index, 0, gl.values, alpha=0.1, color="navy")
        ax2.set_title("Gross Leverage", fontsize=12, fontweight="bold")
        ax2.grid(True, alpha=0.3)

        ax3 = fig.add_subplot(gs[2], sharex=ax1)
        mc = s["pos_max_concentration"]
        med_c = s["pos_median_concentration"]
        ax3.plot(mc.index, mc.values, color="darkred", linewidth=0.9, label="Max")
        ax3.plot(med_c.index, med_c.values, color="orange", linewidth=0.9, label="Median")
        ax3.legend()
        ax3.set_title("Position Concentration", fontsize=12, fontweight="bold")
        ax3.set_ylabel("Fraction of Portfolio")
        ax3.grid(True, alpha=0.3)
        save_page(fig)

        # Holdings over time (stacked area)
        pos_alloc = s["pos_alloc"]
        if len(pos_alloc.columns) <= 20:
            fig, ax = plt.subplots(figsize=(14, 6))
            ax.stackplot(
                pos_alloc.index, *[pos_alloc[c].values for c in pos_alloc.columns], labels=pos_alloc.columns, alpha=0.7
            )
            ax.legend(loc="upper left", fontsize=8, ncol=min(5, len(pos_alloc.columns)))
            ax.set_title("Holdings Allocation Over Time", fontsize=12, fontweight="bold")
            ax.set_ylabel("Allocation")
            ax.grid(True, alpha=0.3)
            save_page(fig)

    # === P13: Transaction Analysis (if available) ===
    if "has_transactions" in s:
        save_page(dict_to_fig(s["txn_summary"], "Transaction Summary"))

        n_txn_plots = 2
        if "turnover" in s:
            n_txn_plots += 1
        if "txn_hours" in s:
            n_txn_plots += 1

        fig = plt.figure(figsize=(14, 5 * n_txn_plots))
        gs = gridspec.GridSpec(n_txn_plots, 1, hspace=0.4)
        pi = 0

        ax = fig.add_subplot(gs[pi])
        dv = s["daily_txn_value"]
        ax.bar(dv.index, dv.values, color="steelblue", alpha=0.7, width=1)
        ax.set_title("Daily Transaction Volume", fontsize=12, fontweight="bold")
        ax.grid(True, alpha=0.3)
        pi += 1

        ax = fig.add_subplot(gs[pi])
        dc = s["daily_txn_count"]
        ax.bar(dc.index, dc.values, color="orange", alpha=0.7, width=1)
        ax.set_title("Daily Transaction Count", fontsize=12, fontweight="bold")
        ax.grid(True, alpha=0.3)
        pi += 1

        if "turnover" in s:
            ax = fig.add_subplot(gs[pi])
            to = s["turnover"].dropna()
            ax.plot(to.index, to.values, color="purple", linewidth=0.9)
            ax.fill_between(to.index, 0, to.values, alpha=0.1, color="purple")
            ax.set_title("Daily Turnover", fontsize=12, fontweight="bold")
            ax.grid(True, alpha=0.3)
            pi += 1

        if "txn_hours" in s:
            ax = fig.add_subplot(gs[pi])
            hours = s["txn_hours"]
            ax.hist(hours, bins=range(25), color="teal", alpha=0.7, edgecolor="white")
            ax.set_title("Transaction Time Distribution (Hour)", fontsize=12, fontweight="bold")
            ax.set_xlabel("Hour of Day")
            ax.set_xticks(range(0, 24, 2))
            ax.grid(True, alpha=0.3)
            pi += 1

        save_page(fig)

    # === P14: Trade Statistics (if available) ===
    if "trade_stats" in s:
        ts = s["trade_stats"]
        pct_keys_t = {"Win Rate", "Long Win Rate", "Short Win Rate"}
        save_page(dict_to_fig(ts, "Trade Statistics", pct_keys=pct_keys_t))

        # PnL Distribution (overall + long/short if available)
        if "trade_pnl" in s:
            pnl = s["trade_pnl"]
            has_ls = "trade_pnl_long" in s

            n_pnl_rows = 2 if has_ls else 1
            fig = plt.figure(figsize=(14, 5 * n_pnl_rows))
            gs = gridspec.GridSpec(n_pnl_rows, 2 if has_ls else 1, hspace=0.4, wspace=0.3)

            # Overall PnL distribution
            ax = fig.add_subplot(gs[0, :] if not has_ls else gs[0, :])
            n_bins = max(10, min(50, len(pnl) // 3 + 1))
            ax.hist(pnl, bins=n_bins, color="steelblue", alpha=0.7, edgecolor="white")
            ax.axvline(x=0, color="gray", linestyle="--", linewidth=1)
            ax.axvline(x=np.mean(pnl), color="blue", linestyle="-", linewidth=1, label=f"Mean: {np.mean(pnl):,.0f}")
            ax.axvline(
                x=np.median(pnl), color="orange", linestyle="-", linewidth=1, label=f"Median: {np.median(pnl):,.0f}"
            )
            ax.legend()
            ax.set_title("Trade PnL Distribution (All Trades)", fontsize=12, fontweight="bold")
            ax.set_xlabel("PnL (after commission)")
            ax.grid(True, alpha=0.3)

            if has_ls:
                pnl_l = s["trade_pnl_long"]
                pnl_s = s["trade_pnl_short"]

                ax2 = fig.add_subplot(gs[1, 0])
                if len(pnl_l) > 0:
                    ax2.hist(
                        pnl_l, bins=max(5, min(30, len(pnl_l) // 3 + 1)), color="green", alpha=0.7, edgecolor="white"
                    )
                    ax2.axvline(x=0, color="gray", linestyle="--", linewidth=0.8)
                    ax2.axvline(
                        x=np.mean(pnl_l),
                        color="darkgreen",
                        linestyle="-",
                        linewidth=1,
                        label=f"Mean: {np.mean(pnl_l):,.0f}",
                    )
                    ax2.legend(fontsize=9)
                ax2.set_title("Long Trades PnL", fontsize=11, fontweight="bold")
                ax2.set_xlabel("PnL")
                ax2.grid(True, alpha=0.3)

                ax3 = fig.add_subplot(gs[1, 1])
                if len(pnl_s) > 0:
                    ax3.hist(
                        pnl_s, bins=max(5, min(30, len(pnl_s) // 3 + 1)), color="red", alpha=0.7, edgecolor="white"
                    )
                    ax3.axvline(x=0, color="gray", linestyle="--", linewidth=0.8)
                    ax3.axvline(
                        x=np.mean(pnl_s),
                        color="darkred",
                        linestyle="-",
                        linewidth=1,
                        label=f"Mean: {np.mean(pnl_s):,.0f}",
                    )
                    ax3.legend(fontsize=9)
                ax3.set_title("Short Trades PnL", fontsize=11, fontweight="bold")
                ax3.set_xlabel("PnL")
                ax3.grid(True, alpha=0.3)

            save_page(fig)

        # Holding time distribution
        if "trade_barlen" in s:
            fig, ax = plt.subplots(figsize=(12, 5))
            barlen = s["trade_barlen"]
            ax.hist(barlen, bins=max(10, min(50, len(barlen) // 3 + 1)), color="teal", alpha=0.7, edgecolor="white")
            ax.axvline(
                x=np.mean(barlen), color="blue", linestyle="-", linewidth=1, label=f"Mean: {np.mean(barlen):.1f} bars"
            )
            ax.axvline(
                x=np.median(barlen),
                color="orange",
                linestyle="-",
                linewidth=1,
                label=f"Median: {np.median(barlen):.0f} bars",
            )
            ax.legend()
            ax.set_title("Trade Holding Time Distribution", fontsize=12, fontweight="bold")
            ax.set_xlabel("Holding Time (bars)")
            ax.grid(True, alpha=0.3)
            save_page(fig)

    pdf.close()
    return output
