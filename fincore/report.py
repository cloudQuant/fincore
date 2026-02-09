"""
策略报告生成器 — 根据传入数据动态生成 HTML 或 PDF 策略分析报告。

传入的数据越多，生成的报告越详细：

- **returns** (必需): 基础绩效指标 + 收益图表
- **+ benchmark_rets**: Alpha/Beta、信息比率、跟踪误差、滚动Beta
- **+ positions**: 持仓分析、多空暴露、杠杆率
- **+ transactions**: 换手率、交易量分析
- **+ trades**: 交易统计（胜率、盈亏比、持仓时长等）

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

import warnings
from typing import Any, Optional, Union

import numpy as np
import pandas as pd


def create_strategy_report(
    returns: pd.Series,
    *,
    benchmark_rets: Optional[pd.Series] = None,
    positions: Optional[pd.DataFrame] = None,
    transactions: Optional[pd.DataFrame] = None,
    trades: Optional[pd.DataFrame] = None,
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
        交易记录 DataFrame（需含 amount, price 列）。传入后增加交易分析。
    trades : pd.DataFrame, optional
        已平仓交易记录（需含 pnlcomm 列，可选 long, barlen 列）。
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
            returns, benchmark_rets=benchmark_rets, positions=positions,
            transactions=transactions, trades=trades, title=title,
            output=output, rolling_window=rolling_window,
        )
    else:
        return _generate_html(
            returns, benchmark_rets=benchmark_rets, positions=positions,
            transactions=transactions, trades=trades, title=title,
            output=output, rolling_window=rolling_window,
        )


# =========================================================================
# 内部：计算引擎
# =========================================================================

def _compute_sections(
    returns, benchmark_rets, positions, transactions, trades, rolling_window,
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

    # ------ 核心绩效 ------
    perf = {}
    perf["Annual Return"] = Empyrical.annual_return(returns)
    perf["Cumulative Returns"] = Empyrical.cum_returns_final(returns)
    perf["Annual Volatility"] = Empyrical.annual_volatility(returns)
    perf["Sharpe Ratio"] = Empyrical.sharpe_ratio(returns)
    perf["Sortino Ratio"] = Empyrical.sortino_ratio(returns)
    perf["Calmar Ratio"] = Empyrical.calmar_ratio(returns)
    perf["Max Drawdown"] = Empyrical.max_drawdown(returns)
    perf["Omega Ratio"] = Empyrical.omega_ratio(returns)
    perf["Stability"] = Empyrical.stability_of_timeseries(returns)
    perf["Skewness"] = Empyrical.skewness(returns)
    perf["Kurtosis"] = Empyrical.kurtosis(returns)
    perf["Tail Ratio"] = Empyrical.tail_ratio(returns)
    perf["Value at Risk (5%)"] = Empyrical.value_at_risk(returns)
    perf["Downside Risk"] = Empyrical.downside_risk(returns)
    sections["perf_stats"] = perf

    # ------ 扩展指标 ------
    ext = {}
    emp = Empyrical(returns=returns)
    ext["Win Rate"] = emp.win_rate()
    ext["Loss Rate"] = emp.loss_rate()
    ext["Serial Correlation"] = emp.serial_correlation()
    ext["Common Sense Ratio"] = emp.common_sense_ratio()
    ext["Sterling Ratio"] = emp.sterling_ratio()
    ext["Burke Ratio"] = emp.burke_ratio()
    ext["Max Drawdown Days"] = emp.max_drawdown_days()
    ext["2nd Max Drawdown"] = emp.second_max_drawdown()
    ext["Max Consecutive Up Days"] = Empyrical.max_consecutive_up_days(returns)
    ext["Max Consecutive Down Days"] = Empyrical.max_consecutive_down_days(returns)
    ext["Max Single Day Gain"] = Empyrical.max_single_day_gain(returns)
    ext["Max Single Day Loss"] = Empyrical.max_single_day_loss(returns)
    ext["Hurst Exponent"] = Empyrical.hurst_exponent(returns)
    sections["extended_stats"] = ext

    # ------ 时间序列数据 ------
    sections["cum_returns"] = Empyrical.cum_returns(returns, starting_value=1.0)
    cum_ret_0 = Empyrical.cum_returns(returns, starting_value=0)
    running_max = (1 + cum_ret_0).cummax()
    sections["drawdown"] = (1 + cum_ret_0) / running_max - 1
    sections["rolling_sharpe"] = Empyrical.rolling_sharpe(
        returns, rolling_sharpe_window=rolling_window
    )
    sections["rolling_volatility"] = Empyrical.rolling_volatility(
        returns, rolling_vol_window=rolling_window
    )
    sections["dd_table"] = Empyrical.gen_drawdown_table(returns, top=5)

    # ------ 按年/按月 ------
    sections["yearly_stats"] = pd.DataFrame({
        "Annual Return": Empyrical.annual_return_by_year(returns),
        "Sharpe Ratio": Empyrical.sharpe_ratio_by_year(returns),
        "Max Drawdown": Empyrical.max_drawdown_by_year(returns),
    })
    sections["monthly_returns"] = Empyrical.aggregate_returns(returns, "monthly")

    # ------ Benchmark 相关 ------
    if benchmark_rets is not None:
        bm = {}
        a, b = Empyrical.alpha_beta(returns, benchmark_rets)
        bm["Alpha"] = a
        bm["Beta"] = b
        bm["Information Ratio"] = Empyrical.information_ratio(returns, benchmark_rets)
        bm["Tracking Error"] = Empyrical.tracking_error(returns, benchmark_rets)
        bm["Up Capture"] = Empyrical.up_capture(returns, benchmark_rets)
        bm["Down Capture"] = Empyrical.down_capture(returns, benchmark_rets)
        sections["benchmark_stats"] = bm

        sections["benchmark_cum"] = Empyrical.cum_returns(benchmark_rets, starting_value=1.0)
        sections["rolling_beta"] = Empyrical.rolling_beta(
            returns, benchmark_rets, rolling_window=rolling_window,
        )

    # ------ Positions 相关 ------
    if positions is not None:
        sections["has_positions"] = True
        pos_no_cash = positions.drop("cash", axis=1, errors="ignore")
        sections["pos_long"] = pos_no_cash.where(pos_no_cash > 0, 0).sum(axis=1)
        sections["pos_short"] = pos_no_cash.where(pos_no_cash < 0, 0).sum(axis=1)
        total = positions.sum(axis=1).replace(0, np.nan)
        exposure = pos_no_cash.abs().sum(axis=1)
        sections["gross_leverage"] = (exposure / total).replace([np.inf, -np.inf], np.nan)

    # ------ Transactions 相关 ------
    if transactions is not None:
        sections["has_transactions"] = True
        txn = transactions.copy()
        txn.index = txn.index.normalize()
        sections["daily_txn_count"] = txn.groupby(txn.index).size()
        sections["daily_txn_value"] = (txn["amount"].abs() * txn["price"]).groupby(txn.index).sum()

    # ------ Trades 相关 ------
    if trades is not None and len(trades) > 0:
        ts = {}
        ts["Total Trades"] = len(trades)
        winners = trades[trades["pnlcomm"] > 0]
        losers = trades[trades["pnlcomm"] <= 0]
        ts["Winning Trades"] = len(winners)
        ts["Losing Trades"] = len(losers)
        ts["Win Rate"] = len(winners) / len(trades) if len(trades) > 0 else 0
        ts["Total PnL"] = trades["pnlcomm"].sum()
        ts["Avg Win"] = winners["pnlcomm"].mean() if len(winners) > 0 else 0
        ts["Avg Loss"] = losers["pnlcomm"].mean() if len(losers) > 0 else 0
        ts["Max Win"] = winners["pnlcomm"].max() if len(winners) > 0 else 0
        ts["Max Loss"] = losers["pnlcomm"].min() if len(losers) > 0 else 0
        if ts["Avg Loss"] != 0:
            ts["Profit/Loss Ratio"] = abs(ts["Avg Win"] / ts["Avg Loss"])
        else:
            ts["Profit/Loss Ratio"] = np.nan
        ts["Total Commission"] = trades["commission"].sum() if "commission" in trades.columns else 0
        if "long" in trades.columns:
            ts["Long Trades"] = int((trades["long"] == 1).sum())
            ts["Short Trades"] = int((trades["long"] == 0).sum())
        if "barlen" in trades.columns:
            ts["Avg Holding Bars"] = trades["barlen"].mean()
            ts["Max Holding Bars"] = trades["barlen"].max()
            ts["Min Holding Bars"] = trades["barlen"].min()
        sections["trade_stats"] = ts
        sections["trade_pnl"] = trades["pnlcomm"].values

    return sections


# =========================================================================
# HTML 报告生成
# =========================================================================

_HTML_CSS = """\
<style>
:root { --blue: #1e40af; --green: #16a34a; --red: #dc2626; --gray: #6b7280; --bg: #f9fafb; }
* { box-sizing: border-box; margin: 0; padding: 0; }
body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
       max-width: 1100px; margin: 0 auto; padding: 24px; color: #1f2937; line-height: 1.6; background: #fff; }
h1 { color: var(--blue); border-bottom: 3px solid var(--blue); padding-bottom: 8px; margin-bottom: 16px; }
h2 { color: var(--blue); margin: 28px 0 12px; font-size: 1.25em; }
h3 { color: #374151; margin: 16px 0 8px; font-size: 1.05em; }
.meta { color: var(--gray); font-size: 0.9em; margin-bottom: 20px; }
.cards { display: flex; flex-wrap: wrap; gap: 10px; margin: 12px 0; }
.card { border: 1px solid #e5e7eb; border-radius: 8px; padding: 12px 18px;
        min-width: 150px; text-align: center; background: var(--bg); }
.card .val { font-size: 1.4em; font-weight: 700; }
.card .lbl { font-size: 0.8em; color: var(--gray); }
.pos { color: var(--green); }
.neg { color: var(--red); }
table { border-collapse: collapse; width: 100%; margin: 10px 0; font-size: 0.9em; }
th, td { border: 1px solid #d1d5db; padding: 6px 10px; }
th { background: #eff6ff; text-align: left; font-weight: 600; }
td { text-align: right; }
tr:nth-child(even) { background: var(--bg); }
.section { margin-bottom: 24px; }
.two-col { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }
@media (max-width: 700px) { .two-col { grid-template-columns: 1fr; } }
.footer { margin-top: 32px; padding-top: 12px; border-top: 1px solid #e5e7eb;
          font-size: 0.8em; color: #9ca3af; text-align: center; }
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


def _css_class(v):
    if isinstance(v, (int, float, np.integer, np.floating)):
        if np.isnan(v) if isinstance(v, float) else False:
            return ""
        return "pos" if v > 0 else ("neg" if v < 0 else "")
    return ""


def _dict_to_table(d, pct_keys=None):
    """dict → HTML table."""
    pct_keys = set(pct_keys or [])
    rows = []
    for k, v in d.items():
        pct = k in pct_keys
        css = _css_class(v)
        rows.append(f'<tr><th>{k}</th><td class="{css}">{_fmt(v, pct=pct)}</td></tr>')
    return "<table>" + "\n".join(rows) + "</table>"


def _df_to_table(df, float_format=".4f"):
    """DataFrame → HTML table."""
    rows = []
    # header
    header = "<tr><th></th>" + "".join(f"<th>{c}</th>" for c in df.columns) + "</tr>"
    rows.append(header)
    for idx, row in df.iterrows():
        cells = f"<th>{idx}</th>"
        for v in row:
            if isinstance(v, (float, np.floating)):
                css = _css_class(v)
                cells += f'<td class="{css}">{v:{float_format}}</td>'
            else:
                cells += f"<td>{v}</td>"
        rows.append(f"<tr>{cells}</tr>")
    return "<table>" + "\n".join(rows) + "</table>"


def _metric_cards(d, keys, pct_keys=None):
    """生成核心指标卡片 HTML。"""
    pct_keys = set(pct_keys or [])
    cards = []
    for k in keys:
        v = d.get(k, np.nan)
        css = _css_class(v)
        pct = k in pct_keys
        cards.append(
            f'<div class="card"><div class="val {css}">{_fmt(v, pct=pct)}</div>'
            f'<div class="lbl">{k}</div></div>'
        )
    return '<div class="cards">' + "\n".join(cards) + "</div>"


def _generate_html(returns, benchmark_rets, positions, transactions, trades,
                   title, output, rolling_window):
    """生成 HTML 报告。"""
    s = _compute_sections(returns, benchmark_rets, positions, transactions,
                          trades, rolling_window)

    parts = []
    parts.append(f"<h1>{title}</h1>")
    parts.append(f'<div class="meta">{s["date_range"][0]} → {s["date_range"][1]} '
                 f'| {s["n_days"]} trading days</div>')

    # --- 核心指标卡片 ---
    parts.append(_metric_cards(
        s["perf_stats"],
        ["Sharpe Ratio", "Annual Return", "Max Drawdown", "Annual Volatility",
         "Sortino Ratio", "Calmar Ratio"],
        pct_keys={"Annual Return", "Max Drawdown", "Annual Volatility"},
    ))

    # --- 绩效统计 ---
    parts.append('<div class="two-col"><div class="section">')
    parts.append("<h2>Performance Statistics</h2>")
    pct_keys = {"Annual Return", "Cumulative Returns", "Annual Volatility",
                "Max Drawdown", "Downside Risk", "Value at Risk (5%)"}
    parts.append(_dict_to_table(s["perf_stats"], pct_keys=pct_keys))
    parts.append("</div><div class='section'>")
    parts.append("<h2>Extended Risk Metrics</h2>")
    parts.append(_dict_to_table(s["extended_stats"]))
    parts.append("</div></div>")

    # --- Benchmark ---
    if "benchmark_stats" in s:
        parts.append('<div class="section">')
        parts.append("<h2>Benchmark Comparison</h2>")
        parts.append(_metric_cards(
            s["benchmark_stats"],
            ["Alpha", "Beta", "Information Ratio", "Tracking Error", "Up Capture", "Down Capture"],
        ))
        parts.append(_dict_to_table(s["benchmark_stats"]))
        parts.append("</div>")

    # --- 按年统计 ---
    parts.append('<div class="section">')
    parts.append("<h2>Yearly Statistics</h2>")
    parts.append(_df_to_table(s["yearly_stats"]))
    parts.append("</div>")

    # --- Top 回撤 ---
    parts.append('<div class="section">')
    parts.append("<h2>Worst Drawdown Periods</h2>")
    parts.append(_df_to_table(s["dd_table"], float_format=".2f"))
    parts.append("</div>")

    # --- 月度收益表 ---
    parts.append('<div class="section">')
    parts.append("<h2>Monthly Returns</h2>")
    monthly = s["monthly_returns"]
    monthly_tbl = monthly.unstack()
    if monthly_tbl is not None and len(monthly_tbl) > 0:
        monthly_tbl = monthly_tbl.fillna(0)
        monthly_tbl.columns = [f"{int(c):02d}" for c in monthly_tbl.columns]
        parts.append(_df_to_table(monthly_tbl * 100, float_format=".2f"))
    parts.append("</div>")

    # --- 持仓分析 ---
    if "has_positions" in s:
        parts.append('<div class="section">')
        parts.append("<h2>Position Analysis</h2>")
        pos_summary = {
            "Avg Gross Leverage": s["gross_leverage"].mean(),
            "Max Gross Leverage": s["gross_leverage"].max(),
            "Avg Long Exposure": s["pos_long"].mean(),
            "Avg Short Exposure": s["pos_short"].mean(),
        }
        parts.append(_dict_to_table(pos_summary))
        parts.append("</div>")

    # --- 交易分析 ---
    if "has_transactions" in s:
        parts.append('<div class="section">')
        parts.append("<h2>Transaction Analysis</h2>")
        txn_summary = {
            "Total Transaction Days": len(s["daily_txn_count"]),
            "Avg Daily Trades": s["daily_txn_count"].mean(),
            "Avg Daily Volume": s["daily_txn_value"].mean(),
            "Max Daily Volume": s["daily_txn_value"].max(),
        }
        parts.append(_dict_to_table(txn_summary))
        parts.append("</div>")

    # --- 交易统计 ---
    if "trade_stats" in s:
        parts.append('<div class="section">')
        parts.append("<h2>Trade Statistics</h2>")
        ts = s["trade_stats"]
        pct_keys_t = {"Win Rate"}
        parts.append(_metric_cards(
            ts,
            ["Total Trades", "Win Rate", "Profit/Loss Ratio", "Total PnL"],
            pct_keys=pct_keys_t,
        ))
        parts.append(_dict_to_table(ts, pct_keys=pct_keys_t))
        parts.append("</div>")

    # --- Footer ---
    parts.append('<div class="footer">Generated by <strong>fincore</strong> | '
                 'create_strategy_report()</div>')

    html = (
        "<!DOCTYPE html>\n<html lang='zh'><head><meta charset='utf-8'>\n"
        f"<title>{title}</title>\n{_HTML_CSS}\n</head>\n<body>\n"
        + "\n".join(parts)
        + "\n</body></html>"
    )

    with open(output, "w", encoding="utf-8") as f:
        f.write(html)

    return output


# =========================================================================
# PDF 报告生成
# =========================================================================

def _generate_pdf(returns, benchmark_rets, positions, transactions, trades,
                  title, output, rolling_window):
    """生成 PDF 报告（使用 matplotlib）。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    from matplotlib.backends.backend_pdf import PdfPages

    s = _compute_sections(returns, benchmark_rets, positions, transactions,
                          trades, rolling_window)

    pdf = PdfPages(output)
    page_count = 0

    def save_page(fig):
        nonlocal page_count
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)
        page_count += 1

    def dict_to_fig(d, fig_title, pct_keys=None):
        """将 dict 渲染为 matplotlib 表格页。"""
        pct_keys = set(pct_keys or [])
        n = len(d)
        fig, ax = plt.subplots(figsize=(10, max(3, 0.38 * n + 1.5)))
        ax.axis("off")
        ax.set_title(fig_title, fontsize=13, fontweight="bold", pad=12)
        cell_text = []
        for k, v in d.items():
            pct = k in pct_keys
            cell_text.append([k, _fmt(v, pct=pct)])
        tbl = ax.table(cellText=cell_text, colLabels=["Metric", "Value"],
                       cellLoc="center", loc="center")
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(9)
        tbl.scale(1.2, 1.4)
        fig.tight_layout()
        return fig

    def df_to_fig(df, fig_title, float_fmt=".4f"):
        """将 DataFrame 渲染为 matplotlib 表格页。"""
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
        tbl = ax.table(cellText=cell_text,
                       rowLabels=[str(i) for i in df.index],
                       colLabels=[str(c) for c in df.columns],
                       cellLoc="center", loc="center")
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(9)
        tbl.scale(1.1, 1.4)
        fig.tight_layout()
        return fig

    # === Page: Performance Statistics ===
    pct_keys = {"Annual Return", "Cumulative Returns", "Annual Volatility",
                "Max Drawdown", "Downside Risk", "Value at Risk (5%)"}
    save_page(dict_to_fig(s["perf_stats"],
              f"{title}\nPerformance Statistics", pct_keys=pct_keys))

    # === Page: Extended Metrics ===
    save_page(dict_to_fig(s["extended_stats"], "Extended Risk Metrics"))

    # === Page: Benchmark (if available) ===
    if "benchmark_stats" in s:
        save_page(dict_to_fig(s["benchmark_stats"], "Benchmark Comparison"))

    # === Page: Yearly + Drawdown Tables ===
    save_page(df_to_fig(s["yearly_stats"], "Yearly Statistics"))
    save_page(df_to_fig(s["dd_table"], "Worst Drawdown Periods", float_fmt=".2f"))

    # === Page: Cumulative Returns + Drawdown ===
    fig = plt.figure(figsize=(14, 10))
    gs = gridspec.GridSpec(2, 1, hspace=0.35)

    ax1 = fig.add_subplot(gs[0])
    ax1.plot(s["cum_returns"].index, s["cum_returns"].values,
             color="steelblue", linewidth=1.2, label="Strategy")
    if "benchmark_cum" in s:
        ax1.plot(s["benchmark_cum"].index, s["benchmark_cum"].values,
                 color="gray", linewidth=0.9, alpha=0.7, label="Benchmark")
        ax1.legend()
    ax1.axhline(y=1.0, color="gray", linestyle="--", linewidth=0.6)
    ax1.set_title("Cumulative Returns", fontsize=12, fontweight="bold")
    ax1.set_ylabel("Growth of $1")
    ax1.grid(True, alpha=0.3)

    ax2 = fig.add_subplot(gs[1], sharex=ax1)
    dd = s["drawdown"]
    ax2.fill_between(dd.index, dd.values, 0, color="red", alpha=0.3)
    ax2.plot(dd.index, dd.values, color="red", linewidth=0.8)
    ax2.set_title("Drawdown", fontsize=12, fontweight="bold")
    ax2.set_ylabel("Drawdown")
    ax2.grid(True, alpha=0.3)
    save_page(fig)

    # === Page: Rolling Sharpe + Volatility ===
    fig = plt.figure(figsize=(14, 10))
    gs = gridspec.GridSpec(2, 1, hspace=0.35)

    ax1 = fig.add_subplot(gs[0])
    rs = s["rolling_sharpe"].dropna()
    ax1.plot(rs.index, rs.values, color="steelblue", linewidth=0.9)
    ax1.axhline(y=0, color="gray", linestyle="--", linewidth=0.6)
    ax1.set_title(f"Rolling Sharpe Ratio ({rolling_window}d)", fontsize=12, fontweight="bold")
    ax1.grid(True, alpha=0.3)

    ax2 = fig.add_subplot(gs[1], sharex=ax1)
    rv = s["rolling_volatility"].dropna()
    ax2.plot(rv.index, rv.values, color="orange", linewidth=0.9)
    ax2.set_title(f"Rolling Volatility ({rolling_window}d)", fontsize=12, fontweight="bold")
    ax2.grid(True, alpha=0.3)
    save_page(fig)

    # === Page: Rolling Beta (if benchmark) ===
    if "rolling_beta" in s:
        fig, ax = plt.subplots(figsize=(14, 5))
        rb = s["rolling_beta"].dropna()
        ax.plot(rb.index, rb.values, color="purple", linewidth=0.9)
        ax.axhline(y=1.0, color="gray", linestyle="--", linewidth=0.6)
        ax.set_title(f"Rolling Beta ({rolling_window}d)", fontsize=12, fontweight="bold")
        ax.grid(True, alpha=0.3)
        save_page(fig)

    # === Page: Monthly Heatmap + Annual Returns ===
    monthly = s["monthly_returns"]
    monthly_tbl = monthly.unstack().fillna(0)

    fig = plt.figure(figsize=(14, 6))
    gs = gridspec.GridSpec(1, 2, wspace=0.3)

    ax1 = fig.add_subplot(gs[0])
    import matplotlib.colors as mcolors
    vals = monthly_tbl.values * 100
    try:
        norm = mcolors.TwoSlopeNorm(vmin=vals.min(), vcenter=0, vmax=max(vals.max(), 0.01))
    except ValueError:
        norm = None
    im = ax1.imshow(vals, cmap="RdYlGn", norm=norm, aspect="auto")
    ax1.set_xticks(range(vals.shape[1]))
    ax1.set_xticklabels([f"{int(c):02d}" for c in monthly_tbl.columns], fontsize=8)
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
    save_page(fig)

    # === Page: Position Analysis (if available) ===
    if "has_positions" in s:
        fig = plt.figure(figsize=(14, 10))
        gs = gridspec.GridSpec(2, 1, hspace=0.35)

        ax1 = fig.add_subplot(gs[0])
        ax1.fill_between(s["pos_long"].index, s["pos_long"].values, 0,
                         color="green", alpha=0.3, label="Long")
        ax1.fill_between(s["pos_short"].index, s["pos_short"].values, 0,
                         color="red", alpha=0.3, label="Short")
        ax1.legend()
        ax1.set_title("Long / Short Exposure", fontsize=12, fontweight="bold")
        ax1.grid(True, alpha=0.3)

        ax2 = fig.add_subplot(gs[1], sharex=ax1)
        gl = s["gross_leverage"].dropna()
        ax2.plot(gl.index, gl.values, color="navy", linewidth=0.9)
        ax2.set_title("Gross Leverage", fontsize=12, fontweight="bold")
        ax2.grid(True, alpha=0.3)
        save_page(fig)

    # === Page: Transaction Analysis (if available) ===
    if "has_transactions" in s:
        fig = plt.figure(figsize=(14, 10))
        gs = gridspec.GridSpec(2, 1, hspace=0.35)

        ax1 = fig.add_subplot(gs[0])
        dv = s["daily_txn_value"]
        ax1.bar(dv.index, dv.values, color="steelblue", alpha=0.7, width=1)
        ax1.set_title("Daily Transaction Volume", fontsize=12, fontweight="bold")
        ax1.grid(True, alpha=0.3)

        ax2 = fig.add_subplot(gs[1])
        dc = s["daily_txn_count"]
        ax2.bar(dc.index, dc.values, color="orange", alpha=0.7, width=1)
        ax2.set_title("Daily Transaction Count", fontsize=12, fontweight="bold")
        ax2.grid(True, alpha=0.3)
        save_page(fig)

    # === Page: Trade Statistics (if available) ===
    if "trade_stats" in s:
        ts = s["trade_stats"]
        pct_keys_t = {"Win Rate"}
        save_page(dict_to_fig(ts, "Trade Statistics", pct_keys=pct_keys_t))

        # PnL Distribution
        if "trade_pnl" in s:
            fig, ax = plt.subplots(figsize=(12, 5))
            pnl = s["trade_pnl"]
            colors_hist = ["green" if v > 0 else "red" for v in sorted(pnl)]
            ax.hist(pnl, bins=min(50, len(pnl) // 2 + 1), color="steelblue",
                    alpha=0.7, edgecolor="white")
            ax.axvline(x=0, color="gray", linestyle="--", linewidth=1)
            ax.axvline(x=np.mean(pnl), color="blue", linestyle="-", linewidth=1,
                       label=f"Mean: {np.mean(pnl):,.0f}")
            ax.legend()
            ax.set_title("Trade PnL Distribution", fontsize=12, fontweight="bold")
            ax.set_xlabel("PnL (after commission)")
            ax.grid(True, alpha=0.3)
            save_page(fig)

    pdf.close()
    return output
