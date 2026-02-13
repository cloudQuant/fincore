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

Modular structure
-----------------
- ``compute``     – statistics computation engine
- ``format``      – CSS styles and HTML formatting helpers
- ``render_html`` – HTML body assembly + ECharts JavaScript
- ``render_pdf``  – PDF rendering via Playwright + PyPDF2 bookmarks
"""

from __future__ import annotations

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
        from fincore.report.render_pdf import generate_pdf

        return generate_pdf(
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
        from fincore.report.render_html import generate_html

        return generate_html(
            returns,
            benchmark_rets=benchmark_rets,
            positions=positions,
            transactions=transactions,
            trades=trades,
            title=title,
            output=output,
            rolling_window=rolling_window,
        )


__all__ = ["create_strategy_report"]
