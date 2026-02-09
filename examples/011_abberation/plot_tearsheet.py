"""
AbberationStrategy Pyfolio 画图分析脚本

从 logs 目录读取净值、持仓、交易数据，转换为 Pyfolio 所需格式，
调用 Pyfolio 生成 tear sheet 图表并保存为 PDF。

使用方式:
    python plot_tearsheet.py

输出:
    tearsheet_returns.pdf   — 收益分析报告
    tearsheet_positions.pdf — 持仓分析报告
    tearsheet_txn.pdf       — 交易分析报告
    tearsheet_full.pdf      — 完整分析报告
"""
import os
import json
import warnings
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from fincore import Pyfolio

warnings.filterwarnings("ignore", category=UserWarning)

# =========================================================================
# 1. 定位日志目录
# =========================================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
LOGS_DIR = os.path.join(SCRIPT_DIR, "logs")
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

run_dirs = sorted(
    [d for d in os.listdir(LOGS_DIR) if os.path.isdir(os.path.join(LOGS_DIR, d))],
)
if not run_dirs:
    raise FileNotFoundError(f"No run directories found in {LOGS_DIR}")
RUN_DIR = os.path.join(LOGS_DIR, run_dirs[-1])
print(f"分析运行目录: {RUN_DIR}")

with open(os.path.join(RUN_DIR, "run_info.json"), "r") as f:
    run_info = json.load(f)
print(f"策略: {run_info['strategy_name']}")

# =========================================================================
# 2. 读取 value.log → 日收益率 (returns) + 日现金 (cash)
# =========================================================================
print("\n[1/3] 读取 value.log ...")
value_df = pd.read_csv(
    os.path.join(RUN_DIR, "value.log"),
    sep="\t",
    usecols=["dt", "value", "cash"],
)
value_df["dt"] = pd.to_datetime(value_df["dt"])

# 取每日最后一根K线
daily_value = value_df.groupby(value_df["dt"].dt.date).last()
daily_value.index = pd.to_datetime(daily_value.index)

# 日收益率
daily_returns = daily_value["value"].pct_change().dropna()
daily_returns.name = "strategy"
daily_returns.index = daily_returns.index.tz_localize("UTC")
daily_returns.index.name = None

print(f"  日期范围: {daily_returns.index[0].strftime('%Y-%m-%d')} → {daily_returns.index[-1].strftime('%Y-%m-%d')}")
print(f"  交易日数: {len(daily_returns)}")

# =========================================================================
# 3. 读取 position.log + value.log → 日持仓 DataFrame (positions)
#
# Pyfolio 要求 positions 格式:
#   - DatetimeIndex (每日)
#   - 列名为资产名称 + 'cash'
#   - 值为持仓的名义价值 (notional value)
# =========================================================================
print("[2/3] 读取 position.log → 构建 positions DataFrame ...")
pos_df = pd.read_csv(
    os.path.join(RUN_DIR, "position.log"),
    sep="\t",
    usecols=["dt", "data_name", "size", "price"],
)
pos_df["dt"] = pd.to_datetime(pos_df["dt"])

# 取每日最后一根K线的持仓
daily_pos = pos_df.groupby(pos_df["dt"].dt.date).last()
daily_pos.index = pd.to_datetime(daily_pos.index)

# 计算各资产的持仓名义价值: size * price
# 注: 对于期货，这里用 size * price 近似名义价值
# 构建 positions DataFrame: columns = [asset_names..., 'cash']
asset_names = pos_df["data_name"].unique()

positions = pd.DataFrame(index=daily_value.index)
for asset in asset_names:
    asset_pos = pos_df[pos_df["data_name"] == asset].copy()
    asset_daily = asset_pos.groupby(asset_pos["dt"].dt.date).last()
    asset_daily.index = pd.to_datetime(asset_daily.index)
    # 名义价值 = size * price
    positions[asset] = asset_daily["size"] * asset_daily["price"]

positions["cash"] = daily_value["cash"]
positions = positions.fillna(0)
positions.index = positions.index.tz_localize("UTC")

# 对齐到 returns 的日期范围（去掉第一天，因为 returns 是从第二天开始的）
positions = positions.loc[daily_returns.index]

print(f"  资产列:   {list(asset_names)}")
print(f"  持仓行数: {len(positions)}")

# =========================================================================
# 4. 读取 order.log → 交易 DataFrame (transactions)
#
# Pyfolio 要求 transactions 格式:
#   - DatetimeIndex (交易时间)
#   - 列: amount (正=买, 负=卖), price, symbol
# =========================================================================
print("[3/3] 读取 order.log → 构建 transactions DataFrame ...")
order_df = pd.read_csv(
    os.path.join(RUN_DIR, "order.log"),
    sep="\t",
)

# 只取已成交的订单
completed = order_df[order_df["status"] == "Completed"].copy()
completed["dt"] = pd.to_datetime(completed["dt"])

txn_index = pd.DatetimeIndex(completed["dt"].values).tz_localize("UTC")
transactions = pd.DataFrame({
    "amount": completed["size"].values,
    "price": completed["executed_price"].values,
    "symbol": completed["data_name"].values,
}, index=txn_index)
transactions.index.name = None

print(f"  成交笔数: {len(transactions)}")

# =========================================================================
# 5. 创建 Pyfolio 实例并生成 tear sheet
# =========================================================================
print(f"\n{'=' * 60}")
print("生成 Pyfolio Tear Sheet ...")
print(f"{'=' * 60}")

pf = Pyfolio(returns=daily_returns)

from matplotlib.backends.backend_pdf import PdfPages
import fincore.utils.common_utils as _cu

# ---------------------------------------------------------------------------
# Monkey-patch print_table: 在原有 HTML 输出之外，额外生成 matplotlib figure，
# 这样 create_full_tear_sheet 内部的 show_perf_stats / show_worst_drawdown_periods
# 产生的表格也能被 plt.get_fignums() 捕获并写入 PDF。
# ---------------------------------------------------------------------------
_original_print_table = _cu.print_table


def _print_table_with_figure(table, name=None, float_format=None,
                             formatters=None, header_rows=None,
                             run_flask_app=False):
    """print_table 的增强版：同时生成 matplotlib 表格 figure。"""
    if isinstance(table, pd.Series):
        table = pd.DataFrame(table)

    # 构建展示用的 DataFrame（含 header_rows）
    display_df = table.copy()
    if header_rows:
        for k, v in reversed(header_rows.items()):
            row = pd.DataFrame(
                [[v] * len(display_df.columns)],
                index=[k], columns=display_df.columns,
            )
            display_df = pd.concat([row, display_df])

    title = name if name else ""
    nrows = len(display_df)
    fig, ax = plt.subplots(figsize=(12, max(2.5, 0.38 * nrows + 1.5)))
    ax.axis("off")
    if title:
        ax.set_title(title, fontsize=13, fontweight="bold", pad=10)

    # 格式化单元格
    fmt = float_format or "{:.4f}".format
    cell_text = []
    for _, row in display_df.iterrows():
        cells = []
        for v in row:
            if isinstance(v, (int, float, np.floating)):
                try:
                    cells.append(fmt(v))
                except (ValueError, TypeError):
                    cells.append(str(v))
            else:
                cells.append(str(v))
        cell_text.append(cells)

    tbl = ax.table(
        cellText=cell_text,
        rowLabels=[str(i) for i in display_df.index],
        colLabels=[str(c) for c in display_df.columns],
        cellLoc="center",
        loc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1.2, 1.4)
    fig.tight_layout()
    # figure 留在 plt 中，稍后统一收集


# 应用 patch
_cu.print_table = _print_table_with_figure
# tearsheets 模块中的 print_table 引用也需要 patch
import fincore.tearsheets.returns as _tr
import fincore.tearsheets.transactions as _ttxn
_tr.print_table = _print_table_with_figure
if hasattr(_ttxn, 'print_table'):
    _ttxn.print_table = _print_table_with_figure

# --- 只调用一个函数：create_full_tear_sheet ---
print("\n生成 Full Tear Sheet ...")
pf.create_full_tear_sheet(
    daily_returns,
    positions=positions,
    transactions=transactions,
)

# 恢复原始 print_table
_cu.print_table = _original_print_table
_tr.print_table = _original_print_table
if hasattr(_ttxn, 'print_table'):
    _ttxn.print_table = _original_print_table

# --- 收集所有 figure 保存为多页 PDF ---
pdf_path = os.path.join(OUTPUT_DIR, "tearsheet_full.pdf")
all_figs = [plt.figure(n) for n in plt.get_fignums()]

with PdfPages(pdf_path) as pdf:
    for fig in all_figs:
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

print(f"  → 已保存: {pdf_path}  ({len(all_figs)} 页)")

print(f"\n{'=' * 60}")
print("完成")
print(f"{'=' * 60}")
