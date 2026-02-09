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

# --- Returns Tear Sheet ---
print("\n[1/4] 生成 Returns Tear Sheet ...")
fig = pf.create_returns_tear_sheet(
    daily_returns,
    positions=positions,
    transactions=transactions,
    run_flask_app=True,  # 返回 fig 对象而不直接 show
)
if fig is not None:
    fig.savefig(os.path.join(OUTPUT_DIR, "tearsheet_returns.pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"  → 已保存: output/tearsheet_returns.pdf")

# --- Position Tear Sheet ---
print("[2/4] 生成 Position Tear Sheet ...")
fig = pf.create_position_tear_sheet(
    daily_returns,
    positions,
    run_flask_app=True,
)
if fig is not None:
    fig.savefig(os.path.join(OUTPUT_DIR, "tearsheet_positions.pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"  → 已保存: output/tearsheet_positions.pdf")

# --- Transaction Tear Sheet ---
print("[3/4] 生成 Transaction Tear Sheet ...")
fig = pf.create_txn_tear_sheet(
    daily_returns,
    positions,
    transactions,
    run_flask_app=True,
)
if fig is not None:
    fig.savefig(os.path.join(OUTPUT_DIR, "tearsheet_txn.pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"  → 已保存: output/tearsheet_txn.pdf")

# --- Interesting Times Tear Sheet ---
print("[4/4] 生成 Interesting Times Tear Sheet ...")
fig = pf.create_interesting_times_tear_sheet(
    daily_returns,
    run_flask_app=True,
)
if fig is not None:
    fig.savefig(os.path.join(OUTPUT_DIR, "tearsheet_interesting_times.pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"  → 已保存: output/tearsheet_interesting_times.pdf")

print(f"\n{'=' * 60}")
print(f"所有 Tear Sheet 已保存到: {OUTPUT_DIR}/")
print(f"{'=' * 60}")
