"""
使用 fincore.create_strategy_report 生成策略分析报告。

演示 create_strategy_report 根据传入数据动态生成报告：
  - 只传 returns → 基础绩效报告
  - 加 positions → 增加持仓分析
  - 加 transactions → 增加交易分析
  - 加 trades → 增加交易统计（胜率、盈亏比等）

同时演示 HTML 和 PDF 两种输出格式。

使用方式:
    python generate_report.py
"""
import os
import sys
import json
import pandas as pd

# 确保导入本地源码（而非 site-packages 中的旧版本）
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import fincore

# =========================================================================
# 1. 读取数据
# =========================================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, "011_abberation", "logs")
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "011_abberation", "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

run_dirs = sorted(
    [d for d in os.listdir(DATA_DIR) if os.path.isdir(os.path.join(DATA_DIR, d))]
)
RUN_DIR = os.path.join(DATA_DIR, run_dirs[-1])

with open(os.path.join(RUN_DIR, "run_info.json")) as f:
    run_info = json.load(f)

print(f"策略: {run_info['strategy_name']}")

# --- returns ---
value_df = pd.read_csv(os.path.join(RUN_DIR, "value.log"), sep="\t", usecols=["dt", "value", "cash"])
value_df["dt"] = pd.to_datetime(value_df["dt"])
daily_value = value_df.groupby(value_df["dt"].dt.date).last()
daily_value.index = pd.to_datetime(daily_value.index)
daily_returns = daily_value["value"].pct_change().dropna()
daily_returns.name = "strategy"
daily_returns.index = daily_returns.index.tz_localize("UTC")

# --- positions ---
pos_df = pd.read_csv(os.path.join(RUN_DIR, "position.log"), sep="\t", usecols=["dt", "data_name", "size", "price"])
pos_df["dt"] = pd.to_datetime(pos_df["dt"])
positions = pd.DataFrame(index=daily_value.index)
for asset in pos_df["data_name"].unique():
    ap = pos_df[pos_df["data_name"] == asset]
    ad = ap.groupby(ap["dt"].dt.date).last()
    ad.index = pd.to_datetime(ad.index)
    positions[asset] = ad["size"] * ad["price"]
positions["cash"] = daily_value["cash"]
positions = positions.fillna(0)
positions.index = positions.index.tz_localize("UTC")
positions = positions.loc[daily_returns.index]

# --- transactions ---
order_df = pd.read_csv(os.path.join(RUN_DIR, "order.log"), sep="\t")
completed = order_df[order_df["status"] == "Completed"].copy()
completed["dt"] = pd.to_datetime(completed["dt"])
txn_index = pd.DatetimeIndex(completed["dt"].values).tz_localize("UTC")
transactions = pd.DataFrame({
    "amount": completed["size"].values,
    "price": completed["executed_price"].values,
    "symbol": completed["data_name"].values,
}, index=txn_index)

# --- trades (已平仓交易记录) ---
trade_df = pd.read_csv(os.path.join(RUN_DIR, "trade.log"), sep="\t")
closed_trades = trade_df[trade_df["status"] == "Closed"].copy()

print(f"日期: {daily_returns.index[0].strftime('%Y-%m-%d')} → {daily_returns.index[-1].strftime('%Y-%m-%d')}")
print(f"交易日: {len(daily_returns)}, 成交: {len(transactions)}, 已平仓交易: {len(closed_trades)}")

# =========================================================================
# 2. 生成报告 — 逐步增加数据，展示动态内容
# =========================================================================

# --- 报告 1: 只有 returns（最简单） ---
print("\n[报告 1] 只传 returns → HTML ...")
out1 = fincore.create_strategy_report(
    daily_returns,
    title=f"{run_info['strategy_name']} — Returns Only",
    output=os.path.join(OUTPUT_DIR, "report_returns_only.html"),
)
print(f"  → {out1}")

# --- 报告 2: returns + positions + transactions ---
print("[报告 2] returns + positions + transactions → HTML ...")
out2 = fincore.create_strategy_report(
    daily_returns,
    positions=positions,
    transactions=transactions,
    title=f"{run_info['strategy_name']} — With Positions & Transactions",
    output=os.path.join(OUTPUT_DIR, "report_with_pos_txn.html"),
)
print(f"  → {out2}")

# --- 报告 3: 完整数据 → HTML ---
print("[报告 3] 完整数据 → HTML ...")
out3 = fincore.create_strategy_report(
    daily_returns,
    positions=positions,
    transactions=transactions,
    trades=closed_trades,
    title=f"{run_info['strategy_name']} — Full Report",
    output=os.path.join(OUTPUT_DIR, "report_full.html"),
)
print(f"  → {out3}")

# --- 报告 4: 完整数据 → PDF ---
print("[报告 4] 完整数据 → PDF ...")
out4 = fincore.create_strategy_report(
    daily_returns,
    positions=positions,
    transactions=transactions,
    trades=closed_trades,
    title=f"{run_info['strategy_name']} — Full Report",
    output=os.path.join(OUTPUT_DIR, "report_full.pdf"),
)
print(f"  → {out4}")

print(f"\n{'=' * 60}")
print("所有报告已生成！")
print(f"{'=' * 60}")
print(f"""
报告对比:
  报告 1 (returns only):         只有基础绩效 + 收益图表
  报告 2 (+ pos + txn):          增加持仓分析 + 交易分析
  报告 3 (+ trades, HTML):       增加交易统计（胜率、盈亏比、PnL分布）
  报告 4 (+ trades, PDF):        同报告 3，PDF 格式
""")
