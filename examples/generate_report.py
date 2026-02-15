"""
Generate strategy performance reports using fincore.create_strategy_report.

Demonstrates how create_strategy_report builds different sections based on
the inputs you provide:
  - returns only -> basic performance report
  - + positions -> adds holdings/positions analysis
  - + transactions -> adds transaction analysis
  - + trades -> adds trade statistics (win rate, profit factor, etc.)

Also demonstrates both HTML and PDF outputs.

Usage:
    python generate_report.py
"""

import json
import os
import sys

import pandas as pd

# Ensure we import local source (not an older site-packages install).
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import fincore

# =========================================================================
# 1. Load data
# =========================================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, "011_abberation", "logs")
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "011_abberation", "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

run_dirs = sorted([d for d in os.listdir(DATA_DIR) if os.path.isdir(os.path.join(DATA_DIR, d))])
RUN_DIR = os.path.join(DATA_DIR, run_dirs[-1])

with open(os.path.join(RUN_DIR, "run_info.json")) as f:
    run_info = json.load(f)

print(f"Strategy: {run_info['strategy_name']}")

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
transactions = pd.DataFrame(
    {
        "amount": completed["size"].values,
        "price": completed["executed_price"].values,
        "symbol": completed["data_name"].values,
    },
    index=txn_index,
)

# --- trades (closed trades only) ---
trade_df = pd.read_csv(os.path.join(RUN_DIR, "trade.log"), sep="\t")
closed_trades = trade_df[trade_df["status"] == "Closed"].copy()

print(f"Dates: {daily_returns.index[0].strftime('%Y-%m-%d')} -> {daily_returns.index[-1].strftime('%Y-%m-%d')}")
print(f"Trading days: {len(daily_returns)}, transactions: {len(transactions)}, closed trades: {len(closed_trades)}")

# =========================================================================
# 2. Generate reports - add inputs progressively to show dynamic sections
# =========================================================================

# --- Report 1: returns only (minimal) ---
print("\n[Report 1] returns only -> HTML ...")
out1 = fincore.create_strategy_report(
    daily_returns,
    title=f"{run_info['strategy_name']} — Returns Only",
    output=os.path.join(OUTPUT_DIR, "report_returns_only.html"),
)
print(f"  → {out1}")

# --- Report 2: returns + positions + transactions ---
print("[Report 2] returns + positions + transactions -> HTML ...")
out2 = fincore.create_strategy_report(
    daily_returns,
    positions=positions,
    transactions=transactions,
    title=f"{run_info['strategy_name']} — With Positions & Transactions",
    output=os.path.join(OUTPUT_DIR, "report_with_pos_txn.html"),
)
print(f"  → {out2}")

# --- Report 3: full inputs -> HTML ---
print("[Report 3] full inputs -> HTML ...")
out3 = fincore.create_strategy_report(
    daily_returns,
    positions=positions,
    transactions=transactions,
    trades=closed_trades,
    title=f"{run_info['strategy_name']} — Full Report",
    output=os.path.join(OUTPUT_DIR, "report_full.html"),
)
print(f"  → {out3}")

# --- Report 4: full inputs -> PDF ---
print("[Report 4] full inputs -> PDF ...")
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
print("All reports generated.")
print(f"{'=' * 60}")
print("""
Report comparison:
  Report 1 (returns only):       basic performance + return charts
  Report 2 (+ pos + txn):        adds positions + transactions analysis
  Report 3 (+ trades, HTML):     adds trade stats (win rate, profit factor, PnL distribution)
  Report 4 (+ trades, PDF):      same as report 3, but PDF output
""")
