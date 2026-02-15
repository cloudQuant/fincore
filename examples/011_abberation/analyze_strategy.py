"""
AbberationStrategy analysis script.

Reads a strategy run from the local logs directory and performs a performance
analysis using fincore.

Usage:
    python analyze_strategy.py
"""

import json
import os

import numpy as np
import pandas as pd

import fincore
from fincore import Empyrical

# ============================================================================
# 1. Locate log directory
# ============================================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
LOGS_DIR = os.path.join(SCRIPT_DIR, "logs")

# Auto-select the latest run directory.
run_dirs = sorted(
    [d for d in os.listdir(LOGS_DIR) if os.path.isdir(os.path.join(LOGS_DIR, d))],
)
if not run_dirs:
    raise FileNotFoundError(f"No run directories found in {LOGS_DIR}")
RUN_DIR = os.path.join(LOGS_DIR, run_dirs[-1])
print(f"Analyzing run directory: {RUN_DIR}\n")

# ============================================================================
# 2. Load run info
# ============================================================================
with open(os.path.join(RUN_DIR, "run_info.json")) as f:
    run_info = json.load(f)

print("=" * 70)
print(f"Strategy:   {run_info['strategy_name']}")
print(f"Run time:   {run_info['run_datetime']}")
print(f"Run ID:     {run_info['run_id']}")
print("=" * 70)

# ============================================================================
# 3. Load value.log -> compute daily returns
# ============================================================================
value_df = pd.read_csv(
    os.path.join(RUN_DIR, "value.log"),
    sep="\t",
    usecols=["dt", "value", "cash"],
)
value_df["dt"] = pd.to_datetime(value_df["dt"])

# Use the last bar of each day as end-of-day NAV.
daily_value = value_df.groupby(value_df["dt"].dt.date)["value"].last()
daily_value.index = pd.to_datetime(daily_value.index)
daily_value.name = "portfolio_value"

# Daily returns.
daily_returns = daily_value.pct_change().dropna()
daily_returns.name = "strategy"

initial_capital = value_df["value"].iloc[0]
final_value = value_df["value"].iloc[-1]

print(f"\nInitial capital: {initial_capital:>15,.2f}")
print(f"Final NAV:       {final_value:>15,.2f}")
print(f"Total P&L:       {final_value - initial_capital:>15,.2f}")
print(f"Total return:    {(final_value / initial_capital - 1) * 100:>14.2f}%")
print(f"Trading days:    {len(daily_returns):>15d}")
print(
    f"Date range:      {daily_returns.index[0].strftime('%Y-%m-%d')} -> {daily_returns.index[-1].strftime('%Y-%m-%d')}"
)

# ============================================================================
# 4. Load trade.log -> trade statistics
# ============================================================================
trade_df = pd.read_csv(os.path.join(RUN_DIR, "trade.log"), sep="\t")

# Only closed trades.
closed_trades = trade_df[trade_df["status"] == "Closed"].copy()
closed_trades["dtopen"] = pd.to_datetime(closed_trades["dtopen"])
closed_trades["dtclose"] = pd.to_datetime(closed_trades["dtclose"])

total_trades = len(closed_trades)
winning_trades = closed_trades[closed_trades["pnlcomm"] > 0]
losing_trades = closed_trades[closed_trades["pnlcomm"] <= 0]
long_trades = closed_trades[closed_trades["long"] == 1]
short_trades = closed_trades[closed_trades["long"] == 0]

total_commission = closed_trades["commission"].sum()
total_pnl = closed_trades["pnl"].sum()
total_pnl_after_comm = closed_trades["pnlcomm"].sum()

print(f"\n{'=' * 70}")
print("Trade statistics")
print(f"{'=' * 70}")
print(f"Total trades:    {total_trades}")
print(f"  Long trades:   {len(long_trades)}")
print(f"  Short trades:  {len(short_trades)}")
print(f"Wins:            {len(winning_trades)}  ({len(winning_trades) / total_trades * 100:.1f}%)")
print(f"Losses:          {len(losing_trades)}  ({len(losing_trades) / total_trades * 100:.1f}%)")
print(f"Gross P&L:       {total_pnl:>15,.2f}")
print(f"Commission:      {total_commission:>15,.2f}")
print(f"Net P&L:         {total_pnl_after_comm:>15,.2f}")

if len(winning_trades) > 0:
    avg_win = winning_trades["pnlcomm"].mean()
    max_win = winning_trades["pnlcomm"].max()
else:
    avg_win = max_win = 0

if len(losing_trades) > 0:
    avg_loss = losing_trades["pnlcomm"].mean()
    max_loss = losing_trades["pnlcomm"].min()
else:
    avg_loss = max_loss = 0

print(f"\nAvg win:         {avg_win:>15,.2f}")
print(f"Avg loss:        {avg_loss:>15,.2f}")
print(f"Max win:         {max_win:>15,.2f}")
print(f"Max loss:        {max_loss:>15,.2f}")
if avg_loss != 0:
    print(f"Win/Loss ratio:  {abs(avg_win / avg_loss):>15.2f}")

# Holding duration.
closed_trades["holding_bars"] = closed_trades["barlen"]
print(f"\nAvg holding bars: {closed_trades['holding_bars'].mean():>14.1f}")
print(f"Max holding bars: {closed_trades['holding_bars'].max():>14d}")
print(f"Min holding bars: {closed_trades['holding_bars'].min():>14d}")

# ============================================================================
# 5. Performance analysis with fincore
# ============================================================================
print(f"\n{'=' * 70}")
print("fincore performance analysis")
print(f"{'=' * 70}")

# --- Option 1: Flat API ---
print("\n--- Core metrics (Flat API) ---")
print(f"Annual return:           {fincore.annual_return(daily_returns):.4f}")
print(f"Annual volatility:       {fincore.annual_volatility(daily_returns):.4f}")
print(f"Sharpe ratio:            {fincore.sharpe_ratio(daily_returns):.4f}")
print(f"Max drawdown:            {fincore.max_drawdown(daily_returns):.4f}")
print(f"Sortino ratio:           {fincore.sortino_ratio(daily_returns):.4f}")
print(f"Calmar ratio:            {fincore.calmar_ratio(daily_returns):.4f}")
print(f"Cumulative return:       {fincore.cum_returns_final(daily_returns):.4f}")
print(f"Value at risk (5%):      {fincore.value_at_risk(daily_returns):.4f}")
print(f"Downside risk:           {fincore.downside_risk(daily_returns):.4f}")
print(f"Tail ratio:              {fincore.tail_ratio(daily_returns):.4f}")

# --- Option 2: Empyrical class methods ---
print("\n--- Extended metrics (Empyrical class methods) ---")
print(f"Skewness:                {Empyrical.skewness(daily_returns):.4f}")
print(f"Kurtosis:                {Empyrical.kurtosis(daily_returns):.4f}")
print(f"Omega ratio:             {Empyrical.omega_ratio(daily_returns):.4f}")
print(f"Hurst exponent:          {Empyrical.hurst_exponent(daily_returns):.4f}")
print(f"Stability of time series:{Empyrical.stability_of_timeseries(daily_returns):.4f}")

# Consecutive moves.
print(f"\nMax consecutive up days:  {Empyrical.max_consecutive_up_days(daily_returns)}")
print(f"Max consecutive down days:{Empyrical.max_consecutive_down_days(daily_returns)}")
print(f"Max single-day gain:      {Empyrical.max_single_day_gain(daily_returns):.4f}")
print(f"Max single-day loss:      {Empyrical.max_single_day_loss(daily_returns):.4f}")

# --- Option 3: Empyrical instance methods (returns pre-filled) ---
print("\n--- Instance methods (returns pre-filled) ---")
emp = Empyrical(returns=daily_returns)

print(f"Win rate:                {emp.win_rate():.4f}")
print(f"Loss rate:               {emp.loss_rate():.4f}")
print(f"Serial correlation:       {emp.serial_correlation():.4f}")
print(f"Common sense ratio:       {emp.common_sense_ratio():.4f}")
print(f"Max drawdown days:        {emp.max_drawdown_days()}")
print(f"Drawdown recovery days:   {emp.max_drawdown_recovery_days()}")
print(f"2nd max drawdown:         {emp.second_max_drawdown():.4f}")
print(f"3rd max drawdown:         {emp.third_max_drawdown():.4f}")

# Sterling / Burke ratios.
print(f"Sterling ratio:           {emp.sterling_ratio():.4f}")
print(f"Burke ratio:              {emp.burke_ratio():.4f}")
print(f"Kappa 3 ratio:            {emp.kappa_three_ratio():.4f}")

# Year-by-year analysis.
print("\n--- Year-by-year stats ---")
annual_by_year = Empyrical.annual_return_by_year(daily_returns)
sharpe_by_year = Empyrical.sharpe_ratio_by_year(daily_returns)
dd_by_year = Empyrical.max_drawdown_by_year(daily_returns)

yearly_stats = pd.DataFrame(
    {
        "Annual return": annual_by_year,
        "Sharpe ratio": sharpe_by_year,
        "Max drawdown": dd_by_year,
    }
)
print(yearly_stats.to_string(float_format=lambda x: f"{x:.4f}"))

# Summary stats.
print("\n--- Summary stats (perf_stats) ---")
stats = Empyrical.perf_stats(daily_returns)
print(stats.to_string(float_format=lambda x: f"{x:.4f}"))

# Drawdown analysis.
print("\n--- Top 5 drawdowns ---")
dd_table = Empyrical.gen_drawdown_table(daily_returns, top=5)
print(dd_table.to_string())

# Monthly returns.
print("\n--- Monthly returns ---")
monthly_returns = Empyrical.aggregate_returns(daily_returns, "monthly")
print(monthly_returns.tail(12).to_string(float_format=lambda x: f"{x:.4f}"))

# ============================================================================
# 6. Summary
# ============================================================================
print(f"\n{'=' * 70}")
print("Analysis complete")
print(f"{'=' * 70}")
print(f"Strategy:    {run_info['strategy_name']}")
print("Instrument:  RB (Rebar futures)")
print(f"Date range:  {daily_returns.index[0].strftime('%Y-%m-%d')} -> {daily_returns.index[-1].strftime('%Y-%m-%d')}")
print(f"Annual return: {fincore.annual_return(daily_returns):.2%}")
print(f"Sharpe ratio:  {fincore.sharpe_ratio(daily_returns):.4f}")
print(f"Max drawdown:  {fincore.max_drawdown(daily_returns):.2%}")
print(f"Trades:        {total_trades}")
print(f"Win rate:      {len(winning_trades) / total_trades:.2%}")
