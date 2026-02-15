"""
End-to-end analysis example using fincore.

Loads NAV/positions/transactions from the `examples/011_abberation` backtest logs
and demonstrates multiple fincore APIs for analysis and visualization.

APIs demonstrated:
  1. fincore.analyze()        - AnalysisContext with cached metrics
  2. fincore.Empyrical        - class-level and instance-level metrics
  3. fincore.Pyfolio plotting - compose plot_* calls to build a tear sheet
  4. Flat API                 - package-level functions

Usage:
    python analyze_with_fincore.py

Output:
    011_abberation/output/analysis_report.pdf
"""

import json
import os
import warnings

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

import fincore
from fincore import Empyrical, Pyfolio, analyze

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# =========================================================================
# 1. Load data (reuse 011_abberation logs)
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
print(f"Run ID:   {run_info['run_id']}")

# --- value.log -> daily returns ---
value_df = pd.read_csv(os.path.join(RUN_DIR, "value.log"), sep="\t", usecols=["dt", "value", "cash"])
value_df["dt"] = pd.to_datetime(value_df["dt"])
daily_value = value_df.groupby(value_df["dt"].dt.date).last()
daily_value.index = pd.to_datetime(daily_value.index)

daily_returns = daily_value["value"].pct_change().dropna()
daily_returns.name = "strategy"
daily_returns.index = daily_returns.index.tz_localize("UTC")
daily_returns.index.name = None

# --- position.log → positions DataFrame ---
pos_df = pd.read_csv(os.path.join(RUN_DIR, "position.log"), sep="\t", usecols=["dt", "data_name", "size", "price"])
pos_df["dt"] = pd.to_datetime(pos_df["dt"])
asset_names = pos_df["data_name"].unique()

positions = pd.DataFrame(index=daily_value.index)
for asset in asset_names:
    ap = pos_df[pos_df["data_name"] == asset]
    ad = ap.groupby(ap["dt"].dt.date).last()
    ad.index = pd.to_datetime(ad.index)
    positions[asset] = ad["size"] * ad["price"]
positions["cash"] = daily_value["cash"]
positions = positions.fillna(0)
positions.index = positions.index.tz_localize("UTC")
positions = positions.loc[daily_returns.index]

# --- order.log → transactions DataFrame ---
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

print(f"Dates: {daily_returns.index[0].strftime('%Y-%m-%d')} -> {daily_returns.index[-1].strftime('%Y-%m-%d')}")
print(f"Trading days: {len(daily_returns)}, transactions: {len(transactions)}")

# =========================================================================
# 2. Analysis + plotting
# =========================================================================
pdf_path = os.path.join(OUTPUT_DIR, "analysis_report.pdf")
pdf = PdfPages(pdf_path)
page_count = 0


def save_page(fig, tight=True):
    """Save one page into the PDF."""
    global page_count
    if tight:
        fig.tight_layout()
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)
    page_count += 1


# =========================================================================
# Page 1: AnalysisContext - summary table
# =========================================================================
print("\n[Page 1] AnalysisContext summary ...")

ctx = analyze(daily_returns)

stats_data = {
    "Annual Return": f"{ctx.annual_return:.4f}",
    "Cumulative Returns": f"{ctx.cumulative_returns:.4f}",
    "Annual Volatility": f"{ctx.annual_volatility:.4f}",
    "Sharpe Ratio": f"{ctx.sharpe_ratio:.4f}",
    "Sortino Ratio": f"{ctx.sortino_ratio:.4f}",
    "Calmar Ratio": f"{ctx.calmar_ratio:.4f}",
    "Max Drawdown": f"{ctx.max_drawdown:.4f}",
    "Omega Ratio": f"{ctx.omega_ratio:.4f}",
    "Stability": f"{ctx.stability:.4f}",
    "Skewness": f"{ctx.skew:.4f}",
    "Kurtosis": f"{ctx.kurtosis:.4f}",
    "Tail Ratio": f"{ctx.tail_ratio:.4f}",
    "Daily Value at Risk": f"{ctx.daily_value_at_risk:.4f}",
}

fig, ax = plt.subplots(figsize=(10, 8))
ax.axis("off")
ax.set_title(
    "AbberationStrategy — Performance Summary\n(via fincore.analyze, AnalysisContext with cached_property)",
    fontsize=13,
    fontweight="bold",
    pad=15,
)

cell_text = [[k, v] for k, v in stats_data.items()]
tbl = ax.table(cellText=cell_text, colLabels=["Metric", "Value"], cellLoc="center", loc="center")
tbl.auto_set_font_size(False)
tbl.set_fontsize(10)
tbl.scale(1.3, 1.5)
for i in range(len(cell_text)):
    tbl[i + 1, 0].set_text_props(ha="left")
save_page(fig)

# =========================================================================
# Page 2: Empyrical (class-level) - yearly stats + drawdown table
# =========================================================================
print("[Page 2] Empyrical (class-level) - yearly stats + drawdown table ...")

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

# Yearly stats
yearly_stats = pd.DataFrame(
    {
        "Annual Return": Empyrical.annual_return_by_year(daily_returns),
        "Sharpe Ratio": Empyrical.sharpe_ratio_by_year(daily_returns),
        "Max Drawdown": Empyrical.max_drawdown_by_year(daily_returns),
    }
)

ax1.axis("off")
ax1.set_title("Yearly Statistics (Empyrical.xxx_by_year)", fontsize=12, fontweight="bold")
cell_text = [[f"{v:.4f}" for v in row] for _, row in yearly_stats.iterrows()]
tbl1 = ax1.table(
    cellText=cell_text,
    rowLabels=[str(y) for y in yearly_stats.index],
    colLabels=yearly_stats.columns.tolist(),
    cellLoc="center",
    loc="center",
)
tbl1.auto_set_font_size(False)
tbl1.set_fontsize(10)
tbl1.scale(1.2, 1.6)

# Top 5 drawdowns
dd_table = Empyrical.gen_drawdown_table(daily_returns, top=5)
ax2.axis("off")
ax2.set_title("Top 5 Drawdowns (Empyrical.gen_drawdown_table)", fontsize=12, fontweight="bold")
cell_text2 = []
for _, row in dd_table.iterrows():
    cells = []
    for v in row:
        if isinstance(v, float):
            cells.append(f"{v:.2f}")
        else:
            cells.append(str(v))
    cell_text2.append(cells)
tbl2 = ax2.table(
    cellText=cell_text2,
    rowLabels=[str(i) for i in dd_table.index],
    colLabels=dd_table.columns.tolist(),
    cellLoc="center",
    loc="center",
)
tbl2.auto_set_font_size(False)
tbl2.set_fontsize(9)
tbl2.scale(1.1, 1.5)
save_page(fig)

# =========================================================================
# Page 3: Empyrical (instance) - extended risk metrics
# =========================================================================
print("[Page 3] Empyrical (instance) - extended risk metrics ...")

emp = Empyrical(returns=daily_returns)

instance_stats = {
    "Win Rate": f"{emp.win_rate():.4f}",
    "Loss Rate": f"{emp.loss_rate():.4f}",
    "Serial Correlation": f"{emp.serial_correlation():.4f}",
    "Common Sense Ratio": f"{emp.common_sense_ratio():.4f}",
    "Sterling Ratio": f"{emp.sterling_ratio():.4f}",
    "Burke Ratio": f"{emp.burke_ratio():.4f}",
    "Kappa Three Ratio": f"{emp.kappa_three_ratio():.4f}",
    "Max Drawdown Days": str(emp.max_drawdown_days()),
    "Max Drawdown Recovery Days": str(emp.max_drawdown_recovery_days()),
    "2nd Max Drawdown": f"{emp.second_max_drawdown():.4f}",
    "3rd Max Drawdown": f"{emp.third_max_drawdown():.4f}",
    "Max Consecutive Up Days": str(Empyrical.max_consecutive_up_days(daily_returns)),
    "Max Consecutive Down Days": str(Empyrical.max_consecutive_down_days(daily_returns)),
    "Max Single Day Gain": f"{Empyrical.max_single_day_gain(daily_returns):.4f}",
    "Max Single Day Loss": f"{Empyrical.max_single_day_loss(daily_returns):.4f}",
    "Hurst Exponent": f"{Empyrical.hurst_exponent(daily_returns):.4f}",
}

fig, ax = plt.subplots(figsize=(10, 9))
ax.axis("off")
ax.set_title(
    "Extended Risk Metrics\n(Empyrical instance — auto-fill returns via @_dual_method)",
    fontsize=13,
    fontweight="bold",
    pad=15,
)
cell_text3 = [[k, v] for k, v in instance_stats.items()]
tbl3 = ax.table(cellText=cell_text3, colLabels=["Metric", "Value"], cellLoc="center", loc="center")
tbl3.auto_set_font_size(False)
tbl3.set_fontsize(10)
tbl3.scale(1.3, 1.45)
for i in range(len(cell_text3)):
    tbl3[i + 1, 0].set_text_props(ha="left")
save_page(fig)

# =========================================================================
# Page 4: Flat API - cumulative returns + monthly aggregation
# =========================================================================
print("[Page 4] Flat API - cumulative returns ...")

cum_rets = fincore.cum_returns(daily_returns, starting_value=1.0)

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True)

ax1.plot(cum_rets.index, cum_rets.values, color="steelblue", linewidth=1.2)
ax1.axhline(y=1.0, color="gray", linestyle="--", linewidth=0.8)
ax1.set_title("Cumulative Returns (fincore.cum_returns)", fontsize=12, fontweight="bold")
ax1.set_ylabel("Cumulative Return")
ax1.fill_between(cum_rets.index, 1.0, cum_rets.values, where=cum_rets.values >= 1.0, alpha=0.15, color="green")
ax1.fill_between(cum_rets.index, 1.0, cum_rets.values, where=cum_rets.values < 1.0, alpha=0.15, color="red")

# Monthly aggregation bar chart
monthly_rets = fincore.aggregate_returns(daily_returns, "monthly")
colors = ["green" if v >= 0 else "red" for v in monthly_rets.values]
ax2.bar(range(len(monthly_rets)), monthly_rets.values, color=colors, alpha=0.7, width=0.8)
ax2.set_title("Monthly Returns (fincore.aggregate_returns)", fontsize=12, fontweight="bold")
ax2.set_ylabel("Monthly Return")
ax2.axhline(y=0, color="gray", linestyle="--", linewidth=0.8)
# Put a tick every ~6 months (keeps labels readable)
tick_positions = list(range(0, len(monthly_rets), 6))
tick_labels = [str(monthly_rets.index[i]) for i in tick_positions]
ax2.set_xticks(tick_positions)
ax2.set_xticklabels(tick_labels, rotation=45, fontsize=8)

save_page(fig)

# =========================================================================
# Page 5: Pyfolio (individual plots) - rolling returns/sharpe/volatility
# =========================================================================
print("[Page 5] Pyfolio (individual plots) - rolling metrics ...")

pf = Pyfolio(returns=daily_returns)

fig = plt.figure(figsize=(14, 16))
gs = gridspec.GridSpec(3, 1, hspace=0.35)

ax1 = fig.add_subplot(gs[0])
pf.plot_rolling_returns(daily_returns, ax=ax1)
ax1.set_title("Cumulative Returns (pf.plot_rolling_returns)", fontsize=12, fontweight="bold")

ax2 = fig.add_subplot(gs[1], sharex=ax1)
pf.plot_rolling_sharpe(daily_returns, ax=ax2)
ax2.set_title("Rolling Sharpe (pf.plot_rolling_sharpe)", fontsize=12, fontweight="bold")

ax3 = fig.add_subplot(gs[2], sharex=ax1)
pf.plot_rolling_volatility(daily_returns, ax=ax3)
ax3.set_title("Rolling Volatility (pf.plot_rolling_volatility)", fontsize=12, fontweight="bold")

save_page(fig, tight=False)

# =========================================================================
# Page 6: Pyfolio (individual plots) - drawdown analysis
# =========================================================================
print("[Page 6] Pyfolio (individual plots) - drawdown analysis ...")

fig = plt.figure(figsize=(14, 12))
gs = gridspec.GridSpec(2, 1, hspace=0.35)

ax1 = fig.add_subplot(gs[0])
pf.plot_drawdown_periods(daily_returns, top=5, ax=ax1)
ax1.set_title("Top 5 Drawdown Periods (pf.plot_drawdown_periods)", fontsize=12, fontweight="bold")

ax2 = fig.add_subplot(gs[1], sharex=ax1)
pf.plot_drawdown_underwater(daily_returns, ax=ax2)
ax2.set_title("Underwater Plot (pf.plot_drawdown_underwater)", fontsize=12, fontweight="bold")

save_page(fig, tight=False)

# =========================================================================
# Page 7: Pyfolio (individual plots) - monthly/annual analysis
# =========================================================================
print("[Page 7] Pyfolio (individual plots) - monthly/annual analysis ...")

fig = plt.figure(figsize=(16, 6))
gs = gridspec.GridSpec(1, 3, wspace=0.4)

ax1 = fig.add_subplot(gs[0])
pf.plot_monthly_returns_heatmap(daily_returns, ax=ax1)

ax2 = fig.add_subplot(gs[1])
pf.plot_annual_returns(daily_returns, ax=ax2)

ax3 = fig.add_subplot(gs[2])
pf.plot_monthly_returns_dist(daily_returns, ax=ax3)

fig.suptitle(
    "Monthly & Annual Analysis (pf.plot_monthly_returns_heatmap / plot_annual_returns / plot_monthly_returns_dist)",
    fontsize=11,
    fontweight="bold",
    y=1.02,
)
save_page(fig, tight=False)

# =========================================================================
# Page 8: Pyfolio (individual plots) - positions analysis
# =========================================================================
print("[Page 8] Pyfolio (individual plots) - positions analysis ...")

fig = plt.figure(figsize=(14, 12))
gs = gridspec.GridSpec(2, 1, hspace=0.35)

ax1 = fig.add_subplot(gs[0])
pf.plot_exposures(daily_returns, positions, ax=ax1)
ax1.set_title("Long/Short Exposures (pf.plot_exposures)", fontsize=12, fontweight="bold")

ax2 = fig.add_subplot(gs[1])
pf.plot_gross_leverage(daily_returns, positions, ax=ax2)
ax2.set_title("Gross Leverage (pf.plot_gross_leverage)", fontsize=12, fontweight="bold")

save_page(fig, tight=False)

# =========================================================================
# Page 9: Pyfolio (individual plots) - transactions analysis
# =========================================================================
print("[Page 9] Pyfolio (individual plots) - transactions analysis ...")

fig = plt.figure(figsize=(14, 12))
gs = gridspec.GridSpec(2, 1, hspace=0.35)

ax1 = fig.add_subplot(gs[0])
pf.plot_turnover(daily_returns, transactions, positions, ax=ax1)
ax1.set_title("Daily Turnover (pf.plot_turnover)", fontsize=12, fontweight="bold")

ax2 = fig.add_subplot(gs[1])
pf.plot_daily_volume(daily_returns, transactions, ax=ax2)
ax2.set_title("Daily Trading Volume (pf.plot_daily_volume)", fontsize=12, fontweight="bold")

save_page(fig, tight=False)

# =========================================================================
# Page 10: Pyfolio - return quantiles
# =========================================================================
print("[Page 10] Pyfolio - return quantiles ...")

fig, ax = plt.subplots(figsize=(14, 6))
pf.plot_return_quantiles(daily_returns, ax=ax)
ax.set_title("Return Quantiles (pf.plot_return_quantiles)", fontsize=12, fontweight="bold")
save_page(fig)

# =========================================================================
# Page 11: fincore.viz - MatplotlibBackend plots
# =========================================================================
print("[Page 11] fincore.viz - MatplotlibBackend ...")

from fincore.viz import get_backend

viz = get_backend("matplotlib")

# Prepare inputs
cum_ret = fincore.cum_returns(daily_returns, starting_value=0)
running_max = (1 + cum_ret).cummax()
drawdown = (1 + cum_ret) / running_max - 1
rolling_sr = Empyrical.rolling_sharpe(daily_returns, rolling_sharpe_window=63)

fig = plt.figure(figsize=(14, 16))
gs = gridspec.GridSpec(4, 1, hspace=0.4)

ax1 = fig.add_subplot(gs[0])
viz.plot_returns(cum_ret, title="Cumulative Returns (viz.plot_returns)", ax=ax1)

ax2 = fig.add_subplot(gs[1], sharex=ax1)
viz.plot_drawdown(drawdown, title="Drawdown (viz.plot_drawdown)", ax=ax2)

ax3 = fig.add_subplot(gs[2], sharex=ax1)
viz.plot_rolling_sharpe(rolling_sr, title="Rolling Sharpe (viz.plot_rolling_sharpe)", ax=ax3)

# Monthly returns heatmap
monthly_agg = Empyrical.aggregate_returns(daily_returns, "monthly")
monthly_table = monthly_agg.unstack().fillna(0) * 100  # percent
ax4 = fig.add_subplot(gs[3])
viz.plot_monthly_heatmap(monthly_table, title="Monthly Returns % (viz.plot_monthly_heatmap)", ax=ax4)

save_page(fig, tight=False)

# =========================================================================
# Page 12: fincore.viz - ctx.plot() + ctx.to_html()
# =========================================================================
print("[Page 12] AnalysisContext.plot() (via fincore.viz) ...")

# ctx.plot() uses MatplotlibBackend to render cumulative returns + drawdown
viz_backend = ctx.plot(backend="matplotlib")

# Collect figures created by ctx.plot()
for n in plt.get_fignums():
    save_page(plt.figure(n), tight=True)

# ctx.to_html() exports an HTML report
html_path = os.path.join(OUTPUT_DIR, "analysis_report.html")
ctx.to_html(path=html_path)
print(f"  -> HTML report: {html_path}")

# =========================================================================
# Done
# =========================================================================
pdf.close()

print(f"\n{'=' * 60}")
print(f"Saved analysis report: {pdf_path} ({page_count} pages)")
print(f"{'=' * 60}")
print("""
fincore APIs used:
  [Page 1]  fincore.analyze() -> AnalysisContext (cached_property)
  [Page 2]  Empyrical.annual_return_by_year / sharpe_ratio_by_year / max_drawdown_by_year / gen_drawdown_table
  [Page 3]  Empyrical(returns=...) instance methods: win_rate / sterling_ratio / burke_ratio / ...
  [Page 4]  fincore.cum_returns / fincore.aggregate_returns (Flat API)
  [Page 5]  pf.plot_rolling_returns / plot_rolling_sharpe / plot_rolling_volatility
  [Page 6]  pf.plot_drawdown_periods / plot_drawdown_underwater
  [Page 7]  pf.plot_monthly_returns_heatmap / plot_annual_returns / plot_monthly_returns_dist
  [Page 8]  pf.plot_exposures / plot_gross_leverage
  [Page 9]  pf.plot_turnover / plot_daily_volume
  [Page 10] pf.plot_return_quantiles
  [Page 11] fincore.viz.get_backend("matplotlib") -> viz.plot_returns / plot_drawdown / plot_rolling_sharpe / plot_monthly_heatmap
  [Page 12] ctx.plot(backend="matplotlib") + ctx.to_html() -> HTML report
""")
