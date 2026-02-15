"""
AbberationStrategy Pyfolio tear sheet generation script.

Reads NAV/positions/transactions from the local logs directory, converts them
to the format expected by Pyfolio, and renders tear sheet figures into a PDF.

Usage:
    python plot_tearsheet.py

Outputs:
    tearsheet_returns.pdf   - returns report
    tearsheet_positions.pdf - positions report
    tearsheet_txn.pdf       - transactions report
    tearsheet_full.pdf      - full report
"""

import json
import os
import warnings

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from fincore import Pyfolio

warnings.filterwarnings("ignore", category=UserWarning)

# ============================================================================
# 1. Locate log directory
# ============================================================================
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
print(f"Analyzing run directory: {RUN_DIR}")

with open(os.path.join(RUN_DIR, "run_info.json")) as f:
    run_info = json.load(f)
print(f"Strategy: {run_info['strategy_name']}")

# ============================================================================
# 2. Load value.log -> daily returns (returns) + daily cash (cash)
# ============================================================================
print("\n[1/3] Loading value.log ...")
value_df = pd.read_csv(
    os.path.join(RUN_DIR, "value.log"),
    sep="\t",
    usecols=["dt", "value", "cash"],
)
value_df["dt"] = pd.to_datetime(value_df["dt"])

# Use the last bar of each day.
daily_value = value_df.groupby(value_df["dt"].dt.date).last()
daily_value.index = pd.to_datetime(daily_value.index)

# Daily returns.
daily_returns = daily_value["value"].pct_change().dropna()
daily_returns.name = "strategy"
daily_returns.index = daily_returns.index.tz_localize("UTC")
daily_returns.index.name = None

print(f"  Date range: {daily_returns.index[0].strftime('%Y-%m-%d')} -> {daily_returns.index[-1].strftime('%Y-%m-%d')}")
print(f"  Trading days: {len(daily_returns)}")

# ============================================================================
# 3. Load position.log + value.log -> daily positions DataFrame (positions)
#
# Pyfolio expects positions:
#   - DatetimeIndex (daily)
#   - columns = asset names + 'cash'
#   - values = notional position value
# ============================================================================
print("[2/3] Loading position.log -> building positions DataFrame ...")
pos_df = pd.read_csv(
    os.path.join(RUN_DIR, "position.log"),
    sep="\t",
    usecols=["dt", "data_name", "size", "price"],
)
pos_df["dt"] = pd.to_datetime(pos_df["dt"])

# Use the last bar of each day.
daily_pos = pos_df.groupby(pos_df["dt"].dt.date).last()
daily_pos.index = pd.to_datetime(daily_pos.index)

# Compute notional position value: size * price
# Note: for futures, size * price is used as an approximation.
# Build positions DataFrame: columns = [asset_names..., 'cash']
asset_names = pos_df["data_name"].unique()

positions = pd.DataFrame(index=daily_value.index)
for asset in asset_names:
    asset_pos = pos_df[pos_df["data_name"] == asset].copy()
    asset_daily = asset_pos.groupby(asset_pos["dt"].dt.date).last()
    asset_daily.index = pd.to_datetime(asset_daily.index)
    # Notional value = size * price
    positions[asset] = asset_daily["size"] * asset_daily["price"]

positions["cash"] = daily_value["cash"]
positions = positions.fillna(0)
positions.index = positions.index.tz_localize("UTC")

# Align to the returns date range (returns start from the second day).
positions = positions.loc[daily_returns.index]

print(f"  Asset columns: {list(asset_names)}")
print(f"  Position rows: {len(positions)}")

# ============================================================================
# 4. Load order.log -> transactions DataFrame (transactions)
#
# Pyfolio expects transactions:
#   - DatetimeIndex (execution time)
#   - columns: amount (positive=buy, negative=sell), price, symbol
# ============================================================================
print("[3/3] Loading order.log -> building transactions DataFrame ...")
order_df = pd.read_csv(
    os.path.join(RUN_DIR, "order.log"),
    sep="\t",
)

# Only completed orders.
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
transactions.index.name = None

print(f"  Completed fills: {len(transactions)}")

# ============================================================================
# 5. Create Pyfolio instance and render tear sheet
# ============================================================================
print(f"\n{'=' * 60}")
print("Generating Pyfolio Tear Sheet ...")
print(f"{'=' * 60}")

pf = Pyfolio(returns=daily_returns)

from matplotlib.backends.backend_pdf import PdfPages

import fincore.utils.common_utils as _cu

# ---------------------------------------------------------------------------
# Monkey-patch print_table: besides the original HTML output, also create a
# matplotlib table figure so that tables generated inside create_full_tear_sheet
# can be captured via plt.get_fignums() and written into the PDF.
# ---------------------------------------------------------------------------
_original_print_table = _cu.print_table


def _print_table_with_figure(
    table, name=None, float_format=None, formatters=None, header_rows=None, run_flask_app=False
):
    """Enhanced print_table that also creates a matplotlib table figure."""
    if isinstance(table, pd.Series):
        table = pd.DataFrame(table)

    # Build a display DataFrame (including optional header rows).
    display_df = table.copy()
    if header_rows:
        for k, v in reversed(header_rows.items()):
            row = pd.DataFrame(
                [[v] * len(display_df.columns)],
                index=[k],
                columns=display_df.columns,
            )
            display_df = pd.concat([row, display_df])

    title = name if name else ""
    nrows = len(display_df)
    fig, ax = plt.subplots(figsize=(12, max(2.5, 0.38 * nrows + 1.5)))
    ax.axis("off")
    if title:
        ax.set_title(title, fontsize=13, fontweight="bold", pad=10)

    # Format table cells.
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
    # Keep the figure alive in pyplot; it will be collected later.


# Apply patch.
_cu.print_table = _print_table_with_figure
# Tearsheets modules also need patching (they import print_table directly).
import fincore.tearsheets.returns as _tr
import fincore.tearsheets.transactions as _ttxn

_tr.print_table = _print_table_with_figure
if hasattr(_ttxn, "print_table"):
    _ttxn.print_table = _print_table_with_figure

# Call a single entrypoint: create_full_tear_sheet
print("\nGenerating Full Tear Sheet ...")
pf.create_full_tear_sheet(
    daily_returns,
    positions=positions,
    transactions=transactions,
)

# Restore original print_table.
_cu.print_table = _original_print_table
_tr.print_table = _original_print_table
if hasattr(_ttxn, "print_table"):
    _ttxn.print_table = _original_print_table

# Collect all figures and save to a multi-page PDF.
pdf_path = os.path.join(OUTPUT_DIR, "tearsheet_full.pdf")
all_figs = [plt.figure(n) for n in plt.get_fignums()]

with PdfPages(pdf_path) as pdf:
    for fig in all_figs:
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

print(f"  Saved: {pdf_path}  ({len(all_figs)} pages)")

print(f"\n{'=' * 60}")
print("Done")
print(f"{'=' * 60}")
