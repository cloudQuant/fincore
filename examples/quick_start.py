"""
Quick Start Example — All 5 ways to use fincore

Demonstrates the complete API surface in one script:
  1. Flat API (function style)
  2. Empyrical class-level calls
  3. Empyrical instance (OOP)
  4. AnalysisContext (cached analysis)
  5. Strategy report generation

Usage:
    python examples/quick_start.py
"""

import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

import fincore
from fincore import Empyrical, Pyfolio, analyze

# =========================================================================
# Setup: generate sample data
# =========================================================================
np.random.seed(42)
dates = pd.date_range("2020-01-01", periods=500, freq="B", tz="UTC")
returns = pd.Series(np.random.normal(0.0005, 0.015, 500), index=dates, name="strategy")
benchmark = pd.Series(np.random.normal(0.0003, 0.012, 500), index=dates, name="benchmark")

print(f"Data: {len(returns)} trading days")
print(f"Period: {dates[0].strftime('%Y-%m-%d')} -> {dates[-1].strftime('%Y-%m-%d')}")
print()

# =========================================================================
# Way 1: Flat API — simplest, one function call per metric
# =========================================================================
print("=" * 50)
print("1. Flat API")
print("=" * 50)

print(f"Sharpe ratio:      {fincore.sharpe_ratio(returns):.4f}")
print(f"Max drawdown:      {fincore.max_drawdown(returns):.4f}")
print(f"Annual return:     {fincore.annual_return(returns):.4f}")
print(f"Annual volatility: {fincore.annual_volatility(returns):.4f}")
print(f"Sortino ratio:     {fincore.sortino_ratio(returns):.4f}")
print(f"Calmar ratio:      {fincore.calmar_ratio(returns):.4f}")
print(f"Omega ratio:       {fincore.omega_ratio(returns):.4f}")
print(f"Tail ratio:        {fincore.tail_ratio(returns):.4f}")
print(f"VaR(5%):           {fincore.value_at_risk(returns):.4f}")
alpha, beta = fincore.alpha_beta(returns, benchmark)
print(f"Alpha:             {alpha:.4f}")
print(f"Beta:              {beta:.4f}")

# =========================================================================
# Way 2: Empyrical class-level — 100+ metrics, lazy loaded
# =========================================================================
print(f"\n{'=' * 50}")
print("2. Empyrical class-level")
print("=" * 50)

# All metrics accessible without creating an instance
print(f"Downside risk:     {Empyrical.downside_risk(returns):.4f}")
print(f"CVaR:              {Empyrical.conditional_value_at_risk(returns):.4f}")
print(f"Tracking error:    {Empyrical.tracking_error(returns, benchmark):.4f}")
print(f"Skewness:          {Empyrical.skewness(returns):.4f}")
print(f"Kurtosis:          {Empyrical.kurtosis(returns):.4f}")
print(f"Hurst exponent:    {Empyrical.hurst_exponent(returns):.4f}")

# Yearly breakdown
yearly_ret = Empyrical.annual_return_by_year(returns)
print(f"\nAnnual returns by year:")
for year, ret in yearly_ret.items():
    print(f"  {year}: {ret:.4f}")

# Drawdown table
dd = Empyrical.gen_drawdown_table(returns, top=3)
print(f"\nTop 3 drawdowns:")
print(dd.to_string())

# Performance summary
stats = Empyrical.perf_stats(returns, factor_returns=benchmark)
print(f"\nPerformance stats:")
print(stats.to_string())

# =========================================================================
# Way 3: Empyrical instance — bind data, auto-fill returns
# =========================================================================
print(f"\n{'=' * 50}")
print("3. Empyrical instance")
print("=" * 50)

emp = Empyrical(returns=returns, factor_returns=benchmark)

# No need to pass returns — auto-filled from instance
print(f"Win rate:          {emp.win_rate():.4f}")
print(f"Loss rate:         {emp.loss_rate():.4f}")
print(f"Max DD days:       {emp.max_drawdown_days()}")
print(f"Sterling ratio:    {emp.sterling_ratio():.4f}")
print(f"Burke ratio:       {emp.burke_ratio():.4f}")
print(f"Common sense:      {emp.common_sense_ratio():.4f}")
print(f"Treynor ratio:     {emp.treynor_ratio():.4f}")
print(f"M-squared:         {emp.m_squared():.4f}")

# =========================================================================
# Way 4: AnalysisContext — cached, lazy analysis
# =========================================================================
print(f"\n{'=' * 50}")
print("4. AnalysisContext (fincore.analyze)")
print("=" * 50)

ctx = analyze(returns, factor_returns=benchmark)
print(f"Context: {ctx}")
print(f"Sharpe:    {ctx.sharpe_ratio:.4f}")
print(f"Max DD:    {ctx.max_drawdown:.4f}")
print(f"Alpha:     {ctx.alpha:.4f}")
print(f"Beta:      {ctx.beta:.4f}")

# Export as dict / JSON
d = ctx.to_dict()
print(f"\nJSON export: {len(ctx.to_json())} chars")

# Perf stats summary
print(f"\nPerf stats:")
print(ctx.perf_stats().to_string())

# =========================================================================
# Way 5: Strategy report
# =========================================================================
print(f"\n{'=' * 50}")
print("5. Strategy Report")
print("=" * 50)

import os
import tempfile

with tempfile.TemporaryDirectory() as tmpdir:
    html_path = os.path.join(tmpdir, "report.html")
    out = fincore.create_strategy_report(
        returns,
        title="Quick Start Strategy",
        output=html_path,
    )
    file_size = os.path.getsize(html_path)
    print(f"Generated HTML report: {file_size:,} bytes")
    print(f"Path: {out}")

print(f"\n{'=' * 50}")
print("All 5 API styles demonstrated successfully!")
print("=" * 50)
