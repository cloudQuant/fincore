"""
Performance Attribution Example

Demonstrates fincore's attribution analysis capabilities:
  1. Brinson attribution — allocation vs selection effects
  2. Style analysis — returns-based style decomposition
  3. Regression attribution — factor exposure analysis

Usage:
    python examples/performance_attribution.py
"""

import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# =========================================================================
# 1. Generate sample data
# =========================================================================
np.random.seed(42)
n_days = 500
dates = pd.date_range("2020-01-01", periods=n_days, freq="B")

# Portfolio returns
returns = pd.Series(
    np.random.normal(0.0005, 0.012, n_days),
    index=dates,
    name="portfolio",
)

# Factor / benchmark returns
benchmark = pd.Series(
    np.random.normal(0.0003, 0.01, n_days),
    index=dates,
    name="benchmark",
)

# Style factor returns (value, growth, momentum, quality)
factors = pd.DataFrame({
    "value": np.random.normal(0.0002, 0.008, n_days),
    "growth": np.random.normal(0.0004, 0.01, n_days),
    "momentum": np.random.normal(0.0003, 0.009, n_days),
    "quality": np.random.normal(0.0001, 0.006, n_days),
}, index=dates)

print(f"Data: {n_days} days, {dates[0].strftime('%Y-%m-%d')} -> {dates[-1].strftime('%Y-%m-%d')}")
print()

# =========================================================================
# 2. Brinson Attribution
# =========================================================================
from fincore.attribution import brinson_attribution, brinson_results

print("=" * 60)
print("Brinson Attribution")
print("=" * 60)

# Prepare sector-level data for Brinson model
sectors = ["Technology", "Healthcare", "Finance", "Energy", "Consumer"]

# Portfolio weights and returns by sector
port_weights = pd.Series([0.30, 0.20, 0.20, 0.15, 0.15], index=sectors)
bench_weights = pd.Series([0.25, 0.20, 0.25, 0.15, 0.15], index=sectors)
port_returns_sector = pd.Series([0.12, 0.08, 0.05, -0.03, 0.10], index=sectors)
bench_returns_sector = pd.Series([0.10, 0.07, 0.06, -0.02, 0.08], index=sectors)

result = brinson_attribution(
    portfolio_weights=port_weights,
    benchmark_weights=bench_weights,
    portfolio_returns=port_returns_sector,
    benchmark_returns=bench_returns_sector,
)

print(f"\nAllocation effect:  {result['allocation']:.4f}")
print(f"Selection effect:   {result['selection']:.4f}")
print(f"Interaction effect: {result['interaction']:.4f}")
print(f"Total active:       {result['total_active']:.4f}")

# Detailed breakdown
details = brinson_results(
    portfolio_weights=port_weights,
    benchmark_weights=bench_weights,
    portfolio_returns=port_returns_sector,
    benchmark_returns=bench_returns_sector,
)
print("\nDetailed results:")
print(details.to_string())

# =========================================================================
# 3. Style Analysis
# =========================================================================
from fincore.attribution import calculate_style_tilts, style_analysis

print("\n" + "=" * 60)
print("Style Analysis")
print("=" * 60)

# Returns-based style analysis
style_result = style_analysis(returns, factors)
print(f"\nStyle analysis result type: {type(style_result).__name__}")
print("Style exposures:")
if hasattr(style_result, 'exposures'):
    for factor_name, exposure in style_result.exposures.items():
        print(f"  {factor_name:<12} {exposure:.4f}")
elif hasattr(style_result, 'weights'):
    for factor_name, weight in zip(factors.columns, style_result.weights):
        print(f"  {factor_name:<12} {weight:.4f}")

# Style tilts
tilts = calculate_style_tilts(returns, factors)
print("\nStyle tilts:")
print(tilts)

# =========================================================================
# 4. Regression Attribution
# =========================================================================
from fincore.attribution import calculate_regression_attribution

print("\n" + "=" * 60)
print("Regression Attribution")
print("=" * 60)

reg_result = calculate_regression_attribution(returns, factors)
print("\nRegression attribution:")
print(f"  Alpha:     {reg_result['alpha']:.6f}")
print(f"  R-squared: {reg_result['r_squared']:.4f}")
print("\n  Factor betas:")
for factor_name, beta_val in reg_result['betas'].items():
    print(f"    {factor_name:<12} {beta_val:.4f}")

# =========================================================================
# 5. Summary
# =========================================================================
print("\n" + "=" * 60)
print("Attribution Methods Summary")
print("=" * 60)
print("""
  Brinson:     Decomposes active return into allocation + selection effects
               Best for: sector/asset class level attribution

  Style:       Constrained regression to identify return style exposures
               Best for: identifying investment style (value, growth, etc.)

  Regression:  Unconstrained factor regression
               Best for: factor exposure and alpha estimation
""")
print("Done! All attribution examples executed successfully.")
