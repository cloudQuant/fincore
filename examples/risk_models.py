"""
Risk Models Example — EVT and GARCH

Demonstrates fincore's advanced risk modeling capabilities:
  1. Extreme Value Theory (EVT) — tail risk via GPD/GEV fitting
  2. GARCH family — conditional volatility forecasting
  3. Combined risk measures — EVT-VaR, GARCH-VaR

Usage:
    python examples/risk_models.py
"""

import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# =========================================================================
# 1. Generate sample return data
# =========================================================================
np.random.seed(42)
dates = pd.date_range("2018-01-01", periods=1000, freq="B")
# Simulate returns with fat tails (mixture of normals)
normal_rets = np.random.normal(0.0003, 0.01, 1000)
shock_mask = np.random.random(1000) < 0.05  # 5% chance of shock
shocks = np.random.normal(-0.02, 0.03, 1000)
raw_rets = np.where(shock_mask, shocks, normal_rets)
returns = pd.Series(raw_rets, index=dates, name="strategy")

print(f"Sample data: {len(returns)} days, mean={returns.mean():.6f}, std={returns.std():.4f}")
print(f"Min={returns.min():.4f}, Max={returns.max():.4f}")
print()

# =========================================================================
# 2. Extreme Value Theory (EVT)
# =========================================================================
from fincore.risk import evt_cvar, evt_var, extreme_risk, gpd_fit, hill_estimator

print("=" * 60)
print("Extreme Value Theory (EVT)")
print("=" * 60)

# Hill estimator — estimate tail index
tail_index = hill_estimator(returns)
print(f"\nHill tail index: {tail_index:.4f}")

# GPD fit — fit Generalized Pareto Distribution to tail losses
gpd_result = gpd_fit(returns)
print(f"GPD fit: shape={gpd_result['shape']:.4f}, scale={gpd_result['scale']:.4f}")

# EVT-based VaR and CVaR
var_95 = evt_var(returns, alpha=0.05)
var_99 = evt_var(returns, alpha=0.01)
cvar_95 = evt_cvar(returns, alpha=0.05)
cvar_99 = evt_cvar(returns, alpha=0.01)

print("\nEVT Value at Risk:")
print(f"  VaR(95%):  {var_95:.4f}")
print(f"  VaR(99%):  {var_99:.4f}")
print(f"  CVaR(95%): {cvar_95:.4f}")
print(f"  CVaR(99%): {cvar_99:.4f}")

# Comprehensive extreme risk report
report = extreme_risk(returns)
print(f"\nExtreme risk report keys: {list(report.keys())}")

# =========================================================================
# 3. GARCH Models
# =========================================================================
from fincore.risk import EGARCH, GARCH, GJRGARCH, conditional_var, forecast_volatility

print("\n" + "=" * 60)
print("GARCH Models")
print("=" * 60)

# Standard GARCH(1,1)
garch = GARCH(returns)
print(f"\nGARCH(1,1) params: {garch.params}")
print("  Conditional volatility (last 5):")
cond_vol = garch.conditional_volatility
for d, v in list(cond_vol.tail().items()):
    print(f"    {d.strftime('%Y-%m-%d')}: {v:.6f}")

# EGARCH (asymmetric — captures leverage effect)
egarch = EGARCH(returns)
print(f"\nEGARCH params: {egarch.params}")

# GJR-GARCH (another asymmetric specification)
gjr = GJRGARCH(returns)
print(f"GJR-GARCH params: {gjr.params}")

# Forecast volatility
vol_forecast = forecast_volatility(returns, horizon=5)
print(f"\nVolatility forecast (5 days): {vol_forecast}")

# Conditional VaR using GARCH
cvar_garch = conditional_var(returns, alpha=0.05)
print(f"GARCH Conditional VaR(95%): {cvar_garch:.4f}")

# =========================================================================
# 4. Comparison table
# =========================================================================
print("\n" + "=" * 60)
print("Risk Measure Comparison")
print("=" * 60)

from fincore import Empyrical

historical_var = Empyrical.value_at_risk(returns, cutoff=0.05)
print(f"\n{'Method':<25} {'VaR(95%)':<12} {'Note'}")
print("-" * 55)
print(f"{'Historical':<25} {historical_var:<12.4f} {'Empirical quantile'}")
print(f"{'EVT (GPD)':<25} {var_95:<12.4f} {'Tail extrapolation'}")
print(f"{'GARCH':<25} {cvar_garch:<12.4f} {'Conditional on vol regime'}")

print("\nDone! All risk models executed successfully.")
