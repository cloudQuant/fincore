"""
Portfolio Optimization Example

Demonstrates fincore's portfolio optimization capabilities:
  1. Efficient frontier computation
  2. Risk parity (equal risk contribution) portfolios
  3. Constrained optimization (max Sharpe, min variance, target return)

Usage:
    python examples/portfolio_optimization.py
"""

import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# =========================================================================
# 1. Generate multi-asset return data
# =========================================================================
np.random.seed(42)
n_days = 500
dates = pd.date_range("2020-01-01", periods=n_days, freq="B")

# Simulate 5 correlated assets
mean_returns = [0.0005, 0.0003, 0.0008, 0.0002, 0.0006]
cov_matrix = np.array([
    [0.0004, 0.0001, 0.0002, 0.00005, 0.00015],
    [0.0001, 0.0003, 0.00008, 0.0001, 0.0001],
    [0.0002, 0.00008, 0.0006, 0.0001, 0.00025],
    [0.00005, 0.0001, 0.0001, 0.0002, 0.00008],
    [0.00015, 0.0001, 0.00025, 0.00008, 0.0005],
])

raw = np.random.multivariate_normal(mean_returns, cov_matrix, n_days)
asset_names = ["Equity_US", "Bonds_US", "Equity_EM", "Bonds_EM", "Commodities"]
returns = pd.DataFrame(raw, index=dates, columns=asset_names)

print("Multi-asset return data:")
print(f"  Assets: {asset_names}")
print(f"  Period: {dates[0].strftime('%Y-%m-%d')} -> {dates[-1].strftime('%Y-%m-%d')}")
print(f"  Days:   {n_days}")
print(f"\nAnnualized returns:")
for col in returns.columns:
    ann_ret = (1 + returns[col].mean()) ** 252 - 1
    ann_vol = returns[col].std() * np.sqrt(252)
    print(f"  {col:<15} return={ann_ret:.2%}  vol={ann_vol:.2%}")
print()

# =========================================================================
# 2. Efficient Frontier
# =========================================================================
from fincore.optimization import efficient_frontier

print("=" * 60)
print("Efficient Frontier")
print("=" * 60)

ef = efficient_frontier(returns, n_points=20)
print(f"\nComputed {len(ef)} points on the efficient frontier.")
print(f"Frontier columns: {list(ef.columns)}")
print(f"\nFirst 5 points (return vs risk):")
for i, (_, row) in enumerate(ef.head().iterrows()):
    print(f"  Point {i+1}: return={row['return']:.4f}, risk={row['risk']:.4f}")

# =========================================================================
# 3. Risk Parity
# =========================================================================
from fincore.optimization import risk_parity

print("\n" + "=" * 60)
print("Risk Parity Portfolio")
print("=" * 60)

rp_weights = risk_parity(returns)
print(f"\nRisk parity weights:")
for asset, w in zip(asset_names, rp_weights):
    print(f"  {asset:<15} {w:.4f} ({w:.1%})")
print(f"  Sum: {sum(rp_weights):.4f}")

# Verify equal risk contribution
cov = returns.cov().values * 252
port_vol = np.sqrt(rp_weights @ cov @ rp_weights)
marginal_contrib = cov @ rp_weights
risk_contrib = rp_weights * marginal_contrib / port_vol
print(f"\nRisk contributions (should be roughly equal):")
for asset, rc in zip(asset_names, risk_contrib):
    print(f"  {asset:<15} {rc:.4f}")

# =========================================================================
# 4. Constrained Optimization
# =========================================================================
from fincore.optimization import optimize

print("\n" + "=" * 60)
print("Constrained Optimization")
print("=" * 60)

# Max Sharpe portfolio
max_sharpe = optimize(returns, objective="max_sharpe")
print(f"\nMax Sharpe portfolio:")
print(f"  Weights: {dict(zip(asset_names, [f'{w:.4f}' for w in max_sharpe['weights']]))}")
print(f"  Expected return: {max_sharpe['return']:.4f}")
print(f"  Risk:            {max_sharpe['risk']:.4f}")
print(f"  Sharpe ratio:    {max_sharpe['sharpe']:.4f}")

# Min Variance portfolio
min_var = optimize(returns, objective="min_variance")
print(f"\nMin Variance portfolio:")
print(f"  Weights: {dict(zip(asset_names, [f'{w:.4f}' for w in min_var['weights']]))}")
print(f"  Expected return: {min_var['return']:.4f}")
print(f"  Risk:            {min_var['risk']:.4f}")

# Target return portfolio
target_ret = optimize(returns, objective="target_return", target=0.10)
print(f"\nTarget Return (10%) portfolio:")
print(f"  Weights: {dict(zip(asset_names, [f'{w:.4f}' for w in target_ret['weights']]))}")
print(f"  Expected return: {target_ret['return']:.4f}")
print(f"  Risk:            {target_ret['risk']:.4f}")

# =========================================================================
# 5. Summary comparison
# =========================================================================
print("\n" + "=" * 60)
print("Portfolio Comparison")
print("=" * 60)

equal_weight = np.ones(5) / 5
eq_ret = (returns.mean() * 252) @ equal_weight
eq_risk = np.sqrt(equal_weight @ (returns.cov().values * 252) @ equal_weight)

print(f"\n{'Portfolio':<20} {'Return':<10} {'Risk':<10} {'Sharpe':<10}")
print("-" * 50)
print(f"{'Equal Weight':<20} {eq_ret:<10.4f} {eq_risk:<10.4f} {eq_ret/eq_risk:<10.4f}")
print(f"{'Risk Parity':<20} {'-':<10} {port_vol:<10.4f} {'-':<10}")
print(f"{'Max Sharpe':<20} {max_sharpe['return']:<10.4f} {max_sharpe['risk']:<10.4f} {max_sharpe['sharpe']:<10.4f}")
print(f"{'Min Variance':<20} {min_var['return']:<10.4f} {min_var['risk']:<10.4f} {min_var['return']/min_var['risk']:<10.4f}")

print("\nDone! All optimization examples executed successfully.")
