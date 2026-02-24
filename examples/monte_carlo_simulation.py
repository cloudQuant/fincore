"""
Monte Carlo Simulation & Bootstrap Example

Demonstrates fincore's simulation capabilities:
  1. Monte Carlo path simulation
  2. Bootstrap confidence intervals
  3. VaR/CVaR estimation via simulation

Usage:
    python examples/monte_carlo_simulation.py
"""

import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# =========================================================================
# 1. Generate sample return data
# =========================================================================
np.random.seed(42)
dates = pd.date_range("2019-01-01", periods=750, freq="B")
returns = pd.Series(
    np.random.normal(0.0004, 0.012, len(dates)),
    index=dates,
    name="strategy",
)

print(f"Sample data: {len(returns)} days")
print(f"  Mean daily return: {returns.mean():.6f}")
print(f"  Daily volatility:  {returns.std():.6f}")
print()

# =========================================================================
# 2. Monte Carlo Path Simulation
# =========================================================================
from fincore.simulation import MonteCarlo

print("=" * 60)
print("Monte Carlo Path Simulation")
print("=" * 60)

mc = MonteCarlo.simulate(returns, n_paths=1000, horizon=252)
print(f"\nSimulated {mc.n_paths} paths over {mc.horizon} days")
print("Terminal wealth statistics (starting at 1.0):")
terminal = mc.terminal_values
print(f"  Mean:   {terminal.mean():.4f}")
print(f"  Median: {np.median(terminal):.4f}")
print(f"  Std:    {terminal.std():.4f}")
print(f"  Min:    {terminal.min():.4f}")
print(f"  Max:    {terminal.max():.4f}")

# Percentiles
for p in [5, 25, 50, 75, 95]:
    print(f"  {p}th percentile: {np.percentile(terminal, p):.4f}")

# Simulated VaR and CVaR
sim_var = mc.var(alpha=0.05)
sim_cvar = mc.cvar(alpha=0.05)
print("\nSimulation-based risk measures:")
print(f"  VaR(95%):  {sim_var:.4f}")
print(f"  CVaR(95%): {sim_cvar:.4f}")

# =========================================================================
# 3. Bootstrap Confidence Intervals
# =========================================================================
from fincore.simulation import bootstrap, bootstrap_ci

print("\n" + "=" * 60)
print("Bootstrap Statistical Inference")
print("=" * 60)

# Bootstrap the Sharpe ratio
from fincore import Empyrical


def sharpe_func(r):
    return Empyrical.sharpe_ratio(r)

ci = bootstrap_ci(returns, func=sharpe_func, n_samples=5000, alpha=0.05)
print("\nSharpe ratio bootstrap (5000 samples):")
print(f"  Point estimate: {sharpe_func(returns):.4f}")
print(f"  95% CI: [{ci[0]:.4f}, {ci[1]:.4f}]")

# Bootstrap the max drawdown
def max_dd_func(r):
    return Empyrical.max_drawdown(r)

ci_dd = bootstrap_ci(returns, func=max_dd_func, n_samples=5000, alpha=0.05)
print("\nMax drawdown bootstrap (5000 samples):")
print(f"  Point estimate: {max_dd_func(returns):.4f}")
print(f"  95% CI: [{ci_dd[0]:.4f}, {ci_dd[1]:.4f}]")

# Generic bootstrap distribution
boot_dist = bootstrap(returns, func=sharpe_func, n_samples=5000)
print("\nBootstrap distribution of Sharpe ratio:")
print(f"  Mean:   {boot_dist.mean():.4f}")
print(f"  Std:    {boot_dist.std():.4f}")
print(f"  Skew:   {pd.Series(boot_dist).skew():.4f}")

# =========================================================================
# 4. Summary
# =========================================================================
print("\n" + "=" * 60)
print("Summary")
print("=" * 60)
print("""
Monte Carlo simulation and bootstrap provide complementary tools:

  Monte Carlo:
    - Forward-looking path simulation
    - Terminal wealth distribution
    - Simulation-based VaR/CVaR

  Bootstrap:
    - Confidence intervals for any statistic
    - Model-free (non-parametric)
    - Assess estimation uncertainty
""")
print("Done! All simulation examples executed successfully.")
