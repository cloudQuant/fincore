"""Example 3: Factor Attribution Analysis

This example demonstrates how to perform factor-based performance attribution.
"""

import numpy as np
import pandas as pd

import fincore


def generate_factor_data(
    n_days: int = 252 * 3,
    seed: int = 42
) -> tuple[pd.Series, pd.Series]:
    """Generate synthetic strategy and factor returns."""
    np.random.seed(seed)

    dates = pd.bdate_range('2021-01-01', periods=n_days)

    # Factor returns (market)
    market_daily_return = 0.08 / 252
    market_daily_vol = 0.15 / np.sqrt(252)
    factor_returns = np.random.normal(market_daily_return, market_daily_vol, n_days)

    # Strategy returns with beta exposure to market
    beta = 1.2
    alpha = 0.02  # 2% annual alpha
    strategy_daily_return = (alpha / 252) + beta * factor_returns
    strategy_vol = 0.18 / np.sqrt(252)
    strategy_returns = np.random.normal(strategy_daily_return, strategy_vol, n_days)

    return pd.Series(strategy_returns, index=dates), pd.Series(factor_returns, index=dates)


def main():
    """Perform factor attribution analysis."""

    print("=" * 60)
    print("Factor Attribution Analysis")
    print("=" * 60)

    # Generate data
    strategy_returns, factor_returns = generate_factor_data()
    print(f"\nData: {len(strategy_returns)} days")

    # Calculate alpha and beta
    print("\n--- Alpha & Beta ---")
    alpha, beta = fincore.alpha_beta(strategy_returns, factor_returns)
    print(f"Alpha:  {alpha:.4f} ({alpha*252:.2%} annual)")
    print(f"Beta:   {beta:.4f}")

    # Calculate rolling alpha and beta
    print("\n--- Rolling Alpha & Beta (126-day window) ---")
    rolling_alpha = fincore.empyrical.roll_alpha(
        strategy_returns, factor_returns, window=126
    )
    rolling_beta = fincore.empyrical.roll_beta(
        strategy_returns, factor_returns, window=126
    )

    print(f"Rolling Alpha (latest): {rolling_alpha.iloc[-1]:.4f}")
    print(f"Rolling Beta (latest):  {rolling_beta.iloc[-1]:.4f}")

    # Calculate capture ratios
    print("\n--- Capture Ratios ---")
    up_capture = fincore.up_capture(strategy_returns, factor_returns)
    down_capture = fincore.down_capture(strategy_returns, factor_returns)

    print(f"Up Capture:   {up_capture:.4f}")
    print(f"Down Capture: {down_capture:.4f}")

    # Explain the strategy performance
    print("\n--- Performance Decomposition ---")

    # Calculate annual returns
    strategy_annual = fincore.annual_return(strategy_returns)
    factor_annual = fincore.annual_return(factor_returns)

    print(f"Strategy Annual Return: {strategy_annual:.2%}")
    print(f"Factor Annual Return:   {factor_annual:.2%}")

    # Decompose: alpha + beta * factor_return
    explained_return = alpha + beta * factor_annual
    residual = strategy_annual - explained_return

    print("\nReturn Decomposition:")
    print(f"  Alpha (stock picking):     {alpha * 252:.2%}")
    print(f"  Beta * Factor (timing):    {beta * factor_annual:.2%}")
    print(f"  Explained Return:          {explained_return:.2%}")
    print(f"  Residual:                  {residual:.2%}")

    # Information ratio
    print("\n--- Information Ratio ---")
    ir = fincore.information_ratio(strategy_returns, factor_returns)
    tracking_error = fincore.empyrical.tracking_error(strategy_returns, factor_returns)

    print(f"Information Ratio: {ir:.4f}")
    print(f"Tracking Error:   {tracking_error:.4f}")

    # Treynor ratio
    print("\n--- Treynor Ratio ---")
    treynor = fincore.empyrical.treynor_ratio(strategy_returns, factor_returns)
    print(f"Treynor Ratio: {treynor:.4f}")

    # Compare to benchmark
    print("\n--- Benchmark Comparison ---")
    active_return = fincore.empyrical.annual_active_return(
        strategy_returns, factor_returns
    )

    print(f"Active Return: {active_return:.2%}")

    if active_return > 0:
        print("Strategy outperformed the benchmark (factor).")
    else:
        print("Strategy underperformed the benchmark (factor).")

    # Monthly analysis
    print("\n--- Monthly Alpha & Beta ---")
    annual_alpha = fincore.empyrical.annual_alpha(
        strategy_returns, factor_returns
    )
    annual_beta = fincore.empyrical.annual_beta(
        strategy_returns, factor_returns
    )

    print("Alpha by Year:")
    if len(annual_alpha) > 0:
        for year, alpha_val in annual_alpha.items():
            print(f"  {year}: {alpha_val:.4f}")

    print("\nBeta by Year:")
    if len(annual_beta) > 0:
        for year, beta_val in annual_beta.items():
            print(f"  {year}: {beta_val:.4f}")

    return strategy_returns, factor_returns


if __name__ == "__main__":
    main()
