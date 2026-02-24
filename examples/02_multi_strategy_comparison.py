"""Example 2: Multi-Strategy Comparison

This example demonstrates how to compare multiple trading strategies.
"""

import numpy as np
import pandas as pd

import fincore


def generate_multiple_strategies(
    n_days: int = 252 * 3,
    n_strategies: int = 5,
    seed: int = 42
) -> pd.DataFrame:
    """Generate synthetic returns for multiple strategies."""
    np.random.seed(seed)

    strategies = {}
    strategy_configs = [
        {"name": "Conservative", "ret": 0.06, "vol": 0.10},
        {"name": "Balanced", "ret": 0.10, "vol": 0.15},
        {"name": "Aggressive", "ret": 0.15, "vol": 0.22},
        {"name": "Momentum", "ret": 0.12, "vol": 0.18},
        {"name": "MeanReversion", "ret": 0.08, "vol": 0.12},
    ]

    dates = pd.bdate_range('2021-01-01', periods=n_days)

    for config in strategy_configs[:n_strategies]:
        daily_ret = config["ret"] / 252
        daily_vol = config["vol"] / np.sqrt(252)
        returns = np.random.normal(daily_ret, daily_vol, n_days)
        strategies[config["name"]] = returns

    return pd.DataFrame(strategies, index=dates)


def main():
    """Compare multiple trading strategies."""

    print("=" * 60)
    print("Multi-Strategy Comparison")
    print("=" * 60)

    # Generate strategy returns
    returns = generate_multiple_strategies()
    print(f"\nComparing {len(returns.columns)} strategies over {len(returns)} days")
    print(f"Strategies: {', '.join(returns.columns)}")

    # Calculate metrics for all strategies
    print("\n--- Risk-Adjusted Performance ---")

    sharpe_ratios = fincore.sharpe_ratio(returns)
    sortino_ratios = fincore.sortino_ratio(returns)
    max_drawdowns = fincore.max_drawdown(returns)
    calmar_ratios = fincore.calmar_ratio(returns)

    # Create comparison table
    comparison = pd.DataFrame({
        "Sharpe Ratio": sharpe_ratios,
        "Sortino Ratio": sortino_ratios,
        "Max Drawdown": max_drawdowns,
        "Calmar Ratio": calmar_ratios,
    })

    # Sort by Sharpe Ratio
    comparison = comparison.sort_values("Sharpe Ratio", ascending=False)
    print(comparison.to_string())

    # Annual returns and volatility
    print("\n--- Returns & Volatility ---")
    annual_returns = fincore.annual_return(returns)
    annual_vols = fincore.annual_volatility(returns)

    returns_vol_comparison = pd.DataFrame({
        "Annual Return": annual_returns,
        "Annual Volatility": annual_vols,
        "Return/Vol": annual_returns / annual_vols,
    })
    print(returns_vol_comparison.sort_values("Annual Return", ascending=False))

    # Find best strategy by different metrics
    print("\n--- Best Strategies by Metric ---")
    print(f"Best Sharpe Ratio: {sharpe_ratios.idxmax()} ({sharpe_ratios.max():.4f})")
    print(f"Best Sortino Ratio: {sortino_ratios.idxmax()} ({sortino_ratios.max():.4f})")
    print(f"Lowest Max Drawdown: {max_drawdowns.idxmin()} ({max_drawdowns.min():.2%})")
    print(f"Highest Calmar Ratio: {calmar_ratios.idxmax()} ({calmar_ratios.max():.4f})")

    # Rolling comparison
    print("\n--- Rolling Sharpe Ratios (252-day window) ---")
    rolling_sharpe = fincore.empyrical.roll_sharpe_ratio(returns, window=252)

    # Print latest rolling Sharpe ratios
    latest_rolling = rolling_sharpe.iloc[-1]
    print("Latest Rolling Sharpe Ratios:")
    for strategy in latest_rolling.index:
        print(f"  {strategy}: {latest_rolling[strategy]:.4f}")

    # Visualize cumulative returns
    print("\n--- Cumulative Returns ---")
    cum_returns = fincore.cum_returns(returns)

    # Final cumulative returns
    final_returns = cum_returns.iloc[-1]
    print("Final Cumulative Returns:")
    for strategy in final_returns.index:
        print(f"  {strategy}: {final_returns[strategy]:.2%}")

    # Win rate analysis
    print("\n--- Win Rate Analysis ---")
    from fincore.metrics.stats import win_rate

    win_rates = returns.apply(win_rate)
    print("Win Rates (days with positive returns):")
    for strategy in win_rates.index:
        print(f"  {strategy}: {win_rates[strategy]:.1%}")

    # Correlation analysis
    print("\n--- Strategy Correlations ---")
    corr_matrix = returns.corr()
    print(corr_matrix.to_string())

    return returns, comparison


if __name__ == "__main__":
    main()
