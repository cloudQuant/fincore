"""Example 1: Single Strategy Backtest Analysis

This example demonstrates how to analyze the performance of a single trading strategy.
"""

import numpy as np
import pandas as pd

import fincore


def generate_strategy_returns(
    n_days: int = 252 * 3,  # 3 years of daily data
    annual_return: float = 0.12,
    annual_volatility: float = 0.18,
    seed: int = 42
) -> pd.Series:
    """Generate synthetic strategy returns for demonstration."""
    np.random.seed(seed)

    # Generate daily returns with given annual characteristics
    daily_return = annual_return / 252
    daily_vol = annual_volatility / np.sqrt(252)

    returns = np.random.normal(daily_return, daily_vol, n_days)

    # Add a few drawdown periods
    returns[50:60] -= 0.02  # First drawdown
    returns[150:165] -= 0.03  # Second drawdown

    dates = pd.bdate_range('2021-01-01', periods=n_days)
    return pd.Series(returns, index=dates)


def main():
    """Analyze a single trading strategy."""

    print("=" * 60)
    print("Single Strategy Backtest Analysis")
    print("=" * 60)

    # Generate strategy returns
    strategy_returns = generate_strategy_returns()
    print(f"\nStrategy data: {len(strategy_returns)} days from {strategy_returns.index[0].date()} to {strategy_returns.index[-1].date()}")

    # Method 1: Using individual metrics
    print("\n--- Individual Metrics ---")
    print(f"Sharpe Ratio:        {fincore.sharpe_ratio(strategy_returns):.4f}")
    print(f"Sortino Ratio:       {fincore.sortino_ratio(strategy_returns):.4f}")
    print(f"Max Drawdown:        {fincore.max_drawdown(strategy_returns):.2%}")
    print(f"Annual Return:       {fincore.annual_return(strategy_returns):.2%}")
    print(f"Annual Volatility:   {fincore.annual_volatility(strategy_returns):.2%}")
    print(f"Calmar Ratio:        {fincore.calmar_ratio(strategy_returns):.4f}")
    print(f"Omega Ratio:         {fincore.omega_ratio(strategy_returns):.4f}")

    # Method 2: Using AnalysisContext (Recommended)
    print("\n--- AnalysisContext (Recommended) ---")
    ctx = fincore.analyze(strategy_returns)

    # Access individual metrics (cached)
    print(f"Sharpe Ratio:        {ctx.sharpe_ratio:.4f}")
    print(f"Max Drawdown:        {ctx.max_drawdown:.2%}")
    print(f"Annual Return:       {ctx.annual_return:.2%}")

    # Get all performance stats at once
    stats = ctx.perf_stats()
    print("\n--- Full Performance Stats ---")
    print(stats.to_string())

    # Export to JSON
    import json
    stats_dict = ctx.to_dict()
    print("\n--- JSON Export ---")
    print(json.dumps(stats_dict, indent=2, default=str))

    # Generate HTML report
    html_file = "/tmp/strategy_report.html"
    ctx.to_html(path=html_file)
    print(f"\nHTML report saved to: {html_file}")

    # Drawdown analysis
    print("\n--- Drawdown Analysis ---")
    from fincore.metrics.drawdown import get_all_drawdowns
    drawdowns = get_all_drawdowns(strategy_returns)
    print(f"Number of drawdowns: {len(drawdowns)}")
    for i, dd_val in enumerate(sorted(drawdowns)[:3], 1):
        print(f"  DD #{i}: {dd_val * 100:.2f}%")

    # Monthly returns heatmap
    print("\n--- Monthly Returns ---")
    from fincore.metrics.returns import aggregate_returns
    monthly_returns = aggregate_returns(strategy_returns, 'monthly')
    print(f"Average monthly return: {monthly_returns.mean() * 100:.2f}%")
    print(f"Best month: {monthly_returns.max() * 100:.2f}%")
    print(f"Worst month: {monthly_returns.min() * 100:.2f}%")

    return ctx


if __name__ == "__main__":
    main()
