"""Example 4: Risk Management Report

This example demonstrates comprehensive risk management analysis.
"""

import numpy as np
import pandas as pd

import fincore


def generate_portfolio_data(
    n_days: int = 252 * 5,  # 5 years
    seed: int = 42
) -> pd.Series:
    """Generate synthetic portfolio returns with various risk characteristics."""
    np.random.seed(seed)

    dates = pd.bdate_range('2019-01-01', periods=n_days)

    # Base returns
    daily_return = 0.10 / 252
    daily_vol = 0.16 / np.sqrt(252)
    returns = np.random.normal(daily_return, daily_vol, n_days)

    # Add stress periods
    returns[252:270] -= 0.15  # COVID-like crash
    returns[270:280] += 0.10  # Partial recovery
    returns[756:770] -= 0.08  # Another drawdown

    return pd.Series(returns, index=dates)


def main():
    """Generate a comprehensive risk management report."""

    print("=" * 70)
    print(" " * 20 + "RISK MANAGEMENT REPORT")
    print("=" * 70)

    returns = generate_portfolio_data()

    # Basic Performance
    print("\n" + "=" * 70)
    print("PERFORMANCE SUMMARY")
    print("=" * 70)

    ann_return = fincore.annual_return(returns)
    ann_vol = fincore.annual_volatility(returns)
    sharpe = fincore.sharpe_ratio(returns)
    sortino = fincore.sortino_ratio(returns)
    calmar = fincore.calmar_ratio(returns)

    print(f"Annual Return:       {ann_return:.2%}")
    print(f"Annual Volatility:   {ann_vol:.2%}")
    print(f"Sharpe Ratio:        {sharpe:.4f}")
    print(f"Sortino Ratio:       {sortino:.4f}")
    print(f"Calmar Ratio:        {calmar:.4f}")

    # Drawdown Analysis
    print("\n" + "=" * 70)
    print("DRAWDOWN ANALYSIS")
    print("=" * 70)

    from fincore.metrics.drawdown import (
        get_all_drawdowns,
        get_max_drawdown_period,
        max_drawdown_days,
        max_drawdown_recovery_days,
    )

    max_dd = fincore.max_drawdown(returns)
    print(f"Maximum Drawdown:           {max_dd:.2%}")

    peak, valley, recovery = get_max_drawdown_period(returns)
    print(f"Max DD Peak Date:            {peak.date()}")
    print(f"Max DD Valley Date:           {valley.date()}")
    if pd.notna(recovery):
        print(f"Max DD Recovery Date:        {recovery.date()}")
    else:
        print("Max DD Recovery Date:        Not yet recovered")

    dd_days = max_drawdown_days(returns)
    recovery_days = max_drawdown_recovery_days(returns)
    print(f"Max DD Duration (days):       {dd_days}")
    print(f"Max DD Recovery (days):      {recovery_days if recovery_days > 0 else 'N/A'}")

    # All drawdowns
    all_drawdowns = get_all_drawdowns(returns)
    print(f"\nTotal Drawdowns:             {len(all_drawdowns)}")

    if len(all_drawdowns) > 1:
        from fincore.metrics.drawdown import second_max_drawdown, second_max_drawdown_days, third_max_drawdown

        print(f"Second Max Drawdown:        {second_max_drawdown(returns):.2%}")
        print(f"Third Max Drawdown:         {third_max_drawdown(returns):.2%}")

    # Value at Risk
    print("\n" + "=" * 70)
    print("VALUE AT RISK")
    print("=" * 70)

    var_95 = fincore.value_at_risk(returns)
    cvar_95 = fincore.conditional_value_at_risk(returns)

    print(f"VaR (95%):                   {var_95:.2%}")
    print(f"CVaR (95%):                  {cvar_95:.2%}")

    # Downside risk
    print("\n" + "=" * 70)
    print("DOWNSIDE RISK")
    print("=" * 70)

    downside_risk = fincore.downside_risk(returns)
    print(f"Downside Risk:                {downside_risk:.4f}")

    # Tail ratio
    tail_ratio = fincore.tail_ratio(returns)
    print(f"Tail Ratio (95th/5th):        {tail_ratio:.4f}")

    # Skewness and Kurtosis
    print("\n" + "=" * 70)
    print("DISTRIBUTION STATISTICS")
    print("=" * 70)

    from fincore.metrics.stats import kurtosis, skewness
    print(f"Skewness:                   {skewness(returns):.4f}")
    print(f"Kurtosis:                   {kurtosis(returns):.4f}")

    # Win/Loss analysis
    print("\n" + "=" * 70)
    print("WIN/LOSS ANALYSIS")
    print("=" * 70)

    from fincore.metrics.stats import loss_rate, win_rate

    win_rt = win_rate(returns)
    loss_rt = loss_rate(returns)

    print(f"Win Rate:                   {win_rt:.1%}")
    print(f"Loss Rate:                  {loss_rt:.1%}")

    # Consecutive streaks
    from fincore.metrics.consecutive import (
        max_consecutive_down_days,
        max_consecutive_gain,
        max_consecutive_loss,
        max_consecutive_up_days,
    )

    max_up_streak = max_consecutive_up_days(returns)
    max_down_streak = max_consecutive_down_days(returns)
    max_gain = max_consecutive_gain(returns)
    max_loss = max_consecutive_loss(returns)

    print(f"Max Consecutive Up Days:    {max_up_streak}")
    print(f"Max Consecutive Down Days:  {max_down_streak}")
    print(f"Max Consecutive Gain:        {max_gain:.2%}")
    print(f"Max Consecutive Loss:        {max_loss:.2%}")

    # Stability
    print("\n" + "=" * 70)
    print("STABILITY ANALYSIS")
    print("=" * 70)

    stability = fincore.stability_of_timeseries(returns)
    print(f"R-squared (stability):       {stability:.4f}")

    # Hurst exponent
    from fincore.metrics.stats import hurst_exponent
    hurst = hurst_exponent(returns)
    print(f"Hurst Exponent:             {hurst:.4f}")
    if hurst < 0.5:
        print("  → Mean-reverting behavior")
    elif hurst > 0.5:
        print("  → Trending behavior")
    else:
        print("  → Random walk")

    # Rolling risk metrics
    print("\n" + "=" * 70)
    print("ROLLING RISK METRICS (252-day window)")
    print("=" * 70)

    rolling_vol = fincore.empyrical.rolling_volatility(returns, window=252)
    rolling_sharpe = fincore.empyrical.roll_sharpe_ratio(returns, window=252)
    rolling_dd = fincore.empyrical.roll_max_drawdown(returns, window=252)

    print(f"Latest Rolling Volatility:    {rolling_vol.iloc[-1]:.2%}")
    print(f"Latest Rolling Sharpe:       {rolling_sharpe.iloc[-1]:.4f}")
    print(f"Latest Rolling Max DD:       {rolling_dd.iloc[-1]:.2%}")

    # Risk assessment
    print("\n" + "=" * 70)
    print("RISK ASSESSMENT")
    print("=" * 70)

    # Calculate risk score
    risk_factors = []
    risk_score = 0

    if max_dd < -0.10:
        risk_factors.append("High drawdown (>10%)")
        risk_score += 2
    elif max_dd < -0.20:
        risk_factors.append("Very high drawdown (>20%)")
        risk_score += 4

    if ann_vol > 0.20:
        risk_factors.append("High volatility (>20%)")
        risk_score += 2

    if skewness(returns) < -1:
        risk_factors.append("Negative skew (fat left tail)")
        risk_score += 2

    if kurtosis(returns) > 5:
        risk_factors.append("High kurtosis (fat tails)")
        risk_score += 1

    print(f"Risk Score:                  {risk_score}/10")
    if risk_factors:
        print("Risk Factors:")
        for factor in risk_factors:
            print(f"  - {factor}")
    else:
        print("  No significant risk factors detected")

    if risk_score <= 3:
        print("\nOverall Risk Level:        LOW")
    elif risk_score <= 6:
        print("\nOverall Risk Level:        MEDIUM")
    else:
        print("\nOverall Risk Level:        HIGH")

    return returns


if __name__ == "__main__":
    main()
