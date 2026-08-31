"""Canonical enhanced-performance scenarios for source and wheel evidence."""

from __future__ import annotations

from typing import get_args

import numpy as np
import pandas as pd

from fincore.performance import (
    CashflowTiming,
    DisclosureContext,
    FeeTreatment,
    cashflow_adjusted_returns,
    cashflow_adjusted_twr,
    render_disclosure,
    sharpe_confidence_interval,
    sharpe_standard_error,
    standard_error_of_mean,
)


def _valuation_index(periods: int) -> pd.DatetimeIndex:
    return pd.date_range("2025-01-02", periods=periods, freq="D", tz="UTC")


def test_cashflow_adjusted_returns() -> None:
    """End-timed contributions are removed before each period return is measured."""
    index = _valuation_index(3)
    valuations = pd.Series([100.0, 110.0, 121.0], index=index)
    cashflows = pd.Series([0.0, 10.0, 0.0], index=index)

    actual = cashflow_adjusted_returns(valuations, cashflows, timing="end")

    pd.testing.assert_series_equal(actual, pd.Series([0.0, 0.1], index=index[1:]))


def test_cashflow_adjusted_twr() -> None:
    """The total return compounds the same cashflow-adjusted period factors."""
    index = _valuation_index(3)
    valuations = pd.Series([100.0, 110.0, 121.0], index=index)
    cashflows = pd.Series([0.0, 10.0, 0.0], index=index)

    np.testing.assert_allclose(cashflow_adjusted_twr(valuations, cashflows, timing="end"), 0.1, rtol=0.0, atol=1e-12)


def test_cashflow_timing_and_fee_treatment() -> None:
    """Timing and fee modes are explicit rather than implicit accounting choices."""
    timing_index = _valuation_index(2)
    timing_valuations = pd.Series([100.0, 121.0], index=timing_index)
    timing_cashflow = pd.Series([0.0, 10.0], index=timing_index)
    fee_valuations = pd.Series([100.0, 108.0], index=timing_index)
    fees = pd.Series([0.0, 2.0], index=timing_index)

    start = cashflow_adjusted_returns(timing_valuations, timing_cashflow, timing="start")
    end = cashflow_adjusted_returns(timing_valuations, timing_cashflow, timing="end")
    net = cashflow_adjusted_returns(fee_valuations, fees=fees, fee_treatment="net")
    gross = cashflow_adjusted_returns(fee_valuations, fees=fees, fee_treatment="gross")

    assert set(get_args(CashflowTiming)) == {"end", "start"}
    assert set(get_args(FeeTreatment)) == {"gross", "net"}
    np.testing.assert_allclose(start.iloc[0], 0.1, rtol=0.0, atol=1e-12)
    np.testing.assert_allclose(end.iloc[0], 0.11, rtol=0.0, atol=1e-12)
    np.testing.assert_allclose(net.iloc[0], 0.08, rtol=0.0, atol=1e-12)
    np.testing.assert_allclose(gross.iloc[0], 0.1, rtol=0.0, atol=1e-12)


def test_disclosure_context() -> None:
    """The disclosure model preserves explicit user assertions as data."""
    context = DisclosureContext(
        convention="TWR",
        sample_period="2024",
        fees="net-of-fees",
        cashflows="end-of-day",
        annualized=False,
        notes=("reviewed",),
    )

    assert context.convention == "TWR"
    assert context.sample_period == "2024"
    assert context.annualized is False
    assert context.notes == ("reviewed",)


def test_render_disclosure() -> None:
    """The rendered block exposes the declared convention and all material caveats."""
    context = DisclosureContext(
        convention="MWR",
        return_type="simple",
        units="decimal",
        frequency="monthly",
        notes=("estimated", "not audited"),
    )

    rendered = render_disclosure(context)

    assert "Convention: MWR" in rendered
    assert "Return type: simple" in rendered
    assert "Units: decimal" in rendered
    assert "Frequency: monthly" in rendered
    assert "Notes: estimated; not audited" in rendered


def test_sharpe_inference() -> None:
    """Sharpe uncertainty and mean uncertainty agree with their formulas."""
    returns = np.array([-0.01, 0.00, 0.02, 0.03])

    mean_standard_error = standard_error_of_mean(returns)
    sharpe_standard_error_value = sharpe_standard_error(returns)
    lower, upper = sharpe_confidence_interval(returns)
    expected_sharpe = returns.mean() / returns.std(ddof=1)

    assert mean_standard_error == np.std(returns, ddof=1) / np.sqrt(len(returns))
    assert sharpe_standard_error_value > 0.0
    assert lower < upper
    assert np.isclose((lower + upper) / 2.0, expected_sharpe)
    assert np.isclose(upper - lower, 2.0 * 1.96 * sharpe_standard_error_value)
