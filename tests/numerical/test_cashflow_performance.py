"""Independent numerical contracts for enhanced cashflow performance metrics."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from fincore.performance.cashflows import cashflow_adjusted_returns, cashflow_adjusted_twr
from tests.oracles.performance.cashflow_oracle import cashflow_adjusted_period_return


def _valuations(values: list[float]) -> pd.Series:
    return pd.Series(
        values,
        index=pd.date_range("2024-01-31", periods=len(values), freq="ME", tz="UTC"),
        dtype=float,
    )


def test_end_timed_cashflows_are_removed_from_period_return_with_hand_oracle() -> None:
    valuations = _valuations([100.0, 110.0, 121.0])
    cashflows = pd.Series([10.0], index=[valuations.index[1]])

    actual = cashflow_adjusted_returns(valuations, cashflows, timing="end")
    expected = pd.Series(
        [
            cashflow_adjusted_period_return(100.0, 110.0, 10.0, 0.0, timing="end", fee_treatment="net"),
            cashflow_adjusted_period_return(110.0, 121.0, 0.0, 0.0, timing="end", fee_treatment="net"),
        ],
        index=valuations.index[1:],
    )

    pd.testing.assert_series_equal(actual, expected)
    assert cashflow_adjusted_twr(valuations, cashflows, timing="end") == pytest.approx(0.1, abs=1e-12)


def test_start_timed_cashflows_adjust_the_opening_base_with_hand_oracle() -> None:
    valuations = _valuations([100.0, 121.0])
    cashflows = pd.Series([10.0], index=[valuations.index[1]])

    actual = cashflow_adjusted_returns(valuations, cashflows, timing="start")
    expected = cashflow_adjusted_period_return(100.0, 121.0, 10.0, 0.0, timing="start", fee_treatment="net")

    assert actual.iloc[0] == pytest.approx(expected, abs=1e-12)
    assert cashflow_adjusted_twr(valuations, cashflows, timing="start") == pytest.approx(0.1, abs=1e-12)


def test_per_cashflow_timings_can_reconcile_mixed_start_and_end_of_day_flows() -> None:
    valuations = _valuations([100.0, 121.0, 133.1])
    cashflows = pd.Series([10.0, 12.1], index=valuations.index[1:])
    cashflow_timings = pd.Series(["start", "end"], index=valuations.index[1:])

    actual = cashflow_adjusted_returns(
        valuations,
        cashflows,
        timing="end",
        cashflow_timings=cashflow_timings,
    )

    assert actual.tolist() == pytest.approx([0.1, 0.0], abs=1e-12)
    assert cashflow_adjusted_twr(
        valuations,
        cashflows,
        timing="end",
        cashflow_timings=cashflow_timings,
    ) == pytest.approx(0.1, abs=1e-12)


def test_transaction_ledger_preserves_opposite_timing_flows_at_one_valuation() -> None:
    valuations = _valuations([100.0, 116.0])
    ledger = pd.DataFrame(
        {"amount": [10.0, -5.0], "timing": ["start", "end"]},
        index=[valuations.index[1], valuations.index[1]],
    )

    actual = cashflow_adjusted_returns(valuations, ledger)

    assert actual.iloc[0] == pytest.approx(0.1, abs=1e-12)
    assert cashflow_adjusted_twr(valuations, ledger) == pytest.approx(0.1, abs=1e-12)


def test_transaction_ledger_rejects_ambiguous_schema_and_opening_flows() -> None:
    valuations = _valuations([100.0, 116.0])
    valid_ledger = pd.DataFrame(
        {"amount": [10.0, -5.0], "timing": ["start", "end"]},
        index=[valuations.index[1], valuations.index[1]],
    )

    with pytest.raises(ValueError, match="columns must be exactly"):
        cashflow_adjusted_returns(valuations, valid_ledger.assign(currency="USD"))
    with pytest.raises(ValueError, match="not valid with a transaction ledger"):
        cashflow_adjusted_returns(
            valuations,
            valid_ledger,
            cashflow_timings=pd.Series(["end"], index=[valuations.index[1]]),
        )
    opening_flows = pd.DataFrame(
        {"amount": [10.0, -10.0], "timing": ["start", "end"]},
        index=[valuations.index[0], valuations.index[0]],
    )
    with pytest.raises(ValueError, match="first valuation date"):
        cashflow_adjusted_returns(valuations, opening_flows)


def test_transaction_ledger_converts_each_duplicate_timestamp_event_with_fx() -> None:
    valuations = _valuations([100.0, 117.2])
    ledger = pd.DataFrame(
        {"amount": [10.0, -5.0], "timing": ["start", "end"]},
        index=[valuations.index[1], valuations.index[1]],
    )
    fx_rates = pd.Series([1.2, 1.2], index=valuations.index)

    actual = cashflow_adjusted_twr(
        valuations,
        ledger,
        cashflow_currency="EUR",
        reporting_currency="USD",
        fx_rates=fx_rates,
    )

    assert actual == pytest.approx(0.1, abs=1e-12)


def test_per_cashflow_timing_ledger_rejects_partial_or_mismatched_metadata() -> None:
    valuations = _valuations([100.0, 121.0, 133.1])
    cashflows = pd.Series([10.0, 12.1], index=valuations.index[1:])

    with pytest.raises(ValueError, match="cover every and only"):
        cashflow_adjusted_returns(
            valuations,
            cashflows,
            cashflow_timings=pd.Series(["start"], index=[valuations.index[1]]),
        )
    with pytest.raises(ValueError, match="timezone must match"):
        cashflow_adjusted_returns(
            valuations,
            cashflows,
            cashflow_timings=pd.Series(
                ["start", "end"],
                index=pd.date_range("2024-02-29", periods=2, freq="ME", tz="America/New_York"),
            ),
        )


def test_gross_fee_treatment_adds_explicit_fees_back_to_closing_value() -> None:
    valuations = _valuations([100.0, 108.0])
    fees = pd.Series([2.0], index=[valuations.index[1]])

    net = cashflow_adjusted_returns(valuations, fees=fees, fee_treatment="net")
    gross = cashflow_adjusted_returns(valuations, fees=fees, fee_treatment="gross")

    assert net.iloc[0] == pytest.approx(
        cashflow_adjusted_period_return(100.0, 108.0, 0.0, 2.0, timing="end", fee_treatment="net"),
        abs=1e-12,
    )
    assert gross.iloc[0] == pytest.approx(
        cashflow_adjusted_period_return(100.0, 108.0, 0.0, 2.0, timing="end", fee_treatment="gross"),
        abs=1e-12,
    )
    assert net.iloc[0] == pytest.approx(0.08, abs=1e-12)
    assert gross.iloc[0] == pytest.approx(0.10, abs=1e-12)


def test_cross_currency_cashflows_require_and_use_an_explicit_fx_series() -> None:
    valuations = _valuations([100.0, 112.0])
    cashflows = pd.Series([10.0], index=[valuations.index[1]])
    fx_rates = pd.Series([1.2, 1.2], index=valuations.index)

    with pytest.raises(ValueError, match="FX"):
        cashflow_adjusted_returns(
            valuations,
            cashflows,
            cashflow_currency="EUR",
            reporting_currency="USD",
        )

    actual = cashflow_adjusted_returns(
        valuations,
        cashflows,
        cashflow_currency="EUR",
        reporting_currency="USD",
        fx_rates=fx_rates,
    )

    assert actual.iloc[0] == pytest.approx(0.0, abs=1e-12)


def test_cashflow_contract_rejects_empty_currency_and_naive_time_axis() -> None:
    valuations = _valuations([100.0, 110.0])
    cashflows = pd.Series([10.0], index=[valuations.index[1]])
    naive_valuations = pd.Series([100.0, 110.0], index=pd.date_range("2024-01-31", periods=2, freq="ME"))

    with pytest.raises(ValueError, match="currency code"):
        cashflow_adjusted_returns(valuations, cashflows, cashflow_currency="")
    with pytest.raises(ValueError, match="timezone-aware"):
        cashflow_adjusted_returns(naive_valuations)


def test_cashflow_contract_rejects_overflowing_fx_and_nonrepresentable_return_factors() -> None:
    valuations = _valuations([100.0, 100.0])
    cashflows = pd.Series([np.finfo(float).max], index=[valuations.index[1]])
    fx_rates = pd.Series([1.0, 2.0], index=valuations.index)

    with pytest.raises(ValueError, match="FX conversion"):
        cashflow_adjusted_returns(
            valuations,
            cashflows,
            cashflow_currency="EUR",
            reporting_currency="USD",
            fx_rates=fx_rates,
        )

    with pytest.raises(ValueError, match="finite"):
        cashflow_adjusted_returns(_valuations([np.nextafter(0.0, 1.0), np.finfo(float).max]))
    with pytest.raises(ValueError, match="representable"):
        cashflow_adjusted_returns(_valuations([np.finfo(float).max, np.nextafter(0.0, 1.0)]))


def test_cashflow_contract_rejects_unvalued_dates_and_nonpositive_capital_base() -> None:
    valuations = _valuations([100.0, 110.0])
    unvalued_cashflow = pd.Series([5.0], index=[pd.Timestamp("2024-02-15", tz="UTC")])

    with pytest.raises(ValueError, match="valuation index"):
        cashflow_adjusted_returns(valuations, unvalued_cashflow)
    with pytest.raises(ValueError, match="positive"):
        cashflow_adjusted_returns(_valuations([0.0, 110.0]))


def test_terminal_zero_valuation_represents_a_valid_total_loss() -> None:
    valuations = _valuations([100.0, 0.0])

    actual = cashflow_adjusted_returns(valuations)

    assert actual.iloc[0] == pytest.approx(-1.0, abs=1e-12)
    assert cashflow_adjusted_twr(valuations) == pytest.approx(-1.0, abs=1e-12)


def test_cashflow_contract_rejects_bad_fx_index_and_nonfinite_values() -> None:
    valuations = _valuations([100.0, 112.0])
    cashflows = pd.Series([10.0], index=[valuations.index[1]])

    with pytest.raises(ValueError, match="same valuation index"):
        cashflow_adjusted_returns(
            valuations,
            cashflows,
            cashflow_currency="EUR",
            reporting_currency="USD",
            fx_rates=pd.Series([1.2], index=[valuations.index[1]]),
        )
    with pytest.raises(ValueError, match="finite"):
        cashflow_adjusted_returns(valuations, pd.Series([np.nan], index=[valuations.index[1]]))
