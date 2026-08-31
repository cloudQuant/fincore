"""Independent arithmetic and adversarial tests for enhanced factor costs."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from fincore.factor_analysis.costs import FactorCostModel, apply_factor_costs, estimate_factor_capacity
from tests.oracles.factor.costs_oracle import factor_cost_ledger_reference


def _inputs() -> tuple[pd.Series, pd.Series, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    dates = pd.date_range("2024-01-02", periods=3, freq="B", tz="UTC", name="date")
    assets = ("A", "B")
    weights = pd.Series(
        [0.60, -0.40, 0.20, -0.80, 0.50, -0.50],
        index=pd.MultiIndex.from_product((dates, assets), names=("date", "asset")),
        name="weight",
    )
    gross_returns = pd.Series([0.02, 0.01, -0.01], index=dates, name="gross_return")
    dollar_volume = pd.DataFrame(
        {"A": [1_000.0, 1_500.0, 2_000.0], "B": [2_000.0, 1_000.0, 2_000.0]},
        index=dates,
    )
    borrow_rates = pd.DataFrame({"A": [0.0, 0.0, 0.0], "B": [0.002, 0.003, 0.004]}, index=dates)
    borrow_available = pd.DataFrame(True, index=dates, columns=assets)
    return gross_returns, weights, dollar_volume, borrow_rates, borrow_available


def test_costs_and_capacity_reconcile_to_a_hand_computed_ledger() -> None:
    gross_returns, weights, dollar_volume, borrow_rates, borrow_available = _inputs()
    model = FactorCostModel(
        half_spread_bps=10.0,
        impact_coefficient=0.01,
        impact_exponent=0.5,
        max_participation=0.75,
    )

    result = apply_factor_costs(
        gross_returns,
        weights,
        dollar_volume,
        portfolio_value=1_000.0,
        model=model,
        borrow_rates=borrow_rates,
        borrow_available=borrow_available,
    )

    dates = gross_returns.index
    reference = factor_cost_ledger_reference(
        gross_returns.to_numpy(dtype=float),
        np.array([[0.60, -0.40], [0.20, -0.80], [0.50, -0.50]], dtype=float),
        dollar_volume.loc[:, ["A", "B"]].to_numpy(dtype=float),
        portfolio_value=1_000.0,
        half_spread_bps=10.0,
        impact_coefficient=0.01,
        impact_exponent=0.5,
        max_participation=0.75,
        borrow_rates=borrow_rates.loc[:, ["A", "B"]].to_numpy(dtype=float),
    )
    expected_trade_weights = pd.DataFrame(
        reference.trade_weights,
        index=dates,
        columns=("A", "B"),
    )
    expected_trade_weights.columns.name = "asset"
    expected_participation = pd.DataFrame(reference.participation, index=dates, columns=("A", "B"))
    expected_participation.columns.name = "asset"
    expected_turnover = pd.Series(reference.turnover, index=dates, name="turnover")
    expected_spread = pd.Series(reference.spread_cost, index=dates, name="spread_cost")
    expected_impact = pd.Series(reference.impact_cost, index=dates, name="impact_cost")
    expected_borrow = pd.Series(reference.borrow_cost, index=dates, name="borrow_cost")
    expected_total = pd.Series(reference.total_cost, index=dates, name="total_cost")
    expected_net = pd.Series(reference.net_returns, index=dates, name="net_return")

    pd.testing.assert_frame_equal(result.trade_weights, expected_trade_weights, check_freq=False)
    pd.testing.assert_frame_equal(result.participation, expected_participation, check_freq=False)
    pd.testing.assert_series_equal(
        result.turnover,
        expected_turnover,
        check_freq=False,
    )
    pd.testing.assert_series_equal(result.spread_cost, expected_spread, check_freq=False)
    pd.testing.assert_series_equal(result.impact_cost, expected_impact, check_freq=False)
    pd.testing.assert_series_equal(result.borrow_cost, expected_borrow, check_freq=False)
    pd.testing.assert_series_equal(result.total_cost, expected_total, check_freq=False)
    pd.testing.assert_series_equal(
        result.net_returns,
        expected_net,
        check_freq=False,
    )
    assert result.capacity.maximum_portfolio_value == pytest.approx(reference.maximum_capacity)
    assert (result.participation <= model.max_participation).all().all()


def test_capacity_is_the_minimum_trade_liquidity_bound_and_respects_labels() -> None:
    gross_returns, weights, dollar_volume, borrow_rates, borrow_available = _inputs()

    result = estimate_factor_capacity(
        weights,
        dollar_volume.loc[:, ["B", "A"]].iloc[::-1],
        max_participation=0.20,
    )

    expected_by_date = pd.Series(
        [200.0 / 0.60, 200.0 / 0.40, 400.0 / 0.30],
        index=dollar_volume.index,
        name="maximum_portfolio_value",
    )
    pd.testing.assert_series_equal(result.maximum_portfolio_value_by_date, expected_by_date, check_freq=False)
    assert result.maximum_portfolio_value == pytest.approx(expected_by_date.min())
    assert result.binding_date == expected_by_date.index[0]
    assert result.binding_asset == "A"

    with pytest.raises(ValueError, match="maximum capacity"):
        apply_factor_costs(
            gross_returns,
            weights,
            dollar_volume,
            portfolio_value=np.nextafter(result.maximum_portfolio_value, np.inf),
            model=FactorCostModel(half_spread_bps=0.0, impact_coefficient=0.0, max_participation=0.20),
            borrow_rates=borrow_rates,
            borrow_available=borrow_available,
        )


def test_rejects_unavailable_borrow_and_over_capacity_without_silent_net_return() -> None:
    gross_returns, weights, dollar_volume, borrow_rates, borrow_available = _inputs()
    model = FactorCostModel(half_spread_bps=0.0, impact_coefficient=0.0, max_participation=0.20)

    unavailable = borrow_available.copy(deep=True)
    unavailable.iloc[0, unavailable.columns.get_loc("B")] = False
    with pytest.raises(ValueError, match="borrow.*unavailable"):
        apply_factor_costs(
            gross_returns,
            weights,
            dollar_volume,
            portfolio_value=100.0,
            model=model,
            borrow_rates=borrow_rates,
            borrow_available=unavailable,
        )

    with pytest.raises(ValueError, match="maximum capacity"):
        apply_factor_costs(
            gross_returns,
            weights,
            dollar_volume,
            portfolio_value=1_000.0,
            model=model,
            borrow_rates=borrow_rates,
            borrow_available=borrow_available,
        )


def test_short_positions_require_explicit_borrow_terms() -> None:
    gross_returns, weights, dollar_volume, _, _ = _inputs()
    model = FactorCostModel(half_spread_bps=0.0, impact_coefficient=0.0, max_participation=1.0)

    with pytest.raises(ValueError, match="borrow_rates and borrow_available"):
        apply_factor_costs(
            gross_returns,
            weights,
            dollar_volume,
            portfolio_value=100.0,
            model=model,
        )


def test_sparse_weight_ledger_treats_an_omitted_asset_as_a_zero_position() -> None:
    dates = pd.date_range("2024-01-02", periods=2, freq="B", tz="UTC", name="date")
    weights = pd.Series(
        [1.0, 1.0],
        index=pd.MultiIndex.from_tuples(((dates[0], "A"), (dates[1], "B")), names=("date", "asset")),
    )
    dollar_volume = pd.DataFrame({"A": [1_000.0, 1_000.0], "B": [1_000.0, 1_000.0]}, index=dates)

    result = apply_factor_costs(
        pd.Series([0.0, 0.0], index=dates),
        weights,
        dollar_volume,
        portfolio_value=1_000.0,
        model=FactorCostModel(half_spread_bps=0.0, impact_coefficient=0.0, max_participation=1.0),
    )

    expected = pd.DataFrame({"A": [1.0, 1.0], "B": [0.0, 1.0]}, index=dates)
    expected.columns.name = "asset"
    pd.testing.assert_frame_equal(result.trade_weights, expected, check_freq=False)
    assert np.isfinite(result.participation.to_numpy(dtype=float)).all()


def test_rejects_cost_assumptions_that_overflow_the_return_ledger() -> None:
    dates = pd.date_range("2024-01-02", periods=2, freq="B", tz="UTC", name="date")
    weights = pd.Series(
        [0.5, -0.5, -0.5, 0.5],
        index=pd.MultiIndex.from_product((dates, ("A", "B")), names=("date", "asset")),
    )
    dollar_volume = pd.DataFrame(1_000.0, index=dates, columns=("A", "B"))
    borrow_rates = pd.DataFrame(0.0, index=dates, columns=("A", "B"))
    borrow_available = pd.DataFrame(True, index=dates, columns=("A", "B"))

    with pytest.raises(ValueError, match="cost ledger.*finite"):
        apply_factor_costs(
            pd.Series([0.0, 0.0], index=dates),
            weights,
            dollar_volume,
            portfolio_value=1_000.0,
            model=FactorCostModel(
                half_spread_bps=0.0,
                impact_coefficient=np.finfo(float).max,
                impact_exponent=1.0,
                max_participation=1.0,
            ),
            borrow_rates=borrow_rates,
            borrow_available=borrow_available,
        )


def test_result_snapshots_do_not_expose_mutable_internal_ledgers() -> None:
    gross_returns, weights, dollar_volume, borrow_rates, borrow_available = _inputs()
    result = apply_factor_costs(
        gross_returns,
        weights,
        dollar_volume,
        portfolio_value=1_000.0,
        model=FactorCostModel(half_spread_bps=1.0, impact_coefficient=0.0, max_participation=1.0),
        borrow_rates=borrow_rates,
        borrow_available=borrow_available,
    )

    baseline = result.net_returns
    exposed = result.net_returns
    exposed.iloc[0] = 999.0
    pd.testing.assert_series_equal(result.net_returns, baseline)

    capacity_baseline = result.capacity.trade_weights
    capacity_exposed = result.capacity.trade_weights
    capacity_exposed.iloc[0, 0] = 999.0
    pd.testing.assert_frame_equal(result.capacity.trade_weights, capacity_baseline)
