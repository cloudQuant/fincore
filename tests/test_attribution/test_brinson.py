"""Tests for fincore.attribution.brinson.

These tests validate the core attribution math and basic class behavior.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from fincore.attribution.brinson import (
    BrinsonAttribution,
    brinson_attribution,
    brinson_cumulative,
    brinson_results,
)


def test_brinson_attribution_bhb_identity_holds():
    # 3 "sectors/assets" for a single period.
    rp = np.array([0.02, 0.01, -0.01], dtype=float)
    rb = np.array([0.015, 0.005, -0.005], dtype=float)
    wp = np.array([0.5, 0.3, 0.2], dtype=float)
    wb = np.array([0.4, 0.4, 0.2], dtype=float)

    out = brinson_attribution(rp, rb, wp, wb)
    assert set(out.keys()) >= {
        "allocation",
        "selection",
        "interaction",
        "total",
        "portfolio_return",
        "benchmark_return",
    }

    active_return = out["portfolio_return"] - out["benchmark_return"]
    assert np.isclose(out["total"], active_return)


def test_brinson_attribution_shape_mismatch_raises():
    rp = np.array([0.01, 0.02])
    rb = np.array([0.01, 0.02, 0.03])
    wp = np.array([0.5, 0.5])
    wb = np.array([0.5, 0.5])
    with pytest.raises(ValueError, match="same shape"):
        brinson_attribution(rp, rb, wp, wb)


def test_brinson_attribution_residual_branch_via_monkeypatch(monkeypatch):
    monkeypatch.setattr("fincore.attribution.brinson.np.isclose", lambda *a, **k: False)

    rp = np.array([0.02, 0.01], dtype=float)
    rb = np.array([0.015, 0.005], dtype=float)
    wp = np.array([0.6, 0.4], dtype=float)
    wb = np.array([0.5, 0.5], dtype=float)

    out = brinson_attribution(rp, rb, wp, wb)
    assert "residual" in out


def test_brinson_results_multiple_periods():
    rp = np.array([[0.02, 0.01], [0.00, 0.03]], dtype=float)
    rb = np.array([[0.01, 0.005], [0.01, 0.02]], dtype=float)
    wp = np.array([[0.6, 0.4], [0.5, 0.5]], dtype=float)
    wb = np.array([[0.5, 0.5], [0.5, 0.5]], dtype=float)

    df = brinson_results(rp, rb, wp, wb, periods=["t0", "t1"])
    assert list(df.columns) == [
        "period",
        "allocation",
        "selection",
        "interaction",
        "total",
        "portfolio_return",
        "benchmark_return",
    ]
    assert df.shape[0] == 2
    assert df["period"].tolist() == ["t0", "t1"]


def test_brinson_results_single_period_from_1d_inputs():
    rp = np.array([0.02, 0.01], dtype=float)
    rb = np.array([0.01, 0.005], dtype=float)
    wp = np.array([0.6, 0.4], dtype=float)
    wb = np.array([0.5, 0.5], dtype=float)

    df = brinson_results(rp, rb, wp, wb)
    assert df.shape[0] == 1
    assert df["period"].tolist() == ["0"]


def test_brinson_cumulative_uses_carino_linking_and_reports_absolute_cumulative_active_return():
    rp = np.array([[0.02, 0.01], [0.00, 0.03]], dtype=float)
    rb = np.array([[0.01, 0.005], [0.01, 0.02]], dtype=float)
    wp = np.array([[0.6, 0.4], [0.5, 0.5]], dtype=float)
    wb = np.array([[0.5, 0.5], [0.5, 0.5]], dtype=float)

    cum = brinson_cumulative(rp, rb, wp, wb)

    port_period = np.sum(wp * rp, axis=1)
    bench_period = np.sum(wb * rb, axis=1)

    total = cum["allocation"] + cum["selection"] + cum["interaction"]

    # Geometric cumulative of per-period weighted returns is preserved.
    portfolio_cumulative = float(np.prod(1.0 + port_period) - 1.0)
    benchmark_cumulative = float(np.prod(1.0 + bench_period) - 1.0)
    assert np.isclose(cum["portfolio_cumulative"], portfolio_cumulative)
    assert np.isclose(cum["benchmark_cumulative"], benchmark_cumulative)
    assert np.isclose(total, portfolio_cumulative - benchmark_cumulative, rtol=1e-12, atol=1e-12)


def test_brinson_cumulative_single_period_preserves_absolute_active_return():
    rp = np.array([0.04, 0.01], dtype=float)
    rb = np.array([0.02, 0.00], dtype=float)
    wp = np.array([0.7, 0.3], dtype=float)
    wb = np.array([0.5, 0.5], dtype=float)

    result = brinson_cumulative(rp, rb, wp, wb)

    portfolio_return = float(np.sum(wp * rp))
    benchmark_return = float(np.sum(wb * rb))
    expected_active = portfolio_return - benchmark_return
    assert np.isclose(result["total"], expected_active, rtol=1e-12, atol=1e-12)


def test_brinson_cumulative_handles_equal_return_period_without_nan():
    rp = np.array([[0.03, 0.01], [0.04, 0.00]], dtype=float)
    rb = np.array([[0.02, 0.02], [0.01, 0.01]], dtype=float)
    wp = np.array([[0.5, 0.5], [0.7, 0.3]], dtype=float)
    wb = np.array([[0.5, 0.5], [0.5, 0.5]], dtype=float)

    result = brinson_cumulative(rp, rb, wp, wb)

    portfolio_period = np.sum(wp * rp, axis=1)
    benchmark_period = np.sum(wb * rb, axis=1)
    expected_active = np.prod(1.0 + portfolio_period) - np.prod(1.0 + benchmark_period)
    assert np.isfinite(result["total"])
    assert np.isclose(result["total"], expected_active, rtol=1e-12, atol=1e-12)


def test_brinson_cumulative_zero_cumulative_active_return_is_finite():
    # The two exactly representable gross returns multiply to one, ensuring an
    # exact zero-active fixture instead of relying on an approximate tolerance.
    rp = np.array([[1.0], [-0.5]], dtype=float)
    rb = np.array([[0.00], [0.00]], dtype=float)
    weights = np.ones((2, 1), dtype=float)

    result = brinson_cumulative(rp, rb, weights, weights)

    assert np.isfinite(result["total"])
    assert np.isclose(result["total"], 0.0, rtol=0.0, atol=0.0)


def test_brinson_cumulative_near_loss_boundary_preserves_absolute_active_return():
    rp = np.array([[-0.999999, 0.0]], dtype=float)
    rb = np.array([[-0.999998, 0.0]], dtype=float)
    weights = np.array([[1.0, 0.0]], dtype=float)

    result = brinson_cumulative(rp, rb, weights, weights)

    assert np.isfinite(result["total"])
    assert np.isclose(result["total"], -0.000001, rtol=0.0, atol=1e-12)


def test_brinson_cumulative_accepts_total_loss_components_when_aggregate_returns_are_valid():
    """Carino is defined on aggregate returns, not individual sector returns."""
    rp = np.array([[-1.0, 0.04], [0.02, -1.0]], dtype=float)
    rb = np.array([[-1.0, 0.01], [0.01, -1.0]], dtype=float)
    weights = np.array([[0.1, 0.9], [0.9, 0.1]], dtype=float)

    result = brinson_cumulative(rp, rb, weights, weights)

    assert np.isfinite(result["total"])
    assert np.isclose(
        result["total"],
        result["portfolio_cumulative"] - result["benchmark_cumulative"],
        rtol=1e-12,
        atol=1e-12,
    )


@pytest.mark.parametrize("bad_return", [np.nan, np.inf, -np.inf])
@pytest.mark.parametrize("target", ["portfolio", "benchmark"])
def test_brinson_cumulative_rejects_nonfinite_component_returns(target, bad_return):
    rp = np.array([[0.02, bad_return]], dtype=float)
    rb = np.array([[0.01, 0.00]], dtype=float)
    if target == "benchmark":
        rp[0, 1] = 0.00
        rb[0, 1] = bad_return
    weights = np.array([[0.5, 0.5]], dtype=float)

    with pytest.raises(ValueError, match="must be finite"):
        brinson_cumulative(rp, rb, weights, weights)


def test_brinson_cumulative_rejects_weighted_period_return_at_or_below_total_loss():
    rp = np.array([[-0.75, 0.00]], dtype=float)
    rb = np.array([[0.00, 0.00]], dtype=float)
    portfolio_weights = np.array([[2.0, -1.0]], dtype=float)
    benchmark_weights = np.array([[1.0, 0.0]], dtype=float)

    with pytest.raises(ValueError, match="period returns must be finite and greater than -1"):
        brinson_cumulative(rp, rb, portfolio_weights, benchmark_weights)


def test_brinson_cumulative_single_period_from_1d_inputs():
    rp = np.array([0.02, 0.01], dtype=float)
    rb = np.array([0.01, 0.005], dtype=float)
    wp = np.array([0.6, 0.4], dtype=float)
    wb = np.array([0.5, 0.5], dtype=float)

    out = brinson_cumulative(rp, rb, wp, wb)
    assert set(out.keys()) >= {
        "allocation",
        "selection",
        "interaction",
        "total",
        "portfolio_cumulative",
        "benchmark_cumulative",
    }


def test_brinson_cumulative_shape_mismatch_raises():
    rp = np.array([[0.02, 0.01], [0.00, 0.03]], dtype=float)
    rb = np.array([[0.01, 0.005], [0.01, 0.02]], dtype=float)
    wp = np.array([[0.6, 0.4]], dtype=float)  # wrong shape
    wb = np.array([[0.5, 0.5], [0.5, 0.5]], dtype=float)
    with pytest.raises(ValueError, match="consistent shapes"):
        brinson_cumulative(rp, rb, wp, wb)


def test_brinson_attribution_class_method_validation():
    idx = pd.date_range("2020-01-01", periods=2)
    returns = pd.DataFrame({"A": [0.01, 0.02], "B": [0.00, 0.01]}, index=idx)

    ba = BrinsonAttribution()
    with pytest.raises(ValueError, match="Unknown attribution method"):
        ba.calculate(returns, method="nope")


def test_brinson_hood_method_is_a_bhb_alias_with_independent_oracle():
    """The historical alias must be an implemented BHB decomposition, not a stub."""
    from tests.oracles.attribution.brinson_oracle import brinson_bhb_reference

    idx = pd.date_range("2024-01-31", periods=2, freq="ME")
    returns = pd.DataFrame({"A": [0.04, -0.01], "B": [0.01, 0.03]}, index=idx)
    benchmark_returns = pd.DataFrame({"A": [0.02, 0.00], "B": [0.00, 0.01]}, index=idx)
    weights = pd.DataFrame({"A": [0.65, 0.40], "B": [0.35, 0.60]}, index=idx)

    result = BrinsonAttribution().calculate(
        returns,
        benchmark_returns=benchmark_returns,
        weights=weights,
        method="brinson_hood",
    )
    canonical = BrinsonAttribution().calculate(
        returns,
        benchmark_returns=benchmark_returns,
        weights=weights,
        method="brinson",
    )
    pd.testing.assert_frame_equal(result, canonical)

    benchmark_weights = np.full(2, 0.5)
    for row, portfolio_row, benchmark_row, weight_row in zip(
        result.itertuples(index=False),
        returns.to_numpy(),
        benchmark_returns.to_numpy(),
        weights.to_numpy(),
        strict=True,
    ):
        expected = brinson_bhb_reference(portfolio_row, benchmark_row, weight_row, benchmark_weights)
        assert row.allocation == pytest.approx(expected["allocation"], abs=1e-15)
        assert row.selection == pytest.approx(expected["selection"], abs=1e-15)
        assert row.interaction == pytest.approx(expected["interaction"], abs=1e-15)
        assert row.total == pytest.approx(expected["total"], abs=1e-15)


def test_brinson_attribution_class_sector_mapping_aggregates_columns():
    idx = pd.date_range("2020-01-01", periods=2)
    returns = pd.DataFrame({"A": [0.01, 0.02], "B": [0.03, 0.01], "C": [-0.01, 0.00]}, index=idx)
    weights = pd.DataFrame({"A": [0.5, 0.5], "B": [0.3, 0.2], "C": [0.2, 0.3]}, index=idx)

    ba = BrinsonAttribution(sector_mapping={"S1": ["A", "B"], "S2": ["C"]})
    df = ba.calculate(returns=returns, weights=weights)
    assert df.shape[0] == 2
    assert set(df.columns) == {"period", "allocation", "selection", "interaction", "total"}


def test_brinson_attribution_class_sector_mapping_with_benchmark_returns():
    idx = pd.date_range("2020-01-01", periods=2)
    returns = pd.DataFrame({"A": [0.01, 0.02], "B": [0.03, 0.01], "C": [-0.01, 0.00]}, index=idx)
    bench = pd.DataFrame({"A": [0.00, 0.01], "B": [0.01, 0.00], "C": [0.00, 0.00]}, index=idx)
    weights = pd.DataFrame({"A": [0.5, 0.5], "B": [0.3, 0.2], "C": [0.2, 0.3]}, index=idx)

    ba = BrinsonAttribution(sector_mapping={"S1": ["A", "B"], "S2": ["C"]})
    df = ba.calculate(returns=returns, benchmark_returns=bench, weights=weights)
    assert df.shape[0] == 2


def test_brinson_attribution_class_creates_equal_weights_when_missing():
    idx = pd.date_range("2020-01-01", periods=2)
    returns = pd.DataFrame({"A": [0.01, 0.02], "B": [0.03, 0.01], "C": [-0.01, 0.00]}, index=idx)

    ba = BrinsonAttribution(sector_mapping={"S1": ["A", "B"], "S2": ["C"]})
    df = ba.calculate(returns=returns)
    assert df.shape[0] == 2


def test_brinson_attribution_apply_sector_mapping_rejects_bad_agg():
    idx = pd.date_range("2020-01-01", periods=2)
    returns = pd.DataFrame({"A": [0.01, 0.02]}, index=idx)
    ba = BrinsonAttribution(sector_mapping={"S1": ["A"]})
    with pytest.raises(ValueError, match="agg must be"):
        ba._apply_sector_mapping(returns, agg="nope")


def test_brinson_attribution_repr_includes_sector_count():
    assert "0 sectors" in repr(BrinsonAttribution())
    assert "2 sectors" in repr(BrinsonAttribution(sector_mapping={"S1": ["A"], "S2": ["B"]}))
