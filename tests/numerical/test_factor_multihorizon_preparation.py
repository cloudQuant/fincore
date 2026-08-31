"""Enhanced per-horizon factor-preparation regression tests."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from fincore.factor_analysis.analysis import analyze_factor
from fincore.factor_analysis.data import prepare_factor_data, prepare_factor_data_by_horizon
from fincore.factor_analysis.exceptions import FactorLossExceededError


def _inputs() -> tuple[pd.Series, pd.DataFrame]:
    dates = pd.date_range("2024-01-01", periods=3, tz="UTC")
    assets = ("A", "B", "C", "D")
    factor = pd.Series(
        np.tile([-1.0, -0.25, 0.25, 1.0], len(dates)),
        index=pd.MultiIndex.from_product((dates, assets), names=("date", "asset")),
        name="factor",
    )
    prices = pd.DataFrame(
        {asset: 100.0 + (np.arange(5, dtype=float) * (position + 1.0)) for position, asset in enumerate(assets)},
        index=pd.date_range("2024-01-01", periods=5, tz="UTC"),
    )
    return factor, prices


def test_prepares_each_horizon_without_discarding_short_horizon_rows() -> None:
    factor, prices = _inputs()

    result = prepare_factor_data_by_horizon(factor, prices, periods=(1, 3), quantiles=2, max_loss=1.0)

    assert tuple(result.by_horizon) == ("1D", "3D")
    assert len(result.by_horizon["1D"].data) == 12
    assert len(result.by_horizon["3D"].data) == 8
    assert set(result.by_horizon["1D"].data.index.get_level_values("date")) == set(
        factor.index.get_level_values("date")
    )
    assert set(result.by_horizon["3D"].data.index.get_level_values("date")) == set(
        factor.index.get_level_values("date")[:8]
    )

    short_report = result.by_horizon["1D"].loss_report
    long_report = result.by_horizon["3D"].loss_report
    assert (short_report.input_count, short_report.forward_returns_count, short_report.binning_count) == (12, 12, 12)
    assert (long_report.input_count, long_report.forward_returns_count, long_report.binning_count) == (12, 8, 8)
    assert long_report.forward_returns_loss == pytest.approx(1.0 / 3.0)

    model = analyze_factor(result.by_horizon["1D"].data, periods=("1D",), include_portfolio_inputs=False)
    assert model.forward_periods == ("1D",)

    legacy_shape = prepare_factor_data(factor, prices, periods=(1, 3), quantiles=2, max_loss=1.0)
    assert len(legacy_shape.data) == len(result.by_horizon["3D"].data)


def test_late_long_horizon_price_perturbation_cannot_remove_short_horizon_observations() -> None:
    factor, prices = _inputs()
    baseline = prepare_factor_data_by_horizon(factor, prices, periods=(1, 3), quantiles=2, max_loss=1.0)
    perturbed_prices = prices.copy(deep=True)
    perturbed_prices.loc[perturbed_prices.index[-1], "A"] *= 1.5

    perturbed = prepare_factor_data_by_horizon(factor, perturbed_prices, periods=(1, 3), quantiles=2, max_loss=1.0)

    pd.testing.assert_frame_equal(baseline.by_horizon["1D"].data, perturbed.by_horizon["1D"].data)
    assert not baseline.by_horizon["3D"].data.equals(perturbed.by_horizon["3D"].data)


def test_rejects_ambiguous_duplicate_horizons_and_enforces_each_loss_budget() -> None:
    factor, prices = _inputs()

    with pytest.raises(ValueError, match="unique"):
        prepare_factor_data_by_horizon(factor, prices, periods=(1, 1), quantiles=2, max_loss=1.0)
    with pytest.raises(FactorLossExceededError, match="3D") as error:
        prepare_factor_data_by_horizon(factor, prices, periods=(1, 3), quantiles=2, max_loss=0.3)

    assert error.value.report is not None
    assert error.value.report.total_loss == pytest.approx(1.0 / 3.0)


def test_rejects_full_sample_filtering_on_the_causal_enhanced_route() -> None:
    factor, prices = _inputs()

    with pytest.raises(ValueError, match="filter_zscore"):
        prepare_factor_data_by_horizon(factor, prices, periods=(1, 3), filter_zscore=3.0)


def test_result_horizon_mapping_is_read_only() -> None:
    factor, prices = _inputs()
    result = prepare_factor_data_by_horizon(factor, prices, periods=(1, 3), quantiles=2, max_loss=1.0)

    with pytest.raises(TypeError):
        result.by_horizon["late"] = result.by_horizon["1D"]  # type: ignore[index]
