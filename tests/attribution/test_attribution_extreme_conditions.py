"""Extreme-condition contracts for the direct attribution kernel."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from fincore.attribution.performance import perf_attrib


def _attribution_inputs(
    *,
    periods: int = 24,
    factor_names: tuple[str, ...] = ("value", "momentum"),
) -> tuple[pd.Series, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Build a deterministic, fully-covered attribution input set."""
    dates = pd.bdate_range("2024-01-02", periods=periods)
    assets = ("AAA", "BBB", "CCC")
    steps = np.arange(periods, dtype=float)

    returns = pd.Series(0.003 * np.sin(steps / 2.0), index=dates, name="returns")
    positions = pd.DataFrame(
        {
            "AAA": 110.0 + steps,
            "BBB": 75.0 + 0.5 * steps,
            "CCC": 45.0 - 0.25 * steps,
        },
        index=dates,
    )
    factor_returns = pd.DataFrame(
        {
            factor: (factor_number + 1) * 0.001 + 0.0002 * np.cos(steps / 3.0)
            for factor_number, factor in enumerate(factor_names)
        },
        index=dates,
    )
    loadings_index = pd.MultiIndex.from_product([dates, assets], names=["dt", "ticker"])
    factor_loadings = pd.DataFrame(
        {
            factor: [
                0.05 * (asset_number + 1) * (factor_number + 1) + 0.001 * date_number
                for date_number in range(periods)
                for asset_number in range(len(assets))
            ]
            for factor_number, factor in enumerate(factor_names)
        },
        index=loadings_index,
    )
    return returns, positions, factor_returns, factor_loadings


def _assert_attribution_is_complete(
    returns: pd.Series,
    risk_exposures: pd.DataFrame,
    attribution: pd.DataFrame,
) -> None:
    assert risk_exposures.index.equals(returns.index)
    assert attribution.index.equals(returns.index)
    assert {"total_returns", "common_returns", "specific_returns"} <= set(attribution.columns)
    np.testing.assert_allclose(attribution["total_returns"], returns)
    assert attribution.notna().all().all()


@pytest.mark.parametrize(
    "scenario",
    ("extreme_bull", "extreme_bear", "high_volatility"),
)
def test_direct_attribution_handles_extreme_return_regimes(scenario: str) -> None:
    returns, positions, factor_returns, factor_loadings = _attribution_inputs(periods=100)
    trend = 1.0 + np.arange(len(returns), dtype=float) / 100.0

    if scenario == "extreme_bull":
        returns = returns.abs() * 0.5 * trend
    elif scenario == "extreme_bear":
        returns = -returns.abs() * 0.5 * trend
    else:
        returns = returns * np.where(np.arange(len(returns)) % 2 == 0, 8.0, -8.0)

    risk_exposures, attribution = perf_attrib(returns, positions, factor_returns, factor_loadings)

    _assert_attribution_is_complete(returns, risk_exposures, attribution)


def test_direct_attribution_handles_multiple_assets_and_short_series() -> None:
    returns, positions, factor_returns, factor_loadings = _attribution_inputs(
        periods=5,
        factor_names=("market",),
    )

    risk_exposures, attribution = perf_attrib(returns, positions, factor_returns, factor_loadings)

    _assert_attribution_is_complete(returns, risk_exposures, attribution)
    assert list(risk_exposures.columns) == ["market"]


def test_direct_attribution_preserves_zero_return_observations() -> None:
    returns, positions, factor_returns, factor_loadings = _attribution_inputs(periods=24)
    returns.iloc[8:13] = 0.0

    risk_exposures, attribution = perf_attrib(returns, positions, factor_returns, factor_loadings)

    _assert_attribution_is_complete(returns, risk_exposures, attribution)
    assert (attribution.loc[returns.iloc[8:13].index, "total_returns"] == 0.0).all()
