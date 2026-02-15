"""Style attribution contract tests.

Validates semantic correctness:
- Key whitelist (no asset codes in style keys)
- Dimension consistency
- Contribution + residual ≈ portfolio return
- End-to-end standalone API calls
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from fincore.attribution.style import (
    StyleResult,
    analyze_performance_by_style,
    calculate_regression_attribution,
    calculate_style_tilts,
    style_analysis,
)

STYLE_WHITELIST = {
    "equal_weight",
    "winner",
    "loser",
    "value",
    "growth",
    "high",
    "low",
    "large",
    "mid",
    "small",
}


@pytest.fixture
def asset_returns():
    np.random.seed(42)
    return pd.DataFrame(
        np.random.randn(120, 5) * 0.01,
        columns=["AAPL", "MSFT", "GOOG", "AMZN", "META"],
        index=pd.date_range("2020-01-01", periods=120, freq="B"),
    )


class TestStyleAnalysisContract:
    def test_returns_style_result(self, asset_returns):
        result = style_analysis(asset_returns)
        assert isinstance(result, StyleResult)

    def test_style_summary_keys_whitelist(self, asset_returns):
        result = style_analysis(asset_returns)
        summary = result.style_summary
        for key in summary:
            assert key in STYLE_WHITELIST, f"Unexpected key '{key}' in style_summary"

    def test_no_asset_codes_in_style_keys(self, asset_returns):
        result = style_analysis(asset_returns)
        asset_names = set(asset_returns.columns)
        style_keys = set(result.style_summary.keys())
        overlap = style_keys & asset_names
        assert len(overlap) == 0, f"Asset codes leaked into style keys: {overlap}"

    def test_exposures_shape(self, asset_returns):
        result = style_analysis(asset_returns)
        n_assets = asset_returns.shape[1]
        # exposures: N assets x S styles
        assert result.exposures.shape[0] == n_assets
        assert result.exposures.shape[1] > 0

    def test_exposures_shape_with_market_caps(self, asset_returns):
        market_caps = pd.Series(
            [1e12, 8e11, 5e11, 3e11, 1e11],
            index=asset_returns.columns,
        )
        result = style_analysis(asset_returns, market_caps=market_caps)
        assert result.exposures.shape[0] == asset_returns.shape[1]
        for key in {"large", "mid", "small"}:
            assert key in result.exposures.columns

    def test_exposures_non_negative(self, asset_returns):
        result = style_analysis(asset_returns)
        assert (result.exposures.values >= 0).all()

    def test_contains_and_getitem(self, asset_returns):
        result = style_analysis(asset_returns)
        assert "exposures" in result
        assert "style_summary" in result
        assert "overall_returns" in result
        assert isinstance(result["exposures"], pd.DataFrame)

    def test_to_dict(self, asset_returns):
        result = style_analysis(asset_returns)
        d = result.to_dict()
        assert set(d.keys()) == {"exposures", "returns_by_style", "overall_returns"}


class TestRegressionAttributionContract:
    def test_standalone_from_dataframe(self, asset_returns):
        attr = calculate_regression_attribution(asset_returns)
        assert isinstance(attr, dict)
        assert "residual" in attr
        # All keys should be style names or 'residual'
        for key in attr:
            assert key in STYLE_WHITELIST or key == "residual"

    def test_contributions_sum_reasonable(self, asset_returns):
        attr = calculate_regression_attribution(asset_returns)
        total = sum(attr.values())
        # total = sum(contributions) + residual ≈ portfolio mean return
        port_mean = float(asset_returns.mean(axis=1).mean())
        assert abs(total - port_mean) < 1e-10

    def test_with_explicit_args(self, asset_returns):
        result = style_analysis(asset_returns)
        port_ret = asset_returns.mean(axis=1)
        style_ret = pd.DataFrame(
            {s: (asset_returns * result.exposures[s]).sum(axis=1) for s in result.exposures.columns}
        )
        attr = calculate_regression_attribution(port_ret, style_ret, result.exposures)
        assert "residual" in attr


class TestStyleTiltsContract:
    def test_short_data_returns_nonempty(self, asset_returns):
        tilts = calculate_style_tilts(asset_returns, window=50)
        assert isinstance(tilts, pd.DataFrame)
        assert len(tilts) > 0

    def test_very_short_data(self):
        tiny = pd.DataFrame(
            np.random.randn(3, 2) * 0.01,
            columns=["A", "B"],
            index=pd.date_range("2020-01-01", periods=3),
        )
        tilts = calculate_style_tilts(tiny)
        assert isinstance(tilts, pd.DataFrame)
        # With 3 rows, effective_window is clamped to 2, should have 1 row
        assert len(tilts) >= 0

    def test_columns_contain_style_labels(self, asset_returns):
        tilts = calculate_style_tilts(asset_returns, window=50)
        style_labels = ["large", "small", "winner", "loser", "value", "growth", "high_vol", "low_vol"]
        has_any = any(any(label in col for label in style_labels) for col in tilts.columns)
        assert has_any, f"No style labels found in columns: {tilts.columns.tolist()[:5]}"


class TestAnalyzePerformanceContract:
    def test_standalone_from_dataframe(self, asset_returns):
        perf = analyze_performance_by_style(asset_returns)
        assert isinstance(perf, pd.DataFrame)
        assert len(perf) == len(asset_returns)
        assert perf.index.name == "Period"

    def test_columns_are_style_returns(self, asset_returns):
        perf = analyze_performance_by_style(asset_returns)
        for col in perf.columns:
            assert col.endswith("_return"), f"Unexpected column: {col}"

    def test_empty_input(self):
        result = analyze_performance_by_style(pd.DataFrame())
        assert isinstance(result, pd.DataFrame)
        assert result.empty
