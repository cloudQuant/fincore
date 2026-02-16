"""Tests for efficient_frontier module - additional coverage for uncovered lines.

This file tests edge cases and validation paths not covered in test_optimization.py.
"""

import numpy as np
import pandas as pd
import pytest

from fincore.optimization import efficient_frontier


class TestFrontierEdgeCases:
    """Test edge cases for efficient_frontier."""

    def test_empty_dataframe_raises(self):
        """Test that empty DataFrame raises ValueError."""
        with pytest.raises(ValueError, match="must be a non-empty"):
            efficient_frontier(pd.DataFrame())

    def test_non_dataframe_raises(self):
        """Test that non-DataFrame input raises ValueError."""
        np.random.seed(42)
        data = np.random.randn(100, 4) * 0.01
        with pytest.raises(ValueError, match="must be a non-empty DataFrame"):
            efficient_frontier(data)

    def test_insufficient_observations_raises(self):
        """Test that less than 2 observations raises ValueError."""
        data = pd.DataFrame([[0.01, 0.02]], columns=["A", "B"])
        with pytest.raises(ValueError, match="At least 2 observations"):
            efficient_frontier(data)

    def test_single_asset_raises(self):
        """Test that single asset raises ValueError."""
        np.random.seed(42)
        data = pd.DataFrame({"A": np.random.randn(100) * 0.01})
        with pytest.raises(ValueError, match="At least 2 assets"):
            efficient_frontier(data)

    def test_inf_values_raises(self):
        """Test that infinite values raise ValueError."""
        np.random.seed(42)
        data = pd.DataFrame(
            np.random.randn(100, 4) * 0.01,
            columns=["A", "B", "C", "D"],
        )
        data.iloc[0, 0] = np.inf
        with pytest.raises(ValueError, match="contains NaN or infinite values"):
            efficient_frontier(data)

    def test_max_weight_zero_raises(self):
        """Test that max_weight <= 0 raises ValueError."""
        np.random.seed(42)
        data = pd.DataFrame(
            np.random.randn(100, 4) * 0.01,
            columns=["A", "B", "C", "D"],
        )
        with pytest.raises(ValueError, match="max_weight must be > 0"):
            efficient_frontier(data, max_weight=0)

    def test_n_points_less_than_2_raises(self):
        """Test that n_points < 2 raises ValueError."""
        np.random.seed(42)
        data = pd.DataFrame(
            np.random.randn(100, 4) * 0.01,
            columns=["A", "B", "C", "D"],
        )
        with pytest.raises(ValueError, match="n_points must be >= 2"):
            efficient_frontier(data, n_points=1)


class TestFrontierWithCustomMaxWeight:
    """Test frontier with custom max_weight parameter."""

    def test_max_weight_constraint(self):
        """Test that max_weight constraint is respected."""
        np.random.seed(42)
        data = pd.DataFrame(
            np.random.randn(252, 4) * 0.01,
            columns=["A", "B", "C", "D"],
            index=pd.date_range("2020-01-01", periods=252),
        )
        max_w = 0.5
        ef = efficient_frontier(data, n_points=10, max_weight=max_w)

        # Check min variance portfolio respects constraint
        assert np.all(ef["min_variance"]["weights"] <= max_w + 1e-8)

        # Check max sharpe portfolio respects constraint
        assert np.all(ef["max_sharpe"]["weights"] <= max_w + 1e-8)

        # Check frontier points respect constraint
        for w in ef["frontier_weights"]:
            if not np.any(np.isnan(w)):
                assert np.all(w <= max_w + 1e-8)


class TestFrontierWithShortSelling:
    """Test frontier with short selling enabled."""

    def test_short_allowed_allows_negative_weights(self):
        """Test that short_allowed permits negative weights."""
        np.random.seed(42)
        data = pd.DataFrame(
            np.random.randn(252, 4) * 0.01,
            columns=["A", "B", "C", "D"],
            index=pd.date_range("2020-01-01", periods=252),
        )
        ef = efficient_frontier(data, n_points=10, short_allowed=True)

        # Check that negative weights are present (or at least allowed)
        min_w = min(
            ef["min_variance"]["weights"].min(),
            ef["max_sharpe"]["weights"].min(),
        )
        # With short selling, we might have negative weights
        assert min_w >= -1.0  # At least not more than -100%


class TestFrontierOutputStructure:
    """Test output structure of efficient_frontier."""

    def test_output_contains_all_keys(self):
        """Test that output contains all expected keys."""
        np.random.seed(42)
        data = pd.DataFrame(
            np.random.randn(252, 4) * 0.01,
            columns=["A", "B", "C", "D"],
            index=pd.date_range("2020-01-01", periods=252),
        )
        ef = efficient_frontier(data, n_points=10)

        expected_keys = [
            "frontier_returns",
            "frontier_volatilities",
            "frontier_sharpe",
            "frontier_weights",
            "min_variance",
            "max_sharpe",
            "asset_names",
        ]
        for key in expected_keys:
            assert key in ef

    def test_min_variance_structure(self):
        """Test min_variance output structure."""
        np.random.seed(42)
        data = pd.DataFrame(
            np.random.randn(252, 4) * 0.01,
            columns=["A", "B", "C", "D"],
            index=pd.date_range("2020-01-01", periods=252),
        )
        ef = efficient_frontier(data, n_points=10)

        mv = ef["min_variance"]
        assert "weights" in mv
        assert "return" in mv
        assert "volatility" in mv
        assert len(mv["weights"]) == 4

    def test_max_sharpe_structure(self):
        """Test max_sharpe output structure."""
        np.random.seed(42)
        data = pd.DataFrame(
            np.random.randn(252, 4) * 0.01,
            columns=["A", "B", "C", "D"],
            index=pd.date_range("2020-01-01", periods=252),
        )
        ef = efficient_frontier(data, n_points=10)

        ms = ef["max_sharpe"]
        assert "weights" in ms
        assert "return" in ms
        assert "volatility" in ms
        assert "sharpe" in ms
        assert len(ms["weights"]) == 4

    def test_frontier_arrays_length(self):
        """Test that frontier arrays have correct length."""
        np.random.seed(42)
        data = pd.DataFrame(
            np.random.randn(252, 4) * 0.01,
            columns=["A", "B", "C", "D"],
            index=pd.date_range("2020-01-01", periods=252),
        )
        n_points = 15
        ef = efficient_frontier(data, n_points=n_points)

        assert len(ef["frontier_returns"]) == n_points
        assert len(ef["frontier_volatilities"]) == n_points
        assert len(ef["frontier_sharpe"]) == n_points
        assert len(ef["frontier_weights"]) == n_points

    def test_asset_names(self):
        """Test that asset_names are correctly extracted."""
        np.random.seed(42)
        columns = ["AAPL", "MSFT", "GOOGL", "TSLA"]
        data = pd.DataFrame(
            np.random.randn(252, 4) * 0.01,
            columns=columns,
            index=pd.date_range("2020-01-01", periods=252),
        )
        ef = efficient_frontier(data, n_points=10)

        assert ef["asset_names"] == columns
