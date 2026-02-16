"""Tests for optimization objectives module - 100% coverage."""

import numpy as np
import pandas as pd
import pytest

from fincore.optimization.objectives import optimize


@pytest.fixture
def sample_returns():
    """Create sample returns data for testing."""
    np.random.seed(42)
    return pd.DataFrame(
        np.random.randn(252, 4) * 0.01,
        columns=["AAPL", "MSFT", "GOOGL", "TSLA"],
        index=pd.date_range("2020-01-01", periods=252),
    )


class TestOptimizeEdgeCases:
    """Test edge cases for optimize function."""

    def test_empty_dataframe(self):
        """Test that empty DataFrame raises ValueError."""
        with pytest.raises(ValueError, match="must be a non-empty"):
            optimize(pd.DataFrame())

    def test_not_dataframe(self):
        """Test that non-DataFrame input raises ValueError."""
        with pytest.raises(ValueError, match="must be a non-empty DataFrame"):
            optimize(np.random.randn(100, 4))

    def test_insufficient_observations(self):
        """Test that less than 2 observations raises ValueError."""
        data = pd.DataFrame(
            [[0.01, 0.02]],
            columns=["A", "B"],
        )
        with pytest.raises(ValueError, match="At least 2 observations"):
            optimize(data)

    def test_nan_values(self):
        """Test that NaN values raise ValueError."""
        data = pd.DataFrame(
            [[0.01, np.nan], [0.03, 0.04], [0.02, 0.01]],
            columns=["A", "B"],
        )
        with pytest.raises(ValueError, match="NaN or infinite"):
            optimize(data)

    def test_inf_values(self):
        """Test that infinite values raise ValueError."""
        data = pd.DataFrame(
            [[0.01, np.inf], [0.03, 0.04], [0.02, 0.01]],
            columns=["A", "B"],
        )
        with pytest.raises(ValueError, match="NaN or infinite"):
            optimize(data)

    def test_max_weight_non_positive(self):
        """Test that non-positive max_weight raises ValueError."""
        data = pd.DataFrame(
            np.random.randn(10, 2) * 0.01,
            columns=["A", "B"],
        )
        with pytest.raises(ValueError, match="max_weight must be > 0"):
            optimize(data, max_weight=0)

    def test_min_weight_greater_than_max(self):
        """Test that min_weight > max_weight raises ValueError."""
        data = pd.DataFrame(
            np.random.randn(10, 2) * 0.01,
            columns=["A", "B"],
        )
        with pytest.raises(ValueError, match="min_weight must be <= max_weight"):
            optimize(data, max_weight=0.5, min_weight=0.6)

    def test_sector_constraints_without_map(self):
        """Test that sector_constraints without sector_map raises ValueError."""
        data = pd.DataFrame(
            np.random.randn(10, 2) * 0.01,
            columns=["A", "B"],
        )
        with pytest.raises(ValueError, match="sector_map is required"):
            optimize(data, sector_constraints={"Tech": (0.1, 0.5)})

    def test_unknown_objective(self, sample_returns):
        """Test that unknown objective raises ValueError."""
        with pytest.raises(ValueError, match="Unknown objective"):
            optimize(sample_returns, objective="unknown_objective")

    def test_target_return_without_value(self, sample_returns):
        """Test that target_return objective without target_return parameter raises ValueError."""
        with pytest.raises(ValueError, match="target_return must be specified"):
            optimize(sample_returns, objective="target_return")

    def test_target_risk_without_value(self, sample_returns):
        """Test that target_risk objective without target_volatility parameter raises ValueError."""
        with pytest.raises(ValueError, match="target_volatility must be specified"):
            optimize(sample_returns, objective="target_risk")


class TestOptimizeWithSectorConstraints:
    """Test optimize with sector constraints."""

    def test_sector_constraints_valid(self, sample_returns):
        """Test optimization with valid sector constraints."""
        sector_map = {"AAPL": "Tech", "MSFT": "Tech", "GOOGL": "Tech", "TSLA": "Auto"}
        sector_constraints = {"Tech": (0.3, 0.7), "Auto": (0.0, 0.3)}

        result = optimize(
            sample_returns,
            objective="max_sharpe",
            sector_constraints=sector_constraints,
            sector_map=sector_map,
        )

        assert "weights" in result
        assert len(result["weights"]) == 4
        assert result["objective"] == "max_sharpe"

    def test_sector_constraints_min_variance(self, sample_returns):
        """Test min_variance with sector constraints."""
        sector_map = {"AAPL": "Tech", "MSFT": "Tech", "GOOGL": "Tech", "TSLA": "Auto"}
        sector_constraints = {"Tech": (0.5, 1.0)}

        result = optimize(
            sample_returns,
            objective="min_variance",
            sector_constraints=sector_constraints,
            sector_map=sector_map,
        )

        assert result["objective"] == "min_variance"
        assert np.allclose(result["weights"].sum(), 1.0, atol=1e-6)


class TestOptimizeAllObjectives:
    """Test all optimization objectives."""

    def test_max_sharpe(self, sample_returns):
        """Test max_sharpe objective."""
        result = optimize(sample_returns, objective="max_sharpe")

        assert "weights" in result
        assert "return" in result
        assert "volatility" in result
        assert "sharpe" in result
        assert len(result["weights"]) == 4

    def test_min_variance(self, sample_returns):
        """Test min_variance objective."""
        result = optimize(sample_returns, objective="min_variance")

        assert result["objective"] == "min_variance"
        assert result["volatility"] >= 0

    def test_target_return(self, sample_returns):
        """Test target_return objective."""
        target = 0.10
        result = optimize(sample_returns, objective="target_return", target_return=target)

        assert result["objective"] == "target_return"
        assert abs(result["return"] - target) < 0.01  # Allow small numerical error

    def test_target_risk(self, sample_returns):
        """Test target_risk objective."""
        target = 0.15
        result = optimize(sample_returns, objective="target_risk", target_volatility=target)

        assert result["objective"] == "target_risk"
        assert abs(result["volatility"] - target) < 0.01  # Allow small numerical error


class TestOptimizeWithBounds:
    """Test optimization with custom bounds."""

    def test_short_allowed(self, sample_returns):
        """Test optimization with short selling allowed."""
        result = optimize(sample_returns, objective="max_sharpe", short_allowed=True)

        # Check that short positions are allowed
        assert result["weights"].min() >= -1.0

    def test_custom_min_weight(self, sample_returns):
        """Test optimization with custom min_weight."""
        result = optimize(sample_returns, objective="max_sharpe", min_weight=0.1)

        assert result["weights"].min() >= 0.1

    def test_custom_max_weight(self, sample_returns):
        """Test optimization with custom max_weight."""
        result = optimize(sample_returns, objective="max_sharpe", max_weight=0.5)

        assert result["weights"].max() <= 0.5
