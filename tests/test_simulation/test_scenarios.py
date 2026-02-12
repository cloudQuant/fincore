"""Tests for stress testing scenarios."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from fincore.simulation.scenarios import (
    stress_test,
    _apply_crash_scenario,
    _apply_spike_scenario,
    generate_correlation_breakdown,
    scenario_table,
)


@pytest.fixture
def sample_returns():
    """Create sample returns for testing."""
    np.random.seed(42)
    return np.random.normal(0.001, 0.02, 100)


class TestStressTest:
    """Tests for stress testing functions."""

    def test_stress_test_returns_dict(self, sample_returns):
        """Test that stress test returns a dictionary."""
        result = stress_test(sample_returns, scenarios=["crash", "spike"])

        assert isinstance(result, dict)
        assert "crash" in result
        assert "spike" in result

    def test_stress_test_all_scenarios(self, sample_returns):
        """Test stress test with all predefined scenarios."""
        result = stress_test(sample_returns)

        # Should apply all 4 default scenarios
        assert len(result) == 4
        assert "crash" in result
        assert "spike" in result
        assert "vol_crush" in result
        assert "vol_spike" in result

    def test_stress_test_custom_scenario(self, sample_returns):
        """Test custom scenario application."""
        custom = {
            "double": {"return_mult": 2.0},
            "shift": {"return_shift": 0.01},
        }
        result = stress_test(sample_returns, custom_scenarios=custom)

        assert "double" in result
        assert "shift" in result

    def test_stress_test_empty_returns(self):
        """Test that empty returns raise error."""
        with pytest.raises(ValueError):
            stress_test(np.array([]))

    def test_crash_scenario(self, sample_returns):
        """Test crash scenario."""
        result = _apply_crash_scenario(sample_returns)

        assert "max_drawdown" in result
        assert result["max_drawdown"] < 0  # Should have negative drawdown

    def test_spike_scenario(self, sample_returns):
        """Test spike scenario."""
        result = _apply_spike_scenario(sample_returns)

        assert "cumulative_return" in result
        # Spike should improve returns
        assert result["cumulative_return"] > np.sum(sample_returns)

    def test_scenario_table_format(self, sample_returns):
        """Test scenario table generation."""
        results = stress_test(sample_returns, scenarios=["crash", "spike"])
        table = scenario_table(results)

        assert isinstance(table, pd.DataFrame)
        assert len(table) == 2
        assert "Scenario" in table.columns
        assert "Cumulative Return" in table.columns
        assert "Max Drawdown" in table.columns


class TestCorrelationBreakdown:
    """Tests for correlation breakdown matrix."""

    def test_correlation_breakdown_shape(self):
        """Test matrix shape."""
        n = 5
        matrix = generate_correlation_breakdown(n_assets=n, target_correlation=0.8)

        assert matrix.shape == (n, n)

    def test_correlation_breakdown_diagonal(self):
        """Test that diagonal elements are 1."""
        matrix = generate_correlation_breakdown(n_assets=10, target_correlation=0.5)

        np.testing.assert_array_equal(np.diag(matrix), np.ones(10))

    def test_correlation_breakdown_off_diagonal(self):
        """Test that off-diagonal elements match target."""
        target = 0.7
        matrix = generate_correlation_breakdown(n_assets=10, target_correlation=target)

        # Extract off-diagonal elements
        off_diag = matrix[~np.eye(10, dtype=bool)]
        np.testing.assert_array_almost_equal(off_diag, np.full(90, target))

    def test_correlation_breakdown_perfect(self):
        """Test perfect correlation (target=1)."""
        matrix = generate_correlation_breakdown(n_assets=5, target_correlation=1.0)

        # All elements should be 1
        np.testing.assert_array_almost_equal(matrix, np.ones((5, 5)))
