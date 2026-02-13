"""Tests for Monte Carlo simulation."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from fincore.simulation import MonteCarlo


@pytest.fixture
def sample_returns():
    """Create sample returns for testing."""
    np.random.seed(42)
    return pd.Series(np.random.normal(0.001, 0.02, 252))


class TestMonteCarlo:
    """Tests for MonteCarlo class."""

    def test_init(self, sample_returns):
        """Test initialization with valid returns."""
        mc = MonteCarlo(sample_returns)
        assert len(mc.returns) == 252
        assert mc.returns is not sample_returns  # Should be a copy

    def test_init_with_nan(self):
        """Test that NaN values are filtered out."""
        returns = pd.Series([0.01, np.nan, 0.02, np.nan, 0.03])
        mc = MonteCarlo(returns)
        assert len(mc.returns) == 3  # Only non-NaN values

    def test_init_empty(self):
        """Test that empty returns raise error."""
        with pytest.raises(ValueError, match="empty"):
            MonteCarlo(np.array([]))

    def test_simulate_shape(self, sample_returns):
        """Test that simulation returns correct shape."""
        mc = MonteCarlo(sample_returns)
        result = mc.simulate(n_paths=100, horizon=60, seed=42)
        assert result.paths.shape == (100, 60)

    def test_simulate_reproducible(self, sample_returns):
        """Test that seed produces reproducible results."""
        mc = MonteCarlo(sample_returns)
        result1 = mc.simulate(n_paths=100, horizon=60, seed=42)
        result2 = mc.simulate(n_paths=100, horizon=60, seed=42)
        np.testing.assert_array_equal(result1.paths, result2.paths)

    def test_var_calculation(self, sample_returns):
        """Test VaR calculation."""
        mc = MonteCarlo(sample_returns)
        var_95 = mc.var(alpha=0.05, n_paths=10000, seed=42)
        var_99 = mc.var(alpha=0.01, n_paths=10000, seed=42)

        # 99% VaR should be more negative than 95% VaR
        assert var_99 < var_95

    def test_cvar_calculation(self, sample_returns):
        """Test CVaR calculation."""
        mc = MonteCarlo(sample_returns)
        cvar = mc.cvar(alpha=0.05, n_paths=10000, seed=42)

        # CVaR should be more negative than VaR
        var = mc.var(alpha=0.05, n_paths=10000, seed=42)
        assert cvar <= var

    def test_price_paths(self, sample_returns):
        """Test price path simulation."""
        mc = MonteCarlo(sample_returns)
        S0 = 100.0
        result = mc.price_paths(S0=S0, n_paths=100, horizon=60, seed=42)

        # All paths should start at S0
        np.testing.assert_array_almost_equal(result.paths[:, 0], S0, decimal=10)

        # Paths should be positive
        assert np.all(result.paths > 0)

    def test_antithetic_variates(self, sample_returns):
        """Test antithetic variates doubles paths."""
        mc = MonteCarlo(sample_returns)
        result_normal = mc.simulate(n_paths=100, horizon=60, seed=42)
        result_antithetic = mc.simulate(n_paths=100, horizon=60, seed=42, antithetic=True)

        # Antithetic should have double the paths
        assert result_antithetic.n_paths == result_normal.n_paths * 2

    def test_stress_test(self, sample_returns):
        """Test stress testing."""
        mc = MonteCarlo(sample_returns)
        results = mc.stress(scenarios=["crash", "spike"])

        assert "crash" in results
        assert "spike" in results
        assert "max_drawdown" in results["crash"]
        assert "volatility" in results["crash"]

    def test_stress_table(self, sample_returns):
        """Test stress table generation."""
        mc = MonteCarlo(sample_returns)
        table = mc.stress_table()

        assert isinstance(table, pd.DataFrame)
        assert len(table) > 0
        assert "Scenario" in table.columns
        assert "Max Drawdown" in table.columns

    @staticmethod
    def test_from_parameters():
        """Test simulation from known parameters."""
        result = MonteCarlo.from_parameters(mu=0.10, sigma=0.20, S0=100.0, n_paths=100, horizon=60, seed=42)

        assert result.n_paths == 100
        np.testing.assert_array_almost_equal(result.paths[:, 0], 100.0)

    def test_sim_result_repr(self, sample_returns):
        """Test SimResult string representation."""
        mc = MonteCarlo(sample_returns)
        result = mc.simulate(n_paths=100, horizon=60)
        repr_str = repr(result)

        assert "n_paths=100" in repr_str
        assert "horizon=60" in repr_str
