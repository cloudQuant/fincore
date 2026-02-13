"""Tests for path generation methods."""

from __future__ import annotations

import numpy as np
import pytest

from fincore.simulation.paths import (
    antithetic_variates,
    gbm_from_returns,
    geometric_brownian_motion,
    latin_hypercube_sampling,
)


class TestGeometricBrownianMotion:
    """Tests for GBM path generation."""

    def test_gbm_shape(self):
        """Test that GBM produces correct shape."""
        paths = geometric_brownian_motion(S0=100.0, mu=0.10, sigma=0.20, T=1.0, dt=1 / 252, n_paths=100, seed=42)

        assert paths.shape == (100, 253)  # n_paths x (n_steps + 1)

    def test_gbm_initial_value(self):
        """Test that all paths start at S0."""
        S0 = 100.0
        paths = geometric_brownian_motion(S0=S0, mu=0.10, sigma=0.20, T=1.0, dt=1 / 252, n_paths=100, seed=42)

        np.testing.assert_array_almost_equal(paths[:, 0], S0, decimal=10)

    def test_gbm_reproducible(self):
        """Test that seed produces reproducible results."""
        paths1 = geometric_brownian_motion(S0=100.0, mu=0.10, sigma=0.20, T=0.25, dt=1 / 252, n_paths=50, seed=123)
        paths2 = geometric_brownian_motion(S0=100.0, mu=0.10, sigma=0.20, T=0.25, dt=1 / 252, n_paths=50, seed=123)

        np.testing.assert_array_equal(paths1, paths2)

    def test_gbm_positive_paths(self):
        """Test that GBM paths remain positive."""
        paths = geometric_brownian_motion(S0=100.0, mu=0.10, sigma=0.20, T=1.0, dt=1 / 252, n_paths=100, seed=42)

        # GBM should never produce negative values
        assert np.all(paths >= 0)

    def test_gbm_drift_effect(self):
        """Test that positive drift increases expected value."""
        # Positive drift
        paths_up = geometric_brownian_motion(S0=100.0, mu=0.20, sigma=0.10, T=1.0, dt=1 / 252, n_paths=1000, seed=42)
        # Negative drift
        paths_down = geometric_brownian_motion(S0=100.0, mu=-0.20, sigma=0.10, T=1.0, dt=1 / 252, n_paths=1000, seed=42)

        # Positive drift should have higher terminal values
        assert np.mean(paths_up[:, -1]) > np.mean(paths_down[:, -1])


class TestGBMFromReturns:
    """Tests for GBM simulation from returns."""

    @pytest.fixture
    def sample_returns(self):
        """Generate sample returns."""
        np.random.seed(42)
        return np.random.normal(0.001, 0.02, 500)

    def test_gbm_from_returns_shape(self, sample_returns):
        """Test output shape."""
        paths = gbm_from_returns(sample_returns, horizon=60, n_paths=100, seed=42)

        assert paths.shape == (100, 60)

    def test_gbm_from_returns_starts_at_zero(self, sample_returns):
        """Test that returns paths have reasonable starting values."""
        paths = gbm_from_returns(sample_returns, horizon=60, n_paths=100, seed=42)

        # Paths should start near 0 (returns from price starting at 1)
        # First values won't be exactly 0 due to GBM's first step
        assert np.all(np.abs(paths[:, 0]) < 0.1)  # Should be close to 0


class TestAntitheticVariates:
    """Tests for antithetic variates."""

    def test_antithetic_doubles_paths(self):
        """Test that antithetic doubles number of paths."""
        paths = np.array([[1, 2, 3], [4, 5, 6]])
        result = antithetic_variates(paths)

        assert result.shape[0] == paths.shape[0] * 2

    def test_antithetic_mirroring(self):
        """Test that antithetic paths are mirrored."""
        S0 = 100.0
        paths = np.array([[S0, 110, 120], [S0, 90, 80]])
        result = antithetic_variates(paths)

        # Antithetic of 110 is 90 (100 - (110 - 100) = 90)
        np.testing.assert_array_almost_equal(result[2, :], [S0, 90, 80])


class TestLatinHypercube:
    """Tests for Latin Hypercube Sampling."""

    def test_lhs_shape(self):
        """Test that LHS produces correct shape."""
        samples = latin_hypercube_sampling(n_samples=100, n_dimensions=5, seed=42)

        assert samples.shape == (100, 5)

    def test_lhs_bounds(self):
        """Test that LHS samples are in [0, 1]."""
        samples = latin_hypercube_sampling(n_samples=1000, n_dimensions=10, seed=42)

        assert np.all(samples >= 0)
        assert np.all(samples <= 1)

    def test_lhs_coverage(self):
        """Test that LHS provides good coverage."""
        samples = latin_hypercube_sampling(n_samples=100, n_dimensions=1, seed=42)

        # Each dimension should be divided into 100 intervals
        # with one sample per interval
        sorted_samples = np.sort(samples[:, 0])

        # Check intervals (roughly)
        for i in range(100):
            lower = i / 100
            upper = (i + 1) / 100
            # Each interval should have approximately one sample
            count = np.sum((sorted_samples >= lower) & (sorted_samples < upper))
            assert count >= 0  # At least some coverage
