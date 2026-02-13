"""Tests for bootstrap methods."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from fincore.simulation.bootstrap import (
    _get_statistic_fn,
    bootstrap,
    bootstrap_ci,
    bootstrap_summary,
)


@pytest.fixture
def sample_returns():
    """Create sample returns for testing."""
    np.random.seed(42)
    return np.random.normal(0.001, 0.02, 252)


class TestBootstrap:
    """Tests for bootstrap functions."""

    def test_bootstrap_returns_distribution(self, sample_returns):
        """Test that bootstrap returns distribution."""
        boot_dist = bootstrap(sample_returns, n_samples=1000, statistic="mean", seed=42)

        assert len(boot_dist) == 1000
        assert boot_dist is not sample_returns

    def test_bootstrap_mean_convergence(self, sample_returns):
        """Test that bootstrap mean converges to sample mean."""
        boot_dist = bootstrap(sample_returns, n_samples=10000, statistic="mean", seed=42)
        sample_mean = np.mean(sample_returns)
        boot_mean = np.mean(boot_dist)

        # Should be close
        assert abs(boot_mean - sample_mean) < 0.001

    def test_bootstrap_sharpe(self, sample_returns):
        """Test bootstrap with Sharpe ratio."""
        boot_dist = bootstrap(sample_returns, n_samples=1000, statistic="sharpe", seed=42)

        assert len(boot_dist) == 1000
        # Sharpe should be positive for these returns (mean > 0)
        assert np.mean(boot_dist) > 0

    def test_bootstrap_custom_statistic(self, sample_returns):
        """Test bootstrap with custom statistic function."""
        custom_stat = lambda x: np.percentile(x, 75)
        boot_dist = bootstrap(sample_returns, n_samples=1000, statistic=custom_stat, seed=42)

        assert len(boot_dist) == 1000

    def test_bootstrap_unknown_statistic(self, sample_returns):
        """Test that unknown statistic raises error."""
        with pytest.raises(ValueError, match="Unknown statistic"):
            bootstrap(sample_returns, statistic="unknown")

    def test_bootstrap_empty(self):
        """Test that empty array raises error."""
        with pytest.raises(ValueError, match="empty"):
            bootstrap(np.array([]))

    def test_bootstrap_ci(self, sample_returns):
        """Test confidence interval calculation."""
        ci_lower, ci_upper = bootstrap_ci(sample_returns, n_samples=10000, alpha=0.05, seed=42)

        # Lower should be less than upper
        assert ci_lower < ci_upper

        # Sample mean should be within CI
        sample_mean = np.mean(sample_returns)
        assert ci_lower <= sample_mean <= ci_upper

    def test_bootstrap_ci_different_alpha(self, sample_returns):
        """Test CI with different alpha levels."""
        ci_90 = bootstrap_ci(sample_returns, n_samples=5000, alpha=0.10, seed=42)
        ci_99 = bootstrap_ci(sample_returns, n_samples=5000, alpha=0.01, seed=42)

        # 99% CI should be wider than 90% CI
        assert (ci_99[1] - ci_99[0]) > (ci_90[1] - ci_90[0])

    def test_bootstrap_summary(self, sample_returns):
        """Test bootstrap summary."""
        summary = bootstrap_summary(sample_returns, n_samples=1000, seed=42)

        expected_keys = ["mean", "std", "sharpe", "sortino"]
        for key in expected_keys:
            assert key in summary
            assert "value" in summary[key]
            assert "se" in summary[key]
            assert "ci_lower" in summary[key]
            assert "ci_upper" in summary[key]


class TestStatisticFunctions:
    """Tests for internal statistic functions."""

    def test_get_statistic_fn_mean(self):
        """Test getting mean statistic function."""
        fn = _get_statistic_fn("mean")
        test_data = np.array([1, 2, 3, 4, 5])
        assert fn(test_data) == 3.0

    def test_get_statistic_fn_std(self):
        """Test getting std statistic function."""
        fn = _get_statistic_fn("std")
        test_data = np.array([1, 2, 3, 4, 5])
        result = fn(test_data)
        # Population std is sqrt(2)
        np.testing.assert_almost_equal(result, np.std(test_data, ddof=1))

    def test_get_statistic_fn_sharpe(self):
        """Test getting Sharpe statistic function."""
        fn = _get_statistic_fn("sharpe")
        # Returns with mean > 0
        test_data = np.array([0.01, 0.02, 0.015, 0.03, -0.01])
        result = fn(test_data)
        assert result > 0

    def test_get_statistic_fn_invalid(self):
        """Test that invalid statistic name raises error."""
        with pytest.raises(ValueError, match="Unknown statistic"):
            _get_statistic_fn("invalid")
