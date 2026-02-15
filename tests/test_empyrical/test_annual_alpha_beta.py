"""
Tests for annual alpha/beta and residual risk metrics.
"""

from unittest import TestCase

import numpy as np
import pandas as pd

from fincore.empyrical import Empyrical

DECIMAL_PLACES = 4

# Pandas frequency alias compatibility
try:
    pd.date_range("2000-1-1", periods=1, freq="ME")
    MONTH_FREQ = "ME"
    YEAR_FREQ = "YE"
except ValueError:
    MONTH_FREQ = "M"
    YEAR_FREQ = "A"


class TestAnnualAlphaBeta(TestCase):
    """Test cases for annual alpha/beta and residual risk."""

    # Multi-year returns
    multi_year_returns = pd.Series(
        np.random.randn(500) / 100,  # ~2 years of daily data
        index=pd.date_range("2020-1-1", periods=500, freq="D"),
    )

    # Market returns (correlated with strategy)
    multi_year_market = pd.Series(
        np.random.randn(500) / 100 * 0.8 + multi_year_returns.values * 0.3,
        index=pd.date_range("2020-1-1", periods=500, freq="D"),
    )

    # Short series
    short_returns = pd.Series(
        np.array([1.0, 2.0, 1.0, 1.0, 1.0]) / 100, index=pd.date_range("2020-1-1", periods=5, freq="D")
    )

    short_market = pd.Series(
        np.array([0.8, 1.9, 1.1, 0.9, 1.0]) / 100, index=pd.date_range("2020-1-1", periods=5, freq="D")
    )

    empty_returns = pd.Series([], dtype=float)

    # Test annual alpha
    def test_annual_alpha(self):
        """Test annual alpha calculation."""
        emp = Empyrical()
        result = emp.annual_alpha(self.multi_year_returns, self.multi_year_market)
        # Should return a Series with years as index
        assert isinstance(result, pd.Series)
        assert len(result) > 0
        # All values should be valid numbers
        assert all(isinstance(v, (float, np.floating)) for v in result)

    def test_annual_alpha_short_series(self):
        """Test annual alpha with short series (less than 1 year)."""
        emp = Empyrical()
        result = emp.annual_alpha(self.short_returns, self.short_market)
        # Should still work but may have only 1 year
        assert isinstance(result, pd.Series)

    def test_annual_alpha_empty(self):
        """Test that empty returns give empty Series."""
        emp = Empyrical()
        result = emp.annual_alpha(self.empty_returns, self.short_market)
        assert isinstance(result, pd.Series)
        assert len(result) == 0

    # Test annual beta
    def test_annual_beta(self):
        """Test annual beta calculation."""
        emp = Empyrical()
        result = emp.annual_beta(self.multi_year_returns, self.multi_year_market)
        # Should return a Series with years as index
        assert isinstance(result, pd.Series)
        assert len(result) > 0
        # Beta values should be reasonable
        assert all(isinstance(v, (float, np.floating)) for v in result)

    def test_annual_beta_short_series(self):
        """Test annual beta with short series."""
        emp = Empyrical()
        result = emp.annual_beta(self.short_returns, self.short_market)
        assert isinstance(result, pd.Series)

    def test_annual_beta_empty(self):
        """Test that empty returns give empty Series."""
        emp = Empyrical()
        result = emp.annual_beta(self.empty_returns, self.short_market)
        assert isinstance(result, pd.Series)
        assert len(result) == 0

    # Test residual risk
    def test_residual_risk(self):
        """Test residual risk calculation."""
        emp = Empyrical()
        result = emp.residual_risk(self.multi_year_returns, self.multi_year_market)
        # Residual risk should be a positive number
        assert isinstance(result, (float, np.floating))
        if not np.isnan(result):
            assert result >= 0, f"Residual risk should be non-negative, got {result}"

    def test_residual_risk_perfect_correlation(self):
        """Test residual risk with perfect correlation (should be low)."""
        returns = self.short_returns
        market = returns.copy()  # Perfect correlation
        emp = Empyrical()
        result = emp.residual_risk(returns, market)
        # Should be very small or zero
        if not np.isnan(result):
            assert result < 0.001, f"Expected low residual risk, got {result}"

    def test_residual_risk_short_series(self):
        """Test residual risk with short series."""
        emp = Empyrical()
        result = emp.residual_risk(self.short_returns, self.short_market)
        assert isinstance(result, (float, np.floating))

    def test_residual_risk_empty(self):
        """Test that empty returns give NaN."""
        emp = Empyrical()
        result = emp.residual_risk(self.empty_returns, self.short_market)
        assert np.isnan(result)

    # Integration tests
    def test_annual_metrics_consistency(self):
        """Test that annual alpha and beta are consistent with overall metrics."""
        # Get annual values
        emp = Empyrical()
        annual_alphas = emp.annual_alpha(self.multi_year_returns, self.multi_year_market)
        annual_betas = emp.annual_beta(self.multi_year_returns, self.multi_year_market)

        # Get overall values
        overall_alpha = emp.alpha(self.multi_year_returns, self.multi_year_market)
        overall_beta = emp.beta(self.multi_year_returns, self.multi_year_market)

        # Annual alphas should vary around the overall alpha
        if len(annual_alphas) > 0 and not np.isnan(overall_alpha):
            mean_annual_alpha = annual_alphas.mean()
            assert abs(mean_annual_alpha - overall_alpha) < 1.0, (
                f"Mean annual alpha ({mean_annual_alpha}) differs too much from overall ({overall_alpha})"
            )

        # Annual betas should vary around the overall beta
        if len(annual_betas) > 0 and not np.isnan(overall_beta):
            mean_annual_beta = annual_betas.mean()
            # Should be relatively close (within reasonable range)
            assert abs(mean_annual_beta - overall_beta) < 1.0, (
                f"Mean annual beta ({mean_annual_beta}) differs too much from overall ({overall_beta})"
            )
