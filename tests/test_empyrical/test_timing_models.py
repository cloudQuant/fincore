"""
Tests for market timing regression models.
"""

from unittest import TestCase

import numpy as np
import pandas as pd

from fincore.empyrical import Empyrical

DECIMAL_PLACES = 4


class TestTimingModels(TestCase):
    """Test cases for market timing regression models."""

    # Multi-year returns for regression
    np.random.seed(42)
    multi_year_returns = pd.Series(
        np.random.randn(500) / 100 + 0.0003, index=pd.date_range("2020-1-1", periods=500, freq="D")
    )

    multi_year_market = pd.Series(
        np.random.randn(500) / 100 + 0.0002, index=pd.date_range("2020-1-1", periods=500, freq="D")
    )

    # Simulated returns with timing ability (higher returns when market is up)
    timing_returns = pd.Series(
        np.random.randn(500) / 100 + 0.0003 + np.where(multi_year_market.values > 0, 0.0002, -0.0001),
        index=pd.date_range("2020-1-1", periods=500, freq="D"),
    )

    # Simple returns
    simple_returns = pd.Series(
        np.array([1.0, 2.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]) / 100,
        index=pd.date_range("2000-1-1", periods=10, freq="D"),
    )

    simple_market = pd.Series(
        np.array([0.8, 1.9, 1.1, 0.9, 1.0, 1.8, 1.1, 1.4, 0.9, 1.0]) / 100,
        index=pd.date_range("2000-1-1", periods=10, freq="D"),
    )

    empty_returns = pd.Series([], dtype=float)

    # Test Treynor-Mazuy model
    def test_treynor_mazuy_timing(self):
        """Test Treynor-Mazuy timing coefficient calculation."""
        emp = Empyrical()
        result = emp.treynor_mazuy_timing(self.multi_year_returns, self.multi_year_market)
        # Should return a valid number
        assert isinstance(result, (float, np.floating))

    def test_treynor_mazuy_timing_empty(self):
        """Test that empty returns give NaN."""
        emp = Empyrical()
        result = emp.treynor_mazuy_timing(self.empty_returns, self.multi_year_market)
        assert np.isnan(result)

    def test_treynor_mazuy_timing_values(self):
        """Test Treynor-Mazuy with known timing behavior."""
        # Timing returns should have positive gamma
        emp = Empyrical()
        gamma = emp.treynor_mazuy_timing(self.timing_returns, self.multi_year_market)
        # Timing coefficient should be a number
        assert not np.isnan(gamma)

    # Test Henriksson-Merton model
    def test_henriksson_merton_timing(self):
        """Test Henriksson-Merton timing coefficient calculation."""
        emp = Empyrical()
        result = emp.henriksson_merton_timing(self.multi_year_returns, self.multi_year_market)
        # Should return a valid number
        assert isinstance(result, (float, np.floating))

    def test_henriksson_merton_timing_empty(self):
        """Test that empty returns give NaN."""
        emp = Empyrical()
        result = emp.henriksson_merton_timing(self.empty_returns, self.multi_year_market)
        assert np.isnan(result)

    def test_henriksson_merton_timing_values(self):
        """Test Henriksson-Merton with known timing behavior."""
        # Timing returns should have positive timing coefficient
        emp = Empyrical()
        timing_coef = emp.henriksson_merton_timing(self.timing_returns, self.multi_year_market)
        # Should be a valid number
        assert not np.isnan(timing_coef)

    # Test Cornell-Letang model (if different from H-M)
    def test_market_timing_return(self):
        """Test market timing return calculation."""
        emp = Empyrical()
        result = emp.market_timing_return(self.multi_year_returns, self.multi_year_market)
        # Should return a valid number
        assert isinstance(result, (float, np.floating))

    def test_market_timing_return_empty(self):
        """Test that empty returns give NaN."""
        emp = Empyrical()
        result = emp.market_timing_return(self.empty_returns, self.multi_year_market)
        assert np.isnan(result)

    # Test with short series
    def test_treynor_mazuy_short_series(self):
        """Test T-M with short series (should handle gracefully)."""
        emp = Empyrical()
        result = emp.treynor_mazuy_timing(self.simple_returns, self.simple_market)
        # May return NaN or a value depending on implementation
        assert isinstance(result, (float, np.floating))

    def test_henriksson_merton_short_series(self):
        """Test H-M with short series (should handle gracefully)."""
        emp = Empyrical()
        result = emp.henriksson_merton_timing(self.simple_returns, self.simple_market)
        # May return NaN or a value depending on implementation
        assert isinstance(result, (float, np.floating))
