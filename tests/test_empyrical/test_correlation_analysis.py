"""
Tests for correlation analysis functions.
"""

import warnings
from unittest import TestCase

import numpy as np
import pandas as pd
from numpy.testing import assert_almost_equal

from fincore.empyrical import Empyrical

DECIMAL_PLACES = 4


class TestCorrelationAnalysis(TestCase):
    """Test cases for correlation analysis."""

    # Portfolio returns
    portfolio_returns = pd.Series(
        np.array([1.0, 2.0, -1.0, 3.0, -2.0, 4.0, 1.0, -1.0]) / 100,
        index=pd.date_range("2000-1-1", periods=8, freq="D"),
    )

    # Highly correlated market returns
    stock_market_returns = pd.Series(
        np.array([1.2, 2.1, -0.9, 3.2, -2.1, 4.1, 1.1, -0.8]) / 100,
        index=pd.date_range("2000-1-1", periods=8, freq="D"),
    )

    # Negatively correlated bond returns
    bond_market_returns = pd.Series(
        np.array([-1.0, -2.0, 1.0, -3.0, 2.0, -4.0, -1.0, 1.0]) / 100,
        index=pd.date_range("2000-1-1", periods=8, freq="D"),
    )

    # Uncorrelated futures returns
    futures_market_returns = pd.Series(
        np.array([0.5, -1.5, 2.0, 0.5, -1.0, 0.5, 2.0, -0.5]) / 100,
        index=pd.date_range("2000-1-1", periods=8, freq="D"),
    )

    # Test data for serial correlation
    autocorr_returns = pd.Series(
        np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0]) / 100,
        index=pd.date_range("2000-1-1", periods=12, freq="W"),
    )

    empty_returns = pd.Series([], dtype=float)

    # Test stock market correlation
    def test_stock_market_correlation(self):
        emp = Empyrical()
        result = emp.stock_market_correlation(self.portfolio_returns, self.stock_market_returns)
        # Should be positive correlation
        assert result > 0.5, f"Expected high positive correlation, got {result}"
        assert -1 <= result <= 1, f"Correlation {result} should be between -1 and 1"

    def test_stock_market_correlation_perfect(self):
        """Test perfect correlation."""
        emp = Empyrical()
        result = emp.stock_market_correlation(self.portfolio_returns, self.portfolio_returns)
        assert_almost_equal(result, 1.0, DECIMAL_PLACES)

    # Test bond market correlation
    def test_bond_market_correlation(self):
        emp = Empyrical()
        result = emp.bond_market_correlation(self.portfolio_returns, self.bond_market_returns)
        # Should be negative correlation
        assert result < 0, f"Expected negative correlation, got {result}"
        assert -1 <= result <= 1, f"Correlation {result} should be between -1 and 1"

    # Test futures market correlation
    def test_futures_market_correlation(self):
        emp = Empyrical()
        result = emp.futures_market_correlation(self.portfolio_returns, self.futures_market_returns)
        # Should be low correlation
        assert -1 <= result <= 1, f"Correlation {result} should be between -1 and 1"

    # Test serial correlation
    def test_serial_correlation_one_week(self):
        emp = Empyrical()
        result = emp.serial_correlation(self.autocorr_returns, lag=1)
        # Trending series should have positive autocorrelation
        assert result > 0, f"Expected positive autocorrelation, got {result}"
        assert -1 <= result <= 1, f"Correlation {result} should be between -1 and 1"

    def test_serial_correlation_default(self):
        """Test serial correlation with default 1-week lag."""
        emp = Empyrical()
        result = emp.serial_correlation(self.autocorr_returns)
        assert -1 <= result <= 1, f"Correlation {result} should be between -1 and 1"

    # Test edge cases
    def test_correlation_empty_returns(self):
        """Test that empty returns give NaN."""
        emp = Empyrical()
        result = emp.stock_market_correlation(self.empty_returns, self.stock_market_returns)
        assert np.isnan(result)

    def test_correlation_mismatched_length(self):
        """Test with mismatched length series."""
        short_returns = self.portfolio_returns[:4]
        emp = Empyrical()
        result = emp.stock_market_correlation(short_returns, self.stock_market_returns)
        # Should handle mismatched lengths (align indices)
        assert not np.isnan(result)

    def test_serial_correlation_insufficient_data(self):
        """Test serial correlation with insufficient data."""
        short_returns = pd.Series([0.01, 0.02], index=pd.date_range("2000-1-1", periods=2, freq="W"))
        emp = Empyrical()
        result = emp.serial_correlation(short_returns, lag=1)
        assert np.isnan(result)

    def test_serial_correlation_constant_returns_no_warning(self):
        """Constant returns should yield NaN without runtime warnings."""
        const = pd.Series(np.zeros(20), index=pd.date_range("2000-1-1", periods=20, freq="D"))
        emp = Empyrical()

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = emp.serial_correlation(const, lag=1)

        assert np.isnan(result)
        runtime_warnings = [w for w in caught if issubclass(w.category, RuntimeWarning)]
        assert len(runtime_warnings) == 0

    def test_market_correlation_constant_returns_no_warning(self):
        """Correlation with a constant series should yield NaN without runtime warnings."""
        const = pd.Series(np.zeros(20), index=pd.date_range("2000-1-1", periods=20, freq="D"))
        market = pd.Series(np.linspace(-0.01, 0.01, 20), index=pd.date_range("2000-1-1", periods=20, freq="D"))
        emp = Empyrical()

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = emp.stock_market_correlation(const, market)

        assert np.isnan(result)
        runtime_warnings = [w for w in caught if issubclass(w.category, RuntimeWarning)]
        assert len(runtime_warnings) == 0
