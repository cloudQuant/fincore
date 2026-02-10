"""
Tests for special indicators.
"""

from unittest import TestCase

import numpy as np
import pandas as pd

from fincore.empyrical import Empyrical

DECIMAL_PLACES = 4


class TestSpecialIndicators(TestCase):
    """Test cases for special indicators."""

    # Standard returns
    normal_returns = pd.Series(
        np.array([0.5, -0.3, 0.4, -0.2, 0.3, 0.1, -0.1, 0.2, -0.2, 0.3]) / 100,
        index=pd.date_range("2000-1-1", periods=10, freq="D"),
    )

    # Returns with some extreme values
    extreme_returns = pd.Series(
        np.array([0.5, -0.3, 5.0, -0.2, 0.3, -4.0, 0.1, 0.2, -0.2, 0.3]) / 100,
        index=pd.date_range("2000-1-1", periods=10, freq="D"),
    )

    # Multi-year returns for RAR
    multi_year_returns = pd.Series(np.random.randn(500) / 100, index=pd.date_range("2020-1-1", periods=500, freq="D"))

    multi_year_market = pd.Series(
        np.random.randn(500) / 100 * 0.8, index=pd.date_range("2020-1-1", periods=500, freq="D")
    )

    # Positive returns
    positive_returns = pd.Series(
        np.array([1.0, 2.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]) / 100,
        index=pd.date_range("2000-1-1", periods=10, freq="D"),
    )

    empty_returns = pd.Series([], dtype=float)

    # Test conditional Sharpe ratio
    def test_conditional_sharpe_ratio(self):
        """Test conditional Sharpe ratio calculation."""
        emp = Empyrical()
        result = emp.conditional_sharpe_ratio(self.normal_returns)
        # Should return a valid number
        assert isinstance(result, (float, np.floating))
        if not np.isnan(result):
            # Should be positive for positive mean returns
            assert result != 0

    def test_conditional_sharpe_ratio_positive(self):
        """Test that positive returns give positive conditional Sharpe."""
        emp = Empyrical()
        result = emp.conditional_sharpe_ratio(self.positive_returns)
        if not np.isnan(result):
            assert result > 0, f"Expected positive conditional Sharpe, got {result}"

    def test_conditional_sharpe_ratio_empty(self):
        """Test that empty returns give NaN."""
        emp = Empyrical()
        result = emp.conditional_sharpe_ratio(self.empty_returns)
        assert np.isnan(result)

    def test_conditional_sharpe_vs_regular_sharpe(self):
        """Test relationship with regular Sharpe ratio."""
        emp = Empyrical()
        cond_sharpe = emp.conditional_sharpe_ratio(self.normal_returns)
        reg_sharpe = emp.sharpe_ratio(self.normal_returns)
        # Both should have the same sign
        if not (np.isnan(cond_sharpe) or np.isnan(reg_sharpe)):
            assert np.sign(cond_sharpe) == np.sign(reg_sharpe)

    # Test VaR excess return
    def test_var_excess_return(self):
        """Test VaR excess return calculation."""
        emp = Empyrical()
        result = emp.var_excess_return(self.normal_returns)
        # Should return a valid number
        assert isinstance(result, (float, np.floating))

    def test_var_excess_return_extreme(self):
        """Test VaR excess return with extreme values."""
        emp = Empyrical()
        result = emp.var_excess_return(self.extreme_returns)
        # Should capture extreme downside
        assert isinstance(result, (float, np.floating))

    def test_var_excess_return_empty(self):
        """Test that empty returns give NaN."""
        emp = Empyrical()
        result = emp.var_excess_return(self.empty_returns)
        assert np.isnan(result)

    # Test regression annual return (RAR)
    def test_regression_annual_return(self):
        """Test regression annual return calculation."""
        emp = Empyrical()
        result = emp.regression_annual_return(self.multi_year_returns, self.multi_year_market)
        # Should return a valid number
        assert isinstance(result, (float, np.floating))

    def test_regression_annual_return_empty(self):
        """Test that empty returns give NaN."""
        emp = Empyrical()
        result = emp.regression_annual_return(self.empty_returns, self.multi_year_market)
        assert np.isnan(result)

    # Test R-cubed
    def test_r_cubed(self):
        """Test R-cubed calculation."""
        emp = Empyrical()
        result = emp.r_cubed(self.multi_year_returns, self.multi_year_market)
        # Should return a valid number
        assert isinstance(result, (float, np.floating))
        if not np.isnan(result):
            # R-cubed should be between 0 and 1
            assert 0 <= result <= 1, f"R-cubed should be in [0,1], got {result}"

    def test_r_cubed_perfect_fit(self):
        """Test R-cubed with perfect fit (returns = market)."""
        returns = self.positive_returns
        market = returns.copy()
        emp = Empyrical()
        result = emp.r_cubed(returns, market)
        if not np.isnan(result):
            # Should be close to 1 for perfect fit
            assert result > 0.9, f"Expected R-cubed close to 1, got {result}"

    def test_r_cubed_empty(self):
        """Test that empty returns give NaN."""
        emp = Empyrical()
        result = emp.r_cubed(self.empty_returns, self.multi_year_market)
        assert np.isnan(result)

    # Test with different cutoffs
    def test_conditional_sharpe_custom_cutoff(self):
        """Test conditional Sharpe with custom cutoff."""
        emp = Empyrical()
        result = emp.conditional_sharpe_ratio(self.normal_returns, cutoff=0.01)
        assert isinstance(result, (float, np.floating))

    def test_var_excess_return_custom_cutoff(self):
        """Test VaR excess return with custom cutoff."""
        emp = Empyrical()
        result = emp.var_excess_return(self.normal_returns, cutoff=0.01)
        assert isinstance(result, (float, np.floating))
