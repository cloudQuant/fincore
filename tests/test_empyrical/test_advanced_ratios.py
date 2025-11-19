"""
Tests for advanced risk-adjusted return ratios.
"""
from __future__ import division

import numpy as np
import pandas as pd
from unittest import TestCase

from fincore.empyrical import Empyrical

DECIMAL_PLACES = 4


class TestAdvancedRatios(TestCase):
    """Test cases for advanced risk-adjusted ratios."""

    # Standard returns with drawdowns
    returns_with_drawdown = pd.Series(
        np.array([1., 2., -5., 3., 2., -3., 4., 1., 2., -2.]) / 100,
        index=pd.date_range('2000-1-1', periods=10, freq='D'))

    # Positive returns
    positive_returns = pd.Series(
        np.array([1., 2., 1., 1., 1., 1., 1., 1., 1., 1.]) / 100,
        index=pd.date_range('2000-1-1', periods=10, freq='D'))

    # High volatility returns
    high_vol_returns = pd.Series(
        np.array([5., -4., 6., -5., 7., -3., 4., -2., 3., -1.]) / 100,
        index=pd.date_range('2000-1-1', periods=10, freq='D'))

    # Normal-like returns
    normal_returns = pd.Series(
        np.array([0.5, -0.3, 0.4, -0.2, 0.3, 0.1, -0.1, 0.2, -0.2, 0.3]) / 100,
        index=pd.date_range('2000-1-1', periods=10, freq='D'))

    empty_returns = pd.Series([], dtype=float)

    # Test Sterling Ratio
    def test_sterling_ratio(self):
        """Test Sterling ratio calculation."""
        emp = Empyrical()
        result = emp.sterling_ratio(self.returns_with_drawdown)
        # Sterling ratio should be a valid number
        assert isinstance(result, (float, np.floating))
        assert not np.isnan(result) or len(self.returns_with_drawdown) < 2

    def test_sterling_ratio_positive_returns(self):
        """Test that positive returns give positive Sterling ratio."""
        emp = Empyrical()
        result = emp.sterling_ratio(self.positive_returns)
        assert result > 0, f"Expected positive Sterling ratio, got {result}"

    def test_sterling_ratio_empty(self):
        """Test that empty returns give NaN."""
        emp = Empyrical()
        result = emp.sterling_ratio(self.empty_returns)
        assert np.isnan(result)

    # Test Burke Ratio
    def test_burke_ratio(self):
        """Test Burke ratio calculation."""
        emp = Empyrical()
        result = emp.burke_ratio(self.returns_with_drawdown)
        # Burke ratio should be a valid number
        assert isinstance(result, (float, np.floating))
        assert not np.isnan(result) or len(self.returns_with_drawdown) < 2

    def test_burke_ratio_positive_returns(self):
        """Test that positive returns give positive Burke ratio."""
        emp = Empyrical()
        result = emp.burke_ratio(self.positive_returns)
        assert result > 0, f"Expected positive Burke ratio, got {result}"

    def test_burke_ratio_empty(self):
        """Test that empty returns give NaN."""
        emp = Empyrical()
        result = emp.burke_ratio(self.empty_returns)
        assert np.isnan(result)

    # Test Kappa 3 Ratio
    def test_kappa_three_ratio(self):
        """Test Kappa 3 ratio calculation."""
        emp = Empyrical()
        result = emp.kappa_three_ratio(self.returns_with_drawdown)
        # Kappa 3 ratio should be a valid number
        assert isinstance(result, (float, np.floating))
        assert not np.isnan(result) or len(self.returns_with_drawdown) < 2

    def test_kappa_three_ratio_positive_returns(self):
        """Test that positive returns give positive Kappa 3 ratio."""
        emp = Empyrical()
        result = emp.kappa_three_ratio(self.positive_returns)
        assert result > 0, f"Expected positive Kappa 3 ratio, got {result}"

    def test_kappa_three_ratio_empty(self):
        """Test that empty returns give NaN."""
        emp = Empyrical()
        result = emp.kappa_three_ratio(self.empty_returns)
        assert np.isnan(result)

    # Test Adjusted Sharpe Ratio
    def test_adjusted_sharpe_ratio(self):
        """Test adjusted Sharpe ratio calculation."""
        emp = Empyrical()
        result = emp.adjusted_sharpe_ratio(self.normal_returns)
        # Adjusted Sharpe should be a valid number
        assert isinstance(result, (float, np.floating))
        assert not np.isnan(result) or len(self.normal_returns) < 3

    def test_adjusted_sharpe_ratio_vs_regular(self):
        """Test that adjusted Sharpe is close to regular Sharpe for normal returns."""
        emp = Empyrical()
        adjusted = emp.adjusted_sharpe_ratio(self.normal_returns)
        regular = emp.sharpe_ratio(self.normal_returns)
        # For near-normal returns, they should be similar
        # (adjusted applies correction for skewness and kurtosis)
        assert abs(adjusted - regular) < 1.0, \
            f"Adjusted ({adjusted}) and regular ({regular}) Sharpe differ too much"

    def test_adjusted_sharpe_ratio_empty(self):
        """Test that empty returns give NaN."""
        emp = Empyrical()
        result = emp.adjusted_sharpe_ratio(self.empty_returns)
        assert np.isnan(result)

    # Test Stutzer Index
    def test_stutzer_index(self):
        """Test Stutzer index calculation."""
        emp = Empyrical()
        result = emp.stutzer_index(self.normal_returns)
        # Stutzer index should be a valid number
        assert isinstance(result, (float, np.floating))
        # Allow NaN for insufficient data
        if not np.isnan(result):
            assert result != 0  # Should have a meaningful value

    def test_stutzer_index_positive_returns(self):
        """Test that positive returns give positive Stutzer index."""
        emp = Empyrical()
        result = emp.stutzer_index(self.positive_returns)
        if not np.isnan(result):
            assert result > 0, f"Expected positive Stutzer index, got {result}"

    def test_stutzer_index_empty(self):
        """Test that empty returns give NaN."""
        emp = Empyrical()
        result = emp.stutzer_index(self.empty_returns)
        assert np.isnan(result)

    # Test with risk-free rate
    def test_sterling_ratio_with_risk_free(self):
        """Test Sterling ratio with non-zero risk-free rate."""
        emp = Empyrical()
        result = emp.sterling_ratio(self.returns_with_drawdown, risk_free=0.02)
        assert isinstance(result, (float, np.floating))

    def test_burke_ratio_with_risk_free(self):
        """Test Burke ratio with non-zero risk-free rate."""
        emp = Empyrical()
        result = emp.burke_ratio(self.returns_with_drawdown, risk_free=0.02)
        assert isinstance(result, (float, np.floating))
