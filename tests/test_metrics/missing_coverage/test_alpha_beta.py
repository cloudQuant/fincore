"""Tests for missing coverage in alpha_beta module.

Test cases for previously uncovered code paths in alpha_beta.py.
Split from test_missing_coverage.py for maintainability.
"""

import unittest

import numpy as np
import pandas as pd

from fincore.metrics import alpha_beta


class AlphaBetaMissingCoverageTestCase(unittest.TestCase):
    """Test cases for previously uncovered code paths in alpha_beta.py."""

    def test_alpha_beta_aligned_insufficient_data(self):
        """Test alpha_beta_aligned with insufficient data (line 408-411)."""
        returns = np.array([0.01])
        factor_returns = np.array([0.01])

        result = alpha_beta.alpha_beta_aligned(returns, factor_returns)
        self.assertTrue(np.isnan(result).all())

    def test_alpha_beta_aligned_zero_factor_variance(self):
        """Test alpha_beta_aligned with zero factor variance (line 419-422)."""
        returns = np.array([0.01, 0.02, 0.015])
        factor_returns = np.array([0.01, 0.01, 0.01])  # Zero variance

        result = alpha_beta.alpha_beta_aligned(returns, factor_returns)
        self.assertTrue(np.isnan(result).all())

    def test_annual_alpha_empty_returns(self):
        """Test annual_alpha with empty returns (line 526-527)."""
        returns = pd.Series([], dtype=float)
        factor_returns = pd.Series([], dtype=float)

        result = alpha_beta.annual_alpha(returns, factor_returns)
        self.assertEqual(len(result), 0)

    def test_annual_alpha_non_datetime_index(self):
        """Test annual_alpha with non-DatetimeIndex (line 529-530)."""
        returns = pd.Series([0.01, 0.02, 0.015])
        factor_returns = pd.Series([0.01, 0.015, 0.01])

        result = alpha_beta.annual_alpha(returns, factor_returns)
        self.assertEqual(len(result), 0)

    def test_annual_beta_empty_returns(self):
        """Test annual_beta with empty returns (line 583)."""
        returns = pd.Series([], dtype=float)
        factor_returns = pd.Series([], dtype=float)

        result = alpha_beta.annual_beta(returns, factor_returns)
        self.assertEqual(len(result), 0)

    def test_up_alpha_beta_no_up_periods(self):
        """Test up_alpha_beta when there are no up periods."""
        returns = pd.Series([0.01, 0.02, 0.015])
        factor_returns = pd.Series([-0.01, -0.02, -0.015])  # All negative

        result = alpha_beta.up_alpha_beta(returns, factor_returns)
        self.assertTrue(np.isnan(result).all())

    def test_down_alpha_beta_no_down_periods(self):
        """Test down_alpha_beta when there are no down periods."""
        returns = pd.Series([0.01, 0.02, 0.015])
        factor_returns = pd.Series([0.01, 0.02, 0.015])  # All positive

        result = alpha_beta.down_alpha_beta(returns, factor_returns)
        self.assertTrue(np.isnan(result).all())

    def test_up_alpha_beta_insufficient_clean_data(self):
        """Test up_alpha_beta when filtered data has < 2 points after NaN removal (line 418-420)."""
        returns = pd.Series([0.01, np.nan, np.nan, np.nan])
        factor_returns = pd.Series([0.01, np.nan, np.nan, np.nan])

        result = alpha_beta.up_alpha_beta(returns, factor_returns)
        self.assertEqual(len(result), 2)
        self.assertTrue(np.isnan(result).all())

    def test_up_alpha_beta_zero_variance(self):
        """Test up_alpha_beta when factor variance is zero (line 429-431)."""
        returns = pd.Series([0.01, 0.02, 0.015])
        factor_returns = pd.Series([0.01, 0.01, 0.01])

        result = alpha_beta.up_alpha_beta(returns, factor_returns)
        self.assertEqual(len(result), 2)
        self.assertTrue(np.isnan(result).all())

    def test_down_alpha_beta_insufficient_clean_data(self):
        """Test down_alpha_beta when filtered data has < 2 points after NaN removal (line 418-420)."""
        returns = pd.Series([-0.01, np.nan, np.nan, np.nan])
        factor_returns = pd.Series([-0.01, np.nan, np.nan, np.nan])

        result = alpha_beta.down_alpha_beta(returns, factor_returns)
        self.assertEqual(len(result), 2)
        self.assertTrue(np.isnan(result).all())

    def test_down_alpha_beta_zero_variance(self):
        """Test down_alpha_beta when factor variance is zero (line 429-431)."""
        returns = pd.Series([-0.01, -0.02, -0.015])
        factor_returns = pd.Series([-0.01, -0.01, -0.01])

        result = alpha_beta.down_alpha_beta(returns, factor_returns)
        self.assertEqual(len(result), 2)
        self.assertTrue(np.isnan(result).all())

    def test_annual_alpha_with_no_overlap_years(self):
        """Test annual_alpha when returns and factor have no overlapping years."""
        returns = pd.Series([0.01, 0.02], index=pd.date_range("2020-01-01", periods=2))
        factor_returns = pd.Series([0.005, 0.01], index=pd.date_range("2021-01-01", periods=2))

        result = alpha_beta.annual_alpha(returns, factor_returns)
        self.assertEqual(len(result), 2)
        self.assertTrue(result.isna().all())

    def test_annual_beta_with_no_overlap_years(self):
        """Test annual_beta when returns and factor have no overlapping years."""
        returns = pd.Series([0.01, 0.02], index=pd.date_range("2020-01-01", periods=2))
        factor_returns = pd.Series([0.005, 0.01], index=pd.date_range("2021-01-01", periods=2))

        result = alpha_beta.annual_beta(returns, factor_returns)
        self.assertEqual(len(result), 2)
        self.assertTrue(result.isna().all())
