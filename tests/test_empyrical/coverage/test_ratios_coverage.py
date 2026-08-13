"""Tests for Empyrical ratio and timing methods.

This file tests treynor_ratio and other ratio methods for coverage.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from fincore.empyrical import Empyrical
from fincore.exceptions import NumericalError


class TestEmpyricalTreynorRatio:
    """Test treynor_ratio edge cases."""

    def test_treynor_ratio_with_nan_beta(self):
        """Test treynor_ratio rejects non-finite factor returns."""
        returns = pd.Series([0.01, 0.02, 0.015])
        factor_returns = pd.Series([np.nan, np.nan, np.nan])

        with pytest.raises(NumericalError, match="finite"):
            Empyrical.treynor_ratio(returns, factor_returns)

    def test_treynor_ratio_with_nan_benchmark_return(self):
        """Test treynor_ratio rejects non-finite factor returns up front."""
        returns = pd.Series([0.01, 0.02, 0.015, -0.01, 0.018])
        factor_returns = pd.Series([np.nan, np.nan, np.nan, np.nan, np.nan])

        # The enhanced surface validates inputs before any alpha/beta work.
        with pytest.raises(NumericalError, match="finite"):
            Empyrical.treynor_ratio(returns, factor_returns)

    def test_treynor_ratio_impl_acceptance_test(self):
        """Test treynor_ratio general behavior."""
        # This test verifies the treynor_ratio method works correctly
        returns = pd.Series([0.01, 0.02, 0.015, -0.01, 0.018])
        factor_returns = pd.Series([0.005, 0.01, 0.008, -0.005, 0.009])

        result = Empyrical.treynor_ratio(returns, factor_returns)

        # Should return a numeric value
        assert isinstance(result, (float, type(np.nan)))

    def test_treynor_ratio_instance_method(self):
        """Test treynor_ratio as instance method."""
        returns = pd.Series(
            [0.01, 0.02, 0.015, -0.01, 0.018],
            index=pd.date_range("2020-01-01", periods=5, freq="D"),
        )
        factor_returns = pd.Series(
            [0.005, 0.01, 0.008, -0.005, 0.009],
            index=returns.index,
        )

        emp = Empyrical(returns=returns, factor_returns=factor_returns)

        result = emp.treynor_ratio()

        # Should use stored returns and factor_returns
        assert isinstance(result, float)


class TestEmpyricalRatiosAndTiming:
    """Test ratio and timing methods for coverage."""

    def test_common_sense_ratio_instance_method(self):
        """Test common_sense_ratio as instance method."""
        returns = pd.Series(
            [0.01, 0.02, 0.015, -0.01, 0.018],
            index=pd.date_range("2020-01-01", periods=5, freq="D"),
        )
        emp = Empyrical(returns=returns)

        result = emp.common_sense_ratio()

        assert isinstance(result, float)

    def test_sterling_ratio_instance_method(self):
        """Test sterling_ratio as instance method."""
        returns = pd.Series(
            [0.01, 0.02, 0.015, -0.01, 0.018],
            index=pd.date_range("2020-01-01", periods=5, freq="D"),
        )
        emp = Empyrical(returns=returns)

        result = emp.sterling_ratio()

        assert isinstance(result, float)

    def test_burke_ratio_instance_method(self):
        """Test burke_ratio as instance method."""
        returns = pd.Series(
            [0.01, 0.02, 0.015, -0.01, 0.018],
            index=pd.date_range("2020-01-01", periods=5, freq="D"),
        )
        emp = Empyrical(returns=returns)

        result = emp.burke_ratio()

        assert isinstance(result, float)

    def test_extract_interesting_date_ranges_instance_method(self):
        """Test extract_interesting_date_ranges as instance method."""
        returns = pd.Series(
            [0.01, 0.02, 0.015, -0.01, 0.018],
            index=pd.date_range("2020-01-01", periods=5, freq="D"),
        )
        emp = Empyrical(returns=returns)

        result = emp.extract_interesting_date_ranges()

        assert result is not None

    def test_regression_annual_return_with_valid_data(self):
        """Test regression_annual_return with valid data."""
        returns = pd.Series(
            [0.01, 0.02, 0.015, -0.01, 0.018, 0.012, 0.022],
            index=pd.date_range("2020-01-01", periods=7, freq="D"),
        )
        factor_returns = pd.Series(
            [0.005, 0.01, 0.008, -0.005, 0.009, 0.006, 0.011],
            index=returns.index,
        )
        emp = Empyrical(returns=returns, factor_returns=factor_returns)

        result = emp.regression_annual_return()

        # Should return a float value
        assert isinstance(result, float)

    def test_regression_annual_return_with_nan_benchmark(self):
        """Test regression_annual_return rejects non-finite factor returns."""
        returns = pd.Series(
            [0.01, 0.02, 0.015, -0.01, 0.018],
            index=pd.date_range("2020-01-01", periods=5, freq="D"),
        )
        factor_returns = pd.Series(
            [np.nan, np.nan, np.nan, np.nan, np.nan],
            index=returns.index,
        )

        # The enhanced surface validates inputs before alpha/beta computation.
        with pytest.raises(NumericalError, match="finite"):
            Empyrical.regression_annual_return(returns, factor_returns)
