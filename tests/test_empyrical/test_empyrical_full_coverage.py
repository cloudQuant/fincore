"""Tests for Empyrical class - full coverage for edge cases.

This file tests edge cases in fincore.empyrical.Empyrical that are not covered
by the main test suite.
"""

import numpy as np
import pandas as pd
import pytest

from fincore.empyrical import Empyrical


class TestEmpyricalInit:
    """Test Empyrical initialization edge cases."""

    def test_init_with_returns_creates_context(self):
        """Test that initializing with returns creates AnalysisContext."""
        returns = pd.Series(
            [0.01, 0.02, 0.015],
            index=pd.date_range("2020-01-01", periods=3, freq="D"),
        )
        emp = Empyrical(returns=returns)

        # Should have _ctx attribute
        assert hasattr(emp, "_ctx")

    def test_init_with_factor_returns(self):
        """Test initializing with factor_returns."""
        returns = pd.Series(
            [0.01, 0.02, 0.015],
            index=pd.date_range("2020-01-01", periods=3, freq="D"),
        )
        factor_returns = pd.Series(
            [0.005, 0.01, 0.008],
            index=returns.index,
        )
        emp = Empyrical(returns=returns, factor_returns=factor_returns)

        assert hasattr(emp, "_ctx")

    def test_init_with_positions(self):
        """Test initializing with positions."""
        returns = pd.Series(
            [0.01, 0.02, 0.015],
            index=pd.date_range("2020-01-01", periods=3, freq="D"),
        )
        positions = pd.DataFrame(
            {"AAPL": [100, 110, 105], "cash": [1000, 1000, 1000]},
            index=returns.index,
        )
        emp = Empyrical(returns=returns, positions=positions)

        assert hasattr(emp, "_ctx")

    def test_init_with_invalid_returns_handles_gracefully(self):
        """Test initializing with invalid returns handles exception gracefully."""
        # Create a Series without a proper DatetimeIndex
        # This will cause AnalysisContext creation to fail
        returns = pd.Series([0.01, 0.02, 0.015])  # No DatetimeIndex

        # Should not raise, just log and set _ctx to None
        emp = Empyrical(returns=returns)

        # Should still have the returns attribute
        assert emp.returns is not None
        # _ctx may be None if AnalysisContext creation failed


class TestEmpyricalGetattrFallback:
    """Test __getattr__ fallback for registry-backed attributes."""

    def test_getattr_for_classmethod_registry(self):
        """Test __getattr__ for classmethod registry methods."""
        returns = pd.Series(
            [0.01, 0.02, 0.015, -0.01, 0.018],
            index=pd.date_range("2020-01-01", periods=5, freq="D"),
        )
        emp = Empyrical(returns=returns)

        # Access a method that would be in CLASSMETHOD_REGISTRY
        # After first access, it should be cached on the class
        result = emp.sharpe_ratio
        assert callable(result)

    def test_getattr_raises_for_unknown_attribute(self):
        """Test __getattr__ raises AttributeError for unknown attributes."""
        emp = Empyrical()

        with pytest.raises(AttributeError, match="has no attribute"):
            _ = emp.unknown_attribute_12345


class TestEmpyricalRollingMetrics:
    """Test rolling metrics with factor_returns."""

    def test_roll_alpha_with_factor_returns(self):
        """Test roll_alpha with factor_returns."""
        returns = pd.Series(
            np.random.randn(300) * 0.01,
            index=pd.date_range("2020-01-01", periods=300, freq="D"),
        )
        factor_returns = pd.Series(
            np.random.randn(300) * 0.01,
            index=returns.index,
        )

        emp = Empyrical(returns=returns, factor_returns=factor_returns)

        result = emp.roll_alpha(window=50)

        assert isinstance(result, (pd.Series, pd.DataFrame))

    def test_roll_beta_with_factor_returns(self):
        """Test roll_beta with factor_returns."""
        returns = pd.Series(
            np.random.randn(300) * 0.01,
            index=pd.date_range("2020-01-01", periods=300, freq="D"),
        )
        factor_returns = pd.Series(
            np.random.randn(300) * 0.01,
            index=returns.index,
        )

        emp = Empyrical(returns=returns, factor_returns=factor_returns)

        result = emp.roll_beta(window=50)

        assert isinstance(result, (pd.Series, pd.DataFrame))

    def test_roll_alpha_instance_with_stored_factor(self):
        """Test roll_alpha using stored factor_returns."""
        returns = pd.Series(
            np.random.randn(300) * 0.01,
            index=pd.date_range("2020-01-01", periods=300, freq="D"),
        )
        factor_returns = pd.Series(
            np.random.randn(300) * 0.01,
            index=returns.index,
        )

        emp = Empyrical(returns=returns, factor_returns=factor_returns)

        # Call without passing factor_returns - should use stored
        result = emp.roll_alpha(window=50)

        assert isinstance(result, (pd.Series, pd.DataFrame))

    def test_roll_beta_instance_with_stored_factor(self):
        """Test roll_beta using stored factor_returns."""
        returns = pd.Series(
            np.random.randn(300) * 0.01,
            index=pd.date_range("2020-01-01", periods=300, freq="D"),
        )
        factor_returns = pd.Series(
            np.random.randn(300) * 0.01,
            index=returns.index,
        )

        emp = Empyrical(returns=returns, factor_returns=factor_returns)

        # Call without passing factor_returns - should use stored
        result = emp.roll_beta(window=50)

        assert isinstance(result, (pd.Series, pd.DataFrame))


class TestEmpyricalTreynorRatio:
    """Test treynor_ratio edge cases."""

    def test_treynor_ratio_with_nan_beta(self):
        """Test treynor_ratio when beta is NaN."""
        returns = pd.Series([0.01, 0.02, 0.015])
        factor_returns = pd.Series([np.nan, np.nan, np.nan])

        result = Empyrical.treynor_ratio(returns, factor_returns)

        # Should return NaN when beta is NaN
        assert np.isnan(result)

    def test_treynor_ratio_with_nan_benchmark_return(self):
        """Test treynor_ratio when benchmark annual return is NaN."""
        returns = pd.Series([0.01, np.nan, 0.015])
        factor_returns = pd.Series([np.nan, np.nan, np.nan])

        result = Empyrical.treynor_ratio(returns, factor_returns)

        # Should return NaN when benchmark annual return is NaN
        assert np.isnan(result)

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


class TestGroupbyConsecutive:
    """Test _groupby_consecutive class method."""

    def test_groupby_consecutive_default_max_delta(self):
        """Test _groupby_consecutive with default max_delta."""
        txn = pd.DataFrame(
            {
                "amount": [10, 5, -5, -10, 8, -8],
                "price": [100.0, 102.0, 105.0, 103.0, 98.0, 99.0],
                "symbol": ["A", "A", "A", "A", "B", "B"],
            },
            index=pd.date_range("2020-01-01", periods=6, freq="h"),
        )

        result = Empyrical._groupby_consecutive(txn)

        assert result is not None
        assert "symbol" in result.columns

    def test_groupby_consecutive_custom_max_delta(self):
        """Test _groupby_consecutive with custom max_delta."""
        txn = pd.DataFrame(
            {
                "amount": [10, 5, -5, -10],
                "price": [100.0, 102.0, 105.0, 103.0],
                "symbol": ["A", "A", "A", "A"],
            },
            index=pd.date_range("2020-01-01", periods=4, freq="h"),
        )

        result = Empyrical._groupby_consecutive(txn, max_delta=pd.Timedelta("4h"))

        assert result is not None


class TestDualMethodBehavior:
    """Test @_dual_method behavior for various methods."""

    def test_get_returns_fallback(self):
        """Test _get_returns falls back to instance returns."""
        returns = pd.Series([0.01, 0.02, 0.015])
        emp = Empyrical(returns=returns)

        # Call without passing returns - should use stored
        result = emp._get_returns(None)

        pd.testing.assert_series_equal(result, returns)

    def test_get_factor_returns_fallback(self):
        """Test _get_factor_returns falls back to instance factor_returns."""
        factor_returns = pd.Series([0.005, 0.01, 0.008])
        emp = Empyrical(factor_returns=factor_returns)

        # Call without passing factor_returns - should use stored
        result = emp._get_factor_returns(None)

        pd.testing.assert_series_equal(result, factor_returns)
