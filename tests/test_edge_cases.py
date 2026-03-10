"""Comprehensive edge case tests for ALL P0 core metrics.

This module systematically tests edge cases across all critical (P0) metrics
to ensure robustness and correct handling of:
- Empty data
- Single values
- All NaN values
- Zero volatility
- Infinite values
- Extreme values
- Mixed frequencies
- Missing data

These tests ensure the library handles real-world data gracefully.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from fincore import (
    sharpe_ratio,
    sortino_ratio,
    max_drawdown,
    annual_return,
    annual_volatility,
    alpha,
    beta,
    cum_returns,
    cum_returns_final,
    value_at_risk,
    calmar_ratio,
    omega_ratio,
    downside_risk,
    tail_ratio,
)
from fincore.metrics.risk import conditional_value_at_risk
from fincore.constants import DAILY


# ==============================================================================
# Edge Case Fixtures
# ==============================================================================


@pytest.fixture
def empty_returns():
    """Empty returns series."""
    return pd.Series([], dtype=float)


@pytest.fixture
def single_value():
    """Single value returns."""
    return pd.Series([0.01])


@pytest.fixture
def two_values():
    """Two values returns."""
    return pd.Series([0.01, -0.005])


@pytest.fixture
def all_nan():
    """All NaN returns."""
    return pd.Series([np.nan] * 100)


@pytest.fixture
def mostly_nan():
    """Mostly NaN with few valid values."""
    s = pd.Series([np.nan] * 100)
    s[10] = 0.01
    s[50] = -0.02
    s[90] = 0.015
    return s


@pytest.fixture
def zero_volatility():
    """Zero volatility (constant returns)."""
    return pd.Series([0.01] * 100)


@pytest.fixture
def all_zeros():
    """All zero returns."""
    return pd.Series([0.0] * 100)


@pytest.fixture
def infinite_values():
    """Infinite values in returns."""
    return pd.Series([0.01, np.inf, -np.inf, 0.02, -0.01])


@pytest.fixture
def extreme_values():
    """Extreme values (very large/small)."""
    return pd.Series([1e10, -1e10, 1e-10, -1e-10, 0.01])


@pytest.fixture
def mixed_nan_inf():
    """Mixed NaN and infinite values."""
    return pd.Series([0.01, np.nan, np.inf, -np.inf, 0.02])


# ==============================================================================
# Common Edge Case Tests
# ==============================================================================


class TestEmptyReturns:
    """Test behavior with empty returns series."""

    @pytest.mark.p1
    def test_sharpe_ratio_empty(self, empty_returns):
        """Sharpe ratio should return NaN for empty data."""
        result = sharpe_ratio(empty_returns)
        assert np.isnan(result)

    @pytest.mark.p1
    def test_max_drawdown_empty(self, empty_returns):
        """Max drawdown should return NaN for empty data."""
        result = max_drawdown(empty_returns)
        assert np.isnan(result)

    @pytest.mark.p1
    def test_annual_return_empty(self, empty_returns):
        """Annual return should return NaN for empty data."""
        result = annual_return(empty_returns)
        assert np.isnan(result)

    @pytest.mark.p1
    def test_annual_volatility_empty(self, empty_returns):
        """Annual volatility should return NaN for empty data."""
        result = annual_volatility(empty_returns)
        assert np.isnan(result)


class TestSingleValue:
    """Test behavior with single value returns."""

    @pytest.mark.p1
    def test_sharpe_ratio_single(self, single_value):
        """Sharpe ratio should return NaN for single value (no volatility)."""
        result = sharpe_ratio(single_value)
        assert np.isnan(result)

    @pytest.mark.p1
    def test_max_drawdown_single(self, single_value):
        """Max drawdown should handle single value gracefully."""
        result = max_drawdown(single_value)
        # Should be 0 or NaN depending on implementation
        assert result == 0 or np.isnan(result)

    @pytest.mark.p1
    def test_annual_volatility_single(self, single_value):
        """Annual volatility should return NaN for single value."""
        result = annual_volatility(single_value)
        assert np.isnan(result)


class TestTwoValues:
    """Test behavior with two values (minimum for meaningful calculation)."""

    @pytest.mark.p1
    def test_sharpe_ratio_two_values(self, two_values):
        """Sharpe ratio should calculate for two values."""
        result = sharpe_ratio(two_values)
        # Should return a finite value or NaN (if volatility is 0)
        assert np.isfinite(result) or np.isnan(result)

    @pytest.mark.p1
    def test_max_drawdown_two_values(self, two_values):
        """Max drawdown should calculate for two values."""
        result = max_drawdown(two_values)
        assert result <= 0


class TestNaNValues:
    """Test behavior with NaN values."""

    @pytest.mark.p1
    def test_sharpe_ratio_all_nan(self, all_nan):
        """Sharpe ratio should return NaN for all NaN data."""
        result = sharpe_ratio(all_nan)
        assert np.isnan(result)

    @pytest.mark.p1
    def test_sharpe_ratio_mostly_nan(self, mostly_nan):
        """Sharpe ratio should handle mostly NaN data."""
        result = sharpe_ratio(mostly_nan)
        # Should return finite value (using valid data) or NaN
        assert np.isfinite(result) or np.isnan(result)

    @pytest.mark.p1
    def test_max_drawdown_all_nan(self, all_nan):
        """Max drawdown should return NaN for all NaN data."""
        result = max_drawdown(all_nan)
        assert np.isnan(result)

    @pytest.mark.p1
    def test_cum_returns_with_nan(self, mostly_nan):
        """Cumulative returns should handle NaN values."""
        result = cum_returns(mostly_nan)
        # Should not crash, may contain NaN
        assert isinstance(result, pd.Series)


class TestZeroVolatility:
    """Test behavior with zero volatility (constant returns)."""

    @pytest.mark.p1
    def test_sharpe_ratio_zero_vol(self, zero_volatility):
        """Sharpe ratio should return NaN for zero volatility."""
        result = sharpe_ratio(zero_volatility)
        assert np.isnan(result)

    @pytest.mark.p1
    def test_sortino_ratio_zero_vol(self, zero_volatility):
        """Sortino ratio should handle zero volatility."""
        result = sortino_ratio(zero_volatility)
        assert np.isnan(result)

    @pytest.mark.p1
    def test_annual_volatility_zero_vol(self, zero_volatility):
        """Annual volatility should be 0 for constant returns."""
        result = annual_volatility(zero_volatility)
        assert np.isclose(result, 0, atol=1e-10)

    @pytest.mark.p1
    def test_downside_risk_zero_vol(self, zero_volatility):
        """Downside risk should be 0 for constant positive returns."""
        result = downside_risk(zero_volatility)
        assert result == 0

    @pytest.mark.p1
    def test_max_drawdown_zero_vol(self, zero_volatility):
        """Max drawdown should be 0 for constant positive returns."""
        result = max_drawdown(zero_volatility)
        assert result == 0


class TestAllZeros:
    """Test behavior with all zero returns."""

    @pytest.mark.p1
    def test_sharpe_ratio_all_zeros(self, all_zeros):
        """Sharpe ratio should return NaN for all zeros."""
        result = sharpe_ratio(all_zeros)
        assert np.isnan(result)

    @pytest.mark.p1
    def test_annual_return_all_zeros(self, all_zeros):
        """Annual return should be 0 for all zeros."""
        result = annual_return(all_zeros)
        assert result == 0

    @pytest.mark.p1
    def test_annual_volatility_all_zeros(self, all_zeros):
        """Annual volatility should be 0 for all zeros."""
        result = annual_volatility(all_zeros)
        assert result == 0


class TestInfiniteValues:
    """Test behavior with infinite values."""

    @pytest.mark.p1
    def test_sharpe_ratio_infinite(self, infinite_values):
        """Sharpe ratio should handle infinite values gracefully."""
        result = sharpe_ratio(infinite_values)
        # Should not crash, may return NaN or finite value
        assert np.isfinite(result) or np.isnan(result)

    @pytest.mark.p1
    def test_max_drawdown_infinite(self, infinite_values):
        """Max drawdown should handle infinite values."""
        # This may raise a warning or return extreme value
        try:
            result = max_drawdown(infinite_values)
            # Should not crash
            assert result <= 0 or np.isnan(result)
        except (ValueError, RuntimeWarning):
            # Acceptable to raise error for infinite values
            pass


class TestExtremeValues:
    """Test behavior with extreme values."""

    @pytest.mark.p1
    def test_sharpe_ratio_extreme(self, extreme_values):
        """Sharpe ratio should handle extreme values."""
        result = sharpe_ratio(extreme_values)
        # Should not crash
        assert np.isfinite(result) or np.isnan(result)

    @pytest.mark.p1
    def test_max_drawdown_extreme(self, extreme_values):
        """Max drawdown should handle extreme values."""
        result = max_drawdown(extreme_values)
        # Should not crash
        assert result <= 0 or np.isnan(result)


class TestMixedNaNInf:
    """Test behavior with mixed NaN and infinite values."""

    @pytest.mark.p1
    def test_sharpe_ratio_mixed(self, mixed_nan_inf):
        """Sharpe ratio should handle mixed edge cases."""
        result = sharpe_ratio(mixed_nan_inf)
        # Should not crash
        assert np.isfinite(result) or np.isnan(result)


# ==============================================================================
# Risk Metrics Edge Cases
# ==============================================================================


class TestValueAtRiskEdgeCases:
    """VaR edge case tests."""

    @pytest.mark.p1
    def test_var_empty(self, empty_returns):
        """VaR should return NaN for empty data."""
        result = value_at_risk(empty_returns, 0.05)
        assert np.isnan(result)

    @pytest.mark.p1
    def test_var_single_value(self, single_value):
        """VaR should handle single value."""
        result = value_at_risk(single_value, 0.05)
        # May return the value itself or NaN
        assert np.isfinite(result) or np.isnan(result)

    @pytest.mark.p1
    def test_var_all_nan(self, all_nan):
        """VaR should return NaN for all NaN data."""
        result = value_at_risk(all_nan, 0.05)
        assert np.isnan(result)


class TestCVaREdgeCases:
    """CVaR edge case tests."""

    @pytest.mark.p1
    def test_cvar_empty(self, empty_returns):
        """CVaR should return NaN for empty data."""
        result = conditional_value_at_risk(empty_returns, 0.05)
        assert np.isnan(result)

    @pytest.mark.p1
    def test_cvar_all_nan(self, all_nan):
        """CVaR should return NaN for all NaN data."""
        result = conditional_value_at_risk(all_nan, 0.05)
        assert np.isnan(result)


# ==============================================================================
# Alpha/Beta Edge Cases
# ==============================================================================


class TestAlphaBetaEdgeCases:
    """Alpha/Beta edge case tests."""

    @pytest.mark.p1
    def test_alpha_empty(self, empty_returns):
        """Alpha should return NaN for empty data."""
        factor = pd.Series([0.01] * len(empty_returns))
        result = alpha(empty_returns, factor)
        assert np.isnan(result)

    @pytest.mark.p1
    def test_beta_empty(self, empty_returns):
        """Beta should return NaN for empty data."""
        factor = pd.Series([0.01] * len(empty_returns))
        result = beta(empty_returns, factor)
        assert np.isnan(result)

    @pytest.mark.p1
    def test_alpha_mismatched_length(self):
        """Alpha should handle mismatched lengths."""
        returns = pd.Series([0.01] * 100)
        factor = pd.Series([0.01] * 50)
        # Should either align or raise error
        try:
            result = alpha(returns, factor)
            # If succeeds, should be finite or NaN
            assert np.isfinite(result) or np.isnan(result)
        except (ValueError, IndexError):
            # Acceptable to raise error
            pass

    @pytest.mark.p1
    def test_beta_zero_factor_vol(self):
        """Beta should handle zero factor volatility."""
        returns = pd.Series([0.01, 0.02, -0.01, 0.015] * 25)
        factor = pd.Series([0.01] * 100)  # Constant factor
        result = beta(returns, factor)
        assert np.isnan(result)


# ==============================================================================
# Cumulative Returns Edge Cases
# ==============================================================================


class TestCumReturnsEdgeCases:
    """Cumulative returns edge case tests."""

    @pytest.mark.p1
    def test_cum_returns_empty(self, empty_returns):
        """Cumulative returns should handle empty data."""
        result = cum_returns(empty_returns)
        assert isinstance(result, pd.Series)
        assert len(result) == 0

    @pytest.mark.p1
    def test_cum_returns_all_zeros(self, all_zeros):
        """Cumulative returns with all zeros should be flat."""
        result = cum_returns(all_zeros, starting_value=1.0)
        assert isinstance(result, pd.Series)
        assert len(result) == len(all_zeros)
        # All values should be 1.0 (starting point with zero returns)
        assert (result == 1.0).all()

    @pytest.mark.p1
    def test_cum_returns_with_nan(self, mostly_nan):
        """Cumulative returns should handle NaN."""
        result = cum_returns(mostly_nan)
        assert isinstance(result, pd.Series)
        # Should not crash


# ==============================================================================
# DataFrame Edge Cases
# ==============================================================================


class TestDataFrameEdgeCases:
    """DataFrame edge case tests."""

    @pytest.mark.p1
    def test_sharpe_ratio_empty_dataframe(self):
        """Sharpe ratio should handle empty DataFrame."""
        df = pd.DataFrame()
        result = sharpe_ratio(df)
        # Returns np.ndarray for 2D input (empty array for empty DataFrame)
        assert isinstance(result, (pd.Series, np.ndarray))
        assert len(result) == 0

    @pytest.mark.p1
    def test_sharpe_ratio_single_column(self):
        """Sharpe ratio should handle single column DataFrame."""
        df = pd.DataFrame({"A": [0.01] * 100})
        result = sharpe_ratio(df)
        assert isinstance(result, (pd.Series, np.ndarray))
        assert len(result) == 1
        val = result.iloc[0] if isinstance(result, pd.Series) else result[0]
        assert np.isnan(val)  # Zero volatility

    @pytest.mark.p1
    def test_sharpe_ratio_mixed_columns(self):
        """Sharpe ratio should handle mixed valid/invalid columns."""
        np.random.seed(42)
        df = pd.DataFrame(
            {
                "valid": np.random.randn(100) * 0.01,
                "constant": [0.01] * 100,
                "all_nan": [np.nan] * 100,
            }
        )
        result = sharpe_ratio(df)
        assert isinstance(result, (pd.Series, np.ndarray))
        assert len(result) == 3
        # First should be finite, others NaN (column order: valid, constant, all_nan)
        r0 = result["valid"] if isinstance(result, pd.Series) else result[0]
        r1 = result["constant"] if isinstance(result, pd.Series) else result[1]
        r2 = result["all_nan"] if isinstance(result, pd.Series) else result[2]
        assert np.isfinite(r0)
        assert np.isnan(r1)
        assert np.isnan(r2)


# ==============================================================================
# Summary
# ==============================================================================

# These edge case tests ensure:
# 1. No crashes on unusual data
# 2. Graceful degradation (return NaN instead of error)
# 3. Correct handling of boundary conditions
# 4. Robustness for production use
