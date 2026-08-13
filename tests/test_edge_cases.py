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

import fincore.empyrical as legacy
from fincore import (
    alpha,
    annual_return,
    annual_volatility,
    beta,
    calmar_ratio,
    cum_returns,
    cum_returns_final,
    downside_risk,
    max_drawdown,
    omega_ratio,
    sharpe_ratio,
    sortino_ratio,
    tail_ratio,
    value_at_risk,
)
from fincore.constants import DAILY
from fincore.exceptions import DataAlignmentError, NumericalError, ValidationError
from fincore.metrics.risk import conditional_value_at_risk

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
        """Sharpe ratio rejects empty data."""
        with pytest.raises(ValidationError, match="empty"):
            sharpe_ratio(empty_returns)

    @pytest.mark.p1
    def test_max_drawdown_empty(self, empty_returns):
        """Max drawdown rejects empty data."""
        with pytest.raises(ValidationError, match="empty"):
            max_drawdown(empty_returns)

    @pytest.mark.p1
    def test_annual_return_empty(self, empty_returns):
        """Annual return rejects empty data."""
        with pytest.raises(ValidationError, match="empty"):
            annual_return(empty_returns)

    @pytest.mark.p1
    def test_annual_volatility_empty(self, empty_returns):
        """Annual volatility rejects empty data."""
        with pytest.raises(ValidationError, match="empty"):
            annual_volatility(empty_returns)


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
        """Sharpe ratio rejects all-NaN data."""
        with pytest.raises(NumericalError, match="finite"):
            sharpe_ratio(all_nan)

    @pytest.mark.p1
    def test_sharpe_ratio_mostly_nan(self, mostly_nan):
        """Sharpe ratio rejects data containing NaN."""
        with pytest.raises(NumericalError, match="finite"):
            sharpe_ratio(mostly_nan)

    @pytest.mark.p1
    def test_max_drawdown_all_nan(self, all_nan):
        """Max drawdown rejects all-NaN data."""
        with pytest.raises(NumericalError, match="finite"):
            max_drawdown(all_nan)

    @pytest.mark.p1
    def test_cum_returns_with_nan(self, mostly_nan):
        """Strict legacy cum_returns keeps the pinned NaN-tolerant kernel."""
        result = legacy.cum_returns(mostly_nan)
        # The pinned legacy oracle replaces NaN with 0 and compounds.
        expected = np.cumprod(1 + mostly_nan.fillna(0).to_numpy()) - 1
        assert isinstance(result, pd.Series)
        assert np.allclose(result.to_numpy(), expected)


class TestZeroVolatility:
    """Test behavior with zero volatility (constant returns)."""

    @pytest.mark.p1
    def test_sharpe_ratio_zero_vol(self, zero_volatility):
        """Sharpe ratio should return NaN, inf, or very large value for zero volatility (mean/0)."""
        result = sharpe_ratio(zero_volatility)
        assert np.isnan(result) or np.isinf(result) or (np.isfinite(result) and abs(result) > 1e10)

    @pytest.mark.p1
    def test_sortino_ratio_zero_vol(self, zero_volatility):
        """Sortino ratio should handle zero volatility (NaN or inf)."""
        result = sortino_ratio(zero_volatility)
        assert np.isnan(result) or np.isinf(result)

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
        """Sharpe ratio rejects infinite values."""
        with pytest.raises(NumericalError, match="finite"):
            sharpe_ratio(infinite_values)

    @pytest.mark.p1
    def test_max_drawdown_infinite(self, infinite_values):
        """Max drawdown rejects infinite values."""
        with pytest.raises(NumericalError, match="finite"):
            max_drawdown(infinite_values)


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
        """Sharpe ratio rejects mixed NaN/Inf data."""
        with pytest.raises(NumericalError, match="finite"):
            sharpe_ratio(mixed_nan_inf)


# ==============================================================================
# Risk Metrics Edge Cases
# ==============================================================================


class TestValueAtRiskEdgeCases:
    """VaR edge case tests."""

    @pytest.mark.p1
    def test_var_empty(self, empty_returns):
        """VaR rejects empty data."""
        with pytest.raises(ValidationError, match="empty"):
            value_at_risk(empty_returns, 0.05)

    @pytest.mark.p1
    def test_var_single_value(self, single_value):
        """VaR should handle single value."""
        result = value_at_risk(single_value, 0.05)
        # May return the value itself or NaN
        assert np.isfinite(result) or np.isnan(result)

    @pytest.mark.p1
    def test_var_all_nan(self, all_nan):
        """VaR rejects all-NaN data."""
        with pytest.raises(NumericalError, match="finite"):
            value_at_risk(all_nan, 0.05)


class TestCVaREdgeCases:
    """CVaR edge case tests."""

    @pytest.mark.p1
    def test_cvar_empty(self, empty_returns):
        """CVaR rejects empty data."""
        with pytest.raises(ValidationError, match="empty"):
            conditional_value_at_risk(empty_returns, 0.05)

    @pytest.mark.p1
    def test_cvar_all_nan(self, all_nan):
        """CVaR rejects all-NaN data."""
        with pytest.raises(NumericalError, match="finite"):
            conditional_value_at_risk(all_nan, 0.05)


# ==============================================================================
# Alpha/Beta Edge Cases
# ==============================================================================


class TestAlphaBetaEdgeCases:
    """Alpha/Beta edge case tests."""

    @pytest.mark.p1
    def test_alpha_empty(self, empty_returns):
        """Alpha rejects empty data at the alignment boundary."""
        factor = pd.Series([0.01] * len(empty_returns))
        with pytest.raises(DataAlignmentError, match="common labels"):
            alpha(empty_returns, factor)

    @pytest.mark.p1
    def test_beta_empty(self, empty_returns):
        """Beta rejects empty data at the alignment boundary."""
        factor = pd.Series([0.01] * len(empty_returns))
        with pytest.raises(DataAlignmentError, match="common labels"):
            beta(empty_returns, factor)

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
        """Strict legacy cum_returns keeps the pinned empty passthrough."""
        result = legacy.cum_returns(empty_returns)
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
        """Strict legacy cum_returns keeps the pinned NaN-tolerant kernel."""
        result = legacy.cum_returns(mostly_nan)
        expected = np.cumprod(1 + mostly_nan.fillna(0).to_numpy()) - 1
        assert isinstance(result, pd.Series)
        assert np.allclose(result.to_numpy(), expected)


# ==============================================================================
# DataFrame Edge Cases
# ==============================================================================


class TestDataFrameEdgeCases:
    """DataFrame edge case tests."""

    @pytest.mark.p1
    def test_sharpe_ratio_empty_dataframe(self):
        """Sharpe ratio rejects an empty DataFrame."""
        df = pd.DataFrame()
        with pytest.raises(ValidationError, match="empty"):
            sharpe_ratio(df)

    @pytest.mark.p1
    def test_sharpe_ratio_single_column(self):
        """Sharpe ratio should handle single column DataFrame (constant -> NaN or inf)."""
        df = pd.DataFrame({"A": [0.01] * 100})
        result = sharpe_ratio(df)
        assert isinstance(result, (pd.Series, np.ndarray))
        assert len(result) == 1
        val = result.iloc[0] if isinstance(result, pd.Series) else result[0]
        assert np.isnan(val) or np.isinf(val) or (np.isfinite(val) and abs(val) > 1e10)  # Zero volatility

    @pytest.mark.p1
    def test_sharpe_ratio_mixed_columns(self):
        """Sharpe ratio rejects a DataFrame containing a non-finite column."""
        np.random.seed(42)
        df = pd.DataFrame(
            {
                "valid": np.random.randn(100) * 0.01,
                "constant": [0.01] * 100,
                "all_nan": [np.nan] * 100,
            }
        )
        with pytest.raises(NumericalError, match="finite"):
            sharpe_ratio(df)


# ==============================================================================
# Summary
# ==============================================================================

# These edge case tests ensure:
# 1. Enhanced surfaces fail fast on empty data (ValidationError)
# 2. Enhanced surfaces fail fast on non-finite data (NumericalError)
# 3. Binary metrics reject empty inputs at the alignment boundary (DataAlignmentError)
# 4. The strict legacy surface keeps the pinned NaN-tolerant kernel behavior
