"""Tests for Empyrical class initialization.

This file tests initialization edge cases and context creation.
"""

from __future__ import annotations

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
        returns = pd.Series([0.01, 0.02, 0.015])  # No DatetimeIndex

        # Should not raise, just log and set _ctx to None
        emp = Empyrical(returns=returns)

        # Should still have the returns attribute
        assert emp.returns is not None
        # _ctx may be None if AnalysisContext creation failed

    def test_init_with_invalid_positions_handles_gracefully(self):
        """Test initializing with invalid positions handles exception gracefully."""
        # Create returns with proper index but positions without matching index
        returns = pd.Series([0.01, 0.02, 0.015])
        positions = pd.DataFrame({"invalid": [1, 2]})  # Mismatched shape

        # Should not raise, just log debug message
        emp = Empyrical(returns=returns, positions=positions)

        # Should still have the returns attribute
        assert emp.returns is not None


class TestEmpyricalInitWithContextFailure:
    """Test Empyrical initialization when AnalysisContext fails."""

    def test_init_with_none_returns(self):
        """Test initializing with None returns doesn't create context."""
        emp = Empyrical(returns=None)

        # Should not have _ctx attribute when returns is None
        assert not hasattr(emp, "_ctx") or emp._ctx is None

    def test_init_without_kwargs(self):
        """Test initializing without any arguments."""
        emp = Empyrical()

        # Should not have _ctx attribute
        assert not hasattr(emp, "_ctx") or emp._ctx is None


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
