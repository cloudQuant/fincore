"""Tests for Empyrical __getattr__ fallback mechanism.

This file tests the registry-backed attribute access.
"""

from __future__ import annotations

import pandas as pd
import pytest

from fincore.empyrical import Empyrical


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


class TestEmpyricalGetattrStaticMethods:
    """Test __getattr__ fallback for static methods registry."""

    def test_getattr_for_static_methods_registry(self):
        """Test __getattr__ for static methods registry."""
        emp = Empyrical()

        # Access a static method like '_flatten' from STATIC_METHODS
        # This should trigger __getattr__ and cache it on the class
        result = emp._flatten
        assert callable(result)

    def test_getattr_unknown_attribute_raises(self):
        """Test __getattr__ raises for unknown attributes."""
        emp = Empyrical()

        with pytest.raises(AttributeError, match="has no attribute"):
            _ = emp.completely_nonexistent_method_xyz

    def test_getattr_static_methods_via_getattribute_bypass(self):
        """Test __getattr__ for static methods by bypassing descriptor."""
        emp = Empyrical()

        # Use object.__getattribute__ to bypass the descriptor and directly trigger __getattr__
        # We access an attribute that exists in STATIC_METHODS registry
        # Delete from class dict first if present to force __getattr__
        if "_flatten" in emp.__class__.__dict__:
            delattr(emp.__class__, "_flatten")

        try:
            # Now accessing should trigger __getattr__
            result = emp._flatten
            assert callable(result)
        finally:
            # Restore the method for other tests
            from fincore.metrics.basic import flatten

            emp.__class__._flatten = staticmethod(flatten)

    def test_getattr_classmethod_via_subclass(self):
        """Test __getattr__ for classmethods via dynamic subclass."""

        # Create a dynamic subclass that doesn't have descriptors set up
        class DynamicSubclass(Empyrical):
            pass

        emp = DynamicSubclass()

        # Access a method that should be in CLASSMETHOD_REGISTRY
        # Since DynamicSubclass doesn't have the descriptor, it should fall through to __getattr__
        # Note: This still might not work because descriptors are inherited from parent
        # Let's try a different approach - delete and then access
        if "cum_returns" in emp.__class__.__dict__:
            delattr(emp.__class__, "cum_returns")

        try:
            result = emp.cum_returns
            assert callable(result)
        finally:
            # Restore for other tests
            from fincore.metrics.returns import cum_returns

            emp.__class__.cum_returns = staticmethod(cum_returns)

    def test_getattr_direct_call_with_deleted_descriptor(self):
        """Test __getattr__ by directly calling it after deleting descriptor."""
        emp = Empyrical()

        # Delete a CLASSMETHOD_REGISTRY method from the class
        if "cum_returns" in emp.__class__.__dict__:
            delattr(emp.__class__, "cum_returns")

        try:
            # Directly call __getattr__ to trigger the fallback path
            result = emp.__getattr__("cum_returns")
            assert callable(result)
        finally:
            # Restore for other tests
            from fincore.metrics.returns import cum_returns

            emp.__class__.cum_returns = staticmethod(cum_returns)

    def test_getattr_static_method_direct_call(self):
        """Test __getattr__ for STATIC_METHODS by directly calling it."""
        emp = Empyrical()

        # Delete a STATIC_METHODS method from the class
        if "_flatten" in emp.__class__.__dict__:
            delattr(emp.__class__, "_flatten")

        try:
            # Directly call __getattr__ to trigger the fallback path
            result = emp.__getattr__("_flatten")
            assert callable(result)
        finally:
            # Restore for other tests
            from fincore.metrics.basic import flatten

            emp.__class__._flatten = staticmethod(flatten)


class TestEmpyricalGetattrClassmethods:
    """Test __getattr__ fallback for classmethod registry."""

    def test_getattr_for_classmethod_via_instance(self):
        """Test __getattr__ for classmethod registry methods."""
        returns = pd.Series([0.01, 0.02, 0.015])
        emp = Empyrical(returns=returns)

        # Access a method via __getattr__ that's in CLASSMETHOD_REGISTRY
        # First access goes through __getattr__, subsequent accesses use cached value
        result = emp.cum_returns
        assert callable(result)

        # Second access should use cached value
        result2 = emp.cum_returns
        assert result is result2


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


class TestEmpyricalDrawdownPeriod:
    """Test get_max_drawdown_period method."""

    def test_get_max_drawdown_period_with_returns(self):
        """Test get_max_drawdown_period with returns."""
        returns = pd.Series(
            [0.01, 0.02, -0.05, 0.01, 0.02, 0.03],
            index=pd.date_range("2020-01-01", periods=6, freq="D"),
        )
        emp = Empyrical(returns=returns)

        result = emp.get_max_drawdown_period()

        # Should return start and end dates of max drawdown period
        assert result is not None
        assert len(result) == 2

    def test_get_max_drawdown_period_class_method(self):
        """Test get_max_drawdown_period as class method."""
        returns = pd.Series(
            [0.01, 0.02, -0.05, 0.01, 0.02, 0.03],
            index=pd.date_range("2020-01-01", periods=6, freq="D"),
        )

        result = Empyrical.get_max_drawdown_period(returns)

        assert result is not None
        assert len(result) == 2
