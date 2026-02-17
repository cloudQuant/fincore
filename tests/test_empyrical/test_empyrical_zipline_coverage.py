"""Tests for zipline-related code in Empyrical."""

from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from fincore.empyrical import Empyrical


class TestEmpyricalZiplineImport:
    """Test zipline import behavior (lines 163-170)."""

    def test_zipline_import_sets_global(self):
        """Test that importing zipline sets ZIPLINE global and Equity/Future (lines 168-170)."""
        # Mock the zipline.assets module
        mock_zip_assets = MagicMock()
        mock_equity = MagicMock()
        mock_future = MagicMock()

        mock_zip_assets.Equity = mock_equity
        mock_zip_assets.Future = mock_future

        with patch("importlib.import_module", return_value=mock_zip_assets):
            # Force re-import by modifying the module's import logic
            import importlib

            import fincore.empyrical

            # Reload the module to trigger the import block
            importlib.reload(fincore.empyrical)

            # Verify ZIPLINE is set to True
            assert fincore.empyrical.ZIPLINE is True
            # Verify Equity and Future are set from zipline
            assert fincore.empyrical.Equity is mock_equity
            assert fincore.empyrical.Future is mock_future


class TestEmpyricalInitContextError:
    """Test AnalysisContext creation error handling (lines 204-207)."""

    def test_init_continues_when_context_creation_fails(self):
        """Test that __init__ continues when AnalysisContext creation fails (line 204-207)."""
        returns = pd.Series([0.01, 0.02, -0.01])

        # Mock AnalysisContext.__init__ to raise an exception
        with patch(
            "fincore.core.context.AnalysisContext.__init__", side_effect=RuntimeError("Context creation failed")
        ):
            # Should not raise, but log debug message and continue
            emp = Empyrical(returns=returns)

            # Instance should be created
            assert emp.returns is not None
            assert emp._ctx is None


class TestEmpyricalGetattrFallback:
    """Test __getattr__ fallback behavior (lines 218-223, 227-230)."""

    def test_getattr_fallback_for_classmethod_registry(self):
        """Test __getattr__ fallback for CLASSMETHOD_REGISTRY (lines 218-223)."""
        returns = pd.Series([0.01, 0.02, -0.01])
        emp = Empyrical(returns=returns)

        # Accessing through instance should work via __getattr__ if not directly set
        # The _dual_method decorator handles most cases, but __getattr__ is a fallback
        # Test a method that exists in CLASSMETHOD_REGISTRY
        result = emp.cum_returns
        # Just verify it doesn't raise and returns a callable or result
        assert callable(result) or isinstance(result, (int, float, pd.Series))

    def test_getattr_fallback_for_static_methods(self):
        """Test __getattr__ fallback for STATIC_METHODS (lines 227-230)."""
        emp = Empyrical()

        # Access a static method that should be in STATIC_METHODS
        result = emp.annualization_factor
        # Just verify it doesn't raise and returns a callable
        assert callable(result)

    def test_getattr_raises_for_unknown_attribute(self):
        """Test __getattr__ raises AttributeError for unknown attributes (line 232)."""
        returns = pd.Series([0.01, 0.02, -0.01])
        emp = Empyrical(returns=returns)

        with pytest.raises(AttributeError, match="has no attribute"):
            emp.unknown_method_12345()


class TestEmpyricalRegressionAnnualReturnNan:
    """Test regression_annual_return with NaN values (line 718)."""

    def test_regression_annual_return_returns_nan_for_nan_benchmark(self):
        """Test regression_annual_return returns NaN when benchmark annual return is NaN (line 718)."""
        # Create returns with finite data
        returns = pd.Series([0.01, 0.02, -0.01, 0.015, 0.005])

        # Create benchmark that will produce NaN annual return
        # We need to create a scenario where annual_return returns NaN
        benchmark = pd.Series([np.nan, np.nan, np.nan, np.nan, np.nan])

        emp = Empyrical(returns=returns, factor_returns=benchmark)
        result = emp.regression_annual_return()

        # Should return NaN
        assert pd.isna(result)
