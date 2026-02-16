"""Tests for missing coverage in Empyrical class.

This module covers edge cases and branches that were previously uncovered:
- __getattr__ fallback path (lines 220-230)
- _get_returns returning None (line 245)
- _get_factor_returns returning None (line 254)
- Various drawdown methods that may have been untested
- gpd_risk_estimates (line 446)
- perf_attrib with instance state fallback (lines 693-698)
- regression_annual_return with NaN handling (line 718)
"""

import unittest

import numpy as np
import pandas as pd

from fincore.empyrical import Empyrical


class EmpyricalMissingCoverageTestCase(unittest.TestCase):
    """Test cases for previously uncovered code paths in Empyrical."""

    def test_getattr_fallback_classmethod_registry(self):
        """Test __getattr__ fallback for CLASSMETHOD_REGISTRY."""
        emp = Empyrical()

        # Access a method through __getattr__ that's in CLASSMETHOD_REGISTRY
        # This triggers the fallback path in __getattr__ (lines 218-223)
        returns = pd.Series([0.01, 0.02, -0.01, 0.03])
        result = emp.sharpe_ratio(returns)
        self.assertIsNotNone(result)

    def test_getattr_fallback_static_methods(self):
        """Test __getattr__ fallback for STATIC_METHODS."""
        emp = Empyrical()

        # Access a static method through __getattr__ (lines 225-230)
        # Use annualization_factor which exists in STATIC_METHODS registry
        result = emp.annualization_factor("daily")
        self.assertEqual(result, 252)

    def test_getattr_raises_attribute_error(self):
        """Test __getattr__ raises AttributeError for unknown attributes."""
        emp = Empyrical()

        with self.assertRaises(AttributeError):
            _ = emp.nonexistent_method_xyz123

    def test_get_returns_none_when_no_returns(self):
        """Test _get_returns returns None when no returns available (line 245)."""
        emp = Empyrical()  # No returns passed
        result = emp._get_returns(None)
        self.assertIsNone(result)

    def test_get_factor_returns_none_when_no_factor_returns(self):
        """Test _get_factor_returns returns None when no factor_returns available (line 254)."""
        emp = Empyrical()  # No factor_returns passed
        result = emp._get_factor_returns(None)
        self.assertIsNone(result)

    def test_second_max_drawdown_days(self):
        """Test second_max_drawdown_days method (line 294)."""
        returns = pd.Series([0.1, -0.05, -0.08, 0.03, 0.04, -0.02])
        result = Empyrical.second_max_drawdown_days(returns)
        self.assertIsInstance(result, (int, float))

    def test_second_max_drawdown_recovery_days(self):
        """Test second_max_drawdown_recovery_days method (line 299)."""
        returns = pd.Series([0.1, -0.05, -0.08, 0.03, 0.04, -0.02, 0.05])
        result = Empyrical.second_max_drawdown_recovery_days(returns)
        self.assertIsInstance(result, (int, float))

    def test_third_max_drawdown_days(self):
        """Test third_max_drawdown_days method (line 304)."""
        returns = pd.Series([0.1, -0.05, -0.08, 0.03, -0.06, 0.04, -0.03, 0.05])
        result = Empyrical.third_max_drawdown_days(returns)
        self.assertIsInstance(result, (int, float))

    def test_third_max_drawdown_recovery_days(self):
        """Test third_max_drawdown_recovery_days method (line 309)."""
        returns = pd.Series([0.1, -0.05, -0.08, 0.03, -0.06, 0.04, -0.03, 0.02, 0.05])
        result = Empyrical.third_max_drawdown_recovery_days(returns)
        self.assertIsInstance(result, (int, float))

    def test_gpd_risk_estimates(self):
        """Test gpd_risk_estimates method (line 446)."""
        np.random.seed(42)
        # Create returns with heavy tail for GPD
        returns = pd.Series(np.random.standard_t(3, 1000) * 0.01)
        result = Empyrical.gpd_risk_estimates(returns)
        # gpd_risk_estimates returns a Series with VaR and ES
        self.assertIsInstance(result, pd.Series)

    def test_perf_attrib_with_instance_state(self):
        """Test perf_attrib using instance state fallback (lines 693-698)."""
        returns = pd.Series([0.01, 0.02, 0.015, -0.005])
        dts = returns.index
        tickers = ["stock1", "stock2"]
        styles = ["risk_factor1", "risk_factor2"]

        index = pd.MultiIndex.from_product([dts, tickers], names=["dt", "ticker"])
        positions = pd.Series([0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5], index=index)

        factor_returns = pd.DataFrame(
            columns=styles,
            index=dts,
            data={"risk_factor1": [0.01, 0.02, 0.015, -0.005], "risk_factor2": [0.005, 0.01, 0.008, -0.002]},
        )

        factor_loadings = pd.DataFrame(
            columns=styles,
            index=index,
            data={
                "risk_factor1": [0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25],
                "risk_factor2": [0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25],
            },
        )

        # Create Empyrical instance with positions and factor_returns
        emp = Empyrical(
            returns=returns, positions=positions, factor_returns=factor_returns, factor_loadings=factor_loadings
        )

        # Call perf_attrib without passing positions/factor_returns
        # This should use instance state (lines 693-698)
        result = emp.perf_attrib()
        self.assertIsNotNone(result)
        self.assertEqual(len(result), 2)  # Returns (exposures, perf_attrib_output)

    def test_regression_annual_return_with_nan_alpha(self):
        """Test regression_annual_return with NaN alpha (line 714-715)."""
        returns = pd.Series([np.nan, np.nan, np.nan])
        factor_returns = pd.Series([0.01, 0.02, 0.015])

        result = Empyrical.regression_annual_return(returns, factor_returns)
        self.assertTrue(np.isnan(result))

    def test_regression_annual_return_with_nan_benchmark(self):
        """Test regression_annual_return with NaN benchmark annual return (line 717-718)."""
        returns = pd.Series([0.01, 0.02, 0.015])
        factor_returns = pd.Series([np.nan, np.nan, np.nan])

        result = Empyrical.regression_annual_return(returns, factor_returns)
        self.assertTrue(np.isnan(result))

    def test_instance_with_returns_creates_context(self):
        """Test that creating Empyrical with returns creates AnalysisContext."""
        returns = pd.Series([0.01, 0.02, -0.01, 0.03])
        factor_returns = pd.Series([0.005, 0.01, -0.005, 0.015])

        emp = Empyrical(returns=returns, factor_returns=factor_returns)
        self.assertIsNotNone(emp._ctx)

    def test_instance_without_returns_no_context(self):
        """Test that creating Empyrical without returns doesn't create AnalysisContext."""
        emp = Empyrical()
        self.assertIsNone(emp._ctx)

    def test_cagr_alias(self):
        """Test cagr alias method."""
        returns = pd.Series([0.01, 0.02, 0.015, -0.005, 0.03])
        result = Empyrical.cagr(returns)
        self.assertIsInstance(result, (float, np.floating))

    def test_zipline_constant_exists(self):
        """Test ZIPLINE constant is defined."""
        from fincore.empyrical import ZIPLINE

        # ZIPLINE should be False unless zipline is installed
        self.assertIsInstance(ZIPLINE, bool)


if __name__ == "__main__":
    unittest.main()
