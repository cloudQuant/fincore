"""Test to cover line 718 in empyrical.py.

Line 718 is: return np.nan (when benchmark_annual is NaN but alpha/beta are not)

Note: Line 718 is covered by other tests in the test suite. The complex
mocking approach needed to reach this specific edge case is fragile and
unnecessary since the code path is already tested through normal usage.
"""

import numpy as np
import pandas as pd
import pytest

from fincore.empyrical import Empyrical


class TestEmpyricalLine718Coverage:
    """Test coverage for line 718 in empyrical.py."""

    def test_regression_annual_return_direct_mock(self):
        """Test line 718 by directly mocking the _yearly module."""
        returns = pd.Series([0.01, 0.02, -0.01, 0.015, 0.005])
        benchmark = pd.Series([0.005, 0.01, -0.005, 0.008, 0.002])

        emp = Empyrical(returns=returns, factor_returns=benchmark)

        # Import the real modules first
        from fincore.metrics import alpha_beta

        # Create a mock yearly module
        from unittest.mock import MagicMock, patch

        mock_yr = MagicMock()

        # annual_return should be called once for factor_returns at line 716
        # We return NaN to trigger line 718
        mock_yr.annual_return.return_value = np.nan

        with patch('fincore.empyrical._resolve_module') as mock_resolve:
            def resolve_side_effect(name):
                if name == '_alpha_beta':
                    return alpha_beta
                elif name == '_yearly':
                    return mock_yr
                return MagicMock()

            mock_resolve.side_effect = resolve_side_effect

            result = emp.regression_annual_return()

            # Should return NaN at line 718
            assert pd.isna(result), f"Expected NaN but got {result}"

    def test_regression_annual_return_negative_ending_value(self):
        """Test with negative ending value that produces -1.0 annual return.

        Note: This doesn't hit line 718 since annual_return returns -1.0 not NaN
        for negative ending values. This test documents that behavior.
        """
        returns = pd.Series([0.05, 0.03, -0.02, 0.04, -0.01])
        benchmark = pd.Series([-0.3, -0.2, -0.15, -0.25, -0.1])

        emp = Empyrical(returns=returns, factor_returns=benchmark)
        result = emp.regression_annual_return()

        # benchmark_annual will be -1.0, not NaN, so line 718 not hit
        # Result will be calculated normally
        assert not pd.isna(result)
