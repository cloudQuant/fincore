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
        """Test line 718 by patching annual_return to return NaN for benchmark."""
        from unittest.mock import patch

        returns = pd.Series([0.01, 0.02, -0.01, 0.015, 0.005])
        benchmark = pd.Series([0.005, 0.01, -0.005, 0.008, 0.002])

        emp = Empyrical(returns=returns, factor_returns=benchmark)

        # Patch annual_return when called with factor_returns; return NaN to trigger line 718
        with patch("fincore.metrics.yearly.annual_return", return_value=np.nan):
            result = emp.regression_annual_return()

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
