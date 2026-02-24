"""Tests for empyrical.py coverage edge cases.

Targets:
- empyrical.py: 718 - regression_annual_return with NaN benchmark_annual
"""

import numpy as np
import pandas as pd
import pytest


class TestEmpyricalRegressionAnnualReturn:
    """Test empyrical.py line 718."""

    def test_regression_annual_return_nan_benchmark_annual(self):
        """Line 718: benchmark_annual is NaN."""
        from fincore.empyrical import Empyrical

        dates = pd.date_range("2020-01-01", periods=100, freq="B")
        returns = pd.Series(np.random.normal(0.001, 0.01, 100), index=dates)
        emp = Empyrical(returns=returns, factor_returns=returns)

        # Single value factor -> alpha/beta computable but annual_return is NaN
        factor = pd.Series([0.001], index=dates[:1])
        result = emp.regression_annual_return(returns, factor)
        assert np.isnan(result)
