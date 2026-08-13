"""Tests for empyrical treynor_ratio missing coverage (line 718)."""

import numpy as np
import pandas as pd
import pytest

from fincore.empyrical import Empyrical
from fincore.exceptions import NumericalError


class TestTreynorRatioCoverage:
    """Tests for treynor_ratio edge cases."""

    def test_treynor_ratio_nan_benchmark_annual(self):
        """Test treynor_ratio rejects non-finite factor returns."""
        emp = Empyrical()

        returns = pd.Series([0.01, 0.02, 0.015])
        factor_returns = pd.Series([np.nan, np.nan, np.nan])

        with pytest.raises(NumericalError, match="finite"):
            emp.treynor_ratio(returns, factor_returns)

    def test_treynor_ratio_zero_beta(self):
        """Test treynor_ratio when beta is zero."""
        emp = Empyrical()

        # Create returns and factor with zero variance
        returns = pd.Series([0.01, 0.02, 0.015])
        factor_returns = pd.Series([0.01, 0.01, 0.01])  # Zero variance

        result = emp.treynor_ratio(returns, factor_returns)

        # Should return NaN when beta is zero or NaN
        assert np.isnan(result)
