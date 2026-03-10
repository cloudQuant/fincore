"""Tests for ratios and yearly line coverage.

Part of test_exact_line_coverage.py split - Ratios and yearly tests with P2 markers.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from fincore.metrics.ratios import mar_ratio
from fincore.metrics.yearly import annual_active_return


@pytest.mark.p2
class TestRatiosYearlyLineCoverage:
    """Test ratios and yearly edge cases for exact line coverage."""

    def test_mar_ratio_line_417(self):
        """ratios.py line 417: return np.nan when returns_clean is empty."""
        # All NaN values -> returns_clean is empty
        returns = pd.Series([np.nan, np.nan, np.nan])
        result = mar_ratio(returns)
        assert np.isnan(result)

    def test_annual_active_return_line_236(self):
        """yearly.py line 236: return np.nan when either annual return is NaN."""
        # Empty series -> annual_return is NaN
        returns = pd.Series([], dtype=float, index=pd.DatetimeIndex([], freq="D"))
        factor_returns = pd.Series([], dtype=float, index=pd.DatetimeIndex([], freq="D"))
        result = annual_active_return(returns, factor_returns)
        assert np.isnan(result)
