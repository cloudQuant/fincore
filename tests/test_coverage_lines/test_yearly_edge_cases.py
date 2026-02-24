"""Tests for metrics/yearly.py edge cases.

Targets:
- metrics/yearly.py: 236 - annual_active_return NaN check
"""

import pandas as pd
import numpy as np


class TestAnnualActiveReturnNanCheck:
    """Test yearly.py line 236."""

    def test_annual_active_return_nan_result(self):
        """Line 236: either annual return is NaN."""
        from fincore.metrics.yearly import annual_active_return

        returns = pd.Series([], dtype=float, index=pd.DatetimeIndex([]))
        factor_returns = pd.Series([], dtype=float, index=pd.DatetimeIndex([]))

        result = annual_active_return(returns, factor_returns)
        assert np.isnan(result)
