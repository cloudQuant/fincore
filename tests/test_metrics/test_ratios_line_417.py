"""Test to cover line 417 in ratios.py.

Line 417 is a defensive check that handles the case where all returns are NaN
after filtering, but max_drawdown returned a negative value (which shouldn't
happen with normal input, but could happen with edge cases or modified behavior).
"""

import numpy as np
import pandas as pd
import pytest
from unittest.mock import patch

from fincore.metrics import ratios


class TestMarRatioLine417:
    """Test coverage for line 417 in ratios.py."""

    def test_mar_ratio_line_417_with_mock(self):
        """Test mar_ratio line 417 by mocking max_drawdown to return negative value.

        This forces the path where:
        1. len(returns) >= 2 (passes line 406)
        2. max_dd < 0 (mocked to pass line 410)
        3. All values are NaN (returns_clean is empty, hits line 417)
        """
        returns = np.array([np.nan, np.nan, np.nan, np.nan, np.nan])

        # Mock max_drawdown to return a negative value
        # This forces execution past line 410 to reach line 417
        with patch('fincore.metrics.drawdown.max_drawdown', return_value=-0.05):
            result = ratios.mar_ratio(returns)

            # Should return NaN at line 417 when returns_clean is empty
            assert np.isnan(result)

    def test_mar_ratio_normal_case(self):
        """Test mar_ratio with valid data."""
        returns = pd.Series([0.01, -0.05, 0.03, 0.02, -0.03])
        result = ratios.mar_ratio(returns)
        # Should return a valid number
        assert isinstance(result, (int, float, np.floating))

    def test_mar_ratio_all_nan_normal_flow(self):
        """Test mar_ratio with all NaN returns (normal flow through line 410)."""
        result = ratios.mar_ratio(np.array([np.nan, np.nan, np.nan]))
        # Returns NaN at line 410 because max_dd >= 0
        assert np.isnan(result)
