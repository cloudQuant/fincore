"""Tests for metrics/ratios.py edge cases.

Targets:
- metrics/ratios.py: 417 - mar_ratio all NaN after cleaning
"""

import numpy as np
import pandas as pd
import pytest

from fincore.exceptions import NumericalError


class TestMarRatioAllNan:
    """Test ratios.py line 417."""

    def test_mar_ratio_all_nan_after_cleaning(self):
        """Line 417: all NaN returns are rejected before any cleaning."""
        from fincore.metrics.ratios import mar_ratio

        returns = pd.Series([np.nan, np.nan, np.nan])
        with pytest.raises(NumericalError, match="finite"):
            mar_ratio(returns)
