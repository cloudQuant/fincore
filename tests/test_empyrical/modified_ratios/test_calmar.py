"""Tests for calmar_ratio with risk_free parameter.

Part of test_modified_ratios.py split.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_almost_equal

from fincore.metrics.ratios import calmar_ratio

DECIMAL_PLACES = 4


def _make_series(values, start="2000-01-03", freq="D"):
    """Helper to create a pd.Series with a DatetimeIndex."""
    return pd.Series(
        np.array(values, dtype=float),
        index=pd.date_range(start, periods=len(values), freq=freq),
    )


@pytest.mark.p1
class TestCalmarRatioRiskFree:
    """Tests for calmar_ratio with risk_free parameter."""

    returns = _make_series([0.01, 0.02, -0.05, 0.03, 0.01, -0.02, 0.04, 0.01, 0.02, -0.03])

    def test_risk_free_zero_default(self):
        """risk_free=0 should match the legacy behaviour."""
        result_default = calmar_ratio(self.returns)
        result_explicit = calmar_ratio(self.returns, risk_free=0)
        assert_almost_equal(result_default, result_explicit, DECIMAL_PLACES)

    def test_risk_free_reduces_ratio(self):
        """Positive risk_free should reduce the Calmar ratio."""
        ratio_0 = calmar_ratio(self.returns, risk_free=0)
        ratio_rf = calmar_ratio(self.returns, risk_free=0.03)
        assert ratio_rf < ratio_0

    def test_risk_free_large_makes_negative(self):
        """A large risk_free can make the ratio negative."""
        np.random.seed(42)
        long_returns = _make_series(np.random.normal(0.0003, 0.01, 252).tolist())
        ratio = calmar_ratio(long_returns, risk_free=0.5)
        assert ratio < 0

    def test_empty_returns_nan(self):
        result = calmar_ratio(pd.Series([], dtype=float))
        assert np.isnan(result)

    def test_single_positive_return(self):
        """Single return has no drawdown => NaN."""
        result = calmar_ratio(_make_series([0.05]))
        assert np.isnan(result)
