"""Tests for conditional_sharpe_ratio with risk_free parameter.

Part of test_modified_ratios.py split.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_almost_equal

from fincore.metrics.ratios import conditional_sharpe_ratio

DECIMAL_PLACES = 4


def _make_series(values, start="2000-01-03", freq="D"):
    """Helper to create a pd.Series with a DatetimeIndex."""
    return pd.Series(
        np.array(values, dtype=float),
        index=pd.date_range(start, periods=len(values), freq=freq),
    )


@pytest.mark.p1
class TestConditionalSharpeRatioRiskFree:
    """Tests for conditional_sharpe_ratio with risk_free parameter."""

    # Use enough data so that cutoff=0.05 yields >= 2 tail observations
    returns = _make_series(np.random.RandomState(42).normal(-0.001, 0.02, 200).tolist())

    def test_risk_free_zero_default(self):
        """risk_free=0 should match the legacy behaviour."""
        result_default = conditional_sharpe_ratio(self.returns)
        result_explicit = conditional_sharpe_ratio(self.returns, risk_free=0)
        assert_almost_equal(result_default, result_explicit, DECIMAL_PLACES)

    def test_risk_free_shifts_mean(self):
        """Positive risk_free should decrease the conditional Sharpe ratio."""
        csr_0 = conditional_sharpe_ratio(self.returns, risk_free=0)
        csr_rf = conditional_sharpe_ratio(self.returns, risk_free=0.01)
        assert not np.isnan(csr_0)
        assert not np.isnan(csr_rf)
        assert csr_rf < csr_0

    def test_empty_returns_nan(self):
        result = conditional_sharpe_ratio(pd.Series([], dtype=float))
        assert np.isnan(result)

    def test_short_returns_nan(self):
        result = conditional_sharpe_ratio(_make_series([0.01]))
        assert np.isnan(result)
