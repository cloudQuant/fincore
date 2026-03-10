"""Tests for kappa_three_ratio and omega_ratio.

Part of test_modified_ratios.py split.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_almost_equal

from fincore.metrics.ratios import kappa_three_ratio, omega_ratio

DECIMAL_PLACES = 4


def _make_series(values, start="2000-01-03", freq="D"):
    """Helper to create a pd.Series with a DatetimeIndex."""
    return pd.Series(
        np.array(values, dtype=float),
        index=pd.date_range(start, periods=len(values), freq=freq),
    )


@pytest.mark.p1
class TestKappaThreeRatio:
    """Tests for kappa_three_ratio after formula rework."""

    returns = _make_series([0.01, 0.02, -0.05, 0.03, 0.01, -0.02, 0.04, 0.01, 0.02, -0.03])
    positive_returns = _make_series([0.01, 0.02, 0.01, 0.015, 0.01, 0.02, 0.01, 0.015, 0.01, 0.01])

    def test_hand_calculation(self):
        """Verify against a hand calculation with mar=0."""
        r = np.array([0.01, 0.02, -0.05, 0.03, 0.01, -0.02, 0.04, 0.01, 0.02, -0.03])
        mu = np.mean(r)
        mar = 0.0
        downside = np.maximum(0, mar - r)
        lpm3 = np.mean(downside**3)
        expected = (mu - mar) / (lpm3 ** (1.0 / 3.0))
        result = kappa_three_ratio(self.returns, mar=0.0)
        assert_almost_equal(result, expected, DECIMAL_PLACES)

    def test_positive_returns_inf(self):
        """All positive returns with mar=0 => LPM3=0 => inf."""
        result = kappa_three_ratio(self.positive_returns, mar=0.0)
        assert np.isinf(result) and result > 0

    def test_mar_above_all_returns(self):
        """When mar is above all returns, expect negative ratio."""
        result = kappa_three_ratio(self.returns, mar=0.10)
        assert result < 0

    def test_empty_returns_nan(self):
        result = kappa_three_ratio(pd.Series([], dtype=float))
        assert np.isnan(result)

    def test_single_return_nan(self):
        result = kappa_three_ratio(_make_series([0.01]))
        assert np.isnan(result)

    def test_risk_free_parameter_accepted(self):
        """risk_free parameter exists for API compatibility."""
        result = kappa_three_ratio(self.returns, risk_free=0.01)
        assert isinstance(result, (float, np.floating))


@pytest.mark.p1
class TestOmegaRatio:
    """Tests for omega_ratio after improvements."""

    returns = _make_series([0.01, 0.02, -0.05, 0.03, 0.01, -0.02, 0.04, 0.01, 0.02, -0.03])

    def test_threshold_zero(self):
        """With threshold=0, omega = sum(gains) / sum(|losses|)."""
        r = np.array([0.01, 0.02, -0.05, 0.03, 0.01, -0.02, 0.04, 0.01, 0.02, -0.03])
        gains = r[r > 0].sum()
        losses = -r[r < 0].sum()
        expected = gains / losses
        result = omega_ratio(self.returns, risk_free=0.0, required_return=0.0)
        assert_almost_equal(result, expected, DECIMAL_PLACES)

    def test_risk_free_shifts_threshold(self):
        """Positive risk_free raises the threshold, reducing the ratio."""
        omega_0 = omega_ratio(self.returns, risk_free=0.0)
        omega_rf = omega_ratio(self.returns, risk_free=0.01)
        assert omega_rf < omega_0

    def test_all_positive_returns(self):
        """All returns above threshold => denom is 0 => NaN."""
        positive = _make_series([0.01, 0.02, 0.03, 0.01, 0.02])
        result = omega_ratio(positive, risk_free=0.0, required_return=0.0)
        assert np.isnan(result)

    def test_empty_returns_nan(self):
        result = omega_ratio(pd.Series([], dtype=float))
        assert np.isnan(result)

    def test_required_return_negative_one(self):
        """required_return <= -1 should return NaN."""
        result = omega_ratio(self.returns, required_return=-1.0)
        assert np.isnan(result)
