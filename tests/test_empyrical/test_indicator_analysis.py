"""
Tests for indicators analyzed in docs/0025-校正指标计算方式/计算downside_risk等指标.md.

Specifically adds missing direct tests for information_ratio.
"""

from unittest import TestCase

import numpy as np
import pandas as pd
from numpy.testing import assert_almost_equal

from fincore.empyrical import Empyrical
from fincore.metrics.ratios import information_ratio

DECIMAL_PLACES = 4


def _make_series(values, start="2000-01-03", freq="D"):
    """Helper to create a pd.Series with a DatetimeIndex."""
    return pd.Series(
        np.array(values, dtype=float),
        index=pd.date_range(start, periods=len(values), freq=freq),
    )


class TestInformationRatio(TestCase):
    """Direct tests for information_ratio function."""

    def test_hand_calculation(self):
        """Verify against hand calculation: IR = mean(active) * sqrt(q) / std(active)."""
        returns = _make_series([0.01, 0.02, -0.01, 0.03, 0.005])
        benchmark = _make_series([0.005, 0.01, 0.0, 0.015, 0.002])
        active = np.array([0.005, 0.01, -0.01, 0.015, 0.003])

        mean_active = np.mean(active)  # 0.0046
        std_active = np.std(active, ddof=1)
        ann_factor = 252
        expected = mean_active * np.sqrt(ann_factor) / std_active

        result = information_ratio(returns, benchmark)
        assert_almost_equal(result, expected, DECIMAL_PLACES)

    def test_identical_returns_nan_or_inf(self):
        """When returns == benchmark, active returns are 0, std=0 => inf or NaN."""
        returns = _make_series([0.01, 0.02, -0.01, 0.03])
        result = information_ratio(returns, returns)
        # std of all zeros is 0, division by 0
        assert np.isnan(result) or np.isinf(result)

    def test_positive_active_return(self):
        """Strategy consistently outperforms benchmark => positive IR."""
        returns = _make_series([0.02, 0.03, 0.01, 0.025, 0.015])
        benchmark = _make_series([0.01, 0.01, 0.01, 0.01, 0.01])
        result = information_ratio(returns, benchmark)
        assert result > 0, f"Expected positive IR, got {result}"

    def test_negative_active_return(self):
        """Strategy consistently underperforms benchmark => negative IR."""
        returns = _make_series([0.005, 0.008, 0.003, 0.006, 0.004])
        benchmark = _make_series([0.02, 0.03, 0.01, 0.025, 0.015])
        result = information_ratio(returns, benchmark)
        assert result < 0, f"Expected negative IR, got {result}"

    def test_short_series_nan(self):
        """Fewer than 2 returns should give NaN."""
        returns = _make_series([0.01])
        benchmark = _make_series([0.005])
        result = information_ratio(returns, benchmark)
        assert np.isnan(result)

    def test_annualization_effect(self):
        """Monthly data should have different annualization factor."""
        np.random.seed(42)
        returns = _make_series(np.random.normal(0.01, 0.03, 36).tolist(), freq="ME")
        benchmark = _make_series(np.random.normal(0.008, 0.025, 36).tolist(), freq="ME")

        ir_daily = information_ratio(returns, benchmark, period="daily")
        ir_monthly = information_ratio(returns, benchmark, period="monthly")

        # sqrt(252) vs sqrt(12) scaling => daily IR should be larger in magnitude
        if not np.isnan(ir_daily) and not np.isnan(ir_monthly):
            assert abs(ir_daily) > abs(ir_monthly), (
                f"Daily IR ({ir_daily}) should have larger magnitude than monthly ({ir_monthly})"
            )

    def test_via_empyrical(self):
        """Test information_ratio accessible via Empyrical class."""
        returns = _make_series([0.01, 0.02, -0.01, 0.03, 0.005, -0.02, 0.015])
        benchmark = _make_series([0.005, 0.01, 0.0, 0.015, 0.002, -0.01, 0.008])
        emp = Empyrical()
        result = emp.information_ratio(returns, benchmark)
        assert isinstance(result, (float, np.floating))
        assert not np.isnan(result)

    def test_high_tracking_error(self):
        """High tracking error (volatile active returns) should give lower |IR|."""
        np.random.seed(42)
        base = np.random.normal(0.001, 0.01, 100)

        # Low tracking error: small diff
        returns_low_te = _make_series((base + 0.002).tolist())
        benchmark = _make_series(base.tolist())

        # High tracking error: large diff with noise
        returns_high_te = _make_series((base + 0.002 + np.random.normal(0, 0.05, 100)).tolist())

        ir_low_te = information_ratio(returns_low_te, benchmark)
        ir_high_te = information_ratio(returns_high_te, benchmark)

        assert abs(ir_low_te) > abs(ir_high_te), (
            f"Low TE IR ({ir_low_te}) should have larger |IR| than high TE ({ir_high_te})"
        )
