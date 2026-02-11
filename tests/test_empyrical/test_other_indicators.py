"""
Tests for indicators analyzed in docs/0025-校正指标计算方式/计算其他指标.md.

Adds missing direct tests for conditional_value_at_risk (CVaR / Expected Shortfall).
"""

from unittest import TestCase

import numpy as np
import pandas as pd
from numpy.testing import assert_almost_equal

from fincore.empyrical import Empyrical
from fincore.metrics.risk import conditional_value_at_risk, value_at_risk

DECIMAL_PLACES = 4


def _make_series(values, start="2000-01-03", freq="D"):
    """Helper to create a pd.Series with a DatetimeIndex."""
    return pd.Series(
        np.array(values, dtype=float),
        index=pd.date_range(start, periods=len(values), freq=freq),
    )


class TestConditionalValueAtRisk(TestCase):
    """Direct tests for conditional_value_at_risk (CVaR / Expected Shortfall)."""

    def test_hand_calculation(self):
        """Verify CVaR against hand calculation with deterministic data."""
        # 10 sorted returns: [-0.10, -0.08, -0.05, -0.03, -0.01, 0.01, 0.02, 0.04, 0.06, 0.08]
        returns = _make_series([-0.10, -0.08, -0.05, -0.03, -0.01, 0.01, 0.02, 0.04, 0.06, 0.08])
        # cutoff=0.20 => VaR = 20th percentile
        # np.percentile(sorted, 20) => between -0.08 and -0.05
        var_20 = np.percentile(returns.values, 20)
        # CVaR = mean of returns <= var_20
        expected = np.mean(returns.values[returns.values <= var_20])
        result = conditional_value_at_risk(returns, cutoff=0.20)
        assert_almost_equal(result, expected, DECIMAL_PLACES)

    def test_hand_calculation_simple(self):
        """Simple 5-element hand calculation at cutoff=0.40."""
        returns = _make_series([-0.05, -0.03, 0.00, 0.02, 0.04])
        # 40th percentile of [-0.05, -0.03, 0.00, 0.02, 0.04]
        var_40 = np.percentile([-0.05, -0.03, 0.00, 0.02, 0.04], 40)
        # returns <= var_40
        tail = np.array([-0.05, -0.03, 0.00, 0.02, 0.04])
        tail_below = tail[tail <= var_40]
        expected = np.mean(tail_below)
        result = conditional_value_at_risk(returns, cutoff=0.40)
        assert_almost_equal(result, expected, DECIMAL_PLACES)

    def test_cvar_leq_var(self):
        """CVaR should always be <= VaR (CVaR is deeper in the tail)."""
        np.random.seed(42)
        returns = _make_series(np.random.normal(0, 0.02, 200).tolist())
        for cutoff in [0.01, 0.05, 0.10, 0.25, 0.50]:
            var = value_at_risk(returns, cutoff=cutoff)
            cvar = conditional_value_at_risk(returns, cutoff=cutoff)
            assert cvar <= var, f"CVaR ({cvar}) should be <= VaR ({var}) at cutoff={cutoff}"

    def test_cutoff_1_equals_mean(self):
        """At cutoff=1.0, all returns are included, so CVaR = mean(returns)."""
        returns = _make_series([-0.05, -0.03, 0.00, 0.02, 0.04])
        result = conditional_value_at_risk(returns, cutoff=1.0)
        expected = np.mean(returns.values)
        assert_almost_equal(result, expected, DECIMAL_PLACES)

    def test_single_return(self):
        """Single return: CVaR should equal that return."""
        returns = _make_series([-0.05])
        result = conditional_value_at_risk(returns, cutoff=0.05)
        assert_almost_equal(result, -0.05, DECIMAL_PLACES)

    def test_empty_returns_nan(self):
        """Empty returns should give NaN."""
        result = conditional_value_at_risk(pd.Series([], dtype=float))
        assert np.isnan(result)

    def test_all_same_returns(self):
        """All identical returns: CVaR = that constant value."""
        returns = _make_series([0.01] * 20)
        result = conditional_value_at_risk(returns, cutoff=0.05)
        assert_almost_equal(result, 0.01, DECIMAL_PLACES)

    def test_negative_skew_larger_gap(self):
        """With negative skew, gap between VaR and CVaR should be larger."""
        np.random.seed(42)
        # Normal returns
        normal_returns = np.random.normal(0, 0.02, 1000)
        # Add negative skew by making some returns very negative
        skewed_returns = normal_returns.copy()
        skewed_returns[:20] = np.random.uniform(-0.15, -0.10, 20)

        normal_s = _make_series(normal_returns.tolist())
        skewed_s = _make_series(skewed_returns.tolist())

        cutoff = 0.05
        normal_gap = value_at_risk(normal_s, cutoff) - conditional_value_at_risk(normal_s, cutoff)
        skewed_gap = value_at_risk(skewed_s, cutoff) - conditional_value_at_risk(skewed_s, cutoff)

        assert skewed_gap > normal_gap, f"Skewed gap ({skewed_gap}) should exceed normal gap ({normal_gap})"

    def test_via_empyrical(self):
        """Test CVaR accessible via Empyrical class."""
        returns = _make_series([-0.05, -0.03, 0.00, 0.02, 0.04, -0.01, 0.03])
        emp = Empyrical()
        result = emp.conditional_value_at_risk(returns, cutoff=0.20)
        assert isinstance(result, (float, np.floating))
        assert result < 0, f"Expected negative CVaR at 20% cutoff, got {result}"
