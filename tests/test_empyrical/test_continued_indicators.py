"""
Tests for indicators analyzed in docs/0025-校正指标计算方式/继续计算其他指标.md.

Tests for modified var_excess_return and stutzer_index.
"""

from unittest import TestCase

import numpy as np
import pandas as pd
from numpy.testing import assert_almost_equal

from fincore.empyrical import Empyrical
from fincore.metrics.risk import value_at_risk, var_excess_return
from fincore.metrics.stats import stutzer_index
from fincore.metrics.yearly import annual_return

DECIMAL_PLACES = 4


def _make_series(values, start="2000-01-03", freq="D"):
    """Helper to create a pd.Series with a DatetimeIndex."""
    return pd.Series(
        np.array(values, dtype=float),
        index=pd.date_range(start, periods=len(values), freq=freq),
    )


# ---------------------------------------------------------------------------
# VaR Excess Return tests
# ---------------------------------------------------------------------------


class TestVarExcessReturn(TestCase):
    """Tests for the modified var_excess_return = (ann_return - rf) / |VaR|."""

    def test_hand_calculation(self):
        """Verify against hand calculation."""
        np.random.seed(42)
        returns = _make_series(np.random.normal(0.001, 0.02, 252).tolist())

        ann_ret = annual_return(returns, period="daily")
        var_val = value_at_risk(returns, cutoff=0.05)
        expected = (ann_ret - 0.0) / abs(var_val)

        result = var_excess_return(returns, cutoff=0.05, risk_free=0.0)
        assert_almost_equal(result, expected, DECIMAL_PLACES)

    def test_with_risk_free(self):
        """Verify risk_free is subtracted from numerator."""
        np.random.seed(42)
        returns = _make_series(np.random.normal(0.001, 0.02, 252).tolist())

        result_no_rf = var_excess_return(returns, risk_free=0.0)
        result_with_rf = var_excess_return(returns, risk_free=0.03)

        # Higher rf => lower numerator => lower ratio
        assert result_with_rf < result_no_rf, f"With rf=0.03 ({result_with_rf}) should be < without rf ({result_no_rf})"

    def test_positive_returns_positive_ratio(self):
        """Strong positive returns with rf=0 should give positive ratio."""
        returns = _make_series([0.005] * 252)
        result = var_excess_return(returns)
        # All returns are 0.005, VaR = 0.005
        # ann_return is very high, VaR is positive
        # But VaR at 5% cutoff of constant returns = 0.005 (positive)
        # (ann_ret - 0) / |0.005| should be a large positive number
        assert result > 0, f"Expected positive ratio, got {result}"

    def test_empty_returns_nan(self):
        """Empty or short returns should give NaN."""
        result = var_excess_return(pd.Series([], dtype=float))
        assert np.isnan(result)

        result = var_excess_return(_make_series([0.01]))
        assert np.isnan(result)

    def test_via_empyrical(self):
        """Test accessible via Empyrical class."""
        np.random.seed(42)
        returns = _make_series(np.random.normal(0.001, 0.02, 252).tolist())
        emp = Empyrical()
        result = emp.var_excess_return(returns)
        assert isinstance(result, (float, np.floating))
        assert not np.isnan(result)

    def test_monthly_annualization(self):
        """Monthly returns should use monthly annualization factor."""
        np.random.seed(42)
        returns = _make_series(np.random.normal(0.01, 0.05, 36).tolist(), freq="ME")

        ann_ret = annual_return(returns, period="monthly")
        var_val = value_at_risk(returns, cutoff=0.05)
        expected = (ann_ret - 0.0) / abs(var_val)

        result = var_excess_return(returns, period="monthly")
        assert_almost_equal(result, expected, DECIMAL_PLACES)


# ---------------------------------------------------------------------------
# Stutzer Index tests
# ---------------------------------------------------------------------------


class TestStutzerIndex(TestCase):
    """Tests for the corrected stutzer_index = sign * sqrt(2 * I_P)."""

    def test_positive_returns_positive_index(self):
        """Positive mean excess returns should give positive Stutzer index."""
        np.random.seed(42)
        returns = _make_series(np.random.normal(0.005, 0.02, 500).tolist())
        result = stutzer_index(returns)
        assert result > 0, f"Expected positive Stutzer index, got {result}"

    def test_negative_returns_negative_index(self):
        """Negative mean excess returns should give negative Stutzer index."""
        np.random.seed(42)
        returns = _make_series(np.random.normal(-0.005, 0.02, 500).tolist())
        result = stutzer_index(returns)
        assert result < 0, f"Expected negative Stutzer index, got {result}"

    def test_zero_mean_returns_zero(self):
        """Zero mean excess returns should give ~0 Stutzer index."""
        result = stutzer_index(np.array([0.01, -0.01, 0.01, -0.01, 0.01, -0.01]))
        # Mean is exactly 0, should return 0.0
        assert_almost_equal(result, 0.0, 4)

    def test_normal_returns_approx_sharpe(self):
        """For normal returns, Stutzer index ≈ Sharpe ratio."""
        np.random.seed(42)
        returns = np.random.normal(0.001, 0.02, 5000)
        stutzer = stutzer_index(returns)
        sharpe_approx = np.mean(returns) / np.std(returns, ddof=1)

        # For normally distributed returns, Stutzer ≈ Sharpe
        assert abs(stutzer - sharpe_approx) < 0.1, (
            f"Stutzer ({stutzer}) should approximate Sharpe ({sharpe_approx}) for normal returns"
        )

    def test_with_target_return(self):
        """Target return shifts excess returns."""
        np.random.seed(42)
        returns = _make_series(np.random.normal(0.005, 0.02, 500).tolist())

        result_zero_target = stutzer_index(returns, target_return=0.0)
        result_high_target = stutzer_index(returns, target_return=0.005)

        # Higher target => lower excess returns => lower Stutzer
        assert result_high_target < result_zero_target, (
            f"High target ({result_high_target}) should give lower Stutzer than zero target ({result_zero_target})"
        )

    def test_empty_returns_nan(self):
        """Empty or short returns should give NaN."""
        assert np.isnan(stutzer_index(np.array([])))
        assert np.isnan(stutzer_index(np.array([0.01])))

    def test_skewed_returns_penalized(self):
        """Negative skew should result in lower Stutzer index vs symmetric."""
        np.random.seed(42)
        n = 2000
        # Symmetric returns
        symmetric = np.random.normal(0.001, 0.02, n)
        # Negatively skewed: same mean but with occasional large losses
        skewed = symmetric.copy()
        skewed[:40] = -0.10  # Add large losses
        # Adjust mean to be similar
        skewed += np.mean(symmetric) - np.mean(skewed)

        stutzer_sym = stutzer_index(symmetric)
        stutzer_skew = stutzer_index(skewed)

        assert stutzer_skew < stutzer_sym, (
            f"Negatively skewed Stutzer ({stutzer_skew}) should be lower than symmetric ({stutzer_sym})"
        )

    def test_via_empyrical(self):
        """Test accessible via Empyrical class."""
        np.random.seed(42)
        returns = _make_series(np.random.normal(0.003, 0.02, 252).tolist())
        emp = Empyrical()
        result = emp.stutzer_index(returns)
        assert isinstance(result, (float, np.floating))
        assert result > 0
