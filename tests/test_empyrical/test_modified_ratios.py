"""
Tests for recently modified ratio indicators:
- calmar_ratio (added risk_free parameter)
- conditional_sharpe_ratio (added risk_free parameter)
- kappa_three_ratio (reworked formula)
- omega_ratio (improved threshold handling)
- deflated_sharpe_ratio (new indicator)
"""

from unittest import TestCase

import numpy as np
import pandas as pd
from numpy.testing import assert_almost_equal

from fincore.empyrical import Empyrical
from fincore.metrics.ratios import (
    calmar_ratio,
    conditional_sharpe_ratio,
    deflated_sharpe_ratio,
    kappa_three_ratio,
    omega_ratio,
)

DECIMAL_PLACES = 4


def _make_series(values, start="2000-01-03", freq="D"):
    """Helper to create a pd.Series with a DatetimeIndex."""
    return pd.Series(
        np.array(values, dtype=float),
        index=pd.date_range(start, periods=len(values), freq=freq),
    )


class TestCalmarRatioRiskFree(TestCase):
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
        assert ratio_rf < ratio_0, f"Expected ratio with rf=0.03 ({ratio_rf}) < ratio with rf=0 ({ratio_0})"

    def test_risk_free_large_makes_negative(self):
        """A large risk_free can make the ratio negative."""
        # Use a longer series so annualized return is modest
        np.random.seed(42)
        long_returns = _make_series(np.random.normal(0.0003, 0.01, 252).tolist())
        ratio = calmar_ratio(long_returns, risk_free=0.5)
        assert ratio < 0, f"Expected negative ratio with large rf, got {ratio}"

    def test_empty_returns_nan(self):
        result = calmar_ratio(pd.Series([], dtype=float))
        assert np.isnan(result)

    def test_single_positive_return(self):
        """Single return has no drawdown => NaN."""
        result = calmar_ratio(_make_series([0.05]))
        assert np.isnan(result)


class TestConditionalSharpeRatioRiskFree(TestCase):
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
        # Both should be valid numbers
        assert not np.isnan(csr_0), "CSR with rf=0 is NaN"
        assert not np.isnan(csr_rf), "CSR with rf=0.01 is NaN"
        assert csr_rf < csr_0, f"Expected CSR with rf=0.01 ({csr_rf}) < CSR with rf=0 ({csr_0})"

    def test_empty_returns_nan(self):
        result = conditional_sharpe_ratio(pd.Series([], dtype=float))
        assert np.isnan(result)

    def test_short_returns_nan(self):
        result = conditional_sharpe_ratio(_make_series([0.01]))
        assert np.isnan(result)


class TestKappaThreeRatio(TestCase):
    """Tests for kappa_three_ratio after formula rework."""

    returns = _make_series([0.01, 0.02, -0.05, 0.03, 0.01, -0.02, 0.04, 0.01, 0.02, -0.03])
    positive_returns = _make_series([0.01, 0.02, 0.01, 0.015, 0.01, 0.02, 0.01, 0.015, 0.01, 0.01])

    def test_hand_calculation(self):
        """Verify against a hand calculation with mar=0."""
        r = np.array([0.01, 0.02, -0.05, 0.03, 0.01, -0.02, 0.04, 0.01, 0.02, -0.03])
        mu = np.mean(r)  # 0.004
        mar = 0.0
        downside = np.maximum(0, mar - r)  # [0, 0, 0.05, 0, 0, 0.02, 0, 0, 0, 0.03]
        lpm3 = np.mean(downside**3)
        expected = (mu - mar) / (lpm3 ** (1.0 / 3.0))
        result = kappa_three_ratio(self.returns, mar=0.0)
        assert_almost_equal(result, expected, DECIMAL_PLACES)

    def test_positive_returns_inf(self):
        """All positive returns with mar=0 => LPM3=0 => inf."""
        result = kappa_three_ratio(self.positive_returns, mar=0.0)
        assert np.isinf(result) and result > 0, f"Expected +inf for all-positive returns, got {result}"

    def test_mar_above_all_returns(self):
        """When mar is above all returns, expect negative ratio."""
        result = kappa_three_ratio(self.returns, mar=0.10)
        assert result < 0, f"Expected negative kappa3 with high mar, got {result}"

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


class TestOmegaRatio(TestCase):
    """Tests for omega_ratio after improvements."""

    returns = _make_series([0.01, 0.02, -0.05, 0.03, 0.01, -0.02, 0.04, 0.01, 0.02, -0.03])

    def test_threshold_zero(self):
        """With threshold=0, omega = sum(gains) / sum(|losses|)."""
        r = np.array([0.01, 0.02, -0.05, 0.03, 0.01, -0.02, 0.04, 0.01, 0.02, -0.03])
        gains = r[r > 0].sum()  # 0.14
        losses = -r[r < 0].sum()  # 0.10
        expected = gains / losses
        result = omega_ratio(self.returns, risk_free=0.0, required_return=0.0)
        assert_almost_equal(result, expected, DECIMAL_PLACES)

    def test_risk_free_shifts_threshold(self):
        """Positive risk_free raises the threshold, reducing the ratio."""
        omega_0 = omega_ratio(self.returns, risk_free=0.0)
        omega_rf = omega_ratio(self.returns, risk_free=0.01)
        assert omega_rf < omega_0, f"Expected omega with rf=0.01 ({omega_rf}) < omega with rf=0 ({omega_0})"

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


class TestDeflatedSharpeRatio(TestCase):
    """Tests for the new deflated_sharpe_ratio function."""

    def test_empty_returns_nan(self):
        result = deflated_sharpe_ratio(pd.Series([], dtype=float))
        assert np.isnan(result)

    def test_two_obs_nan(self):
        result = deflated_sharpe_ratio(_make_series([0.01, -0.01]))
        assert np.isnan(result)

    def test_output_in_zero_one(self):
        """DSR is a probability, should be in [0, 1]."""
        np.random.seed(42)
        r = _make_series(np.random.normal(0.001, 0.02, 100).tolist())
        result = deflated_sharpe_ratio(r)
        assert 0.0 <= result <= 1.0, f"DSR {result} not in [0, 1]"

    def test_single_trial_equals_psr_at_zero(self):
        """With num_trials=1, SR* = 0 => DSR = PSR(0)."""
        np.random.seed(123)
        r = _make_series(np.random.normal(0.002, 0.01, 252).tolist())
        dsr = deflated_sharpe_ratio(r, num_trials=1)
        # Positive mean => DSR should be > 0.5
        assert dsr > 0.5, f"Expected DSR > 0.5 for positive-mean returns, got {dsr}"

    def test_more_trials_lower_dsr(self):
        """More trials should lower the DSR (higher hurdle)."""
        np.random.seed(42)
        r = _make_series(np.random.normal(0.001, 0.02, 252).tolist())
        dsr_1 = deflated_sharpe_ratio(r, num_trials=1)
        dsr_10 = deflated_sharpe_ratio(r, num_trials=10)
        dsr_100 = deflated_sharpe_ratio(r, num_trials=100)
        assert dsr_1 >= dsr_10 >= dsr_100, f"DSR should decrease with more trials: {dsr_1}, {dsr_10}, {dsr_100}"

    def test_risk_free_shifts_sr(self):
        """A higher risk_free should lower DSR for positive-mean returns."""
        np.random.seed(42)
        r = _make_series(np.random.normal(0.002, 0.02, 252).tolist())
        dsr_0 = deflated_sharpe_ratio(r, risk_free=0)
        dsr_rf = deflated_sharpe_ratio(r, risk_free=0.002)
        assert dsr_rf < dsr_0, f"Expected DSR with rf ({dsr_rf}) < DSR without rf ({dsr_0})"

    def test_strong_signal_high_dsr(self):
        """Very strong positive signal => DSR equal to 1.0 (edge case: denom_sq<=0)."""
        np.random.seed(42)
        r = _make_series(np.random.normal(0.05, 0.01, 500).tolist())
        dsr = deflated_sharpe_ratio(r, num_trials=1)
        assert dsr >= 0.99, f"Expected DSR >= 0.99 for very strong signal, got {dsr}"

    def test_zero_mean_returns(self):
        """Zero-mean returns with 1 trial => DSR near 0.5."""
        np.random.seed(42)
        r = _make_series(np.random.normal(0, 0.02, 10000).tolist())
        dsr = deflated_sharpe_ratio(r, num_trials=1)
        assert 0.3 < dsr < 0.7, f"Expected DSR near 0.5 for zero-mean returns, got {dsr}"

    def test_constant_returns(self):
        """Constant returns => std=0 => SR=0, denom_sq=1 => z=0 => DSR=0.5 for 1 trial.
        With num_trials>1, SR*>0 so SR_hat<SR* => DSR=0.0 (edge: denom_sq<=0)."""
        r = _make_series([0.01] * 20)
        dsr_1 = deflated_sharpe_ratio(r, num_trials=1)
        # sr_hat=0, sr_star=0 => z=0 => Φ(0)=0.5
        assert_almost_equal(dsr_1, 0.5, 2)

        dsr_10 = deflated_sharpe_ratio(r, num_trials=10)
        # sr_hat=0 < sr_star>0 => dsr should be low
        assert dsr_10 < 0.5, f"Expected DSR < 0.5 with 10 trials for constant returns, got {dsr_10}"


class TestEmpyricalIntegration(TestCase):
    """Test that modified ratios are accessible via Empyrical class."""

    returns = _make_series([0.01, 0.02, -0.05, 0.03, 0.01, -0.02, 0.04, 0.01, 0.02, -0.03])

    def setUp(self):
        self.emp = Empyrical()

    def test_calmar_via_empyrical(self):
        result = self.emp.calmar_ratio(self.returns, risk_free=0.01)
        assert isinstance(result, (float, np.floating))
        assert not np.isnan(result)

    def test_conditional_sharpe_via_empyrical(self):
        r = _make_series(
            [
                0.01,
                -0.02,
                0.005,
                -0.03,
                0.02,
                -0.01,
                0.015,
                -0.025,
                0.01,
                -0.015,
                0.008,
                -0.012,
                0.003,
                -0.018,
                0.007,
                -0.022,
                0.011,
                -0.009,
                0.004,
                -0.014,
            ]
        )
        result = self.emp.conditional_sharpe_ratio(r, risk_free=0.001)
        assert isinstance(result, (float, np.floating))

    def test_kappa_three_via_empyrical(self):
        result = self.emp.kappa_three_ratio(self.returns, mar=0.001)
        assert isinstance(result, (float, np.floating))

    def test_omega_via_empyrical(self):
        result = self.emp.omega_ratio(self.returns, risk_free=0.001)
        assert isinstance(result, (float, np.floating))
        assert not np.isnan(result)

    def test_deflated_sharpe_via_empyrical(self):
        np.random.seed(42)
        r = _make_series(np.random.normal(0.001, 0.02, 100).tolist())
        result = self.emp.deflated_sharpe_ratio(r, num_trials=5)
        assert isinstance(result, (float, np.floating))
        assert 0.0 <= result <= 1.0

    def test_deflated_sharpe_instance_mode(self):
        """Test DSR with instance-bound returns."""
        np.random.seed(42)
        r = _make_series(np.random.normal(0.001, 0.02, 100).tolist())
        emp = Empyrical(returns=r)
        result = emp.deflated_sharpe_ratio()
        assert isinstance(result, (float, np.floating))
        assert 0.0 <= result <= 1.0
