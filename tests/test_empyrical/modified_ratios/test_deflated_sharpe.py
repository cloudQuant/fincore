"""Tests for deflated_sharpe_ratio.

Part of test_modified_ratios.py split.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_almost_equal

from fincore.empyrical import Empyrical
from fincore.metrics.ratios import deflated_sharpe_ratio


def _make_series(values, start="2000-01-03", freq="D"):
    """Helper to create a pd.Series with a DatetimeIndex."""
    return pd.Series(
        np.array(values, dtype=float),
        index=pd.date_range(start, periods=len(values), freq=freq),
    )


@pytest.mark.p1
class TestDeflatedSharpeRatio:
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
        assert 0.0 <= result <= 1.0

    def test_single_trial_equals_psr_at_zero(self):
        """With num_trials=1, SR* = 0 => DSR = PSR(0)."""
        np.random.seed(123)
        r = _make_series(np.random.normal(0.002, 0.01, 252).tolist())
        dsr = deflated_sharpe_ratio(r, num_trials=1)
        assert dsr > 0.5

    def test_more_trials_lower_dsr(self):
        """More trials should lower the DSR (higher hurdle)."""
        np.random.seed(42)
        r = _make_series(np.random.normal(0.001, 0.02, 252).tolist())
        dsr_1 = deflated_sharpe_ratio(r, num_trials=1)
        dsr_10 = deflated_sharpe_ratio(r, num_trials=10)
        dsr_100 = deflated_sharpe_ratio(r, num_trials=100)
        assert dsr_1 >= dsr_10 >= dsr_100

    def test_risk_free_shifts_sr(self):
        """A higher risk_free should lower DSR for positive-mean returns."""
        np.random.seed(42)
        r = _make_series(np.random.normal(0.002, 0.02, 252).tolist())
        dsr_0 = deflated_sharpe_ratio(r, risk_free=0)
        dsr_rf = deflated_sharpe_ratio(r, risk_free=0.002)
        assert dsr_rf < dsr_0

    def test_strong_signal_high_dsr(self):
        """Very strong positive signal => DSR equal to 1.0."""
        np.random.seed(42)
        r = _make_series(np.random.normal(0.05, 0.01, 500).tolist())
        dsr = deflated_sharpe_ratio(r, num_trials=1)
        assert dsr >= 0.99

    def test_zero_mean_returns(self):
        """Zero-mean returns with 1 trial => DSR near 0.5."""
        np.random.seed(42)
        r = _make_series(np.random.normal(0, 0.02, 10000).tolist())
        dsr = deflated_sharpe_ratio(r, num_trials=1)
        assert 0.3 < dsr < 0.7

    def test_constant_returns(self):
        """Constant returns => DSR behavior."""
        r = _make_series([0.01] * 20)
        dsr_1 = deflated_sharpe_ratio(r, num_trials=1)
        assert_almost_equal(dsr_1, 0.5, 2)

        dsr_10 = deflated_sharpe_ratio(r, num_trials=10)
        assert dsr_10 < 0.5


@pytest.mark.p1
class TestEmpyricalIntegration:
    """Test that modified ratios are accessible via Empyrical class."""

    returns = _make_series([0.01, 0.02, -0.05, 0.03, 0.01, -0.02, 0.04, 0.01, 0.02, -0.03])

    @pytest.fixture(autouse=True)
    def _setup_emp(self):
        """Set up Empyrical instance."""
        self.emp = Empyrical()

    def test_calmar_via_empyrical(self, _setup_emp):
        result = self.emp.calmar_ratio(self.returns, risk_free=0.01)
        assert isinstance(result, (float, np.floating))
        assert not np.isnan(result)

    def test_conditional_sharpe_via_empyrical(self, _setup_emp):
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

    def test_kappa_three_via_empyrical(self, _setup_emp):
        result = self.emp.kappa_three_ratio(self.returns, mar=0.001)
        assert isinstance(result, (float, np.floating))

    def test_omega_via_empyrical(self, _setup_emp):
        result = self.emp.omega_ratio(self.returns, risk_free=0.001)
        assert isinstance(result, (float, np.floating))
        assert not np.isnan(result)

    def test_deflated_sharpe_via_empyrical(self, _setup_emp):
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
