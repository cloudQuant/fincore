"""Risk-domain numerical correctness tests against independent oracles.

These tests freeze the P0 errors identified in the 2026-08-20 audit and assert
the corrected behavior against hand-derived / SciPy / statsmodels references.
Each P0 domain function has (a) an independent oracle, (b) a property-style
invariant, and (c) a wrong-model counter-example where applicable.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from fincore.risk.backtesting import kupiec_lr
from fincore.risk.evt import hill_estimator
from fincore.risk.garch import EGARCH, GARCH, GJRGARCH
from fincore.risk.models import forecast_es, forecast_var
from tests.oracles.risk.evt_oracle import hill_threshold_reference
from tests.oracles.risk.kupiec_oracle import kupiec_lr_brute_reference, kupiec_lr_reference
from tests.oracles.risk.normal_es_oracle import normal_es_reference, normal_var_reference


# --------------------------------------------------------------------------- #
# Kupiec LR-POF
# --------------------------------------------------------------------------- #
class TestKupiecLR:
    """Kupiec LR-POF must be non-negative and match the reference oracle."""

    @pytest.mark.parametrize(
        ("n", "x", "p"),
        [
            (100, 5, 0.99),
            (1000, 10, 0.99),
            (1000, 25, 0.99),
            (250, 2, 0.99),
            (250, 10, 0.99),
            (1000, 50, 0.95),
            (1000, 100, 0.95),
            (500, 0, 0.99),
            (500, 500, 0.99),
        ],
    )
    def test_lr_nonnegative(self, n: int, x: int, p: float) -> None:
        assert kupiec_lr(n, x, p) >= 0.0

    @pytest.mark.parametrize(
        ("n", "x", "p"),
        [
            (100, 5, 0.99),
            (1000, 10, 0.99),
            (1000, 25, 0.99),
            (250, 2, 0.99),
            (1000, 50, 0.95),
            (250, 7, 0.95),
        ],
    )
    def test_lr_matches_reference(self, n: int, x: int, p: float) -> None:
        got = kupiec_lr(n, x, p)
        ref = kupiec_lr_reference(n, x, p)
        assert np.isclose(got, ref, rtol=1e-12, atol=1e-12)

    def test_lr_matches_brute_reference(self) -> None:
        # Two independent reference code paths agree with the implementation.
        for n, x, p in [(100, 5, 0.99), (1000, 10, 0.99), (250, 7, 0.95)]:
            got = kupiec_lr(n, x, p)
            assert np.isclose(got, kupiec_lr_brute_reference(n, x, p), rtol=1e-12, atol=1e-12)

    def test_boundary_zero_exceptions_is_finite(self) -> None:
        # x=0 uses the continuous limit: LR = -2*n*ln(1-p), finite and non-negative.
        lr = kupiec_lr(100, 0, 0.99)
        assert np.isfinite(lr)
        assert lr >= 0.0
        assert np.isclose(lr, kupiec_lr_reference(100, 0, 0.99), rtol=1e-12, atol=1e-12)

    def test_boundary_all_exceptions_is_finite(self) -> None:
        # x=n uses the continuous limit: LR = 2*n*ln(1/p), finite and non-negative.
        lr = kupiec_lr(100, 100, 0.99)
        assert np.isfinite(lr)
        assert lr >= 0.0
        assert np.isclose(lr, kupiec_lr_reference(100, 100, 0.99), rtol=1e-12, atol=1e-12)

    def test_known_value(self) -> None:
        # Hand-computed: 2 * [5*ln(5) + 95*ln(95/99)] ≈ 8.2582.
        assert np.isclose(kupiec_lr(100, 5, 0.99), 8.258217002871657, rtol=1e-12, atol=1e-12)

    def test_wrong_model_rejected(self) -> None:
        # A badly calibrated 99% VaR (5 exceptions in 100 obs) must produce a
        # clearly non-zero statistic that rejects the null.
        assert kupiec_lr(100, 5, 0.99) > 3.84  # chi2(1) 95% critical value


# --------------------------------------------------------------------------- #
# VaR / ES forecast pair
# --------------------------------------------------------------------------- #
class TestVarEsForecastPair:
    """ES must be more extreme than VaR, and match analytic normal reference."""

    def _returns(self) -> pd.Series:
        rng = np.random.default_rng(7)
        return pd.Series(rng.normal(0.0, 0.02, 2000))

    @pytest.mark.parametrize("confidence_level", [0.95, 0.99, 0.975])
    def test_historical_es_leq_var(self, confidence_level: float) -> None:
        returns = self._returns()
        var = forecast_var(returns, method="historical", confidence_level=confidence_level)
        es = forecast_es(returns, method="historical", confidence_level=confidence_level)
        assert es.estimate <= var.estimate + 1e-12

    def test_garch_es_strictly_more_extreme_than_var(self) -> None:
        # P0: GARCH ES must NOT equal GARCH VaR.
        returns = self._returns()
        var = forecast_var(returns, method="garch", confidence_level=0.99)
        es = forecast_es(returns, method="garch", confidence_level=0.99)
        assert es.estimate < var.estimate

    def test_conditional_es_ratio_matches_analytic_normal(self) -> None:
        from fincore.risk.garch import conditional_es, conditional_var

        returns = self._returns()
        es = conditional_es(returns, model="GARCH", alpha=0.01, horizon=1)
        var = conditional_var(returns, model="GARCH", alpha=0.01, horizon=1)
        z = float(stats.norm.ppf(0.01))
        expected_ratio = -stats.norm.pdf(z) / (z * 0.01)
        assert np.isclose(es["es"] / var["var"], expected_ratio, rtol=1e-12)

    def test_horizon_changes_forecast(self) -> None:
        returns = self._returns()
        var1 = forecast_var(returns, method="garch", confidence_level=0.99, horizon=1)
        var10 = forecast_var(returns, method="garch", confidence_level=0.99, horizon=10)
        var20 = forecast_var(returns, method="garch", confidence_level=0.99, horizon=20)
        # Multi-horizon forecasts must aggregate volatility and therefore differ.
        assert not np.isclose(var1.estimate, var10.estimate, rtol=1e-12)
        assert abs(var20.estimate) > abs(var10.estimate) > abs(var1.estimate)


# --------------------------------------------------------------------------- #
# GARCH-family model identity
# --------------------------------------------------------------------------- #
class TestGARCHFamilyIdentity:
    """GARCH models must only expose the orders they actually implement."""

    def test_garch_rejects_unsupported_order(self) -> None:
        rng = np.random.default_rng(42)
        returns = rng.normal(0.0, 0.01, 500)
        with pytest.raises(ValueError):
            GARCH(p=2, q=1).fit(returns)

    def test_egarch_rejects_unsupported_order(self) -> None:
        rng = np.random.default_rng(42)
        returns = rng.normal(0.0, 0.01, 500)
        with pytest.raises(ValueError):
            EGARCH(p=2, q=1).fit(returns)

    def test_gjr_rejects_unsupported_order(self) -> None:
        rng = np.random.default_rng(42)
        returns = rng.normal(0.0, 0.01, 500)
        with pytest.raises(ValueError):
            GJRGARCH(p=1, q=2).fit(returns)


# --------------------------------------------------------------------------- #
# EVT tail selection and GEV shape sign
# --------------------------------------------------------------------------- #
class TestEVTSemantics:
    """EVT must distinguish upper/lower tails and use standard GEV xi sign."""

    def _asymmetric(self) -> np.ndarray:
        rng = np.random.default_rng(1)
        return np.concatenate([rng.normal(0.001, 0.01, 3000), rng.normal(-0.05, 0.03, 1000)])

    def test_upper_and_lower_tail_differ(self) -> None:
        from fincore.risk.evt import evt_var

        data = self._asymmetric()
        var_lower = evt_var(data, alpha=0.05, model="gpd", tail="lower")
        var_upper = evt_var(data, alpha=0.05, model="gpd", tail="upper")
        assert not np.isclose(var_lower, var_upper, rtol=1e-6)

    def test_gpd_lower_tail_negative_upper_tail_positive(self) -> None:
        from fincore.risk.evt import evt_var

        data = self._asymmetric()
        var_lower = evt_var(data, alpha=0.05, model="gpd", tail="lower")
        var_upper = evt_var(data, alpha=0.05, model="gpd", tail="upper")
        assert var_lower < 0.0
        assert var_upper > 0.0

    def test_gev_shape_uses_standard_xi_sign(self) -> None:
        from fincore.risk.evt import gev_fit

        rng = np.random.default_rng(42)
        heavy = rng.standard_t(3, 10000)
        params = gev_fit(heavy, block_size=100)
        # Standard GEV xi > 0 for a Frechet (heavy-tailed) distribution.
        assert params["xi"] > 0.0

    def test_hill_matches_independent_threshold_formula(self) -> None:
        """Hill uses log(tail observation / threshold), not log(excess)."""
        magnitudes = np.array([1.10, 1.25, 1.50, 1.75, 2.00, 2.50, 3.00, 3.50, 4.00, 5.00, 6.00, 8.00])
        expected, expected_observations = hill_threshold_reference(magnitudes, threshold=1.0, tail="upper")

        actual, observations = hill_estimator(magnitudes, threshold=1.0, tail="upper")

        assert np.isclose(actual, expected, rtol=1e-12, atol=1e-12)
        assert np.array_equal(observations, expected_observations)

    def test_hill_lower_tail_is_reflection_of_upper_tail(self) -> None:
        """The same loss magnitudes have the same Hill index after reflection."""
        magnitudes = np.array([1.10, 1.25, 1.50, 1.75, 2.00, 2.50, 3.00, 3.50, 4.00, 5.00, 6.00, 8.00])
        expected, expected_observations = hill_threshold_reference(-magnitudes, threshold=1.0, tail="lower")

        actual, observations = hill_estimator(-magnitudes, threshold=1.0, tail="lower")

        assert np.isclose(actual, expected, rtol=1e-12, atol=1e-12)
        assert np.array_equal(observations, expected_observations)


# --------------------------------------------------------------------------- #
# Deflated Sharpe Ratio (kurtosis + trial-Sharpe-variance semantics)
# --------------------------------------------------------------------------- #
class TestDeflatedSharpeRatio:
    """DSR must use ordinary kurtosis and the exact trial-Sharpe hurdle."""

    @staticmethod
    def _dsr_reference(returns: np.ndarray, num_trials: int) -> float:
        from scipy.stats import norm

        x = np.asarray(returns, dtype=float)
        x = x[~np.isnan(x)]
        t = len(x)
        if t < 3:
            return float("nan")
        mean = float(np.mean(x))
        std = float(np.std(x, ddof=1))
        sr = mean / std if std > 1e-15 else 0.0
        # scipy.stats.skew/kurtosis: kurtosis is Fisher (excess) by default.
        gamma3 = float(stats.skew(x, bias=False))
        gamma4 = float(stats.kurtosis(x, bias=False, fisher=False))  # ordinary kurtosis
        sr_var = 1.0 - gamma3 * sr + (gamma4 - 1) / 4.0 * sr**2

        n = max(num_trials, 1)
        if n <= 1:
            sr_star = 0.0
        else:
            gamma_euler = 0.5772156649015329
            z_n = float(norm.ppf(1.0 - 1.0 / n))
            z_ne = float(norm.ppf(1.0 - 1.0 / (n * np.e)))
            sr_star = np.sqrt(max(sr_var, 0.0)) * ((1.0 - gamma_euler) * z_n + gamma_euler * z_ne)

        if sr_var <= 0:
            return 1.0 if sr > sr_star else 0.0
        z = (sr - sr_star) * np.sqrt(t - 1) / np.sqrt(sr_var)
        return float(norm.cdf(z))

    @pytest.mark.parametrize("num_trials", [1, 3, 10, 50])
    def test_matches_independent_reference(self, num_trials: int) -> None:
        from fincore.metrics.ratios import deflated_sharpe_ratio

        rng = np.random.default_rng(42)
        returns = rng.normal(0.001, 0.02, 500)
        got = deflated_sharpe_ratio(returns, num_trials=num_trials)
        ref = self._dsr_reference(returns, num_trials)
        assert np.isclose(got, ref, rtol=1e-10, atol=1e-10)
