"""Attribution-domain numerical correctness tests against independent oracles."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tests.oracles.attribution.brinson_oracle import brinson_carino_reference
from tests.oracles.attribution.regression_oracle import ols_hac_reference, wls_reference


def _make_factor_data(n: int = 500) -> tuple[pd.Series, pd.DataFrame]:
    rng = np.random.default_rng(42)
    factors = pd.DataFrame(
        {
            "MKT": rng.normal(0.0, 0.01, n),
            "SMB": rng.normal(0.0, 0.005, n),
            "HML": rng.normal(0.0, 0.005, n),
        }
    )
    returns = pd.Series(
        0.001 + 1.2 * factors["MKT"] + 0.3 * factors["SMB"] - 0.2 * factors["HML"] + rng.normal(0.0, 0.01, n)
    )
    return returns, factors


class TestFamaFrenchHAC:
    """Fama-French HAC standard errors must match statsmodels sandwich oracle."""

    def test_hac_standard_errors_match_statsmodels(self) -> None:
        from fincore.attribution.fama_french import FamaFrenchModel

        returns, factors = _make_factor_data()
        model = FamaFrenchModel(model_type="3factor")
        result = model.fit(returns, factors, newey_west_lags=1)

        X = np.column_stack([np.ones(len(returns)), factors[model.factors].to_numpy()])
        _, ref_bse, _ = ols_hac_reference(returns.to_numpy(), X, nlags=1)

        assert np.allclose(result["std_errors"], ref_bse, rtol=1e-10, atol=1e-10), (
            f"got {result['std_errors']} vs statsmodels {ref_bse}"
        )

    def test_hac_standard_errors_not_all_identical(self) -> None:
        from fincore.attribution.fama_french import FamaFrenchModel

        returns, factors = _make_factor_data()
        model = FamaFrenchModel(model_type="3factor")
        result = model.fit(returns, factors, newey_west_lags=1)
        std_errors = np.asarray(result["std_errors"])
        assert len(np.unique(np.round(std_errors, 10))) > 1

    def test_wls_differs_from_ols(self) -> None:
        from fincore.attribution.fama_french import FamaFrenchModel

        returns, factors = _make_factor_data()
        n = len(returns)
        weights = pd.Series(np.linspace(0.5, 2.0, n), index=factors.index)

        model_ols = FamaFrenchModel(model_type="3factor")
        result_ols = model_ols.fit(returns, factors)

        model_wls = FamaFrenchModel(model_type="3factor")
        result_wls = model_wls.fit(returns, factors, method="wls", weights=weights)

        ols_betas = [result_ols["alpha"], *list(result_ols["betas"].values())]
        wls_betas = [result_wls["alpha"], *list(result_wls["betas"].values())]
        assert not np.allclose(ols_betas, wls_betas, rtol=1e-6)

    def test_wls_matches_statsmodels(self) -> None:
        from fincore.attribution.fama_french import FamaFrenchModel

        returns, factors = _make_factor_data()
        n = len(returns)
        weights = np.linspace(0.5, 2.0, n)

        model = FamaFrenchModel(model_type="3factor")
        result = model.fit(returns, factors, method="wls", weights=pd.Series(weights))

        X = np.column_stack([np.ones(n), factors[model.factors].to_numpy()])
        ref_params, _ = wls_reference(returns.to_numpy(), X, weights)

        got_params = np.array([result["alpha"], *list(result["betas"].values())])
        assert np.allclose(got_params, ref_params, rtol=1e-8, atol=1e-8)


class TestBrinsonLinking:
    """Multi-period Brinson must reconcile via geometric (Carino) linking."""

    def test_brinson_cumulative_reconciles_to_absolute_cumulative_active_return(self) -> None:
        """BHB Carino linking reports cumulative portfolio minus benchmark return.

        This fixed numerical regression deliberately distinguishes the
        standard absolute Carino target from the relative active return.  It
        uses only NumPy operations and does not call the oracle under test.
        """
        from fincore.attribution.brinson import brinson_cumulative

        rp = np.array([[0.05, 0.03], [0.02, -0.01], [0.04, 0.06]])
        rb = np.array([[0.03, 0.02], [0.01, 0.0], [0.03, 0.05]])
        wp = np.array([[0.6, 0.4], [0.5, 0.5], [0.4, 0.6]])
        wb = np.array([[0.5, 0.5], [0.5, 0.5], [0.5, 0.5]])

        result = brinson_cumulative(rp, rb, wp, wb)

        portfolio_period = np.sum(wp * rp, axis=1)
        benchmark_period = np.sum(wb * rb, axis=1)
        total = result["allocation"] + result["selection"] + result["interaction"]

        portfolio_cumulative = float(np.prod(1.0 + portfolio_period) - 1.0)
        benchmark_cumulative = float(np.prod(1.0 + benchmark_period) - 1.0)
        absolute_active = portfolio_cumulative - benchmark_cumulative
        relative_active = (1.0 + portfolio_cumulative) / (1.0 + benchmark_cumulative) - 1.0

        assert np.isclose(absolute_active, 0.030334920000000265, rtol=0.0, atol=1e-15)
        assert not np.isclose(absolute_active, relative_active, rtol=1e-12, atol=1e-12)
        assert np.isclose(total, absolute_active, rtol=1e-12, atol=1e-12), (
            f"linked total={total} vs absolute cumulative active {absolute_active}"
        )

    def test_brinson_cumulative_matches_carino_oracle(self) -> None:
        from fincore.attribution.brinson import brinson_cumulative

        rp = np.array([[0.05, 0.03], [0.02, -0.01], [0.04, 0.06]])
        rb = np.array([[0.03, 0.02], [0.01, 0.0], [0.03, 0.05]])
        wp = np.array([[0.6, 0.4], [0.5, 0.5], [0.4, 0.6]])
        wb = np.array([[0.5, 0.5], [0.5, 0.5], [0.5, 0.5]])

        result = brinson_cumulative(rp, rb, wp, wb)

        ref = brinson_carino_reference(rp, rb, wp, wb)

        assert np.isclose(
            ref["allocation"] + ref["selection"] + ref["interaction"],
            ref["active_return"],
            rtol=1e-12,
            atol=1e-12,
        )
        assert np.isclose(result["allocation"], ref["allocation"], rtol=1e-12, atol=1e-12)
        assert np.isclose(result["selection"], ref["selection"], rtol=1e-12, atol=1e-12)
        assert np.isclose(result["interaction"], ref["interaction"], rtol=1e-12, atol=1e-12)
        assert np.isclose(result["total"], ref["active_return"], rtol=1e-12, atol=1e-12)


class TestStyleBeta:
    """Style attribution beta must be cov/var (a slope), not a correlation."""

    def test_regression_attribution_beta_is_slope(self) -> None:
        from fincore.attribution.style import calculate_regression_attribution

        rng = np.random.default_rng(7)
        n = 300
        sr = rng.normal(0.0, 0.01, n)
        pr = 0.5 * sr + rng.normal(0.0, 0.005, n)  # true beta = 0.5

        idx = pd.date_range("2024-01-01", periods=n)
        style_returns = pd.DataFrame({"style_a": sr}, index=idx)
        portfolio = pd.Series(pr, index=idx)

        result = calculate_regression_attribution(
            portfolio, style_returns, style_exposures=pd.DataFrame({"style_a": [1.0]}, index=[0])
        )

        cov = float(np.cov(pr, sr, ddof=1)[0, 1])
        var = float(np.var(sr, ddof=1))
        expected_beta = cov / var
        observed_beta = result["style_a"] / float(np.mean(sr))
        assert np.isclose(observed_beta, expected_beta, rtol=1e-6, atol=1e-6)
