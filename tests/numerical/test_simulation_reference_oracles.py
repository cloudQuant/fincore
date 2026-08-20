"""Simulation-domain numerical correctness tests against analytic GBM oracle."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from fincore.simulation.base import estimate_parameters
from fincore.simulation.paths import gbm_from_returns, geometric_brownian_motion
from tests.oracles.simulation.gbm_oracle import (
    gbm_terminal_log_std,
    gbm_terminal_log_std_ci,
    gbm_terminal_mean,
    gbm_terminal_mean_ci,
)


class TestGBMTerminalVolatility:
    """A 20% annualized volatility over one year must produce ~20% log vol."""

    def test_gbm_from_returns_terminal_log_vol(self) -> None:
        rng = np.random.default_rng(42)
        returns = rng.normal(0.0, 0.02, 5000)  # daily vol 2%
        annual_vol = 0.02 * np.sqrt(252)
        paths = gbm_from_returns(returns, horizon=252, n_paths=5000, frequency=252, seed=7)
        log_ret = np.log(paths[:, -1] + 1.0)
        est_vol = float(np.std(log_ret, ddof=1))
        assert abs(est_vol - annual_vol) / annual_vol < 0.05, (
            f"terminal log vol {est_vol:.4f} vs annual vol {annual_vol:.4f}"
        )

    def test_geometric_brownian_motion_terminal_log_vol(self) -> None:
        paths = geometric_brownian_motion(
            S0=100.0, mu=0.10, sigma=0.20, T=1.0, dt=1 / 252, n_paths=20000, seed=42
        )
        log_ret = np.log(paths[:, -1] / 100.0)
        est_vol = float(np.std(log_ret, ddof=1))
        assert abs(est_vol - 0.20) / 0.20 < 0.05

    def test_gbm_terminal_mean_inside_analytic_ci(self) -> None:
        paths = geometric_brownian_motion(
            S0=100.0, mu=0.10, sigma=0.20, T=1.0, dt=1 / 252, n_paths=20000, seed=42
        )
        lo, hi = gbm_terminal_mean_ci(100.0, 0.10, 0.20, 1.0, 20000)
        observed_mean = float(np.mean(paths[:, -1]))
        assert lo < observed_mean < hi

    def test_gbm_terminal_log_std_inside_analytic_ci(self) -> None:
        paths = geometric_brownian_motion(
            S0=100.0, mu=0.10, sigma=0.20, T=1.0, dt=1 / 252, n_paths=20000, seed=42
        )
        log_ret = np.log(paths[:, -1] / 100.0)
        lo, hi = gbm_terminal_log_std_ci(0.20, 1.0, 20000)
        observed_std = float(np.std(log_ret, ddof=1))
        assert lo < observed_std < hi


class TestMonteCarloParameterRespect:
    """MonteCarlo.simulate must honor drift/volatility parameters."""

    def test_drift_and_volatility_change_output(self) -> None:
        from fincore.simulation import MonteCarlo

        rng = np.random.default_rng(42)
        returns = pd.Series(rng.normal(0.0, 0.02, 1000))
        mc = MonteCarlo(returns)

        up = mc.simulate(n_paths=5000, horizon=252, drift=0.3, volatility=0.1, seed=7)
        down = mc.simulate(n_paths=5000, horizon=252, drift=-0.3, volatility=0.5, seed=7)

        assert np.mean(up.paths[:, -1]) > np.mean(down.paths[:, -1])

    def test_volatility_scales_dispersion(self) -> None:
        from fincore.simulation import MonteCarlo

        rng = np.random.default_rng(42)
        returns = pd.Series(rng.normal(0.0, 0.02, 1000))
        mc = MonteCarlo(returns)

        low = mc.simulate(n_paths=5000, horizon=252, drift=0.0, volatility=0.1, seed=7)
        high = mc.simulate(n_paths=5000, horizon=252, drift=0.0, volatility=0.4, seed=7)

        assert np.std(low.paths[:, -1], ddof=1) < np.std(high.paths[:, -1], ddof=1)


class TestAntitheticVariates:
    """Antithetic paths must be the Z / -Z pairing of the same random stream."""

    def test_antithetic_uses_negated_shocks(self) -> None:
        paths = geometric_brownian_motion(
            S0=100.0, mu=0.10, sigma=0.20, T=0.25, dt=1 / 252, n_paths=1000, seed=42, antithetic=True
        )
        assert paths.shape[0] == 2000

    def test_antithetic_pair_mean_is_deterministic(self) -> None:
        # The paired (Z, -Z) paths should have a symmetric terminal log-return
        # structure: for each pair, log(S+/S0) + log(S-/S0) = 2*(mu-0.5s^2)*T.
        n = 500
        paths = geometric_brownian_motion(
            S0=1.0, mu=0.10, sigma=0.20, T=1.0, dt=1 / 252, n_paths=n, seed=7, antithetic=True
        )
        log_plus = np.log(paths[:n, -1])
        log_minus = np.log(paths[n:, -1])
        target = 2.0 * (0.10 - 0.5 * 0.20**2) * 1.0
        paired_sum = log_plus + log_minus
        assert np.allclose(paired_sum, target, rtol=1e-9, atol=1e-9)


class TestEstimateParameters:
    """estimate_parameters must return annualized values (no double scaling)."""

    def test_annualized_volatility(self) -> None:
        rng = np.random.default_rng(42)
        returns = rng.normal(0.0, 0.02, 10000)
        _, sigma = estimate_parameters(returns, frequency=252)
        assert abs(sigma - 0.02 * np.sqrt(252)) / (0.02 * np.sqrt(252)) < 0.05
