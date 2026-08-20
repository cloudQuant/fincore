"""Performance semantics numerical tests against independent oracles."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from fincore.performance import mwr, sharpe_standard_error, twr, xirr


def _npv(cashflows: np.ndarray, times: np.ndarray, rate: float) -> float:
    return float(np.sum(cashflows / (1.0 + rate) ** times))


class TestTWR:
    def test_twr_matches_hand_computed(self) -> None:
        returns = pd.Series([0.01, -0.02, 0.03, 0.005])
        expected = float((1.01) * (0.98) * (1.03) * (1.005) - 1.0)
        assert np.isclose(twr(returns), expected, rtol=1e-12, atol=1e-12)

    def test_twr_ignores_nan(self) -> None:
        returns = pd.Series([0.01, np.nan, 0.02])
        assert np.isclose(twr(returns), (1.01) * (1.02) - 1.0, rtol=1e-12)


class TestMWR:
    def test_mwr_satisfies_npv_zero(self) -> None:
        cashflows = np.array([-100.0, 10.0, 10.0, 110.0])
        rate = mwr(cashflows)
        times = np.arange(len(cashflows), dtype=float)
        assert np.isclose(_npv(cashflows, times, rate), 0.0, atol=1e-8)

    def test_mwr_known_value(self) -> None:
        cashflows = np.array([-100.0, 110.0])
        rate = mwr(cashflows)
        assert np.isclose(rate, 0.10, rtol=1e-6)


class TestXIRR:
    def test_xirr_matches_mwr_for_annual_dates(self) -> None:
        cashflows = pd.Series([-100.0, 110.0])
        dates = pd.to_datetime(["2019-01-01", "2020-01-01"])
        rate = xirr(cashflows, dates)
        assert np.isclose(rate, 0.10, rtol=1e-6)


class TestSharpeInference:
    def test_sharpe_se_is_positive_and_finite(self) -> None:
        rng = np.random.default_rng(42)
        returns = rng.normal(0.001, 0.02, 252)
        se = sharpe_standard_error(returns)
        assert se > 0.0
        assert np.isfinite(se)

    def test_sharpe_se_scales_with_sample_size(self) -> None:
        rng = np.random.default_rng(7)
        short = rng.normal(0.001, 0.02, 100)
        long_ = rng.normal(0.001, 0.02, 10000)
        assert sharpe_standard_error(long_) < sharpe_standard_error(short)
