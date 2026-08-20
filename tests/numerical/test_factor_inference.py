"""Factor inference and PIT numerical tests."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from fincore.factor_analysis.inference import fama_macbeth, ic_confidence_interval, ic_mean, ic_t_stat
from fincore.factor_analysis.pit import PITPoint, validate_pit_alignment


class TestFamaMacBeth:
    def test_recovers_known_coefficients(self) -> None:
        rng = np.random.default_rng(42)
        n_periods = 200
        n_assets = 50
        true_alpha = 0.001
        true_slope = 1.5
        periods = pd.date_range("2020-01-01", periods=n_periods)
        assets = [f"a{i}" for i in range(n_assets)]

        xs = rng.normal(0.0, 1.0, n_assets)
        exposures = pd.DataFrame(np.tile(xs, (n_periods, 1)), index=periods, columns=assets)
        returns = pd.DataFrame(index=periods, columns=assets)
        for t in periods:
            returns.loc[t] = true_alpha + true_slope * xs + rng.normal(0.0, 0.01, n_assets)

        result = fama_macbeth(returns, exposures)

        assert abs(result.loc["exposure", "mean"] - true_slope) < 0.01
        assert abs(result.loc["intercept", "mean"] - true_alpha) < 0.01
        assert result.loc["exposure", "t_stat"] > 10.0


class TestIC:
    def test_ic_mean_and_t_stat(self) -> None:
        rng = np.random.default_rng(7)
        ic = rng.normal(0.05, 0.01, 100)
        assert np.isclose(ic_mean(ic), np.mean(ic), rtol=1e-9)
        assert ic_t_stat(ic) > 0

    def test_ic_confidence_interval_contains_mean(self) -> None:
        rng = np.random.default_rng(7)
        ic = rng.normal(0.05, 0.01, 100)
        lo, hi = ic_confidence_interval(ic)
        assert lo < np.mean(ic) < hi


class TestPIT:
    def test_rejects_lookahead(self) -> None:
        with pytest.raises(ValueError, match="look-ahead"):
            PITPoint(
                as_of=pd.Timestamp("2020-01-02"),
                known_at=pd.Timestamp("2020-01-01"),
                effective_from=pd.Timestamp("2020-01-01"),
                value=1.0,
            )

    def test_accepts_valid_points(self) -> None:
        points = [
            PITPoint(
                as_of=pd.Timestamp("2020-01-01"),
                known_at=pd.Timestamp("2020-01-01"),
                effective_from=pd.Timestamp("2020-01-01"),
                value=1.0,
            )
        ]
        validate_pit_alignment(points)
