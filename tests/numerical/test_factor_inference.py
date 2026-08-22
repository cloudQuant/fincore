"""Factor inference and PIT numerical tests."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm

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

    def test_broadcasts_one_row_static_exposure_across_return_dates(self) -> None:
        """A documented static cross-section must not disappear in an index intersection."""
        dates = pd.date_range("2024-01-01", periods=8)
        assets = ["a", "b", "c"]
        exposure = np.array([-1.0, 0.0, 1.0])
        returns = pd.DataFrame([0.01 + 2.0 * exposure for _ in dates], index=dates, columns=assets)
        static_exposure = pd.DataFrame([exposure], index=[pd.Timestamp("2000-01-01")], columns=assets)

        result = fama_macbeth(returns, static_exposure)

        assert np.isclose(result.loc["intercept", "mean"], 0.01, atol=1e-12)
        assert np.isclose(result.loc["exposure", "mean"], 2.0, atol=1e-12)

    def test_aligns_exposures_by_asset_label_not_input_column_position(self) -> None:
        """Reordering an exposure panel must not invert a cross-sectional beta."""
        dates = pd.date_range("2024-01-01", periods=8)
        assets = ["a", "b", "c"]
        exposure = np.array([-1.0, 0.0, 1.0])
        returns = pd.DataFrame([0.01 + 2.0 * exposure for _ in dates], index=dates, columns=assets)
        shuffled_exposures = pd.DataFrame(
            [exposure[::-1] for _ in dates],
            index=dates,
            columns=["c", "b", "a"],
        )

        result = fama_macbeth(returns, shuffled_exposures)

        assert np.isclose(result.loc["intercept", "mean"], 0.01, atol=1e-12)
        assert np.isclose(result.loc["exposure", "mean"], 2.0, atol=1e-12)

    def test_matches_statsmodels_cross_sectional_oracle(self) -> None:
        """The panel mean and i.i.d. Fama-MacBeth SE agree with OLS fixtures."""
        rng = np.random.default_rng(31)
        dates = pd.date_range("2024-01-01", periods=12)
        assets = ["a", "b", "c", "d", "e"]
        exposures = pd.DataFrame(rng.normal(size=(len(dates), len(assets))), index=dates, columns=assets)
        returns = (
            0.003
            + 1.25 * exposures
            + pd.DataFrame(rng.normal(scale=0.02, size=exposures.shape), index=dates, columns=assets)
        )

        expected_coefficients = []
        for date in dates:
            design = sm.add_constant(exposures.loc[date].to_numpy(dtype=float), has_constant="add")
            expected_coefficients.append(sm.OLS(returns.loc[date].to_numpy(dtype=float), design).fit().params)
        expected = np.asarray(expected_coefficients)

        result = fama_macbeth(returns, exposures)

        assert np.isclose(result.loc["intercept", "mean"], expected[:, 0].mean(), rtol=1e-12, atol=1e-12)
        assert np.isclose(result.loc["exposure", "mean"], expected[:, 1].mean(), rtol=1e-12, atol=1e-12)
        assert np.isclose(
            result.loc["intercept", "std_error"],
            expected[:, 0].std(ddof=1) / np.sqrt(len(expected)),
            rtol=1e-12,
            atol=1e-12,
        )
        assert np.isclose(
            result.loc["exposure", "std_error"],
            expected[:, 1].std(ddof=1) / np.sqrt(len(expected)),
            rtol=1e-12,
            atol=1e-12,
        )


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
