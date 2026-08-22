"""Factor inference and PIT numerical tests."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm
from scipy import stats
from statsmodels.stats.multitest import multipletests

from fincore.factor_analysis.inference import (
    benjamini_hochberg,
    fama_macbeth,
    ic_confidence_interval,
    ic_mean,
    ic_t_stat,
)
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

    def test_t_stat_matches_scipy_and_zero_constant_sample_is_zero(self) -> None:
        values = np.array([-0.02, 0.01, 0.03, 0.04, -0.01])

        assert np.isclose(ic_t_stat(values), stats.ttest_1samp(values, popmean=0.0).statistic, rtol=1e-12)
        assert ic_t_stat(np.zeros(5)) == 0.0

    def test_rejects_infinite_ic_observations_and_invalid_interval_multiplier(self) -> None:
        infinite = np.array([0.01, np.inf])

        with pytest.raises(ValueError, match="infinite"):
            ic_mean(infinite)
        with pytest.raises(ValueError, match="infinite"):
            ic_t_stat(infinite)
        with pytest.raises(ValueError, match="infinite"):
            ic_confidence_interval(infinite)
        with pytest.raises(ValueError, match="z"):
            ic_confidence_interval(np.array([0.01, 0.02]), z=0.0)


class TestBenjaminiHochberg:
    def test_matches_statsmodels_and_preserves_factor_labels(self) -> None:
        """FDR decisions and q-values agree with an independent implementation."""
        p_values = pd.Series(
            [0.049, 0.001, 0.01, 0.01, 0.2, 0.9],
            index=pd.Index(["value", "quality", "momentum", "size", "low_vol", "noise"], name="factor"),
            name="p_value",
        )
        expected_rejected, expected_adjusted, _, _ = multipletests(p_values.to_numpy(), alpha=0.05, method="fdr_bh")

        result = benjamini_hochberg(p_values, alpha=0.05)

        assert result.method == "benjamini-hochberg"
        assert result.alpha == 0.05
        pd.testing.assert_series_equal(result.p_values, p_values.astype(float))
        pd.testing.assert_series_equal(
            result.adjusted_p_values,
            pd.Series(expected_adjusted, index=p_values.index, name="adjusted_p_value"),
        )
        pd.testing.assert_series_equal(
            result.rejected,
            pd.Series(expected_rejected, index=p_values.index, name="rejected"),
        )

    @pytest.mark.parametrize(
        ("p_values", "alpha"),
        [
            (np.array([0.1, np.nan]), 0.05),
            (np.array([-0.1, 0.1]), 0.05),
            (np.array([0.1, 1.1]), 0.05),
            (np.array([[0.1, 0.2]]), 0.05),
            (np.array([0.1, 0.2]), 0.0),
            (np.array([0.1, 0.2]), 1.1),
        ],
    )
    def test_rejects_invalid_probabilities_and_alpha(self, p_values: np.ndarray, alpha: float) -> None:
        with pytest.raises(ValueError):
            benjamini_hochberg(p_values, alpha=alpha)

    def test_empty_series_has_an_explicit_empty_audit_result(self) -> None:
        empty = pd.Series([], dtype=float, index=pd.Index([], name="factor"), name="p_value")

        result = benjamini_hochberg(empty)

        assert result.n_tests == 0
        assert result.p_values.empty
        assert result.adjusted_p_values.empty
        assert result.rejected.empty

    def test_rejects_duplicate_factor_labels(self) -> None:
        duplicate_labels = pd.Series([0.01, 0.02], index=["value", "value"])

        with pytest.raises(ValueError, match="duplicate"):
            benjamini_hochberg(duplicate_labels)


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
