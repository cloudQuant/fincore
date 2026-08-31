"""Risk model validation tests."""

from __future__ import annotations

from dataclasses import replace
from decimal import Decimal

import numpy as np
import pandas as pd
import pytest

from fincore.risk.backtesting import backtest_var
from fincore.risk.calibration import basel_traffic_light, es_calibration_score, expected_exception_count
from fincore.risk.diagnostics import walk_forward_var
from fincore.risk.specs import RiskModelSpec


def _normal_walk_forward_result():
    index = pd.date_range("2024-01-02", periods=12, freq="B", tz="UTC")
    returns = pd.Series(
        [-0.03, 0.01, -0.02, 0.02, -0.01, 0.03, -0.04, 0.01, -0.02, 0.02, -0.01, 0.01],
        index=index,
    )
    return walk_forward_var(
        returns,
        RiskModelSpec(confidence_level=0.95, distribution="normal", window=4, refit_cadence=2),
    )


class TestRiskModelSpec:
    def test_defaults(self) -> None:
        spec = RiskModelSpec()
        assert spec.confidence_level == 0.99
        assert spec.horizon == 1
        assert spec.sign_convention == "losses_negative"

    def test_rejects_invalid_confidence(self) -> None:
        with pytest.raises(ValueError, match="confidence_level"):
            RiskModelSpec(confidence_level=1.5)

    @pytest.mark.parametrize(
        "confidence_level",
        [float("nan"), float("inf"), float("-inf"), Decimal("NaN"), Decimal("Infinity"), "0.95"],
    )
    def test_rejects_non_finite_or_non_numeric_confidence(self, confidence_level: object) -> None:
        with pytest.raises(ValueError, match="confidence_level"):
            RiskModelSpec(confidence_level=confidence_level)  # type: ignore[arg-type]

    def test_normalizes_finite_decimal_confidence_before_digest_serialization(self) -> None:
        spec = RiskModelSpec(confidence_level=Decimal("0.95"))
        idx = pd.date_range("2022-01-01", periods=3, freq="B", tz="UTC")
        returns = pd.Series([-0.02, 0.01, -0.01], index=idx)

        result = walk_forward_var(returns, replace(spec, window=2))

        assert type(spec.confidence_level) is float
        assert spec.confidence_level == pytest.approx(0.95)
        assert result.inputs_digest.isascii()
        assert len(result.inputs_digest) == 64

    def test_rejects_invalid_target(self) -> None:
        with pytest.raises(ValueError, match="forecast_target"):
            RiskModelSpec(forecast_target="bogus")

    @pytest.mark.parametrize(
        ("field", "value", "message"),
        [
            ("distribution", "garch", "distribution"),
            ("sign_convention", "losses_positive", "sign_convention"),
            ("window", 1, "window"),
            ("refit_cadence", 0, "refit_cadence"),
            ("model_version", "experimental", "model_version"),
        ],
    )
    def test_rejects_unsupported_or_invalid_model_contract(
        self,
        field: str,
        value: object,
        message: str,
    ) -> None:
        with pytest.raises(ValueError, match=message):
            RiskModelSpec(**{field: value})  # type: ignore[arg-type]


class TestBaselTrafficLight:
    def test_green_zone(self) -> None:
        for exceptions in (0, 1, 2, 3, 4):
            assert basel_traffic_light(exceptions, 250, 0.99) == "green"

    def test_yellow_zone(self) -> None:
        for exceptions in (5, 6, 7, 8, 9):
            assert basel_traffic_light(exceptions, 250, 0.99) == "yellow"

    def test_red_zone(self) -> None:
        for exceptions in (10, 11, 20):
            assert basel_traffic_light(exceptions, 250, 0.99) == "red"


class TestESCalibration:
    def test_calibrated_es_scores_near_zero(self) -> None:
        rng = np.random.default_rng(7)
        realized = rng.normal(0.0, 0.02, 2000)
        alpha = 0.01
        var = float(np.quantile(realized, alpha))
        tail = realized[realized <= var]
        forecast_es = float(tail.mean())
        score = es_calibration_score(forecast_es, realized, 0.99)
        assert np.isclose(score, 0.0, atol=1e-6)


class TestExpectedExceptions:
    def test_expected_exception_count(self) -> None:
        assert np.isclose(expected_exception_count(250, 0.99), 2.5)
        assert np.isclose(expected_exception_count(1000, 0.95), 50.0)


class TestWalkForward:
    def test_public_historical_forecast_is_strictly_out_of_sample_and_backtestable(self) -> None:
        rng = np.random.default_rng(7)
        idx = pd.date_range("2020-01-01", periods=160, freq="B", tz="UTC")
        returns = pd.Series(rng.normal(0.0, 0.02, len(idx)), index=idx)
        spec = RiskModelSpec(confidence_level=0.95, distribution="historical", window=40, refit_cadence=5)

        result = walk_forward_var(returns, spec)

        ordered_window = np.sort(returns.iloc[:40].to_numpy())
        rank = (len(ordered_window) + 1) * 0.05
        lower_index = int(np.floor(rank)) - 1
        upper_index = lower_index + 1
        expected_first = float(
            ordered_window[lower_index]
            + (rank - np.floor(rank)) * (ordered_window[upper_index] - ordered_window[lower_index])
        )
        assert result.status == "ok"
        assert result.forecast.index.equals(idx[40:])
        assert result.realized.index.equals(result.forecast.index)
        assert result.realized.equals(returns.iloc[40:])
        assert result.forecast.iloc[0] == pytest.approx(expected_first)
        assert result.refit_timestamps.equals(idx[40::5])
        assert len(result.inputs_digest) == 64
        assert result.backtest is not None
        assert result.backtest.observations == len(result.forecast)

    def test_historical_forecast_uses_weibull_finite_sample_quantile(self) -> None:
        idx = pd.date_range("2020-01-01", periods=253, freq="B", tz="UTC")
        returns = pd.Series(np.arange(253, dtype=float), index=idx)
        spec = RiskModelSpec(confidence_level=0.95, distribution="historical", window=252)

        result = walk_forward_var(returns, spec)

        # Hyndman-Fan type 6 / NumPy ``weibull``: h = (n + 1) * p = 12.65.
        assert result.forecast.iloc[0] == pytest.approx(11.65)
        assert result.diagnostics["quantile_method"] == "weibull"

    @pytest.mark.parametrize("confidence_level", [0.9999, 0.0001])
    def test_historical_weibull_rejects_confidence_outside_finite_sample_support(self, confidence_level: float) -> None:
        idx = pd.date_range("2020-01-01", periods=4, freq="B", tz="UTC")
        returns = pd.Series([-0.02, -0.01, 0.01, 0.02], index=idx)
        spec = RiskModelSpec(confidence_level=confidence_level, distribution="historical", window=2)

        result = walk_forward_var(returns, spec)

        assert result.status == "unsupported"
        assert result.forecast.empty
        assert result.realized.empty
        assert result.refit_timestamps.empty
        assert result.backtest is None
        assert "weibull" in result.diagnostics["reason"].lower()

    @pytest.mark.parametrize("confidence_level", [1.0 / 3.0, 1.0 - 1.0 / 3.0])
    def test_historical_weibull_accepts_finite_sample_support_boundaries(self, confidence_level: float) -> None:
        idx = pd.date_range("2020-01-01", periods=4, freq="B", tz="UTC")
        returns = pd.Series([-0.02, -0.01, 0.01, 0.02], index=idx)
        spec = RiskModelSpec(confidence_level=confidence_level, distribution="historical", window=2)

        result = walk_forward_var(returns, spec)

        assert result.status == "ok"
        assert result.diagnostics["quantile_method"] == "weibull"

    def test_historical_weibull_rejects_material_rank_deficit_at_large_window(self) -> None:
        window = 10_000_000
        alpha = 9.9e-8
        idx = pd.date_range("2020-01-01", periods=2, freq="B", tz="UTC")
        returns = pd.Series([-0.02, 0.01], index=idx)
        spec = RiskModelSpec(
            confidence_level=1.0 - alpha,
            distribution="historical",
            window=window,
        )

        result = walk_forward_var(returns, spec)

        assert result.status == "unsupported"
        assert "weibull" in result.diagnostics["reason"].lower()

    def test_historical_walk_forward_has_calibrated_fixed_stream_coverage(self) -> None:
        """A deterministic iid normal probe independent of the quantile implementation."""
        rng = np.random.default_rng(20260821)
        idx = pd.date_range("2000-01-03", periods=5_000, freq="B", tz="UTC")
        returns = pd.Series(rng.normal(size=len(idx)), index=idx)
        spec = RiskModelSpec(confidence_level=0.99, distribution="historical", window=252)

        result = walk_forward_var(returns, spec)

        assert result.backtest is not None
        assert result.backtest.exception_rate == pytest.approx(0.01095, abs=0.001)
        assert abs(result.backtest.exception_rate - 0.01) < 0.002

    def test_normal_forecast_reuses_fit_until_the_next_deterministic_refit(self) -> None:
        idx = pd.date_range("2021-01-01", periods=80, freq="B", tz="UTC")
        returns = pd.Series(np.linspace(-0.03, 0.04, len(idx)), index=idx)
        spec = RiskModelSpec(distribution="normal", confidence_level=0.95, window=20, refit_cadence=4)

        result = walk_forward_var(returns, spec)

        assert result.status == "ok"
        assert result.forecast.iloc[0] == pytest.approx(result.forecast.iloc[1])
        assert result.forecast.iloc[1] == pytest.approx(result.forecast.iloc[3])
        assert result.forecast.iloc[4] != pytest.approx(result.forecast.iloc[3])
        assert result.refit_timestamps.equals(idx[20::4])

    def test_future_perturbation_cannot_change_completed_forecasts(self) -> None:
        rng = np.random.default_rng(12)
        idx = pd.date_range("2022-01-01", periods=180, freq="B", tz="UTC")
        returns = pd.Series(rng.normal(0.0, 0.02, len(idx)), index=idx)
        spec = RiskModelSpec(distribution="historical", confidence_level=0.95, window=40, refit_cadence=1)
        baseline = walk_forward_var(returns, spec)

        cutoff = idx[120]
        changed = returns.copy()
        changed.loc[cutoff:] = changed.loc[cutoff:] - 0.5
        perturbed = walk_forward_var(changed, spec)

        pd.testing.assert_series_equal(
            baseline.forecast.loc[:cutoff],
            perturbed.forecast.loc[:cutoff],
            check_names=False,
        )
        assert baseline.inputs_digest != perturbed.inputs_digest

    def test_reports_structured_insufficiency_and_unsupported_status(self) -> None:
        idx = pd.date_range("2022-01-01", periods=10, freq="B", tz="UTC")
        returns = pd.Series(np.linspace(-0.01, 0.01, len(idx)), index=idx)

        insufficient = walk_forward_var(returns, RiskModelSpec(window=10))
        unsupported = walk_forward_var(returns, RiskModelSpec(forecast_target="es", window=2))

        assert insufficient.status == "insufficient_data"
        assert insufficient.forecast.empty
        assert insufficient.backtest is None
        assert unsupported.status == "unsupported"
        assert unsupported.forecast.empty
        assert unsupported.backtest is None

    def test_insufficient_data_keeps_its_structured_status_for_a_named_datetime_index(self) -> None:
        index = pd.date_range("2022-01-01", periods=4, freq="B", tz="UTC", name=42)
        returns = pd.Series(np.linspace(-0.02, 0.02, len(index)), index=index)

        result = walk_forward_var(returns, RiskModelSpec(window=4))

        assert result.status == "insufficient_data"
        assert result.forecast.empty
        assert result.realized.empty

    def test_walk_forward_result_enforces_status_specific_state_invariants(self) -> None:
        idx = pd.date_range("2022-01-01", periods=8, freq="B", tz="UTC")
        returns = pd.Series(np.linspace(-0.02, 0.02, len(idx)), index=idx)
        valid = walk_forward_var(returns, RiskModelSpec(window=3, distribution="normal"))
        insufficient = walk_forward_var(returns.iloc[:3], RiskModelSpec(window=3))
        unsupported = walk_forward_var(returns, RiskModelSpec(forecast_target="es", window=3))

        assert valid.backtest is not None
        with pytest.raises(ValueError, match="ok result must contain a non-empty forecast path"):
            replace(valid, forecast=valid.forecast.iloc[:0], realized=valid.realized.iloc[:0])
        with pytest.raises(ValueError, match="ok result must contain at least one refit timestamp"):
            replace(valid, refit_timestamps=valid.refit_timestamps[:0])
        with pytest.raises(ValueError, match="refit_timestamps must follow the configured refit cadence"):
            replace(valid, refit_timestamps=valid.refit_timestamps[:1])
        with pytest.raises(ValueError, match="ok result must contain a backtest"):
            replace(valid, backtest=None)
        with pytest.raises(ValueError, match="inputs_digest must be a lowercase SHA-256 hex digest"):
            replace(valid, inputs_digest="not-a-digest")
        with pytest.raises(ValueError, match="backtest aligned_index must equal forecast index"):
            replace(valid, backtest=replace(valid.backtest, aligned_index=valid.backtest.aligned_index[1:]))
        with pytest.raises(ValueError, match="forecast and realized indexes must have the same name"):
            replace(valid, realized=valid.realized.rename_axis("different_name"))
        with pytest.raises(ValueError, match="ok result forecast and realized must contain only finite values"):
            replace(valid, forecast=valid.forecast * np.nan, realized=valid.realized * np.nan)
        with pytest.raises(ValueError, match="backtest confidence_level must equal the specification"):
            replace(valid, backtest=replace(valid.backtest, confidence_level=0.5))
        with pytest.raises(ValueError, match="backtest must match the forecast and realized path"):
            replace(
                valid,
                backtest=replace(valid.backtest, diagnostics={"significance": 0.99, "small_sample": False}),
            )
        fit_parameters = dict(valid.diagnostics["fit_parameters"])
        fit_parameters[valid.refit_timestamps[0].isoformat()] = {}
        with pytest.raises(ValueError, match="must include mean, standard_deviation, n_observations, and forecast"):
            replace(valid, diagnostics={**valid.diagnostics, "fit_parameters": fit_parameters})
        with pytest.raises(ValueError, match="diagnostics method must equal the specification distribution"):
            replace(valid, diagnostics={**valid.diagnostics, "method": "historical"})
        with pytest.raises(ValueError, match="must reproduce the forecast segment"):
            replace(valid, forecast=valid.forecast + 0.001)
        with pytest.raises(ValueError, match="insufficient_data result must have an empty forecast path"):
            replace(valid, status="insufficient_data")
        with pytest.raises(ValueError, match="insufficient_data result must not contain a backtest"):
            replace(insufficient, backtest=valid.backtest)
        with pytest.raises(ValueError, match="must include a non-empty diagnostic reason"):
            replace(insufficient, diagnostics={})
        with pytest.raises(ValueError, match="must include a non-empty diagnostic reason"):
            replace(unsupported, diagnostics={})

    def test_refit_parameters_reproduce_the_recorded_normal_forecast_segments(self) -> None:
        result = _normal_walk_forward_result()
        fit_parameters = result.diagnostics["fit_parameters"]

        for position, timestamp in enumerate(result.refit_timestamps):
            parameters = fit_parameters[timestamp.isoformat()]
            segment_end = result.refit_timestamps[position + 1] if position + 1 < len(result.refit_timestamps) else None
            segment = (
                result.forecast.loc[timestamp:]
                if segment_end is None
                else result.forecast.loc[timestamp:segment_end].iloc[:-1]
            )
            assert parameters["forecast"] == pytest.approx(segment.iloc[0])
            assert segment.eq(parameters["forecast"]).all()

    def test_refit_parameters_record_the_historical_forecast_threshold(self) -> None:
        index = pd.date_range("2024-01-02", periods=10, freq="B", tz="UTC")
        returns = pd.Series(np.linspace(-0.03, 0.03, len(index)), index=index)
        result = walk_forward_var(
            returns,
            RiskModelSpec(confidence_level=0.8, distribution="historical", window=4, refit_cadence=2),
        )

        for timestamp in result.refit_timestamps:
            parameters = result.diagnostics["fit_parameters"][timestamp.isoformat()]
            assert parameters["quantile_method"] == "weibull"
            assert parameters["forecast"] == pytest.approx(result.forecast.loc[timestamp])

    def test_rejects_refit_parameters_that_do_not_reproduce_normal_forecasts(self) -> None:
        valid = _normal_walk_forward_result()
        fit_parameters = {key: dict(value) for key, value in valid.diagnostics["fit_parameters"].items()}
        first_key = valid.refit_timestamps[0].isoformat()
        fit_parameters[first_key]["mean"] += 0.5

        with pytest.raises(ValueError, match="must reproduce the forecast segment"):
            replace(valid, diagnostics={**valid.diagnostics, "fit_parameters": fit_parameters})

    def test_forecast_uses_only_past_data(self) -> None:
        rng = np.random.default_rng(7)
        n = 400
        window = 100
        returns = pd.Series(rng.normal(0.0, 0.02, n))
        idx = pd.date_range("2020-01-01", periods=n)

        forecast_timestamps = []
        for t in range(window, n):
            train = returns.iloc[t - window : t]
            var = float(np.quantile(train.to_numpy(), 0.01))
            forecast_timestamps.append((idx[t], var))

        # Every forecast was produced from a window strictly before its timestamp.
        for t, (ts, _) in enumerate(forecast_timestamps):
            window_end = window + t
            assert ts == idx[window_end]

    def test_backtest_var_walk_forward(self) -> None:
        rng = np.random.default_rng(7)
        n = 300
        idx = pd.date_range("2020-01-01", periods=n)
        returns = pd.Series(rng.normal(0.0, 0.02, n), index=idx)
        forecast = pd.Series(index=idx, dtype=float)
        window = 100
        for t in range(window, n):
            forecast.iloc[t] = float(np.quantile(returns.iloc[t - window : t].to_numpy(), 0.01))

        result = backtest_var(forecast.iloc[window:], returns.iloc[window:], confidence_level=0.99)
        assert result.observations == n - window
        assert result.exceptions >= 0
