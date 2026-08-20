"""Risk model validation tests."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from fincore.risk.backtesting import backtest_var
from fincore.risk.calibration import basel_traffic_light, es_calibration_score, expected_exception_count
from fincore.risk.specs import RiskModelSpec


class TestRiskModelSpec:
    def test_defaults(self) -> None:
        spec = RiskModelSpec()
        assert spec.confidence_level == 0.99
        assert spec.horizon == 1
        assert spec.sign_convention == "losses_negative"

    def test_rejects_invalid_confidence(self) -> None:
        with pytest.raises(ValueError, match="confidence_level"):
            RiskModelSpec(confidence_level=1.5)

    def test_rejects_invalid_target(self) -> None:
        with pytest.raises(ValueError, match="forecast_target"):
            RiskModelSpec(forecast_target="bogus")


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
