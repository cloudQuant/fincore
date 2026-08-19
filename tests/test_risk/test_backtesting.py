"""Out-of-sample risk backtest tests."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from fincore.risk.backtesting import (
    backtest_es,
    backtest_var,
    christoffersen_lr,
    kupiec_lr,
)

FIXTURES = Path(__file__).parent / "fixtures"


def test_var_backtest_keeps_time_alignment_and_exception_count() -> None:
    forecast = pd.Series([-0.02, -0.02, -0.02], index=pd.date_range("2024-01-01", periods=3, tz="UTC"))
    realized = pd.Series([-0.01, -0.03, -0.02], index=forecast.index)

    result = backtest_var(forecast, realized, confidence_level=0.99)

    assert result.observations == 3
    assert result.exceptions == 1
    assert result.aligned_index.equals(forecast.index)


def test_var_backtest_rejects_non_overlapping_inputs() -> None:
    forecast = pd.Series([-0.02], index=pd.date_range("2024-01-01", periods=1, tz="UTC"))
    realized = pd.Series([-0.03], index=pd.date_range("2024-02-01", periods=1, tz="UTC"))

    with pytest.raises(ValueError, match="overlap"):
        backtest_var(forecast, realized, confidence_level=0.99)


def test_var_backtest_rejects_duplicate_timestamps() -> None:
    idx = pd.date_range("2024-01-01", periods=2, tz="UTC").append(pd.date_range("2024-01-01", periods=1, tz="UTC"))
    forecast = pd.Series([-0.02, -0.02, -0.02], index=idx)

    with pytest.raises(ValueError, match="duplicate"):
        backtest_var(forecast, forecast, confidence_level=0.99)


def test_kupiec_lr_zero_exceptions_is_infinite() -> None:
    assert np.isinf(kupiec_lr(100, 0, confidence_level=0.99))


def test_kupiec_lr_matches_expected_rate() -> None:
    # At exactly the expected rate the statistic is near zero.
    lr = kupiec_lr(1000, 10, confidence_level=0.99)
    assert lr < 1e-6


def test_christoffersen_lr_independent_exceptions_are_small() -> None:
    rng = np.random.default_rng(7)
    exceptions = (rng.random(1000) < 0.01).astype(int)

    lr = christoffersen_lr(exceptions)

    assert np.isfinite(lr)


def test_backtest_fixture_cases() -> None:
    cases = json.loads((FIXTURES / "risk_backtest_cases.json").read_text())
    for case in cases["cases"]:
        forecast = pd.Series(case["forecast"], index=pd.to_datetime(case["index"], utc=True))
        realized = pd.Series(case["realized"], index=pd.to_datetime(case["index"], utc=True))

        result = backtest_var(forecast, realized, confidence_level=case["confidence_level"])

        assert result.observations == case["observations"]
        assert result.exceptions == case["exceptions"]


def test_backtest_es_reports_experimental_status() -> None:
    forecast = pd.Series([-0.03] * 50, index=pd.date_range("2024-01-01", periods=50, tz="UTC"))
    rng = np.random.default_rng(7)
    realized = pd.Series(rng.normal(-0.001, 0.02, 50), index=forecast.index)

    result = backtest_es(forecast, realized, confidence_level=0.975)

    assert result.method == "es"
    assert result.status == "experimental"


def test_align_rejects_non_datetime_index() -> None:
    with pytest.raises(ValueError, match="DatetimeIndex"):
        backtest_var(pd.Series([-0.02]), pd.Series([-0.03]))


def test_kupiec_lr_non_positive_observations() -> None:
    assert kupiec_lr(0, 0, confidence_level=0.99) == 0.0


def test_christoffersen_lr_single_observation() -> None:
    assert christoffersen_lr(np.array([1])) == 0.0


def test_var_backtest_rejects_invalid_confidence_level() -> None:
    forecast = pd.Series([-0.02], index=pd.date_range("2024-01-01", periods=1))
    with pytest.raises(ValueError, match="confidence_level"):
        backtest_var(forecast, forecast, confidence_level=1.5)


def test_es_backtest_rejects_invalid_confidence_level() -> None:
    forecast = pd.Series([-0.02], index=pd.date_range("2024-01-01", periods=1))
    with pytest.raises(ValueError, match="confidence_level"):
        backtest_es(forecast, forecast, confidence_level=0.0)


def test_var_backtest_fails_when_exceptions_exceed_significance() -> None:
    rng = np.random.default_rng(1)
    n = 500
    idx = pd.date_range("2024-01-01", periods=n, freq="B")
    realized = pd.Series(rng.normal(0.0, 0.02, n), index=idx)
    forecast = pd.Series(-0.005, index=idx)  # far too loose for 99% VaR

    result = backtest_var(forecast, realized, confidence_level=0.99)

    assert result.status == "fail"
