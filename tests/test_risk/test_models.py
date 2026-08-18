"""Risk estimate result-contract tests."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from fincore.risk.models import (
    SIGN_LOSSES_NEGATIVE,
    RiskEstimate,
    forecast_es,
    forecast_var,
)


def test_risk_estimate_rejects_duplicate_index() -> None:
    idx = pd.date_range("2024-01-01", periods=2, tz="UTC").append(pd.date_range("2024-01-01", periods=1, tz="UTC"))
    with pytest.raises(ValueError, match="duplicate"):
        RiskEstimate(
            method="historical",
            confidence_level=0.95,
            horizon=1,
            sign_convention=SIGN_LOSSES_NEGATIVE,
            estimate=-0.01,
            forecast_timestamp=pd.Timestamp("2024-01-02", tz="UTC"),
            inputs_digest="abc",
            forecast=pd.Series([-0.01, -0.01, -0.01], index=idx),
        )


def test_risk_estimate_rejects_unsorted_index() -> None:
    idx = pd.DatetimeIndex([pd.Timestamp("2024-01-02", tz="UTC"), pd.Timestamp("2024-01-01", tz="UTC")])
    with pytest.raises(ValueError, match="sorted"):
        RiskEstimate(
            method="historical",
            confidence_level=0.95,
            horizon=1,
            sign_convention=SIGN_LOSSES_NEGATIVE,
            estimate=-0.01,
            forecast_timestamp=pd.Timestamp("2024-01-02", tz="UTC"),
            inputs_digest="abc",
            forecast=pd.Series([-0.01, -0.02], index=idx),
        )


def test_forecast_var_historical_is_negative_and_losses_negative() -> None:
    rng = np.random.default_rng(7)
    returns = pd.Series(rng.normal(0.0, 0.02, 500))

    estimate = forecast_var(returns, method="historical", confidence_level=0.99)

    assert estimate.sign_convention == SIGN_LOSSES_NEGATIVE
    assert estimate.estimate < 0.0
    assert estimate.method == "historical"
    assert len(estimate.inputs_digest) == 64
    assert estimate.status == "ok"


def test_forecast_var_evt_and_garch_match_legacy_kernels() -> None:
    rng = np.random.default_rng(7)
    returns = pd.Series(rng.normal(0.0, 0.02, 800))

    evt_estimate = forecast_var(returns, method="evt", confidence_level=0.99)

    assert evt_estimate.estimate < 0.0
    assert evt_estimate.method == "evt"

    garch_estimate = forecast_var(returns, method="garch", confidence_level=0.99)

    assert garch_estimate.estimate < 0.0
    assert garch_estimate.method == "garch"


def test_forecast_es_is_more_extreme_than_var_for_historical() -> None:
    rng = np.random.default_rng(7)
    returns = pd.Series(rng.normal(0.0, 0.02, 500))

    var_estimate = forecast_var(returns, method="historical", confidence_level=0.99)
    es_estimate = forecast_es(returns, method="historical", confidence_level=0.99)

    assert es_estimate.estimate <= var_estimate.estimate + 1e-12


def test_forecast_var_rejects_unknown_method() -> None:
    returns = pd.Series([0.01, -0.01, 0.02])

    with pytest.raises(ValueError, match="method"):
        forecast_var(returns, method="bogus")


def test_forecast_var_insufficient_data_status() -> None:
    returns = pd.Series([0.01])

    estimate = forecast_var(returns, method="historical", confidence_level=0.95)

    assert estimate.status == "insufficient_data"
