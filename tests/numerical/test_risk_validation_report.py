"""Contract tests for the auditable walk-forward VaR report."""

from __future__ import annotations

import json
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest
from dateutil import tz

from fincore.risk import (
    BASEL_REFERENCE_DISCLOSURE,
    RiskModelSpec,
    build_risk_validation_report,
    walk_forward_var,
)
from fincore.risk.backtesting import backtest_var
from fincore.risk.calibration import basel_traffic_light


def _validated_result():
    index = pd.date_range("2024-01-02", periods=12, freq="B", tz="UTC")
    returns = pd.Series(
        [-0.03, 0.01, -0.02, 0.02, -0.01, 0.03, -0.04, 0.01, -0.02, 0.02, -0.01, 0.01],
        index=index,
    )
    return walk_forward_var(
        returns,
        RiskModelSpec(confidence_level=0.95, distribution="normal", window=4, refit_cadence=2),
    )


def test_report_reconstructs_every_forecast_exception_and_refit() -> None:
    result = _validated_result()

    report = build_risk_validation_report(result)
    payload = report.to_dict()

    assert result.status == "ok"
    assert report.status == result.status
    assert payload["schema_version"] == 1
    assert payload["inputs_digest"] == result.inputs_digest
    assert payload["timestamp_index_name"] is None
    assert payload["timestamp_timezone"] == "UTC"
    assert payload["specification"] == {
        "confidence_level": 0.95,
        "distribution": "normal",
        "forecast_target": "var",
        "horizon": 1,
        "model_version": "1.0",
        "refit_cadence": 2,
        "sign_convention": "losses_negative",
        "tail": "lower",
        "window": 4,
    }
    assert payload["disclosure"] == BASEL_REFERENCE_DISCLOSURE

    assert len(payload["forecast_events"]) == len(result.forecast)
    assert len(payload["refits"]) == len(result.refit_timestamps)
    for event, timestamp in zip(payload["forecast_events"], result.forecast.index, strict=True):
        assert event == {
            "timestamp": timestamp.isoformat(),
            "forecast": float(result.forecast.loc[timestamp]),
            "realized": float(result.realized.loc[timestamp]),
            "exception": bool(result.realized.loc[timestamp] < result.forecast.loc[timestamp]),
            "refit": bool(timestamp in result.refit_timestamps),
        }

    fit_parameters = result.diagnostics["fit_parameters"]
    for refit, timestamp in zip(payload["refits"], result.refit_timestamps, strict=True):
        assert refit == {
            "timestamp": timestamp.isoformat(),
            "parameters": fit_parameters[timestamp.isoformat()],
        }

    assert result.backtest is not None
    assert payload["backtest"] == {
        "method": result.backtest.method,
        "confidence_level": result.backtest.confidence_level,
        "observations": result.backtest.observations,
        "exceptions": result.backtest.exceptions,
        "expected_exceptions": result.backtest.expected_exceptions,
        "inputs_digest": result.backtest.inputs_digest,
        "exception_rate": result.backtest.exception_rate,
        "kupiec_lr": result.backtest.kupiec_lr,
        "kupiec_pvalue": result.backtest.kupiec_pvalue,
        "christoffersen_lr": result.backtest.christoffersen_lr,
        "christoffersen_pvalue": result.backtest.christoffersen_pvalue,
        "diagnostics": result.backtest.diagnostics,
        "status": result.backtest.status,
        "traffic_light": {
            "zone": basel_traffic_light(
                result.backtest.exceptions,
                result.backtest.observations,
                result.backtest.confidence_level,
            ),
            "observations": result.backtest.observations,
            "confidence_level": result.backtest.confidence_level,
        },
    }
    assert "fit_parameters" not in payload["diagnostics"]
    assert json.loads(report.to_json()) == payload


def test_report_preserves_structured_insufficient_data_status() -> None:
    index = pd.date_range("2024-01-02", periods=4, freq="B", tz="UTC")
    result = walk_forward_var(pd.Series(np.linspace(-0.02, 0.02, 4), index=index), RiskModelSpec(window=4))

    report = build_risk_validation_report(result)

    assert report.status == "insufficient_data"
    assert report.forecast_events == ()
    assert report.refits == ()
    assert report.backtest is None
    assert report.to_dict()["diagnostics"]["reason"].startswith("at least one realized")


def test_report_writes_a_deterministic_json_artifact(tmp_path) -> None:
    report = build_risk_validation_report(_validated_result())

    path = report.write_json(tmp_path / "risk-validation.json")

    assert path.name == "risk-validation.json"
    assert json.loads(path.read_text(encoding="utf-8")) == report.to_dict()


def test_report_rejects_non_walk_forward_results() -> None:
    with pytest.raises(TypeError, match="WalkForwardVaRResult"):
        build_risk_validation_report(object())


def test_report_fails_closed_when_a_refit_has_no_recorded_parameters() -> None:
    result = _validated_result()
    diagnostics = {key: value for key, value in result.diagnostics.items() if key != "fit_parameters"}

    with pytest.raises(ValueError, match="record fit_parameters"):
        invalid = replace(result, diagnostics=diagnostics)
        build_risk_validation_report(invalid)


def test_report_rejects_non_json_diagnostics() -> None:
    result = _validated_result()
    invalid = replace(result, diagnostics={**result.diagnostics, "opaque": object()})

    with pytest.raises(TypeError, match="not JSON-compatible"):
        build_risk_validation_report(invalid)


def test_report_revalidates_a_mutated_forecast_before_serializing() -> None:
    result = _validated_result()
    result.forecast.iloc[0] = 1.0

    with pytest.raises(ValueError, match="must reproduce the forecast segment"):
        build_risk_validation_report(result)


def test_report_revalidates_mutated_refit_parameters_before_serializing() -> None:
    result = _validated_result()
    parameters = result.diagnostics["fit_parameters"]
    parameters[result.refit_timestamps[0].isoformat()]["mean"] = float("nan")

    with pytest.raises(ValueError, match="mean must be a finite real number"):
        build_risk_validation_report(result)


def test_built_report_cannot_be_mutated_after_audit_snapshot() -> None:
    report = build_risk_validation_report(_validated_result())

    with pytest.raises(TypeError):
        report.forecast_events[0]["forecast"] = 1.0
    with pytest.raises(TypeError):
        report.refits[0]["parameters"]["forecast"] = 1.0


def test_report_keeps_the_index_name_needed_to_replay_backtest_digest() -> None:
    result = _validated_result()
    forecast = result.forecast.rename_axis("audit_time")
    realized = result.realized.rename_axis("audit_time")
    named_result = replace(
        result,
        forecast=forecast,
        realized=realized,
        backtest=backtest_var(forecast, realized, confidence_level=result.spec.confidence_level),
    )

    payload = build_risk_validation_report(named_result).to_dict()
    timestamp_index = pd.DatetimeIndex(
        [event["timestamp"] for event in payload["forecast_events"]],
        name=payload["timestamp_index_name"],
    )
    replayed_forecast = pd.Series(
        [event["forecast"] for event in payload["forecast_events"]],
        index=timestamp_index,
    )
    replayed_realized = pd.Series(
        [event["realized"] for event in payload["forecast_events"]],
        index=timestamp_index,
    )
    replayed = backtest_var(replayed_forecast, replayed_realized, confidence_level=result.spec.confidence_level)

    assert payload["timestamp_index_name"] == "audit_time"
    assert payload["backtest"]["inputs_digest"] == replayed.inputs_digest


def test_report_rejects_a_non_json_timestamp_index_name_that_cannot_replay_digest() -> None:
    result = _validated_result()
    index_name = pd.Timestamp("2020-01-02T03:04:05+00:00")
    forecast = result.forecast.rename_axis(index_name)
    realized = result.realized.rename_axis(index_name)
    named_result = replace(
        result,
        forecast=forecast,
        realized=realized,
        backtest=backtest_var(forecast, realized, confidence_level=result.spec.confidence_level),
    )

    with pytest.raises(TypeError, match="timestamp_index_name must be a native JSON scalar"):
        build_risk_validation_report(named_result)


def test_report_keeps_timezone_needed_to_replay_a_dst_aware_backtest_digest() -> None:
    index = pd.date_range("2024-03-01", periods=12, freq="B", tz="America/New_York")
    returns = pd.Series(
        [-0.03, 0.01, -0.02, 0.02, -0.01, 0.03, -0.04, 0.01, -0.02, 0.02, -0.01, 0.01],
        index=index,
    )
    result = walk_forward_var(
        returns,
        RiskModelSpec(confidence_level=0.95, distribution="normal", window=4, refit_cadence=2),
    )

    payload = build_risk_validation_report(result).to_dict()
    timestamp_index = pd.DatetimeIndex(
        pd.to_datetime([event["timestamp"] for event in payload["forecast_events"]], utc=True).tz_convert(
            payload["timestamp_timezone"]
        ),
        name=payload["timestamp_index_name"],
    )
    replayed_forecast = pd.Series(
        [event["forecast"] for event in payload["forecast_events"]],
        index=timestamp_index,
    )
    replayed_realized = pd.Series(
        [event["realized"] for event in payload["forecast_events"]],
        index=timestamp_index,
    )
    replayed = backtest_var(replayed_forecast, replayed_realized, confidence_level=result.spec.confidence_level)

    assert payload["timestamp_timezone"] == "America/New_York"
    assert payload["backtest"]["inputs_digest"] == replayed.inputs_digest


@pytest.mark.parametrize(
    ("timezone", "expected_token"),
    [
        (tz.gettz("America/New_York"), "America/New_York"),
        (tz.tzoffset(None, 5 * 60 * 60 + 30 * 60), "UTC+05:30"),
    ],
)
def test_report_serializes_dateutil_timezones_as_portable_tokens(
    timezone,
    expected_token: str,
) -> None:
    index = pd.date_range("2024-03-01", periods=12, freq="B", tz=timezone)
    returns = pd.Series(
        [-0.03, 0.01, -0.02, 0.02, -0.01, 0.03, -0.04, 0.01, -0.02, 0.02, -0.01, 0.01],
        index=index,
    )
    result = walk_forward_var(
        returns,
        RiskModelSpec(confidence_level=0.95, distribution="normal", window=4, refit_cadence=2),
    )

    payload = build_risk_validation_report(result).to_dict()
    timestamp_index = pd.DatetimeIndex(
        pd.to_datetime([event["timestamp"] for event in payload["forecast_events"]], utc=True).tz_convert(
            payload["timestamp_timezone"]
        )
    )
    replayed = backtest_var(
        pd.Series([event["forecast"] for event in payload["forecast_events"]], index=timestamp_index),
        pd.Series([event["realized"] for event in payload["forecast_events"]], index=timestamp_index),
        confidence_level=result.spec.confidence_level,
    )

    assert payload["timestamp_timezone"] == expected_token
    assert "zoneinfo" not in payload["timestamp_timezone"]
    assert not payload["timestamp_timezone"].startswith("/")
    assert payload["backtest"]["inputs_digest"] == replayed.inputs_digest
