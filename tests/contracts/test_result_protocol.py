"""Result protocol and serialization tests."""

from __future__ import annotations

import numpy as np
import pandas as pd

from fincore.exceptions import FincoreError
from fincore.results import (
    STATUS_FAILED,
    STATUS_SUCCESS,
    STATUS_UNSUPPORTED,
    AnalysisResult,
    ResultMetadata,
    from_json,
    to_json,
)


def _meta(operation: str = "sharpe_ratio") -> ResultMetadata:
    return ResultMetadata(operation=operation, profile="enhanced_v1", schema_version="1.0")


def test_success_result_carries_value() -> None:
    result = AnalysisResult.success(0.5, _meta())
    assert result.ok
    assert result.status == STATUS_SUCCESS
    assert result.value == 0.5


def test_unsupported_result() -> None:
    result = AnalysisResult.unsupported(_meta(), "no factor returns")
    assert result.status == STATUS_UNSUPPORTED
    assert result.value is None


def test_failure_result_carries_error() -> None:
    result = AnalysisResult.failure(FincoreError("boom"), _meta())
    assert result.status == STATUS_FAILED
    assert result.error is not None


def test_json_round_trip_scalar() -> None:
    meta = _meta()
    meta = ResultMetadata(**{**meta.__dict__, "units": "ratio"})
    text = to_json(AnalysisResult.success(0.5, meta))
    result = from_json(text)
    assert result.status == STATUS_SUCCESS
    assert result.value == 0.5
    assert result.metadata is not None
    assert result.metadata.operation == "sharpe_ratio"
    assert result.metadata.units == "ratio"


def test_json_round_trip_series() -> None:
    series = pd.Series([0.01, -0.02, 0.03], index=["a", "b", "c"])
    result = from_json(to_json(AnalysisResult.success(series, _meta("cum_returns"))))
    assert isinstance(result.value, pd.Series)
    assert list(result.value) == [0.01, -0.02, 0.03]


def test_json_round_trip_nan() -> None:
    result = from_json(to_json(AnalysisResult.success(float("nan"), _meta())))
    assert np.isnan(result.value)


def test_json_round_trip_digest_preserved() -> None:
    meta = _meta()
    meta = ResultMetadata(**{**meta.__dict__, "input_digest": "abc123"})
    result = from_json(to_json(AnalysisResult.success(1.0, meta)))
    assert result.metadata is not None
    assert result.metadata.input_digest == "abc123"


def test_json_rejects_missing_schema() -> None:
    import pytest

    with pytest.raises(ValueError, match="schema_version"):
        from_json('{"status": "success"}')


def test_json_tolerates_unknown_keys() -> None:
    text = '{"schema_version": "1.0", "status": "success", "value": 1.0, "future_field": true}'
    result = from_json(text)
    assert result.value == 1.0
