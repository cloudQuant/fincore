"""Independent legacy report semantics frozen for the 0042-R2 cutover."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

GOLDEN = Path(__file__).parent / "goldens" / "0042-r2" / "reports" / "portfolio-core.json"


def _series(specification: dict[str, Any]) -> pd.Series:
    index = pd.date_range(
        specification["start"],
        periods=specification["size"],
        freq=specification["frequency"],
        tz=specification["timezone"],
    )
    values = np.resize(np.asarray(specification["pattern"], dtype=float), len(index))
    return pd.Series(values, index=index, dtype=float, name=specification["name"])


def _series_digest(values: pd.Series) -> str:
    payload = {
        "index": [timestamp.isoformat() for timestamp in values.index],
        "name": values.name,
        "values": [float(value) for value in values.to_numpy()],
    }
    return hashlib.sha256(
        json.dumps(payload, ensure_ascii=True, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _assert_metrics(actual: dict[str, Any], expected: dict[str, float | None]) -> None:
    for name, expected_value in expected.items():
        actual_value = actual[name]
        if expected_value is None:
            assert math.isnan(actual_value), name
        else:
            assert actual_value == pytest.approx(expected_value, rel=1e-12, abs=1e-12), name


def test_portfolio_report_matches_the_frozen_legacy_core_semantic_golden() -> None:
    from fincore.report.portfolio.compute import build_portfolio_report

    golden = json.loads(GOLDEN.read_text(encoding="utf-8"))
    inputs = golden["inputs"]
    returns = _series(inputs["returns"])
    benchmark_returns = _series(inputs["benchmark_returns"])
    positions = pd.DataFrame(inputs["positions"], index=returns.index)
    transaction_index = pd.DatetimeIndex(
        [returns.index[offset] + pd.Timedelta(hours=hour) for offset, hour in inputs["transaction_offsets"]]
    )
    transactions = pd.DataFrame(inputs["transactions"], index=transaction_index)

    document = build_portfolio_report(
        returns,
        benchmark_returns=benchmark_returns,
        positions=positions,
        transactions=transactions,
        title=inputs["title"],
        period=inputs["period"],
        rolling_window=inputs["rolling_window"],
    )

    assert golden["artifact_type"] == "report_semantic_golden"
    assert golden["capability_id"] == "report.portfolio.create_strategy_report"
    assert golden["scenario_id"] == "ordinary_inputs"
    assert golden["legacy_source"] == {
        "commit": "66ef3c9ad2fc000229a80ee0dec9762a96ad1770",
        "tree": "5bea028ad3b8d8d2476e0aab9ba94583628cebb1",
    }
    assert document.domain == "portfolio"
    _assert_metrics(document.section("performance").metrics, golden["expected"]["performance_metrics"])
    _assert_metrics(document.section("benchmark").metrics, golden["expected"]["benchmark_metrics"])
    _assert_metrics(document.section("portfolio").metrics, golden["expected"]["portfolio_metrics"])
    _assert_metrics(document.section("transactions").metrics, golden["expected"]["transaction_metrics"])
    assert (
        _series_digest(document.section("performance").series["cumulative_returns"])
        == golden["expected"]["cumulative_returns_digest"]
    )
    assert (
        _series_digest(document.section("portfolio").series["gross_leverage"])
        == golden["expected"]["gross_leverage_digest"]
    )
