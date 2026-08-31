"""Contracts for immutable runtime input snapshots."""

from __future__ import annotations

import pandas as pd
import pytest


def test_snapshot_copy_on_ingest_preserves_the_caller_series_and_returns_fresh_materializations() -> None:
    from fincore.runtime.data import AnalysisSnapshot

    returns = pd.Series(
        [0.01, 0.02],
        index=pd.date_range("2024-01-01", periods=2, tz="UTC"),
        name="returns",
    )

    snapshot = AnalysisSnapshot.from_inputs({"returns": returns, "window": 2})
    returns.iloc[0] = 99.0
    first = snapshot.materialize()
    first["returns"].iloc[1] = 88.0
    second = snapshot.materialize()

    assert first["returns"].iloc[0] == 0.01
    assert second["returns"].tolist() == [0.01, 0.02]
    assert second["returns"].index.tz is not None
    with pytest.raises(TypeError):
        second["new_input"] = "not allowed"  # type: ignore[index]


def test_snapshot_digest_changes_when_pandas_timezone_or_values_change() -> None:
    from fincore.runtime.data import AnalysisSnapshot

    utc = pd.Series([0.01], index=pd.date_range("2024-01-01", periods=1, tz="UTC"))
    shanghai = pd.Series([0.01], index=pd.date_range("2024-01-01", periods=1, tz="Asia/Shanghai"))
    changed_value = pd.Series([0.02], index=pd.date_range("2024-01-01", periods=1, tz="UTC"))

    first = AnalysisSnapshot.from_inputs({"returns": utc})

    assert first.digest != AnalysisSnapshot.from_inputs({"returns": shanghai}).digest
    assert first.digest != AnalysisSnapshot.from_inputs({"returns": changed_value}).digest
    assert first.digest == AnalysisSnapshot.from_inputs({"returns": utc.copy()}).digest
