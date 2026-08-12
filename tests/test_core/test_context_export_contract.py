from __future__ import annotations

import json

import pandas as pd

from fincore.core.context import AnalysisContext


def test_to_json_writes_when_path_is_given(tmp_path) -> None:
    returns = pd.Series([0.01, -0.02, 0.03], index=pd.date_range("2024-01-01", periods=3))
    target = tmp_path / "metrics.json"

    payload = AnalysisContext(returns).to_json(path=target)

    assert target.read_text(encoding="utf-8") == payload
    assert "Sharpe ratio" in json.loads(payload)


def test_to_json_without_path_only_returns_the_payload(tmp_path) -> None:
    returns = pd.Series([0.01, -0.02, 0.03], index=pd.date_range("2024-01-01", periods=3))

    payload = AnalysisContext(returns).to_json()

    assert isinstance(payload, str)
    assert list(tmp_path.iterdir()) == []
