"""Tests for intraday detection and estimation utilities."""

from __future__ import annotations

import pandas as pd
import pytest

from fincore.utils import common_utils as cu


def test_detect_intraday_and_check_intraday_branches(monkeypatch):
    """Test detect_intraday and check_intraday branches."""
    idx = pd.date_range("2024-01-01", periods=3, freq="B", tz="UTC")
    positions = pd.DataFrame({"A": [0, 0, 0], "cash": [100, 100, 100]}, index=idx)
    txn_idx = pd.to_datetime(["2024-01-01 10:00", "2024-01-02 10:00"]).tz_localize("UTC")
    txns = pd.DataFrame({"symbol": ["A", "A"], "amount": [1, -1], "price": [10.0, 10.0]}, index=txn_idx)

    assert bool(cu.detect_intraday(positions, txns)) is True

    monkeypatch.setattr(cu, "detect_intraday", lambda *_a, **_k: True)
    monkeypatch.setattr(cu, "estimate_intraday", lambda *_a, **_k: "EST")
    rets = pd.Series([0.0, 0.0, 0.0], index=idx)
    with pytest.warns(UserWarning, match="Detected intraday strategy"):
        out = cu.check_intraday("infer", rets, positions, txns)
    assert out == "EST"

    with pytest.raises(ValueError, match="Positions and txns needed"):
        cu.check_intraday(True, rets, None, txns)


def test_estimate_intraday_smoke_and_divisor_zero_branch():
    """Test estimate_intraday basic functionality and zero divisor branch."""
    idx = pd.date_range("2024-01-01", periods=2, freq="B", tz="UTC")
    returns = pd.Series([-1.0, 0.0], index=idx)
    positions = pd.DataFrame({"A": [50.0, 0.0], "cash": [50.0, 100.0]}, index=idx)

    txn_idx = pd.to_datetime(
        ["2024-01-01 09:30", "2024-01-01 15:30", "2024-01-02 10:00"],
        utc=True,
    )
    transactions = pd.DataFrame(
        {"symbol": ["A", "A", "A"], "amount": [1, -1, 1], "price": [10.0, 11.0, 12.0]},
        index=txn_idx,
    )

    corrected = cu.estimate_intraday(returns, positions, transactions)
    assert isinstance(corrected, pd.DataFrame)
    assert "cash" in corrected.columns
    assert corrected.index.name == "period_close"


def test_check_intraday_with_estimate_true_and_both_inputs():
    """Test check_intraday with estimate=True and both inputs provided."""
    idx = pd.date_range("2024-01-01", periods=2, freq="B", tz="UTC")
    rets = pd.Series([0.0, 0.0], index=idx)
    positions = pd.DataFrame({"A": [50.0, 0.0], "cash": [50.0, 100.0]}, index=idx)
    txn_idx = pd.to_datetime(["2024-01-01 09:30"], utc=True)
    txns = pd.DataFrame({"symbol": ["A"], "amount": [1], "price": [10.0]}, index=txn_idx)

    result = cu.check_intraday(True, rets, positions, txns)
    assert isinstance(result, pd.DataFrame)
    assert "cash" in result.columns


def test_check_intraday_with_estimate_true_missing_inputs():
    """Test check_intraday with estimate=True but missing positions or transactions."""
    idx = pd.date_range("2024-01-01", periods=2, freq="B", tz="UTC")
    rets = pd.Series([0.0, 0.0], index=idx)
    positions = pd.DataFrame({"A": [50.0, 0.0], "cash": [50.0, 100.0]}, index=idx)

    with pytest.raises(ValueError, match="Positions and txns needed"):
        cu.check_intraday(True, rets, positions, None)


def test_check_intraday_estimate_false_returns_positions():
    """Test check_intraday with estimate=False returns positions."""
    idx = pd.date_range("2024-01-01", periods=2, freq="B", tz="UTC")
    rets = pd.Series([0.0, 0.0], index=idx)
    positions = pd.DataFrame({"A": [50.0, 0.0], "cash": [50.0, 100.0]}, index=idx)

    result = cu.check_intraday(False, rets, positions, None)
    assert result is positions


def test_check_intraday_infer_no_intraday_returns_positions():
    """Test check_intraday with infer when no intraday detected."""
    idx = pd.date_range("2024-01-01", periods=2, freq="B", tz="UTC")
    rets = pd.Series([0.0, 0.0], index=idx)
    positions = pd.DataFrame({"A": [50.0, 50.0], "cash": [50.0, 50.0]}, index=idx)
    # Same-day transaction - no intraday
    txn_idx = pd.to_datetime(["2024-01-01 00:00", "2024-01-02 00:00"], utc=True)
    txns = pd.DataFrame({"symbol": ["A", "A"], "amount": [1, -1], "price": [10.0, 10.0]}, index=txn_idx)

    result = cu.check_intraday("infer", rets, positions, txns)
    assert result is positions


def test_check_intraday_infer_with_only_positions():
    """Test check_intraday with infer but only positions."""
    idx = pd.date_range("2024-01-01", periods=2, freq="B", tz="UTC")
    rets = pd.Series([0.0, 0.0], index=idx)
    positions = pd.DataFrame({"A": [50.0, 50.0], "cash": [50.0, 50.0]}, index=idx)

    # When transactions is None, should return positions
    result = cu.check_intraday("infer", rets, positions, None)
    assert result is positions

    # When positions is None, should also return None
    result = cu.check_intraday("infer", rets, None, None)
    assert result is None
