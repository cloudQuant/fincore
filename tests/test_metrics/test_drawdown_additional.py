from __future__ import annotations

import numpy as np
import pandas as pd

from fincore.metrics import drawdown as dd


def test_get_all_drawdowns_detailed_marks_no_recovery_when_series_ends_in_drawdown():
    idx = pd.date_range("2024-01-01", periods=6, freq="B", tz="UTC")
    # Build a drawdown that never recovers to its prior peak by the end.
    returns = pd.Series([0.01, 0.0, -0.20, 0.01, 0.01, 0.0], index=idx)
    out = dd.get_all_drawdowns_detailed(returns)
    assert out
    assert out[-1]["recovery_duration"] is None


def test_max_drawdown_dataframe_returns_series_when_output_allocated() -> None:
    returns = pd.DataFrame({"a": [0.01, -0.02, 0.0], "b": [0.0, -0.01, 0.01]})
    out = dd.max_drawdown(returns)
    assert isinstance(out, pd.Series)
    assert out.shape[0] == 2


def test_get_all_drawdowns_detailed_returns_empty_list_when_no_drawdowns() -> None:
    returns = pd.Series([0.01, 0.0, 0.02], index=pd.date_range("2024-01-01", periods=3, freq="B"))
    assert dd.get_all_drawdowns_detailed(returns) == []


def test_get_max_drawdown_underwater_falls_back_when_no_zero_before_or_after_valley() -> None:
    idx = pd.date_range("2024-01-01", periods=4, freq="B")
    # All negative: no 0 before/after valley, so both try/except branches are exercised.
    underwater = pd.Series([-0.05, -0.02, -0.03, -0.01], index=idx)
    peak, valley, recovery = dd.get_max_drawdown_underwater(underwater)
    assert peak == idx[0]
    assert valley == idx[0]
    assert pd.isna(recovery)


def test_max_drawdown_days_non_datetime_index_returns_positional_distance():
    returns = pd.Series([0.01, -0.10, 0.02, 0.0], index=[0, 1, 2, 3])
    out = dd.max_drawdown_days(returns)
    assert out == 1


def test_get_max_drawdown_period_returns_none_when_input_not_series():
    returns = np.array([0.01, -0.1, 0.02], dtype=float)
    start, end = dd.get_max_drawdown_period(returns)  # type: ignore[arg-type]
    assert start is None and end is None


def test_get_max_drawdown_period_returns_none_when_empty() -> None:
    returns = pd.Series(dtype=float)
    start, end = dd.get_max_drawdown_period(returns)
    assert start is None and end is None


def test_get_max_drawdown_period_happy_path() -> None:
    idx = pd.date_range("2024-01-01", periods=5, freq="B")
    returns = pd.Series([0.0, -0.10, 0.05, 0.0, 0.0], index=idx)
    start, end = dd.get_max_drawdown_period(returns)
    assert start is not None and end is not None
    assert start <= end


def test_second_and_third_drawdown_helpers_return_nan_when_insufficient_periods():
    returns = pd.Series([0.01, -0.05, 0.06], index=pd.date_range("2024-01-01", periods=3, freq="B", tz="UTC"))
    assert np.isnan(dd.second_max_drawdown(returns))
    assert np.isnan(dd.third_max_drawdown(returns))
    assert np.isnan(dd.second_max_drawdown_days(returns))
    assert np.isnan(dd.third_max_drawdown_days(returns))


def test_second_and_third_drawdown_duration_and_recovery_branches() -> None:
    idx = pd.date_range("2024-01-01", periods=7, freq="B")
    # Three drawdowns, last drawdown ends without recovery and is the least severe.
    # This arrangement exercises both integer recovery duration and None->NaN conversion.
    returns = pd.Series(
        [
            0.0,
            -0.20,
            0.30,  # recover above prior peak (avoid float "almost recovered" issues)
            -0.15,
            0.25,  # recover above prior peak
            -0.05,
            0.0,  # end in drawdown
        ],
        index=idx,
    )

    assert dd.second_max_drawdown_days(returns) == 0
    assert dd.second_max_drawdown_recovery_days(returns) == 1
    assert dd.third_max_drawdown_days(returns) == 0
    assert np.isnan(dd.third_max_drawdown_recovery_days(returns))


def test_second_max_drawdown_recovery_days_returns_nan_when_no_recovery_for_second_worst() -> None:
    idx = pd.date_range("2024-01-01", periods=7, freq="B")
    # Make the final drawdown (no recovery) the second worst by magnitude.
    returns = pd.Series(
        [
            0.0,
            -0.20,
            0.30,  # recover
            -0.05,
            0.10,  # recover well above prior peak
            -0.15,
            0.0,  # end in drawdown
        ],
        index=idx,
    )
    assert np.isnan(dd.second_max_drawdown_recovery_days(returns))


def test_get_top_drawdowns_handles_empty_underwater_during_iteration() -> None:
    """Test get_top_drawdowns when underwater becomes empty (line 327)."""
    # Create returns with only positive values - no drawdowns
    # After processing, underwater becomes empty triggering the break condition
    idx = pd.date_range("2024-01-01", periods=5, freq="B")
    returns = pd.Series([0.01, 0.02, 0.015, 0.03, 0.01], index=idx)

    result = dd.get_top_drawdowns(returns, top=5)
    # Should handle empty underwater gracefully
    assert isinstance(result, list)
    # With all positive returns, should return empty list
    assert len(result) == 0
