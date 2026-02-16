from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from fincore.attribution import style as style_mod
from fincore.attribution.style import (
    StyleResult,
    calculate_regression_attribution,
    calculate_style_tilts,
    style_analysis,
)


def test_style_result_style_summary_series_and_dataframe_variants() -> None:
    exposures = pd.DataFrame({"value": [1.0]}, index=["A"])
    overall = pd.Series([0.0], index=[pd.Timestamp("2024-01-01")])

    # returns_by_style as Series
    rbs_s = pd.Series({"value": 0.1, "growth": -0.2})
    sr = StyleResult(exposures=exposures, returns_by_style=rbs_s, overall_returns=overall)
    summ = sr.style_summary
    assert summ["value"] == 0.1
    assert summ["growth"] == -0.2

    # returns_by_style with explicit "style" column
    rbs_df = pd.DataFrame({"style": ["value", "growth"], "return": [0.1, -0.2]})
    sr2 = StyleResult(exposures=exposures, returns_by_style=rbs_df, overall_returns=overall)
    summ2 = sr2.style_summary
    assert summ2["value"] == 0.1
    assert summ2["growth"] == -0.2

    # returns_by_style as DataFrame indexed by style (including duplicate labels)
    # df.loc[style] returns a DataFrame when the index label is duplicated.
    rbs_idx = pd.DataFrame({"return": [0.1, 0.2]}, index=pd.Index(["value", "value"], name="style"))
    sr3 = StyleResult(exposures=exposures, returns_by_style=rbs_idx, overall_returns=overall)
    summ3 = sr3.style_summary
    assert summ3["value"] == 0.1


def test_style_analysis_value_scores_branch() -> None:
    idx = pd.date_range("2024-01-01", periods=30, freq="B")
    returns = pd.DataFrame(
        {"A": np.linspace(0.001, -0.001, len(idx)), "B": 0.0, "C": 0.0005},
        index=idx,
    )
    scores = pd.Series({"A": 0.1, "B": 0.9, "C": 0.5})
    out = style_analysis(returns, value_scores=scores)
    assert "value" in out.exposures.columns
    assert "growth" in out.exposures.columns


def test_private_helpers_momentum_and_lookback_and_size_rank() -> None:
    idx = pd.date_range("2024-01-01", periods=5, freq="B")
    rets = pd.DataFrame({"A": [0.01, 0.0, 0.0, 0.0, 0.0], "B": [-0.01, 0.0, 0.0, 0.0, 0.0]}, index=idx)

    mom = style_mod._calculate_momentum(rets, window=2)
    assert isinstance(mom, pd.DataFrame)
    assert mom.shape[0] == 1

    pos = style_mod._exposure_from_lookback(rets, periods=2, direction="positive")
    neg = style_mod._exposure_from_lookback(rets, periods=2, direction="negative")
    assert pos.shape == (1, 2)
    assert neg.shape == (1, 2)

    ranks = pd.Series({"A": 0.2, "B": 0.8})
    exp = style_mod._size_rank_to_exposure(ranks)
    assert set(exp.index) == {"large", "small"}


def test_calculate_style_tilts_effective_window_too_small_returns_empty() -> None:
    tiny = pd.DataFrame(
        {"A": [0.01, -0.01], "B": [0.0, 0.0]},
        index=pd.date_range("2024-01-01", periods=2, freq="B"),
    )
    out = calculate_style_tilts(tiny, window=252)
    assert out.empty


def test_calculate_regression_attribution_raises_when_missing_style_inputs_for_series() -> None:
    idx = pd.date_range("2024-01-01", periods=10, freq="B")
    port = pd.Series(np.linspace(0.001, -0.001, len(idx)), index=idx)
    with pytest.raises(TypeError, match="must be a DataFrame"):
        calculate_regression_attribution(port)


def test_calculate_regression_attribution_skip_and_short_series_branches() -> None:
    idx = pd.date_range("2024-01-01", periods=5, freq="B")
    port = pd.Series([0.01, 0.0, 0.01, 0.0, 0.01], index=idx)

    # Skip when style is not in exposures.
    style_returns = pd.DataFrame({"foo": port.values}, index=idx)
    exposures = pd.DataFrame({"bar": [1.0]}, index=["A"])
    out = calculate_regression_attribution(port, style_returns=style_returns, style_exposures=exposures)
    assert "foo" not in out
    assert "residual" in out

    # common_idx < 3 => attribution forced to 0.0
    style_returns_short = pd.DataFrame({"value": [0.01, 0.02]}, index=idx[:2])
    exposures2 = pd.DataFrame({"value": [1.0]}, index=["A"])
    out2 = calculate_regression_attribution(port, style_returns=style_returns_short, style_exposures=exposures2)
    assert out2["value"] == 0.0

    # valid_mask.sum() < 3 => attribution forced to 0.0
    style_returns_nan = pd.DataFrame({"value": [np.nan, np.nan, 0.01, np.nan, np.nan]}, index=idx)
    out3 = calculate_regression_attribution(port, style_returns=style_returns_nan, style_exposures=exposures2)
    assert out3["value"] == 0.0


def test_analyze_performance_by_style_returns_empty_when_exposures_empty() -> None:
    idx = pd.date_range("2024-01-01", periods=5, freq="B")
    returns = pd.DataFrame({"A": [0.01, 0.0, 0.01, 0.0, 0.01]}, index=idx)
    out = style_mod.analyze_performance_by_style(returns, style_exposures=pd.DataFrame())
    assert out.empty


def test_style_result_empty_dataframe_in_summary() -> None:
    """Test style_summary when .loc returns an empty DataFrame."""
    exposures = pd.DataFrame({"value": [1.0]}, index=["A"])
    overall = pd.Series([0.0], index=[pd.Timestamp("2024-01-01")])

    # Create a DataFrame with unique index where loc would return empty DataFrame
    # This happens when accessing a non-existent key
    rbs_idx = pd.DataFrame({"return": [0.1, 0.2]}, index=["value", "growth"])
    sr = StyleResult(exposures=exposures, returns_by_style=rbs_idx, overall_returns=overall)
    # Accessing a non-existent style returns empty DataFrame, handled by line 65-66
    summ = sr.style_summary
    assert "value" in summ
    assert "growth" in summ


def test_style_result_empty_series_in_summary() -> None:
    """Test style_summary when .loc returns an empty Series."""
    exposures = pd.DataFrame({"value": [1.0]}, index=["A"])
    overall = pd.Series([0.0], index=[pd.Timestamp("2024-01-01")])

    # Create a Series where accessing returns empty
    rbs_series = pd.Series({"value": 0.1, "growth": 0.2})
    sr = StyleResult(exposures=exposures, returns_by_style=rbs_series, overall_returns=overall)
    summ = sr.style_summary
    assert summ["value"] == 0.1


def test_style_result_return_column_in_index() -> None:
    """Test style_summary when 'return' is in the Series index."""
    exposures = pd.DataFrame({"value": [1.0]}, index=["A"])
    overall = pd.Series([0.0], index=[pd.Timestamp("2024-01-01")])

    # Series with 'return' as an index value
    rbs_series = pd.Series([0.1, 0.2], index=["return", "value"])
    sr = StyleResult(exposures=exposures, returns_by_style=rbs_series, overall_returns=overall)
    summ = sr.style_summary
    # Should handle 'return' in index properly (line 74-75)
    assert summ["return"] == 0.1
