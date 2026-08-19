"""Branch-completion tests for factor_analysis.analysis validation paths."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from fincore.factor_analysis.analysis import (
    _analyze_factor,
    _copy_clean_factor_data,
    _legacy_quantile_cumulative_returns,
    _normalize_periods,
    _normalize_positive_lags,
    _normalize_time_aggregation,
)


def _clean() -> pd.DataFrame:
    dates = pd.bdate_range("2024-01-02", periods=30)
    assets = ["A", "B", "C", "D"]
    index = pd.MultiIndex.from_product((dates, assets), names=("date", "asset"))
    return pd.DataFrame(
        {
            "factor": np.random.default_rng(1).normal(0, 1, len(index)),
            "factor_quantile": [i % 4 + 1 for i in range(len(index))],
            "1D": np.random.default_rng(2).normal(0, 0.01, len(index)),
            "5D": np.random.default_rng(3).normal(0, 0.02, len(index)),
        },
        index=index,
    )


# ---------------------------------------------------------------------------
# _copy_clean_factor_data
# ---------------------------------------------------------------------------


def test_copy_clean_rejects_non_dataframe() -> None:
    with pytest.raises(TypeError, match="DataFrame"):
        _copy_clean_factor_data([1, 2, 3])  # type: ignore[arg-type]


def test_copy_clean_rejects_non_multiindex() -> None:
    df = pd.DataFrame({"factor": [1.0], "factor_quantile": [1], "1D": [0.01]})
    with pytest.raises(ValueError, match="MultiIndex"):
        _copy_clean_factor_data(df)


def test_copy_clean_rejects_missing_factor_column() -> None:
    df = _clean().drop(columns=["factor"])
    with pytest.raises(ValueError, match="factor"):
        _copy_clean_factor_data(df)


def test_copy_clean_rejects_missing_quantile_column() -> None:
    df = _clean().drop(columns=["factor_quantile"])
    with pytest.raises(ValueError, match="factor_quantile"):
        _copy_clean_factor_data(df)


def test_copy_clean_rejects_missing_forward_column() -> None:
    df = _clean().drop(columns=["1D", "5D"])
    with pytest.raises(ValueError, match="forward-return"):
        _copy_clean_factor_data(df)


# ---------------------------------------------------------------------------
# _normalize_periods
# ---------------------------------------------------------------------------


def test_normalize_periods_rejects_string() -> None:
    with pytest.raises(TypeError, match="sequence"):
        _normalize_periods(("1D", "5D"), "1D")  # type: ignore[arg-type]


def test_normalize_periods_rejects_duplicates() -> None:
    with pytest.raises(ValueError, match="duplicate"):
        _normalize_periods(("1D", "5D"), ("1D", "1D"))


def test_normalize_periods_rejects_unknown() -> None:
    with pytest.raises(ValueError, match="unknown"):
        _normalize_periods(("1D", "5D"), ("9D",))


def test_normalize_periods_empty_selection() -> None:
    with pytest.raises(ValueError, match="at least one"):
        _normalize_periods(("1D", "5D"), ())


# ---------------------------------------------------------------------------
# _normalize_positive_lags
# ---------------------------------------------------------------------------


def test_normalize_lags_rejects_string() -> None:
    with pytest.raises(TypeError, match="sequence"):
        _normalize_positive_lags("1")  # type: ignore[arg-type]


def test_normalize_lags_rejects_empty() -> None:
    with pytest.raises(ValueError, match="positive lag"):
        _normalize_positive_lags(())


def test_normalize_lags_rejects_non_positive() -> None:
    with pytest.raises(ValueError, match="positive integers"):
        _normalize_positive_lags((0,))


def test_normalize_lags_rejects_duplicates() -> None:
    with pytest.raises(ValueError, match="duplicate"):
        _normalize_positive_lags((1, 1))


def test_normalize_lags_allows_legacy_zero() -> None:
    assert _normalize_positive_lags((0, 1), allow_legacy_zero=True) == (0, 1)


# ---------------------------------------------------------------------------
# _normalize_time_aggregation
# ---------------------------------------------------------------------------


def test_normalize_time_aggregation_rejects_string() -> None:
    with pytest.raises(TypeError, match="sequence"):
        _normalize_time_aggregation("M")  # type: ignore[arg-type]


def test_normalize_time_aggregation_rejects_empty_string() -> None:
    with pytest.raises(ValueError, match="frequency"):
        _normalize_time_aggregation(("",))


def test_normalize_time_aggregation_rejects_duplicates() -> None:
    with pytest.raises(ValueError, match="duplicate"):
        _normalize_time_aggregation(("M", "M"))


# ---------------------------------------------------------------------------
# _legacy_quantile_cumulative_returns
# ---------------------------------------------------------------------------


def test_legacy_quantile_cumulative_returns_empty() -> None:
    empty = pd.DataFrame()
    assert _legacy_quantile_cumulative_returns(empty) == {}


# ---------------------------------------------------------------------------
# _analyze_factor type validation
# ---------------------------------------------------------------------------


def test_analyze_factor_rejects_non_string_benchmark_period() -> None:
    with pytest.raises(TypeError, match="pyfolio_benchmark_period"):
        _analyze_factor(_clean(), pyfolio_benchmark_period=1)  # type: ignore[arg-type]


def test_analyze_factor_rejects_non_numeric_capital() -> None:
    with pytest.raises(TypeError, match="pyfolio_capital"):
        _analyze_factor(_clean(), pyfolio_capital="lots")  # type: ignore[arg-type]


def test_analyze_factor_legacy_event_windows_type_error() -> None:
    with pytest.raises(ValueError, match="non-negative integers"):
        _analyze_factor(
            _clean(),
            allow_legacy_event_windows=True,
            event_returns=pd.DataFrame(),
            event_before="x",  # type: ignore[arg-type]
            event_after=1,
        )
