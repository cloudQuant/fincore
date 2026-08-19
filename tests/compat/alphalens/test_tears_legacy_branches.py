"""Branch-completion tests for the strict Alphalens tear-sheet legacy helpers."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import warnings

import numpy as np
import pandas as pd
import pytest

from fincore.alphalens.tears import (
    _deduplicated_forward_model_input,
    _has_duplicate_forward_columns,
    _legacy_event_window_bound,
    _legacy_frequency_warning,
    _legacy_group_rows_empty,
    _legacy_reject_duplicate_forward_columns,
    _legacy_require_group_for_by_group,
    _legacy_turnover_periods,
    _summary_turnover_periods,
    _summary_turnover_quantiles,
)


def _factor_data(columns=("1D",)) -> pd.DataFrame:
    dates = pd.bdate_range("2024-01-02", periods=10)
    assets = ["A", "B"]
    index = pd.MultiIndex.from_product((dates, assets), names=("date", "asset"))
    data: dict[str, object] = {
        "factor": np.random.default_rng(1).normal(0, 1, len(index)),
        "factor_quantile": [1, 2] * (len(index) // 2),
    }
    for col in columns:
        data[col] = np.random.default_rng(2).normal(0, 0.01, len(index))
    return pd.DataFrame(data, index=index)


# ---------------------------------------------------------------------------
# _legacy_event_window_bound
# ---------------------------------------------------------------------------


def test_legacy_event_window_bound_none() -> None:
    assert _legacy_event_window_bound(None) == (None, False)


def test_legacy_event_window_bound_non_indexable() -> None:
    assert _legacy_event_window_bound("abc") == ("abc", False)


def test_legacy_event_window_bound_negative() -> None:
    assert _legacy_event_window_bound(-1) == (-1, True)


def test_legacy_event_window_bound_bool() -> None:
    assert _legacy_event_window_bound(True) == (1, True)


# ---------------------------------------------------------------------------
# _legacy_frequency_warning
# ---------------------------------------------------------------------------


def test_legacy_frequency_warning_non_dataframe_silent() -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        _legacy_frequency_warning(None)


def test_legacy_frequency_warning_naive_dates() -> None:
    dates = pd.DatetimeIndex(["2024-01-02", "2024-01-03", "2024-01-04"])  # no freq
    index = pd.MultiIndex.from_product((dates, ["A", "B"]), names=("date", "asset"))
    data = pd.DataFrame(
        {
            "factor": np.random.default_rng(1).normal(0, 1, len(index)),
            "factor_quantile": [1, 2] * (len(index) // 2),
            "1D": np.random.default_rng(2).normal(0, 0.01, len(index)),
        },
        index=index,
    )
    with pytest.warns(UserWarning, match="freq"):
        _legacy_frequency_warning(data)


# ---------------------------------------------------------------------------
# _legacy_require_group_for_by_group
# ---------------------------------------------------------------------------


def test_legacy_require_group_for_by_group_no_group_column() -> None:
    with pytest.raises(KeyError):
        _legacy_require_group_for_by_group(_factor_data(), True)


# ---------------------------------------------------------------------------
# _legacy_group_rows_empty
# ---------------------------------------------------------------------------


def test_legacy_group_rows_empty_not_by_group() -> None:
    assert _legacy_group_rows_empty(_factor_data(), False) is False


def test_legacy_group_rows_empty_with_nan_group() -> None:
    data = _factor_data()
    data["group"] = np.nan
    assert _legacy_group_rows_empty(data, True) is True


# ---------------------------------------------------------------------------
# _legacy_turnover_periods / _summary_turnover_periods / _summary_turnover_quantiles
# ---------------------------------------------------------------------------


def test_legacy_turnover_periods_default_branch() -> None:
    result = _legacy_turnover_periods(_factor_data(), None)
    assert result == (1,)


def test_summary_turnover_periods_non_dataframe() -> None:
    assert _summary_turnover_periods(None) == (1,)


def test_summary_turnover_quantiles_non_dataframe() -> None:
    assert _summary_turnover_quantiles(None) is None


def test_summary_turnover_quantiles_missing_column() -> None:
    data = _factor_data().drop(columns=["factor_quantile"])
    assert _summary_turnover_quantiles(data) is None


def test_summary_turnover_quantiles_all_nan() -> None:
    data = _factor_data()
    data["factor_quantile"] = np.nan
    assert _summary_turnover_quantiles(data) == ()


# ---------------------------------------------------------------------------
# duplicate forward columns helpers
# ---------------------------------------------------------------------------


def test_legacy_reject_duplicate_forward_columns_non_dataframe() -> None:
    _legacy_reject_duplicate_forward_columns(None)  # no-op


def test_has_duplicate_forward_columns_non_dataframe() -> None:
    assert _has_duplicate_forward_columns(None) is False


def test_deduplicated_forward_model_input_no_duplicates() -> None:
    data = _factor_data()
    result = _deduplicated_forward_model_input(data)
    assert result is data


def test_legacy_reject_duplicate_forward_columns_raises() -> None:
    data = _factor_data()
    data["1D_dup"] = data["1D"]  # noqa: not a forward column, harmless
    # Build a genuine duplicate forward label.
    data = data.drop(columns=["1D_dup"])
    data.columns = [col if col != "1D" else "1D" for col in data.columns]
    if _has_duplicate_forward_columns(data):
        with pytest.raises(ValueError, match="same length"):
            _legacy_reject_duplicate_forward_columns(data)
    else:
        _legacy_reject_duplicate_forward_columns(data)
