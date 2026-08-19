"""Branch-completion tests for factor_analysis.data validation paths."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from fincore.factor_analysis.data import (
    FactorLossReport,
    _loss_report,
    _normalize_groupby,
    _require_factor_series,
    _require_prices,
    compute_forward_returns,
    prepare_factor_data_from_forward_returns,
    quantize_factor,
)


def _factor_series(n: int = 40) -> pd.Series:
    dates = pd.bdate_range("2024-01-02", periods=n)
    assets = ["A", "B", "C", "D"]
    index = pd.MultiIndex.from_product((dates, assets), names=("date", "asset"))
    return pd.Series(np.random.default_rng(1).normal(0, 1, len(index)), index=index)


# ---------------------------------------------------------------------------
# FactorLossReport
# ---------------------------------------------------------------------------


def test_loss_report_legacy_forward_returns_loss_zero_input() -> None:
    report = FactorLossReport(0, 0, 0, 0, 0.0, 0.0, 0.0, 0.0)
    assert report.legacy_forward_returns_loss == 0.0


def test_loss_report_rejects_non_positive_input() -> None:
    with pytest.raises(ValueError, match="at least one"):
        _loss_report(0, 0, 0, 0)


# ---------------------------------------------------------------------------
# _require_factor_series
# ---------------------------------------------------------------------------


def test_require_factor_series_rejects_non_series() -> None:
    with pytest.raises(TypeError, match="Series"):
        _require_factor_series([1, 2])  # type: ignore[arg-type]


def test_require_factor_series_rejects_non_multiindex() -> None:
    with pytest.raises(ValueError, match="MultiIndex"):
        _require_factor_series(pd.Series([1.0, 2.0]))


def test_require_factor_series_rejects_non_numeric() -> None:
    idx = pd.MultiIndex.from_product([["2024-01-01"], ["A"]], names=("date", "asset"))
    with pytest.raises(ValueError, match="numeric"):
        _require_factor_series(pd.Series(["abc"], index=idx))


# ---------------------------------------------------------------------------
# _require_prices
# ---------------------------------------------------------------------------


def test_require_prices_rejects_non_dataframe() -> None:
    with pytest.raises(TypeError, match="DataFrame"):
        _require_prices(pd.Series([1.0]), pd.Index(["A"]))  # type: ignore[arg-type]


def test_require_prices_rejects_duplicate_index() -> None:
    prices = pd.DataFrame(
        {"A": [1.0, 2.0]},
        index=pd.DatetimeIndex(["2024-01-01", "2024-01-01"]),
    )
    with pytest.raises(ValueError, match="unique"):
        _require_prices(prices, pd.Index(["A"]))


def test_require_prices_rejects_non_datetime_index() -> None:
    prices = pd.DataFrame({"A": [1.0, 2.0]}, index=["a", "b"])
    with pytest.raises(ValueError, match="datetimes"):
        _require_prices(prices, pd.Index(["A"]))


def test_require_prices_rejects_duplicate_columns() -> None:
    prices = pd.DataFrame(
        [[1.0, 2.0], [3.0, 4.0]],
        index=pd.DatetimeIndex(["2024-01-01", "2024-01-02"]),
        columns=["A", "A"],
    )
    with pytest.raises(ValueError, match="columns must be unique"):
        _require_prices(prices, pd.Index(["A"]))


def test_require_prices_rejects_non_numeric_values() -> None:
    prices = pd.DataFrame(
        {"A": ["x", "y"]},
        index=pd.DatetimeIndex(["2024-01-01", "2024-01-02"]),
    )
    with pytest.raises(ValueError, match="numeric"):
        _require_prices(prices, pd.Index(["A"]))


# ---------------------------------------------------------------------------
# compute_forward_returns
# ---------------------------------------------------------------------------


def test_compute_forward_returns_rejects_non_positive_periods() -> None:
    factor = _factor_series(5)
    prices = pd.DataFrame(
        {asset: [100.0 + i for i in range(10)] for asset in ["A", "B", "C", "D"]},
        index=pd.date_range("2024-01-02", periods=10, freq="B"),
    )
    with pytest.raises(ValueError, match="positive integers"):
        compute_forward_returns(factor, prices, periods=(0,))


def test_compute_forward_returns_rejects_disjoint_dates() -> None:
    factor = _factor_series(5)
    prices = pd.DataFrame(
        {asset: [100.0] for asset in ["A", "B", "C", "D"]},
        index=pd.date_range("2030-01-01", periods=1, freq="B"),
    )
    with pytest.raises(ValueError, match="indices don't match"):
        compute_forward_returns(factor, prices, periods=(1,))


def test_compute_forward_returns_rejects_bad_filter_zscore() -> None:
    factor = _factor_series(5)
    prices = pd.DataFrame(
        {asset: [100.0 + i for i in range(10)] for asset in ["A", "B", "C", "D"]},
        index=pd.date_range("2024-01-02", periods=10, freq="B"),
    )
    with pytest.raises(ValueError, match="filter_zscore"):
        compute_forward_returns(factor, prices, periods=(1,), filter_zscore=np.inf)


# ---------------------------------------------------------------------------
# quantize_factor
# ---------------------------------------------------------------------------


def test_quantize_factor_rejects_non_dataframe() -> None:
    with pytest.raises(TypeError, match="DataFrame"):
        quantize_factor(pd.Series([1.0]))  # type: ignore[arg-type]


def test_quantize_factor_rejects_missing_factor_column() -> None:
    with pytest.raises(ValueError, match="factor"):
        quantize_factor(pd.DataFrame({"x": [1.0]}))


def test_quantize_factor_rejects_non_multiindex() -> None:
    with pytest.raises(ValueError, match="MultiIndex"):
        quantize_factor(pd.DataFrame({"factor": [1.0]}))


def test_quantize_factor_rejects_both_quantiles_and_bins() -> None:
    df = _factor_series(20).to_frame("factor")
    with pytest.raises(ValueError, match="Either quantiles or bins"):
        quantize_factor(df, quantiles=5, bins=5)


def test_quantize_factor_rejects_zero_aware_non_int() -> None:
    df = _factor_series(20).to_frame("factor")
    with pytest.raises(ValueError, match="zero_aware"):
        quantize_factor(df, quantiles=[0.1, 0.9], zero_aware=True)


def test_quantize_factor_rejects_zero_aware_too_few() -> None:
    df = _factor_series(20).to_frame("factor")
    with pytest.raises(ValueError, match="at least two"):
        quantize_factor(df, quantiles=1, zero_aware=True)


def test_quantize_factor_rejects_by_group_without_group() -> None:
    df = _factor_series(20).to_frame("factor")
    with pytest.raises(ValueError, match="group"):
        quantize_factor(df, by_group=True)


# ---------------------------------------------------------------------------
# _normalize_groupby
# ---------------------------------------------------------------------------


def test_normalize_groupby_mapping_missing_assets() -> None:
    idx = pd.MultiIndex.from_product([["2024-01-01"], ["A", "B"]], names=("date", "asset"))
    with pytest.raises(KeyError, match="not in group mapping"):
        _normalize_groupby({"A": "g1"}, idx, None)


def test_normalize_groupby_series_missing_assets() -> None:
    idx = pd.MultiIndex.from_product([["2024-01-01"], ["A", "B"]], names=("date", "asset"))
    with pytest.raises(KeyError, match="not in group mapping"):
        _normalize_groupby(pd.Series(["g1"], index=["A"]), idx, None)


def test_normalize_groupby_rejects_bad_type() -> None:
    idx = pd.MultiIndex.from_product([["2024-01-01"], ["A"]], names=("date", "asset"))
    with pytest.raises(TypeError, match="mapping, Series, or None"):
        _normalize_groupby(42, idx, None)  # type: ignore[arg-type]


def test_normalize_groupby_missing_labels() -> None:
    idx = pd.MultiIndex.from_product([["2024-01-01"], ["A"]], names=("date", "asset"))
    with pytest.raises(KeyError, match="not in passed group names"):
        _normalize_groupby({"A": "g1"}, idx, {"other": "renamed"})


# ---------------------------------------------------------------------------
# prepare_factor_data_from_forward_returns
# ---------------------------------------------------------------------------


def test_prepare_rejects_bad_max_loss() -> None:
    factor = _factor_series(10)
    forward = factor.to_frame("1D")
    with pytest.raises(ValueError, match="max_loss"):
        prepare_factor_data_from_forward_returns(factor, forward, max_loss=1.5)


def test_prepare_rejects_non_multiindex_forward_returns() -> None:
    factor = _factor_series(10)
    with pytest.raises(TypeError, match="MultiIndex pandas DataFrame"):
        prepare_factor_data_from_forward_returns(
            factor, pd.DataFrame({"1D": [0.01, 0.02]})
        )
