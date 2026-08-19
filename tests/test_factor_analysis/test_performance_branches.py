"""Branch-completion tests for factor_analysis.performance validation paths."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from fincore.factor_analysis.performance import (
    _average_cumulative_return_by_quantile,
    _common_start_returns,
    _copy_factor_data,
    _event_factor_copy,
    compute_mean_returns_spread,
    cumulative_returns,
    factor_alpha_beta,
    factor_information_coefficient,
    factor_rank_autocorrelation,
    mean_return_by_quantile,
    quantile_turnover,
)


def _factor_data(with_group: bool = True) -> pd.DataFrame:
    dates = pd.bdate_range("2024-01-02", periods=15)
    assets = ["A", "B", "C", "D"]
    index = pd.MultiIndex.from_product((dates, assets), names=("date", "asset"))
    data: dict[str, object] = {
        "factor": np.random.default_rng(1).normal(0, 1, len(index)),
        "factor_quantile": [i % 4 + 1 for i in range(len(index))],
        "1D": np.random.default_rng(2).normal(0, 0.01, len(index)),
    }
    if with_group:
        data["group"] = ["g1", "g2"] * (len(index) // 2)
    return pd.DataFrame(data, index=index)


# ---------------------------------------------------------------------------
# _copy_factor_data
# ---------------------------------------------------------------------------


def test_copy_factor_data_rejects_non_dataframe() -> None:
    with pytest.raises(TypeError, match="DataFrame"):
        _copy_factor_data([1, 2])  # type: ignore[arg-type]


def test_copy_factor_data_rejects_non_multiindex() -> None:
    with pytest.raises(ValueError, match="MultiIndex"):
        _copy_factor_data(pd.DataFrame({"factor": [1.0]}))


def test_copy_factor_data_rejects_missing_factor() -> None:
    df = _factor_data().drop(columns=["factor"])
    with pytest.raises(ValueError, match="factor"):
        _copy_factor_data(df)


# ---------------------------------------------------------------------------
# factor_information_coefficient
# ---------------------------------------------------------------------------


def test_factor_ic_rejects_by_group_without_group_column() -> None:
    with pytest.raises(ValueError, match="group"):
        factor_information_coefficient(_factor_data(with_group=False), by_group=True)


# ---------------------------------------------------------------------------
# factor_alpha_beta
# ---------------------------------------------------------------------------


def test_factor_alpha_beta_rejects_series_with_multiple_columns() -> None:
    data = _factor_data()
    data["5D"] = np.random.default_rng(9).normal(0, 0.02, len(data))
    returns = pd.Series([0.01, -0.01], index=data.index.get_level_values("date").unique()[:2])
    with pytest.raises(ValueError, match="exactly one"):
        factor_alpha_beta(data, returns=returns)


def test_factor_alpha_beta_rejects_bad_returns_type() -> None:
    with pytest.raises(TypeError, match="Series, DataFrame, or None"):
        factor_alpha_beta(_factor_data(), returns=42)  # type: ignore[arg-type]


def test_factor_alpha_beta_rejects_non_forward_column() -> None:
    data = _factor_data()
    bad_returns = pd.DataFrame(
        {"NotAForwardCol": [0.01, -0.01]},
        index=data.index.get_level_values("date").unique()[:2],
    )
    with pytest.raises(ValueError, match="forward-return"):
        factor_alpha_beta(data, returns=bad_returns)


def test_factor_alpha_beta_insufficient_valid_rows() -> None:
    data = _factor_data()
    dates = data.index.get_level_values("date").unique()
    returns = pd.DataFrame({"1D": [0.01]}, index=dates[:1])
    result = factor_alpha_beta(data, returns=returns)
    assert np.isnan(result.loc["Ann. alpha", "1D"])


# ---------------------------------------------------------------------------
# quantile_turnover / factor_rank_autocorrelation
# ---------------------------------------------------------------------------


def test_quantile_turnover_rejects_non_series() -> None:
    with pytest.raises(TypeError, match="Series"):
        quantile_turnover([1, 2, 3], 1)  # type: ignore[arg-type]


def test_quantile_turnover_rejects_non_multiindex() -> None:
    with pytest.raises(ValueError, match="MultiIndex"):
        quantile_turnover(pd.Series([1, 2]), 1)


def test_quantile_turnover_rejects_non_positive_period() -> None:
    idx = pd.MultiIndex.from_product([["d1"], ["A", "B"]])
    with pytest.raises(ValueError, match="positive integer"):
        quantile_turnover(pd.Series([1, 1], index=idx), 1, period=0)


def test_factor_rank_autocorrelation_rejects_non_positive_period() -> None:
    with pytest.raises(ValueError, match="positive integer"):
        factor_rank_autocorrelation(_factor_data(), period=0)


# ---------------------------------------------------------------------------
# cumulative_returns
# ---------------------------------------------------------------------------


def test_cumulative_returns_dataframe() -> None:
    df = pd.DataFrame({"a": [0.01, -0.01], "b": [0.02, np.nan]})
    result = cumulative_returns(df)
    assert isinstance(result, pd.DataFrame)


def test_cumulative_returns_ndarray() -> None:
    arr = np.array([[0.01, 0.02], [np.nan, -0.01]])
    result = cumulative_returns(arr)
    assert isinstance(result, np.ndarray)


# ---------------------------------------------------------------------------
# mean_return_by_quantile
# ---------------------------------------------------------------------------


def test_mean_return_by_quantile_rejects_missing_quantile_column() -> None:
    with pytest.raises(ValueError, match="factor_quantile"):
        mean_return_by_quantile(_factor_data().drop(columns=["factor_quantile"]))


def test_mean_return_by_quantile_rejects_by_group_without_group() -> None:
    with pytest.raises(ValueError, match="group"):
        mean_return_by_quantile(_factor_data(with_group=False), by_group=True)


# ---------------------------------------------------------------------------
# compute_mean_returns_spread
# ---------------------------------------------------------------------------


def test_spread_rejects_non_dataframe() -> None:
    with pytest.raises(TypeError, match="DataFrame"):
        compute_mean_returns_spread(pd.Series([1.0]), 2, 1)  # type: ignore[arg-type]


def test_spread_rejects_multiindex_without_quantile_level() -> None:
    idx = pd.MultiIndex.from_product([["x", "z"], ["y"]])
    frame = pd.DataFrame({"p": [1.0, 2.0]}, index=idx)
    with pytest.raises(ValueError, match="factor_quantile"):
        compute_mean_returns_spread(frame, 2, 1)


def test_spread_rejects_wrong_index_name() -> None:
    frame = pd.DataFrame({"p": [1.0, 2.0]}, index=pd.Index([1, 2], name="other"))
    with pytest.raises(ValueError, match="factor_quantile"):
        compute_mean_returns_spread(frame, 2, 1)


def test_spread_returns_none_std_err() -> None:
    frame = pd.DataFrame({"p": [1.0, 2.0]}, index=pd.Index([1, 2], name="factor_quantile"))
    _difference, std_err = compute_mean_returns_spread(frame, 2, 1)
    assert std_err is None


def test_spread_rejects_non_dataframe_std_err() -> None:
    frame = pd.DataFrame({"p": [1.0, 2.0]}, index=pd.Index([1, 2], name="factor_quantile"))
    with pytest.raises(TypeError, match="std_err"):
        compute_mean_returns_spread(frame, 2, 1, std_err=pd.Series([1.0]))  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# _event_factor_copy / _common_start_returns
# ---------------------------------------------------------------------------


def test_event_factor_copy_rejects_non_pandas() -> None:
    with pytest.raises(TypeError, match="Series or DataFrame"):
        _event_factor_copy([1, 2])  # type: ignore[arg-type]


def test_event_factor_copy_rejects_non_multiindex() -> None:
    with pytest.raises(ValueError, match="MultiIndex"):
        _event_factor_copy(pd.Series([1, 2]))


def test_common_start_returns_rejects_non_dataframe_returns() -> None:
    factor = _factor_data()["factor_quantile"]
    with pytest.raises(TypeError, match="DataFrame"):
        _common_start_returns(factor, pd.Series([0.01]), 1, 1)  # type: ignore[arg-type]


def test_common_start_returns_rejects_duplicate_index() -> None:
    factor = _factor_data()["factor_quantile"]
    returns = pd.DataFrame(
        {"A": [0.01, 0.02]},
        index=pd.DatetimeIndex(["2024-01-02", "2024-01-02"]),
    )
    with pytest.raises(ValueError, match="unique"):
        _common_start_returns(factor, returns, 1, 1)


def test_common_start_returns_rejects_missing_assets() -> None:
    factor = _factor_data()["factor_quantile"]
    returns = pd.DataFrame(
        {"NOT_ASSET": [0.01, 0.02, 0.03]},
        index=pd.DatetimeIndex(["2024-01-02", "2024-01-03", "2024-01-04"]),
    )
    with pytest.raises(ValueError, match="do not contain factor assets"):
        _common_start_returns(factor, returns, 1, 1)


def test_common_start_returns_legacy_type_error() -> None:
    factor = _factor_data()["factor_quantile"]
    returns = pd.DataFrame({"A": [0.01]}, index=pd.DatetimeIndex(["2024-01-02"]))
    with pytest.raises(ValueError, match="non-negative integers"):
        _common_start_returns(factor, returns, "x", 1, _allow_legacy_event_windows=True)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# _average_cumulative_return_by_quantile
# ---------------------------------------------------------------------------


def test_avg_cum_return_rejects_missing_quantile_column() -> None:
    with pytest.raises(ValueError, match="factor_quantile"):
        _average_cumulative_return_by_quantile(
            _factor_data().drop(columns=["factor_quantile"]),
            pd.DataFrame({"A": [0.01]}),
        )


def test_avg_cum_return_rejects_non_dataframe_returns() -> None:
    with pytest.raises(TypeError, match="DataFrame"):
        _average_cumulative_return_by_quantile(_factor_data(), pd.Series([0.01]))  # type: ignore[arg-type]


def test_avg_cum_return_rejects_group_analytics_without_group() -> None:
    with pytest.raises(ValueError, match="group"):
        _average_cumulative_return_by_quantile(
            _factor_data(with_group=False),
            pd.DataFrame({"A": [0.01]}),
            by_group=True,
        )


def test_avg_cum_return_legacy_type_error() -> None:
    with pytest.raises(ValueError, match="non-negative integers"):
        _average_cumulative_return_by_quantile(
            _factor_data(),
            pd.DataFrame({"A": [0.01]}),
            periods_before="x",  # type: ignore[arg-type]
            _allow_legacy_event_windows=True,
        )
