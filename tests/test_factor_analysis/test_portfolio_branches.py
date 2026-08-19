"""Branch-completion tests for factor_analysis.portfolio validation paths."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from fincore.factor_analysis.portfolio import (
    _filtered_portfolio_data,
    _require_weight_series,
    positions,
)


def _factor_data() -> pd.DataFrame:
    dates = pd.bdate_range("2024-01-02", periods=20)
    assets = ["A", "B", "C", "D"]
    index = pd.MultiIndex.from_product((dates, assets), names=("date", "asset"))
    return pd.DataFrame(
        {
            "factor": np.random.default_rng(1).normal(0, 1, len(index)),
            "factor_quantile": [i % 4 + 1 for i in range(len(index))],
            "group": ["g1", "g2"] * (len(index) // 2),
            "1D": np.random.default_rng(2).normal(0, 0.01, len(index)),
        },
        index=index,
    )


def test_require_weight_series_rejects_non_series() -> None:
    with pytest.raises(TypeError, match="Series"):
        _require_weight_series([0.5, 0.5])  # type: ignore[arg-type]


def test_require_weight_series_rejects_non_multiindex() -> None:
    with pytest.raises(ValueError, match="MultiIndex"):
        _require_weight_series(pd.Series([0.5, 0.5]))


def test_positions_rejects_non_datetime_date_level() -> None:
    idx = pd.MultiIndex.from_product([["d1", "d2"], ["A", "B"]], names=["date", "asset"])
    weights = pd.Series([0.5, 0.5, 0.3, 0.7], index=idx)
    with pytest.raises(ValueError, match="DatetimeIndex"):
        positions(weights, "1D")


def test_filtered_portfolio_data_rejects_unknown_period() -> None:
    with pytest.raises(ValueError, match="not found"):
        _filtered_portfolio_data(_factor_data(), "9D", quantiles=None, groups=None)


def test_filtered_portfolio_data_rejects_missing_quantile_column() -> None:
    data = _factor_data().drop(columns=["factor_quantile"])
    with pytest.raises(KeyError, match="factor_quantile"):
        _filtered_portfolio_data(data, "1D", quantiles=[1], groups=None)


def test_filtered_portfolio_data_rejects_missing_group_column() -> None:
    data = _factor_data().drop(columns=["group"])
    with pytest.raises(KeyError, match="group"):
        _filtered_portfolio_data(data, "1D", quantiles=None, groups=["g1"])
