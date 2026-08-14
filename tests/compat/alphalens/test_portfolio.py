"""Strict facade regressions for the Task 5 portfolio APIs."""

from __future__ import annotations

import inspect

import numpy as np
import pandas as pd
import pytest
from pandas.tseries.offsets import BDay

from fincore.alphalens import performance as strict_performance
from fincore.factor_analysis.portfolio import (
    factor_cumulative_returns as enhanced_factor_cumulative_returns,
)


def _strict_factor_data() -> pd.DataFrame:
    dates = pd.bdate_range("2024-01-02", periods=4, name="date")
    assets = pd.Index(["A", "B", "C", "D"], name="asset")
    index = pd.MultiIndex.from_product((dates, assets), names=("date", "asset"))
    return pd.DataFrame(
        {
            "factor": np.tile([1.0, 2.0, 3.0, 4.0], len(dates)),
            "1D": [0.10, 0.02, -0.01, 0.03, 0.00, 0.04, 0.01, -0.02, 0.02, -0.02, 0.03, 0.00, 0.01, 0.03, -0.01, 0.02],
            "5D": [0.20, 0.10, -0.05, 0.15, 0.00, 0.08, 0.02, -0.04, 0.04, -0.04, 0.06, 0.00, 0.02, 0.06, -0.02, 0.04],
            "factor_quantile": np.tile([1, 1, 2, 2], len(dates)),
            "group": np.tile(["tech", "tech", "finance", "finance"], len(dates)),
        },
        index=index,
    )


@pytest.mark.parametrize(
    ("name", "expected_signature"),
    [
        ("positions", "(weights, period, freq=None)"),
        (
            "factor_cumulative_returns",
            "(factor_data, period, long_short=True, group_neutral=False, equal_weight=False, quantiles=None, groups=None)",
        ),
        (
            "factor_positions",
            "(factor_data, period, long_short=True, group_neutral=False, equal_weight=False, quantiles=None, groups=None)",
        ),
        (
            "create_pyfolio_input",
            "(factor_data, period, capital=None, long_short=True, group_neutral=False, equal_weight=False, quantiles=None, groups=None, benchmark_period='1D')",
        ),
    ],
)
def test_strict_portfolio_public_signatures(name: str, expected_signature: str) -> None:
    assert str(inspect.signature(getattr(strict_performance, name))) == expected_signature


def test_strict_factor_cumulative_returns_uses_legacy_projection_and_filter() -> None:
    source = _strict_factor_data()
    strict = strict_performance.factor_cumulative_returns(
        source,
        "1D",
        long_short=False,
        equal_weight=True,
        quantiles=[2],
    )
    enhanced = enhanced_factor_cumulative_returns(
        source,
        "1D",
        long_short=False,
        equal_weight=True,
        quantiles=[2],
    )

    pd.testing.assert_series_equal(strict, enhanced.rename(None))
    assert strict.name is None
    assert strict.iloc[0] == pytest.approx(1.01)


def test_strict_factor_positions_and_pyfolio_tuple_match_legacy_shapes() -> None:
    source = _strict_factor_data()
    output = strict_performance.factor_positions(source, "5D", equal_weight=True)
    returns, positions, benchmark = strict_performance.create_pyfolio_input(
        source,
        "1D",
        capital=100_000,
        long_short=False,
        equal_weight=True,
        benchmark_period="missing",
    )

    assert output.index[-1] == source.index.get_level_values("date").max() + BDay(5)
    assert isinstance(returns, pd.Series)
    assert isinstance(positions, pd.DataFrame)
    assert benchmark is None
    assert returns.name is None
    assert positions.columns[-1] == "cash"
    np.testing.assert_allclose(positions.drop(columns="cash").abs().sum(axis=1).iloc[0], 100_000 * 1.035)
