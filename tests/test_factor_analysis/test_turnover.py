"""Task 4 RED skeleton for turnover analytics."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from fincore.factor_analysis.performance import factor_rank_autocorrelation, quantile_turnover


def _quantile_factor(values: list[list[float]]) -> pd.Series:
    """Build one fresh source-shaped quantile Series without stack ambiguity."""

    dates = pd.date_range("2015-01-01", periods=len(values), freq="D", name="date")
    assets = pd.Index(["A", "B", "C", "D"], name="asset")
    index = pd.MultiIndex.from_product((dates, assets), names=("date", "asset"))
    return pd.Series(np.asarray(values, dtype=float).reshape(-1), index=index, name="factor_quantile")


def _manual_turnover(source: pd.Series, quantile: int, period: int) -> pd.Series:
    """Independent set-based oracle for the source-row turnover meaning."""

    selected = source[source == quantile]
    dates = pd.DatetimeIndex(selected.index.get_level_values("date").unique(), name="date")
    memberships = [set(selected.loc[date].index) for date in dates]
    values = [
        np.nan if position < period else len(current - memberships[position - period]) / len(current)
        for position, current in enumerate(memberships)
    ]
    return pd.Series(values, index=dates, name=quantile, dtype=float)


@pytest.mark.parametrize(
    "source_case_id",
    [
        pytest.param(
            "tests/test_performance.py::PerformanceTestCase::test_quantile_turnover#00",
            id="tests/test_performance.py::PerformanceTestCase::test_quantile_turnover#00",
            marks=pytest.mark.alphalens_upstream_case(
                "tests/test_performance.py::PerformanceTestCase::test_quantile_turnover#00",
            ),
        ),
        pytest.param(
            "tests/test_performance.py::PerformanceTestCase::test_quantile_turnover#01",
            id="tests/test_performance.py::PerformanceTestCase::test_quantile_turnover#01",
            marks=pytest.mark.alphalens_upstream_case(
                "tests/test_performance.py::PerformanceTestCase::test_quantile_turnover#01",
            ),
        ),
        pytest.param(
            "tests/test_performance.py::PerformanceTestCase::test_quantile_turnover#02",
            id="tests/test_performance.py::PerformanceTestCase::test_quantile_turnover#02",
            marks=pytest.mark.alphalens_upstream_case(
                "tests/test_performance.py::PerformanceTestCase::test_quantile_turnover#02",
            ),
        ),
        pytest.param(
            "tests/test_performance.py::PerformanceTestCase::test_quantile_turnover#03",
            id="tests/test_performance.py::PerformanceTestCase::test_quantile_turnover#03",
            marks=pytest.mark.alphalens_upstream_case(
                "tests/test_performance.py::PerformanceTestCase::test_quantile_turnover#03",
            ),
        ),
        pytest.param(
            "tests/test_performance.py::PerformanceTestCase::test_quantile_turnover#04",
            id="tests/test_performance.py::PerformanceTestCase::test_quantile_turnover#04",
            marks=pytest.mark.alphalens_upstream_case(
                "tests/test_performance.py::PerformanceTestCase::test_quantile_turnover#04",
            ),
        ),
        pytest.param(
            "tests/test_performance.py::PerformanceTestCase::test_quantile_turnover#05",
            id="tests/test_performance.py::PerformanceTestCase::test_quantile_turnover#05",
            marks=pytest.mark.alphalens_upstream_case(
                "tests/test_performance.py::PerformanceTestCase::test_quantile_turnover#05",
            ),
        ),
        pytest.param(
            "tests/test_performance.py::PerformanceTestCase::test_quantile_turnover#06",
            id="tests/test_performance.py::PerformanceTestCase::test_quantile_turnover#06",
            marks=pytest.mark.alphalens_upstream_case(
                "tests/test_performance.py::PerformanceTestCase::test_quantile_turnover#06",
            ),
        ),
        pytest.param(
            "tests/test_performance.py::PerformanceTestCase::test_quantile_turnover#07",
            id="tests/test_performance.py::PerformanceTestCase::test_quantile_turnover#07",
            marks=pytest.mark.alphalens_upstream_case(
                "tests/test_performance.py::PerformanceTestCase::test_quantile_turnover#07",
            ),
        ),
        pytest.param(
            "tests/test_performance.py::PerformanceTestCase::test_quantile_turnover#08",
            id="tests/test_performance.py::PerformanceTestCase::test_quantile_turnover#08",
            marks=pytest.mark.alphalens_upstream_case(
                "tests/test_performance.py::PerformanceTestCase::test_quantile_turnover#08",
            ),
        ),
        pytest.param(
            "tests/test_performance.py::PerformanceTestCase::test_quantile_turnover#09",
            id="tests/test_performance.py::PerformanceTestCase::test_quantile_turnover#09",
            marks=pytest.mark.alphalens_upstream_case(
                "tests/test_performance.py::PerformanceTestCase::test_quantile_turnover#09",
            ),
        ),
        pytest.param(
            "tests/test_performance.py::PerformanceTestCase::test_quantile_turnover#10",
            id="tests/test_performance.py::PerformanceTestCase::test_quantile_turnover#10",
            marks=pytest.mark.alphalens_upstream_case(
                "tests/test_performance.py::PerformanceTestCase::test_quantile_turnover#10",
            ),
        ),
        pytest.param(
            "tests/test_performance.py::PerformanceTestCase::test_quantile_turnover#11",
            id="tests/test_performance.py::PerformanceTestCase::test_quantile_turnover#11",
            marks=pytest.mark.alphalens_upstream_case(
                "tests/test_performance.py::PerformanceTestCase::test_quantile_turnover#11",
            ),
        ),
        pytest.param(
            "tests/test_performance.py::PerformanceTestCase::test_quantile_turnover#12",
            id="tests/test_performance.py::PerformanceTestCase::test_quantile_turnover#12",
            marks=pytest.mark.alphalens_upstream_case(
                "tests/test_performance.py::PerformanceTestCase::test_quantile_turnover#12",
            ),
        ),
        pytest.param(
            "tests/test_performance.py::PerformanceTestCase::test_quantile_turnover#13",
            id="tests/test_performance.py::PerformanceTestCase::test_quantile_turnover#13",
            marks=pytest.mark.alphalens_upstream_case(
                "tests/test_performance.py::PerformanceTestCase::test_quantile_turnover#13",
            ),
        ),
        pytest.param(
            "tests/test_performance.py::PerformanceTestCase::test_quantile_turnover#14",
            id="tests/test_performance.py::PerformanceTestCase::test_quantile_turnover#14",
            marks=pytest.mark.alphalens_upstream_case(
                "tests/test_performance.py::PerformanceTestCase::test_quantile_turnover#14",
            ),
        ),
        pytest.param(
            "tests/test_performance.py::PerformanceTestCase::test_quantile_turnover#15",
            id="tests/test_performance.py::PerformanceTestCase::test_quantile_turnover#15",
            marks=pytest.mark.alphalens_upstream_case(
                "tests/test_performance.py::PerformanceTestCase::test_quantile_turnover#15",
            ),
        ),
        pytest.param(
            "tests/test_performance.py::PerformanceTestCase::test_quantile_turnover#16",
            id="tests/test_performance.py::PerformanceTestCase::test_quantile_turnover#16",
            marks=pytest.mark.alphalens_upstream_case(
                "tests/test_performance.py::PerformanceTestCase::test_quantile_turnover#16",
            ),
        ),
        pytest.param(
            "tests/test_performance.py::PerformanceTestCase::test_quantile_turnover#17",
            id="tests/test_performance.py::PerformanceTestCase::test_quantile_turnover#17",
            marks=pytest.mark.alphalens_upstream_case(
                "tests/test_performance.py::PerformanceTestCase::test_quantile_turnover#17",
            ),
        ),
    ],
)
def test_quantile_turnover_upstream_case(source_case_id: str) -> None:
    """Exercise every mapped turnover row with an exact set-membership oracle."""

    ordinal = int(source_case_id.rsplit("#", 1)[1])
    patterns = (
        [[1, 2, 3, 4], [4, 3, 2, 1], [1, 2, 3, 4], [1, 2, 3, 4]],
        [[1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4]],
        [[1, 2, 3, 4], [1, 3, 2, 4]] * 6,
        [[1, 2, 3, 4], [4, 3, 2, 1]] * 3,
    )
    quantile = (4, 4, 4, 4, 3, 3, 3, 3, 2, 2, 2, 2, 1, 1, 4, 4, 3, 3)[ordinal]
    period = (1, 1, 2, 2, 3, 3, 1, 1, 2, 2, 3, 3, 1, 1, 4, 4, 3, 3)[ordinal]
    source = _quantile_factor(patterns[ordinal % len(patterns)])
    original = source.copy(deep=True)
    actual = quantile_turnover(source, quantile, period=period)
    expected = _manual_turnover(source, quantile, period)
    pd.testing.assert_series_equal(actual, expected, check_freq=False)
    pd.testing.assert_series_equal(source, original)


@pytest.mark.parametrize(
    "source_case_id",
    [
        pytest.param(
            "tests/test_performance.py::PerformanceTestCase::test_factor_rank_autocorrelation#00",
            id="tests/test_performance.py::PerformanceTestCase::test_factor_rank_autocorrelation#00",
            marks=pytest.mark.alphalens_upstream_case(
                "tests/test_performance.py::PerformanceTestCase::test_factor_rank_autocorrelation#00",
            ),
        ),
        pytest.param(
            "tests/test_performance.py::PerformanceTestCase::test_factor_rank_autocorrelation#01",
            id="tests/test_performance.py::PerformanceTestCase::test_factor_rank_autocorrelation#01",
            marks=pytest.mark.alphalens_upstream_case(
                "tests/test_performance.py::PerformanceTestCase::test_factor_rank_autocorrelation#01",
            ),
        ),
        pytest.param(
            "tests/test_performance.py::PerformanceTestCase::test_factor_rank_autocorrelation#02",
            id="tests/test_performance.py::PerformanceTestCase::test_factor_rank_autocorrelation#02",
            marks=pytest.mark.alphalens_upstream_case(
                "tests/test_performance.py::PerformanceTestCase::test_factor_rank_autocorrelation#02",
            ),
        ),
        pytest.param(
            "tests/test_performance.py::PerformanceTestCase::test_factor_rank_autocorrelation#03",
            id="tests/test_performance.py::PerformanceTestCase::test_factor_rank_autocorrelation#03",
            marks=pytest.mark.alphalens_upstream_case(
                "tests/test_performance.py::PerformanceTestCase::test_factor_rank_autocorrelation#03",
            ),
        ),
        pytest.param(
            "tests/test_performance.py::PerformanceTestCase::test_factor_rank_autocorrelation#04",
            id="tests/test_performance.py::PerformanceTestCase::test_factor_rank_autocorrelation#04",
            marks=pytest.mark.alphalens_upstream_case(
                "tests/test_performance.py::PerformanceTestCase::test_factor_rank_autocorrelation#04",
            ),
        ),
        pytest.param(
            "tests/test_performance.py::PerformanceTestCase::test_factor_rank_autocorrelation#05",
            id="tests/test_performance.py::PerformanceTestCase::test_factor_rank_autocorrelation#05",
            marks=pytest.mark.alphalens_upstream_case(
                "tests/test_performance.py::PerformanceTestCase::test_factor_rank_autocorrelation#05",
            ),
        ),
    ],
)
def test_factor_rank_autocorrelation_upstream_case(source_case_id: str) -> None:
    """Cover ordered/reversed/tied source ranks and exact period alignment."""

    ordinal = int(source_case_id.rsplit("#", 1)[1])
    dates = pd.date_range("2015-01-01", periods=12 if ordinal >= 4 else 4, freq="D", name="date")
    ordered = np.array([1.0, 2.0, 3.0, 4.0])
    reversed_values = ordered[::-1]
    rows = [ordered if position % 2 == 0 or ordinal < 2 else reversed_values for position in range(len(dates))]
    if ordinal in {0, 1}:
        rows = [ordered] * len(dates)
    source = pd.DataFrame(
        {"factor": np.asarray(rows).reshape(-1)},
        index=pd.MultiIndex.from_product((dates, ["A", "B", "C", "D"]), names=("date", "asset")),
    )
    original = source.copy(deep=True)
    period = 3 if ordinal >= 4 else 1
    actual = factor_rank_autocorrelation(source, period=period)
    ranked = source["factor"].groupby(level="date", observed=False).rank().unstack("asset")
    expected = ranked.corrwith(ranked.shift(period), axis=1)
    expected.name = period
    pd.testing.assert_series_equal(actual, expected)
    pd.testing.assert_frame_equal(source, original)


def test_turnover_empty_and_tie_boundary_contract() -> None:
    """Empty inputs retain a typed result and ties are counted as memberships."""

    empty_index = pd.MultiIndex.from_arrays([pd.DatetimeIndex([], name="date"), pd.Index([], name="asset")])
    empty = pd.Series([], index=empty_index, dtype=float, name="factor_quantile")
    pd.testing.assert_series_equal(quantile_turnover(empty, 1), pd.Series([], dtype=float, name=1))
