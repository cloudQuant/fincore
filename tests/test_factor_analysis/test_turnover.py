"""Task 4 RED skeleton for turnover analytics."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from pandas.tseries.offsets import BDay

from fincore.alphalens import performance as strict_performance
from fincore.factor_analysis.performance import factor_rank_autocorrelation, quantile_turnover


def _quantile_factor(values: list[list[float]]) -> pd.Series:
    """Build one fresh source-shaped quantile Series without stack ambiguity."""

    dates = pd.date_range("2015-01-01", periods=len(values), freq="D", name="date")
    assets = pd.Index(["A", "B", "C", "D"], name="asset")
    index = pd.MultiIndex.from_product((dates, assets), names=("date", "asset"))
    return pd.Series(np.asarray(values, dtype=float).reshape(-1), index=index, name="factor_quantile")


_TURNOVER_CASES = (
    (
        [[1, 2, 3, 4], [4, 3, 2, 1], [1, 2, 3, 4], [1, 2, 3, 4]],
        "1B",
        4,
        1,
        [np.nan, 1.0, 1.0, 0.0],
    ),
    (
        [[1, 2, 3, 4], [4, 3, 2, 1], [1, 2, 3, 4], [1, 2, 3, 4]],
        "1D",
        4,
        1,
        [np.nan, 1.0, 1.0, 0.0],
    ),
    (
        [[1, 2, 3, 4], [4, 3, 2, 1], [1, 2, 3, 4], [1, 2, 3, 4]],
        "1B",
        4,
        2,
        [np.nan, np.nan, 0.0, 1.0],
    ),
    (
        [[1, 2, 3, 4], [4, 3, 2, 1], [1, 2, 3, 4], [1, 2, 3, 4]],
        "1D",
        4,
        2,
        [np.nan, np.nan, 0.0, 1.0],
    ),
    (
        [[1, 2, 3, 4], [4, 3, 2, 1], [1, 2, 3, 4], [1, 2, 3, 4]],
        "1B",
        4,
        3,
        [np.nan, np.nan, np.nan, 0.0],
    ),
    (
        [[1, 2, 3, 4], [4, 3, 2, 1], [1, 2, 3, 4], [1, 2, 3, 4]],
        "1D",
        4,
        3,
        [np.nan, np.nan, np.nan, 0.0],
    ),
    (
        [[1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4]],
        "1B",
        3,
        1,
        [np.nan, 0.0, 0.0, 0.0],
    ),
    (
        [[1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4]],
        "1D",
        3,
        1,
        [np.nan, 0.0, 0.0, 0.0],
    ),
    (
        [[1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4]],
        "1B",
        3,
        2,
        [np.nan, np.nan, 0.0, 0.0],
    ),
    (
        [[1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4]],
        "1D",
        3,
        2,
        [np.nan, np.nan, 0.0, 0.0],
    ),
    (
        [[1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4]],
        "1B",
        3,
        3,
        [np.nan, np.nan, np.nan, 0.0],
    ),
    (
        [[1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4]],
        "1D",
        3,
        3,
        [np.nan, np.nan, np.nan, 0.0],
    ),
    (
        [[1, 2, 3, 4], [4, 3, 2, 1], [1, 2, 3, 4], [4, 3, 2, 1]],
        "1B",
        2,
        1,
        [np.nan, 1.0, 1.0, 1.0],
    ),
    (
        [[1, 2, 3, 4], [4, 3, 2, 1], [1, 2, 3, 4], [4, 3, 2, 1]],
        "1D",
        2,
        1,
        [np.nan, 1.0, 1.0, 1.0],
    ),
    (
        [[1, 2, 3, 4], [1, 3, 2, 4]] * 6,
        "1B",
        3,
        4,
        [np.nan, np.nan, np.nan, np.nan, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    ),
    (
        [[1, 2, 3, 4], [1, 3, 2, 4]] * 6,
        "1D",
        3,
        4,
        [np.nan, np.nan, np.nan, np.nan, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    ),
    (
        [[1, 2, 3, 4], [1, 3, 2, 4]] * 5 + [[1, 2, 3, 4], [1, 2, 3, 4]],
        "1B",
        3,
        10,
        [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, 0.0, 1.0],
    ),
    (
        [[1, 2, 3, 4], [1, 3, 2, 4]] * 5 + [[1, 2, 3, 4], [1, 2, 3, 4]],
        "1D",
        3,
        10,
        [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, 0.0, 1.0],
    ),
)


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
    """Rebuild each literal pinned row, including its frequency and result."""

    ordinal = int(source_case_id.rsplit("#", 1)[1])
    values, frequency, quantile, period, expected_values = _TURNOVER_CASES[ordinal]
    source = _quantile_factor(values)
    source.index = source.index.set_levels(
        pd.date_range("2015-01-01", periods=len(values), freq=BDay() if frequency == "1B" else "D", name="date"),
        level="date",
    )
    original = source.copy(deep=True)
    actual = quantile_turnover(source, quantile, period=period)
    expected_dates = pd.DatetimeIndex(
        pd.date_range("2015-01-01", periods=len(values), freq=BDay() if frequency == "1B" else "D").to_numpy(),
        name="date",
    )
    expected = pd.Series(
        expected_values,
        index=expected_dates,
        name=quantile,
        dtype=float,
    )
    pd.testing.assert_series_equal(actual, expected)
    pd.testing.assert_series_equal(strict_performance.quantile_turnover(source, quantile, period=period), expected)
    pd.testing.assert_series_equal(source, original)


_RANK_AUTOCORRELATION_CASES = (
    (
        [[1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4]],
        "1B",
        1,
        [np.nan, 1.0, 1.0, 1.0],
    ),
    (
        [[1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4]],
        "1D",
        1,
        [np.nan, 1.0, 1.0, 1.0],
    ),
    (
        [[4, 3, 2, 1], [1, 2, 3, 4], [4, 3, 2, 1], [1, 2, 3, 4]],
        "1B",
        1,
        [np.nan, -1.0, -1.0, -1.0],
    ),
    (
        [[4, 3, 2, 1], [1, 2, 3, 4], [4, 3, 2, 1], [1, 2, 3, 4]],
        "1D",
        1,
        [np.nan, -1.0, -1.0, -1.0],
    ),
    (
        [
            [1, 2, 3, 4],
            [2, 1, 4, 3],
            [4, 3, 2, 1],
            [1, 2, 3, 4],
            [2, 1, 4, 3],
            [4, 3, 2, 1],
            [2, 1, 4, 3],
            [4, 3, 2, 1],
            [1, 2, 3, 4],
            [2, 1, 4, 3],
            [2, 1, 4, 3],
            [4, 3, 2, 1],
        ],
        "1B",
        3,
        [np.nan, np.nan, np.nan, 1.0, 1.0, 1.0, 0.6, -0.6, -1.0, 1.0, -0.6, -1.0],
    ),
    (
        [
            [1, 2, 3, 4],
            [2, 1, 4, 3],
            [4, 3, 2, 1],
            [1, 2, 3, 4],
            [2, 1, 4, 3],
            [4, 3, 2, 1],
            [2, 1, 4, 3],
            [4, 3, 2, 1],
            [1, 2, 3, 4],
            [2, 1, 4, 3],
            [2, 1, 4, 3],
            [4, 3, 2, 1],
        ],
        "1D",
        3,
        [np.nan, np.nan, np.nan, 1.0, 1.0, 1.0, 0.6, -0.6, -1.0, 1.0, -0.6, -1.0],
    ),
)


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
    """Rebuild the source rank sequences and assert literal autocorrelations."""

    ordinal = int(source_case_id.rsplit("#", 1)[1])
    values, frequency, period, expected_values = _RANK_AUTOCORRELATION_CASES[ordinal]
    dates = pd.date_range("2015-01-01", periods=len(values), freq=BDay() if frequency == "1B" else "D", name="date")
    source = pd.DataFrame(
        {"factor": np.asarray(values, dtype=float).reshape(-1)},
        index=pd.MultiIndex.from_product((dates, ["A", "B", "C", "D"]), names=("date", "asset")),
    )
    original = source.copy(deep=True)
    actual = factor_rank_autocorrelation(source, period=period)
    expected = pd.Series(
        expected_values,
        index=dates,
        name=period,
        dtype=float,
    )
    pd.testing.assert_series_equal(actual, expected)
    pd.testing.assert_series_equal(strict_performance.factor_rank_autocorrelation(source, period=period), expected)
    pd.testing.assert_frame_equal(source, original)


def test_turnover_empty_and_tie_boundary_contract() -> None:
    """Empty inputs retain a typed result and ties are counted as memberships."""

    empty_index = pd.MultiIndex.from_arrays([pd.DatetimeIndex([], name="date"), pd.Index([], name="asset")])
    empty = pd.Series([], index=empty_index, dtype=float, name="factor_quantile")
    pd.testing.assert_series_equal(quantile_turnover(empty, 1), pd.Series([], dtype=float, name=1))
