"""Task 4 RED skeleton for weights, returns, and alpha/beta."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from fincore.alphalens import performance as strict_performance
from fincore.factor_analysis.performance import factor_alpha_beta, factor_returns, factor_weights


def _factor_frame(
    values: list[list[float]],
    tickers: list[str],
    groups: dict[str, str] | None = None,
) -> pd.DataFrame:
    """Create a fresh source-shaped frame while explicitly dropping NaN rows."""

    dates = pd.date_range("2000-01-12", periods=len(values), freq="D", name="date")
    entries = [
        (date, ticker, value)
        for date, row in zip(dates, values, strict=True)
        for ticker, value in zip(tickers, row, strict=True)
        if not pd.isna(value)
    ]
    index = pd.MultiIndex.from_tuples([(date, ticker) for date, ticker, _ in entries], names=("date", "asset"))
    frame = pd.DataFrame({"factor": [value for _, _, value in entries]}, index=index, dtype=float)
    if groups is not None:
        frame["group"] = [groups[str(asset)] for asset in index.get_level_values("asset")]
    return frame


_WEIGHT_CASES = (
    (
        [[3, 4, 2, 1, np.nan], [3, 4, -2, -1, np.nan], [3, np.nan, np.nan, 1, 4]],
        False,
        False,
        False,
        [0.30, 0.40, 0.20, 0.10, 0.30, 0.40, -0.20, -0.10, 0.375, 0.125, 0.50],
    ),
    (
        [[3, 4, 2, 1, np.nan], [3, 4, -2, -1, np.nan], [3, np.nan, np.nan, 1, 4]],
        True,
        False,
        False,
        [0.125, 0.375, -0.125, -0.375, 0.20, 0.30, -0.30, -0.20, 0.10, -0.50, 0.40],
    ),
    (
        [[3, 4, 2, 1, np.nan], [-3, 4, -2, 1, np.nan], [2, 2, 2, 3, 1]],
        False,
        True,
        False,
        [0.30, 0.40, 0.20, 0.10, -0.30, 0.40, -0.20, 0.10, 0.20, 0.20, 0.20, 0.30, 0.10],
    ),
    (
        [[3, 4, 2, 1, np.nan], [3, 4, -2, -1, np.nan], [3, np.nan, np.nan, 1, 4]],
        True,
        True,
        False,
        [0.25, 0.25, -0.25, -0.25, 0.25, 0.25, -0.25, -0.25, -0.50, np.nan, 0.50],
    ),
    (
        [[3, 4, 2, 1, 5], [3, 4, -2, -1, 5], [3, np.nan, np.nan, 1, np.nan]],
        False,
        False,
        True,
        [0.20, 0.20, 0.20, 0.20, 0.20, 0.20, 0.20, -0.20, -0.20, 0.20, 0.50, 0.50],
    ),
    (
        [[1, 4, 2, 3, np.nan], [1, 4, -2, -3, np.nan], [3, np.nan, np.nan, 2, 7]],
        True,
        False,
        True,
        [-0.25, 0.25, -0.25, 0.25, 0.25, 0.25, -0.25, -0.25, 0.0, -0.50, 0.50],
    ),
    (
        [
            [3, 4, 2, 1, np.nan],
            [-3, 4, -2, 1, np.nan],
            [3, np.nan, np.nan, 1, 4],
            [3, np.nan, np.nan, -1, 4],
            [3, np.nan, np.nan, 1, -4],
        ],
        False,
        True,
        True,
        [0.25, 0.25, 0.25, 0.25, -0.25, 0.25, -0.25, 0.25, 0.25, 0.50, 0.25, 0.25, -0.50, 0.25, 0.25, 0.50, -0.25],
    ),
    (
        [[1, 4, 2, 3, np.nan], [3, 4, -2, -1, np.nan], [3, np.nan, np.nan, 2, 7], [3, np.nan, np.nan, 2, -7]],
        True,
        True,
        True,
        [-0.25, 0.25, 0.25, -0.25, 0.25, 0.25, -0.25, -0.25, -0.50, np.nan, 0.50, 0.50, np.nan, -0.50],
    ),
)

_GROUPS = {"A": "Group1", "B": "Group2", "C": "Group1", "D": "Group2", "E": "Group1"}


@pytest.mark.parametrize(
    "source_case_id",
    [
        pytest.param(
            "tests/test_performance.py::PerformanceTestCase::test_factor_weights#00",
            id="tests/test_performance.py::PerformanceTestCase::test_factor_weights#00",
            marks=pytest.mark.alphalens_upstream_case(
                "tests/test_performance.py::PerformanceTestCase::test_factor_weights#00",
            ),
        ),
        pytest.param(
            "tests/test_performance.py::PerformanceTestCase::test_factor_weights#01",
            id="tests/test_performance.py::PerformanceTestCase::test_factor_weights#01",
            marks=pytest.mark.alphalens_upstream_case(
                "tests/test_performance.py::PerformanceTestCase::test_factor_weights#01",
            ),
        ),
        pytest.param(
            "tests/test_performance.py::PerformanceTestCase::test_factor_weights#02",
            id="tests/test_performance.py::PerformanceTestCase::test_factor_weights#02",
            marks=pytest.mark.alphalens_upstream_case(
                "tests/test_performance.py::PerformanceTestCase::test_factor_weights#02",
            ),
        ),
        pytest.param(
            "tests/test_performance.py::PerformanceTestCase::test_factor_weights#03",
            id="tests/test_performance.py::PerformanceTestCase::test_factor_weights#03",
            marks=pytest.mark.alphalens_upstream_case(
                "tests/test_performance.py::PerformanceTestCase::test_factor_weights#03",
            ),
        ),
        pytest.param(
            "tests/test_performance.py::PerformanceTestCase::test_factor_weights#04",
            id="tests/test_performance.py::PerformanceTestCase::test_factor_weights#04",
            marks=pytest.mark.alphalens_upstream_case(
                "tests/test_performance.py::PerformanceTestCase::test_factor_weights#04",
            ),
        ),
        pytest.param(
            "tests/test_performance.py::PerformanceTestCase::test_factor_weights#05",
            id="tests/test_performance.py::PerformanceTestCase::test_factor_weights#05",
            marks=pytest.mark.alphalens_upstream_case(
                "tests/test_performance.py::PerformanceTestCase::test_factor_weights#05",
            ),
        ),
        pytest.param(
            "tests/test_performance.py::PerformanceTestCase::test_factor_weights#06",
            id="tests/test_performance.py::PerformanceTestCase::test_factor_weights#06",
            marks=pytest.mark.alphalens_upstream_case(
                "tests/test_performance.py::PerformanceTestCase::test_factor_weights#06",
            ),
        ),
        pytest.param(
            "tests/test_performance.py::PerformanceTestCase::test_factor_weights#07",
            id="tests/test_performance.py::PerformanceTestCase::test_factor_weights#07",
            marks=pytest.mark.alphalens_upstream_case(
                "tests/test_performance.py::PerformanceTestCase::test_factor_weights#07",
            ),
        ),
    ],
)
def test_factor_weights_upstream_case(source_case_id: str) -> None:
    """Rebuild each pinned weight row with exact source input and values."""

    ordinal = int(source_case_id.rsplit("#", 1)[1])
    values, demeaned, group_adjust, equal_weight, expected_values = _WEIGHT_CASES[ordinal]
    source = _factor_frame(values, ["A", "B", "C", "D", "E"], _GROUPS)
    original = source.copy(deep=True)
    actual = factor_weights(source, demeaned=demeaned, group_adjust=group_adjust, equal_weight=equal_weight)
    expected = pd.Series(expected_values, index=source.index, name="factor", dtype=float)
    pd.testing.assert_series_equal(actual, expected, rtol=1e-12, atol=1e-12)
    pd.testing.assert_series_equal(
        strict_performance.factor_weights(
            source, demeaned=demeaned, group_adjust=group_adjust, equal_weight=equal_weight
        ),
        expected,
        rtol=1e-12,
        atol=1e-12,
    )
    pd.testing.assert_frame_equal(source, original)


@pytest.mark.parametrize(
    "source_case_id",
    [
        pytest.param(
            "tests/test_performance.py::PerformanceTestCase::test_factor_returns#00",
            id="tests/test_performance.py::PerformanceTestCase::test_factor_returns#00",
            marks=pytest.mark.alphalens_upstream_case(
                "tests/test_performance.py::PerformanceTestCase::test_factor_returns#00",
            ),
        ),
        pytest.param(
            "tests/test_performance.py::PerformanceTestCase::test_factor_returns#01",
            id="tests/test_performance.py::PerformanceTestCase::test_factor_returns#01",
            marks=pytest.mark.alphalens_upstream_case(
                "tests/test_performance.py::PerformanceTestCase::test_factor_returns#01",
            ),
        ),
        pytest.param(
            "tests/test_performance.py::PerformanceTestCase::test_factor_returns#02",
            id="tests/test_performance.py::PerformanceTestCase::test_factor_returns#02",
            marks=pytest.mark.alphalens_upstream_case(
                "tests/test_performance.py::PerformanceTestCase::test_factor_returns#02",
            ),
        ),
        pytest.param(
            "tests/test_performance.py::PerformanceTestCase::test_factor_returns#03",
            id="tests/test_performance.py::PerformanceTestCase::test_factor_returns#03",
            marks=pytest.mark.alphalens_upstream_case(
                "tests/test_performance.py::PerformanceTestCase::test_factor_returns#03",
            ),
        ),
        pytest.param(
            "tests/test_performance.py::PerformanceTestCase::test_factor_returns#04",
            id="tests/test_performance.py::PerformanceTestCase::test_factor_returns#04",
            marks=pytest.mark.alphalens_upstream_case(
                "tests/test_performance.py::PerformanceTestCase::test_factor_returns#04",
            ),
        ),
    ],
)
def test_factor_returns_upstream_case(source_case_id: str) -> None:
    """Compare each pinned portfolio-return row with its source fixture table."""

    ordinal = int(source_case_id.rsplit("#", 1)[1])
    factor_values, returns, group_adjust, expected_values = (
        (
            [1, 2, 3, 4, 4, 3, 2, 1],
            [4, 3, 2, 1, 1, 2, 3, 4],
            False,
            [-1.25, -1.25],
        ),
        ([1] * 8, [4, 3, 2, 1, 1, 2, 3, 4], False, [0.0, 0.0]),
        ([1, 2, 3, 4, 4, 3, 2, 1], [4, 3, 2, 1, 1, 2, 3, 4], True, [-0.5, -0.5]),
        ([1, 2, 3, 4, 1, 2, 3, 4], [1, 4, 1, 2, 1, 2, 2, 1], True, [1.0, 0.0]),
        ([1] * 8, [4, 3, 2, 1, 1, 2, 3, 4], True, [0.0, 0.0]),
    )[ordinal]
    source = _factor_frame(
        [factor_values[:4], factor_values[4:]], ["A", "B", "C", "D"], {"A": "one", "B": "one", "C": "two", "D": "two"}
    )
    source["1D"] = returns
    original = source.copy(deep=True)
    actual = factor_returns(source, demeaned=True, group_adjust=group_adjust)
    expected = pd.DataFrame(
        {"1D": expected_values}, index=pd.date_range("2000-01-12", periods=2, freq="D", name="date")
    )
    pd.testing.assert_frame_equal(actual, expected, rtol=1e-12, atol=1e-12, check_freq=False)
    pd.testing.assert_frame_equal(
        strict_performance.factor_returns(source, demeaned=True, group_adjust=group_adjust),
        expected,
        rtol=1e-12,
        atol=1e-12,
        check_freq=False,
    )
    pd.testing.assert_frame_equal(source, original)


def test_factor_returns_projects_an_all_nan_weighted_period_to_zero() -> None:
    """Pinned pandas groupby sum retains the portfolio row as 0.0, not NaN."""

    source = _factor_frame([[1.0, 2.0]], ["A", "B"])
    source["1D"] = np.nan
    original = source.copy(deep=True)
    expected = pd.DataFrame({"1D": [0.0]}, index=pd.DatetimeIndex([pd.Timestamp("2000-01-12")], name="date"))

    pd.testing.assert_frame_equal(factor_returns(source), expected)
    pd.testing.assert_frame_equal(strict_performance.factor_returns(source), expected)
    pd.testing.assert_frame_equal(source, original)


@pytest.mark.parametrize(
    "source_case_id",
    [
        pytest.param(
            "tests/test_performance.py::PerformanceTestCase::test_factor_alpha_beta#00",
            id="tests/test_performance.py::PerformanceTestCase::test_factor_alpha_beta#00",
            marks=pytest.mark.alphalens_upstream_case(
                "tests/test_performance.py::PerformanceTestCase::test_factor_alpha_beta#00",
            ),
        ),
    ],
)
def test_factor_alpha_beta_upstream_case(source_case_id: str) -> None:
    """Check the pinned annual-alpha/beta projection against an exact table."""

    source = _factor_frame(
        [[1, 2, 3, 4], [4, 3, 2, 1]], ["A", "B", "C", "D"], {"A": "one", "B": "one", "C": "two", "D": "two"}
    )
    source["1D"] = [1, 2, 3, 4, 1, 1, 1, 1]
    original = source.copy(deep=True)
    actual = factor_alpha_beta(source)
    expected = pd.DataFrame({"1D": [-1.0, 5.0 / 6.0]}, index=["Ann. alpha", "beta"])
    pd.testing.assert_frame_equal(actual, expected, rtol=1e-10, atol=1e-10)
    pd.testing.assert_frame_equal(strict_performance.factor_alpha_beta(source), expected, rtol=1e-10, atol=1e-10)
    pd.testing.assert_frame_equal(source, original)


def test_factor_weights_enhanced_gross_net_and_nan_contract() -> None:
    """Enhanced invariants cover gross/net exposure and constant-factor NaNs."""

    source = _factor_frame([[1, 2, 3, 4], [4, 3, 2, 1]], ["A", "B", "C", "D"])
    original = source.copy(deep=True)
    actual = factor_weights(source, demeaned=True)
    gross = actual.abs().groupby(level="date").sum()
    net = actual.groupby(level="date").sum()
    pd.testing.assert_series_equal(gross, pd.Series(1.0, index=gross.index, name="factor"))
    np.testing.assert_allclose(net.to_numpy(), 0.0, atol=1e-12)
    constant = _factor_frame([[1, 1, 1, 1]], ["A", "B", "C", "D"])
    assert factor_weights(constant, demeaned=True).isna().all()
    pd.testing.assert_frame_equal(source, original)
