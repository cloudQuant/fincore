"""Task 4 RED skeleton for information analytics."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from fincore.alphalens import performance as strict_performance
from fincore.factor_analysis.performance import (
    factor_information_coefficient,
    mean_information_coefficient,
)


def _information_data(forward_returns: list[float]) -> pd.DataFrame:
    """Build a fresh copy of the pinned two-date, four-asset source fixture."""

    dates = pd.date_range("2015-01-01", periods=2, freq="D", name="date")
    assets = pd.Index(["A", "B", "C", "D"], name="asset")
    index = pd.MultiIndex.from_product((dates, assets), names=("date", "asset"))
    frame = pd.DataFrame(index=index)
    frame["factor"] = [1.0, 2.0, 3.0, 4.0, 4.0, 3.0, 2.0, 1.0]
    frame["group"] = pd.Categorical([1, 1, 2, 2, 1, 1, 2, 2])
    frame["1D"] = forward_returns
    return frame


@pytest.mark.parametrize(
    "source_case_id",
    [
        pytest.param(
            "tests/test_performance.py::PerformanceTestCase::test_information_coefficient#00",
            id="tests/test_performance.py::PerformanceTestCase::test_information_coefficient#00",
            marks=pytest.mark.alphalens_upstream_case(
                "tests/test_performance.py::PerformanceTestCase::test_information_coefficient#00",
            ),
        ),
        pytest.param(
            "tests/test_performance.py::PerformanceTestCase::test_information_coefficient#01",
            id="tests/test_performance.py::PerformanceTestCase::test_information_coefficient#01",
            marks=pytest.mark.alphalens_upstream_case(
                "tests/test_performance.py::PerformanceTestCase::test_information_coefficient#01",
            ),
        ),
        pytest.param(
            "tests/test_performance.py::PerformanceTestCase::test_information_coefficient#02",
            id="tests/test_performance.py::PerformanceTestCase::test_information_coefficient#02",
            marks=pytest.mark.alphalens_upstream_case(
                "tests/test_performance.py::PerformanceTestCase::test_information_coefficient#02",
            ),
        ),
        pytest.param(
            "tests/test_performance.py::PerformanceTestCase::test_information_coefficient#03",
            id="tests/test_performance.py::PerformanceTestCase::test_information_coefficient#03",
            marks=pytest.mark.alphalens_upstream_case(
                "tests/test_performance.py::PerformanceTestCase::test_information_coefficient#03",
            ),
        ),
    ],
)
def test_information_coefficient_upstream_case(source_case_id: str) -> None:
    """Use exact pinned source-row arguments and compare the full IC frame."""

    ordinal = int(source_case_id.rsplit("#", 1)[1])
    forward_returns = [4, 3, 2, 1, 1, 2, 3, 4] if ordinal == 0 else [1, 2, 3, 4, 4, 3, 2, 1]
    source = _information_data(forward_returns)
    original = source.copy(deep=True)
    actual = factor_information_coefficient(
        source,
        group_adjust=ordinal == 3,
        by_group=ordinal >= 2,
    )
    if ordinal >= 2:
        expected_index: pd.Index = pd.MultiIndex.from_product(
            (pd.date_range("2015-01-01", periods=2, freq="D", name="date"), [1, 2]),
            names=("date", "group"),
        )
        expected_values = np.ones(4)
    else:
        expected_index = pd.date_range("2015-01-01", periods=2, freq="D", name="date")
        expected_values = np.full(2, -1.0 if ordinal == 0 else 1.0)
    expected = pd.DataFrame({"1D": expected_values}, index=expected_index)
    pd.testing.assert_frame_equal(actual, expected, check_freq=False)
    strict_expected = expected
    if ordinal >= 2:
        strict_expected = pd.DataFrame(
            {"1D": expected_values},
            index=pd.MultiIndex.from_product(
                (
                    pd.date_range("2015-01-01", periods=2, freq="D", name="date"),
                    pd.CategoricalIndex([1, 2], categories=[1, 2], name="group"),
                ),
                names=("date", "group"),
            ),
        )
    pd.testing.assert_frame_equal(
        strict_performance.factor_information_coefficient(
            source,
            group_adjust=ordinal == 3,
            by_group=ordinal >= 2,
        ),
        strict_expected,
        check_freq=False,
    )
    pd.testing.assert_frame_equal(source, original)


@pytest.mark.parametrize(
    "source_case_id",
    [
        pytest.param(
            "tests/test_performance.py::PerformanceTestCase::test_mean_information_coefficient#00",
            id="tests/test_performance.py::PerformanceTestCase::test_mean_information_coefficient#00",
            marks=pytest.mark.alphalens_upstream_case(
                "tests/test_performance.py::PerformanceTestCase::test_mean_information_coefficient#00",
            ),
        ),
        pytest.param(
            "tests/test_performance.py::PerformanceTestCase::test_mean_information_coefficient#01",
            id="tests/test_performance.py::PerformanceTestCase::test_mean_information_coefficient#01",
            marks=pytest.mark.alphalens_upstream_case(
                "tests/test_performance.py::PerformanceTestCase::test_mean_information_coefficient#01",
            ),
        ),
        pytest.param(
            "tests/test_performance.py::PerformanceTestCase::test_mean_information_coefficient#02",
            id="tests/test_performance.py::PerformanceTestCase::test_mean_information_coefficient#02",
            marks=pytest.mark.alphalens_upstream_case(
                "tests/test_performance.py::PerformanceTestCase::test_mean_information_coefficient#02",
            ),
        ),
        pytest.param(
            "tests/test_performance.py::PerformanceTestCase::test_mean_information_coefficient#03",
            id="tests/test_performance.py::PerformanceTestCase::test_mean_information_coefficient#03",
            marks=pytest.mark.alphalens_upstream_case(
                "tests/test_performance.py::PerformanceTestCase::test_mean_information_coefficient#03",
            ),
        ),
    ],
)
def test_mean_information_coefficient_upstream_case(source_case_id: str) -> None:
    """Use exact daily/weekly/group pinned mean-IC source-row configurations."""

    ordinal = int(source_case_id.rsplit("#", 1)[1])
    forward_returns = [4, 3, 2, 1, 1, 2, 3, 4] if ordinal == 0 else [1, 2, 3, 4, 4, 3, 2, 1]
    source = _information_data(forward_returns)
    original = source.copy(deep=True)
    by_group = ordinal >= 2
    by_time = ("D", "W", None, "W")[ordinal]
    actual = mean_information_coefficient(source, by_group=by_group, by_time=by_time)
    expected_value = -1.0 if ordinal == 0 else 1.0
    if by_time is None:
        expected_index: pd.Index = pd.Index([1, 2], name="group")
    elif by_group:
        expected_index = pd.MultiIndex.from_product(
            (pd.DatetimeIndex(["2015-01-04"], name="date"), [1, 2]),
            names=("date", "group"),
        )
    elif by_time == "D":
        expected_index = pd.date_range("2015-01-01", periods=2, freq="D", name="date")
    else:
        expected_index = pd.DatetimeIndex(["2015-01-04"], name="date")
    expected = pd.DataFrame({"1D": np.full(len(expected_index), expected_value)}, index=expected_index)
    pd.testing.assert_frame_equal(actual, expected, check_freq=False)
    strict_expected = expected
    if ordinal == 2:
        strict_expected = pd.DataFrame(
            {"1D": [expected_value, expected_value]},
            index=pd.CategoricalIndex([1, 2], categories=[1, 2], name="group"),
        )
    elif ordinal == 3:
        strict_expected = pd.DataFrame(
            {"1D": np.full(2, expected_value)},
            index=pd.MultiIndex.from_product(
                (
                    pd.DatetimeIndex(["2015-01-04"], name="date"),
                    pd.CategoricalIndex([1, 2], categories=[1, 2], name="group"),
                ),
                names=("date", "group"),
            ),
        )
    pd.testing.assert_frame_equal(
        strict_performance.mean_information_coefficient(source, by_group=by_group, by_time=by_time),
        strict_expected,
        check_freq=False,
    )
    pd.testing.assert_frame_equal(source, original)


def test_information_coefficient_enhanced_nan_invariant() -> None:
    """The enhanced contract accepts sparse data and keeps IC inside [-1, 1]."""

    source = _information_data([1.0, 2.0, np.nan, 4.0, 4.0, 3.0, 2.0, 1.0])
    original = source.copy(deep=True)
    actual = factor_information_coefficient(source)
    assert (((actual >= -1) & (actual <= 1)) | actual.isna()).all().all()
    pd.testing.assert_frame_equal(source, original)


def test_strict_information_coefficient_propagates_nan_while_enhanced_is_pairwise() -> None:
    """Pinned strict SciPy IC propagates a missing forward return per date."""

    from fincore.alphalens import performance as strict

    date = pd.Timestamp("2024-03-01")
    index = pd.MultiIndex.from_product(([date], ["A", "B", "C"]), names=("date", "asset"))
    source = pd.DataFrame({"factor": [1.0, 2.0, 3.0], "1D": [1.0, np.nan, 3.0]}, index=index)
    expected = pd.DataFrame({"1D": [np.nan]}, index=pd.DatetimeIndex([date], name="date"))

    pd.testing.assert_frame_equal(strict.factor_information_coefficient(source), expected)
    pd.testing.assert_frame_equal(
        factor_information_coefficient(source),
        pd.DataFrame({"1D": [1.0]}, index=pd.DatetimeIndex([date], name="date")),
    )
    pd.testing.assert_series_equal(
        strict.mean_information_coefficient(source),
        pd.Series({"1D": np.nan}),
    )


def test_group_adjustment_replaces_integer_forward_columns_with_float_results() -> None:
    """Pandas 3 group adjustment accepts the pinned integer IC #03 fixture."""

    source = _information_data([1, 2, 3, 4, 4, 3, 2, 1])
    original = source.copy(deep=True)
    dates = pd.date_range("2015-01-01", periods=2, freq="D", name="date")
    enhanced_expected = pd.DataFrame(
        {"1D": np.ones(4)},
        index=pd.MultiIndex.from_product((dates, [1, 2]), names=("date", "group")),
    )
    strict_expected = pd.DataFrame(
        {"1D": np.ones(4)},
        index=pd.MultiIndex.from_product(
            (dates, pd.CategoricalIndex([1, 2], categories=[1, 2], name="group")),
            names=("date", "group"),
        ),
    )

    pd.testing.assert_frame_equal(
        factor_information_coefficient(source, group_adjust=True, by_group=True),
        enhanced_expected,
    )
    pd.testing.assert_frame_equal(
        strict_performance.factor_information_coefficient(source, group_adjust=True, by_group=True),
        strict_expected,
    )
    pd.testing.assert_frame_equal(source, original)
    assert source["1D"].dtype == np.dtype("int64")
