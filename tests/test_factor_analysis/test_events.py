"""Task 4 RED skeleton for event analytics."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from fincore.factor_analysis.performance import average_cumulative_return_by_quantile, common_start_returns


def _common_inputs() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build a fresh event fixture with a single auditable event date."""

    dates = pd.date_range("2015-01-15", periods=18, freq="D", name="date")
    assets = pd.Index(["A", "B", "C", "D"], name="asset")
    returns = pd.DataFrame(
        {asset: base ** np.arange(1, len(dates) + 1) for asset, base in zip(assets, [1.2, 1.4, 0.9, 0.8], strict=True)},
        index=dates,
    )
    event_date = dates[8]
    index = pd.MultiIndex.from_product(([event_date], assets), names=("date", "asset"))
    factor = pd.DataFrame({"factor_quantile": [1, 2, 3, 4]}, index=index)
    return factor, returns


def _manual_common(
    factor: pd.DataFrame,
    returns: pd.DataFrame,
    before: int,
    after: int,
    *,
    mean_by_date: bool,
    demeaned: bool,
) -> pd.DataFrame | pd.Series:
    """Independent event-window construction for the source-row oracle."""

    event_date = factor.index.get_level_values("date")[0]
    day_zero = returns.index.get_loc(event_date)
    window = returns.iloc[max(day_zero - before, 0) : min(day_zero + after + 1, len(returns))]
    event_assets = pd.Index(factor.index.get_level_values("asset").unique())
    selected = window.loc[:, event_assets].copy()
    selected.index = pd.RangeIndex(
        max(day_zero - before, 0) - day_zero, min(day_zero + after + 1, len(returns)) - day_zero
    )
    if demeaned:
        selected = selected.sub(selected.mean(axis=1), axis=0)
    return selected.mean(axis=1) if mean_by_date else selected


def _event_factor(quantiles: int, *, varying_universe: bool) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build deterministic ordinary/varying-universe event data for average tests."""

    _, returns = _common_inputs()
    selected_assets = ["A", "B", "C", "D"][:quantiles]
    if varying_universe:
        second_date = returns.index[9]
        first = pd.MultiIndex.from_product(([returns.index[8]], selected_assets), names=("date", "asset"))
        second = pd.MultiIndex.from_product(([second_date], selected_assets[::-1]), names=("date", "asset"))
        index = first.append(second)
        values = list(range(1, quantiles + 1)) + list(range(quantiles, 0, -1))
    else:
        index = pd.MultiIndex.from_product(([returns.index[8]], selected_assets), names=("date", "asset"))
        values = list(range(1, quantiles + 1))
    data = pd.DataFrame({"factor_quantile": values}, index=index)
    return data, returns


@pytest.mark.parametrize(
    "source_case_id",
    [
        pytest.param(
            "tests/test_performance.py::PerformanceTestCase::test_common_start_returns#00",
            id="tests/test_performance.py::PerformanceTestCase::test_common_start_returns#00",
            marks=pytest.mark.alphalens_upstream_case(
                "tests/test_performance.py::PerformanceTestCase::test_common_start_returns#00",
            ),
        ),
        pytest.param(
            "tests/test_performance.py::PerformanceTestCase::test_common_start_returns#01",
            id="tests/test_performance.py::PerformanceTestCase::test_common_start_returns#01",
            marks=pytest.mark.alphalens_upstream_case(
                "tests/test_performance.py::PerformanceTestCase::test_common_start_returns#01",
            ),
        ),
        pytest.param(
            "tests/test_performance.py::PerformanceTestCase::test_common_start_returns#02",
            id="tests/test_performance.py::PerformanceTestCase::test_common_start_returns#02",
            marks=pytest.mark.alphalens_upstream_case(
                "tests/test_performance.py::PerformanceTestCase::test_common_start_returns#02",
            ),
        ),
        pytest.param(
            "tests/test_performance.py::PerformanceTestCase::test_common_start_returns#03",
            id="tests/test_performance.py::PerformanceTestCase::test_common_start_returns#03",
            marks=pytest.mark.alphalens_upstream_case(
                "tests/test_performance.py::PerformanceTestCase::test_common_start_returns#03",
            ),
        ),
        pytest.param(
            "tests/test_performance.py::PerformanceTestCase::test_common_start_returns#04",
            id="tests/test_performance.py::PerformanceTestCase::test_common_start_returns#04",
            marks=pytest.mark.alphalens_upstream_case(
                "tests/test_performance.py::PerformanceTestCase::test_common_start_returns#04",
            ),
        ),
        pytest.param(
            "tests/test_performance.py::PerformanceTestCase::test_common_start_returns#05",
            id="tests/test_performance.py::PerformanceTestCase::test_common_start_returns#05",
            marks=pytest.mark.alphalens_upstream_case(
                "tests/test_performance.py::PerformanceTestCase::test_common_start_returns#05",
            ),
        ),
        pytest.param(
            "tests/test_performance.py::PerformanceTestCase::test_common_start_returns#06",
            id="tests/test_performance.py::PerformanceTestCase::test_common_start_returns#06",
            marks=pytest.mark.alphalens_upstream_case(
                "tests/test_performance.py::PerformanceTestCase::test_common_start_returns#06",
            ),
        ),
        pytest.param(
            "tests/test_performance.py::PerformanceTestCase::test_common_start_returns#07",
            id="tests/test_performance.py::PerformanceTestCase::test_common_start_returns#07",
            marks=pytest.mark.alphalens_upstream_case(
                "tests/test_performance.py::PerformanceTestCase::test_common_start_returns#07",
            ),
        ),
    ],
)
def test_common_start_returns_upstream_case(source_case_id: str) -> None:
    """Run all eight source window shapes with an independent numeric oracle."""

    ordinal = int(source_case_id.rsplit("#", 1)[1])
    before, after, mean_by_date, demeaned = (
        (2, 3, False, False),
        (3, 2, False, True),
        (3, 5, True, False),
        (1, 4, True, True),
        (6, 6, False, False),
        (6, 6, False, True),
        (6, 6, True, False),
        (6, 6, True, True),
    )[ordinal]
    source, returns = _common_inputs()
    original_source = source.copy(deep=True)
    original_returns = returns.copy(deep=True)
    actual = common_start_returns(
        source,
        returns,
        before,
        after,
        cumulative=True,
        mean_by_date=mean_by_date,
        demean_by=source if demeaned else None,
    )
    expected = _manual_common(source, returns, before, after, mean_by_date=mean_by_date, demeaned=demeaned)
    if mean_by_date:
        pd.testing.assert_series_equal(actual.iloc[:, 0], expected, check_names=False)  # type: ignore[arg-type]
    else:
        pd.testing.assert_frame_equal(actual, expected)  # type: ignore[arg-type]
    pd.testing.assert_frame_equal(source, original_source)
    pd.testing.assert_frame_equal(returns, original_returns)


@pytest.mark.parametrize(
    "source_case_id",
    [
        pytest.param(
            "tests/test_performance.py::PerformanceTestCase::test_average_cumulative_return_by_quantile#00",
            id="tests/test_performance.py::PerformanceTestCase::test_average_cumulative_return_by_quantile#00",
            marks=pytest.mark.alphalens_upstream_case(
                "tests/test_performance.py::PerformanceTestCase::test_average_cumulative_return_by_quantile#00",
            ),
        ),
        pytest.param(
            "tests/test_performance.py::PerformanceTestCase::test_average_cumulative_return_by_quantile#01",
            id="tests/test_performance.py::PerformanceTestCase::test_average_cumulative_return_by_quantile#01",
            marks=pytest.mark.alphalens_upstream_case(
                "tests/test_performance.py::PerformanceTestCase::test_average_cumulative_return_by_quantile#01",
            ),
        ),
        pytest.param(
            "tests/test_performance.py::PerformanceTestCase::test_average_cumulative_return_by_quantile#02",
            id="tests/test_performance.py::PerformanceTestCase::test_average_cumulative_return_by_quantile#02",
            marks=pytest.mark.alphalens_upstream_case(
                "tests/test_performance.py::PerformanceTestCase::test_average_cumulative_return_by_quantile#02",
            ),
        ),
        pytest.param(
            "tests/test_performance.py::PerformanceTestCase::test_average_cumulative_return_by_quantile#03",
            id="tests/test_performance.py::PerformanceTestCase::test_average_cumulative_return_by_quantile#03",
            marks=pytest.mark.alphalens_upstream_case(
                "tests/test_performance.py::PerformanceTestCase::test_average_cumulative_return_by_quantile#03",
            ),
        ),
        pytest.param(
            "tests/test_performance.py::PerformanceTestCase::test_average_cumulative_return_by_quantile#04",
            id="tests/test_performance.py::PerformanceTestCase::test_average_cumulative_return_by_quantile#04",
            marks=pytest.mark.alphalens_upstream_case(
                "tests/test_performance.py::PerformanceTestCase::test_average_cumulative_return_by_quantile#04",
            ),
        ),
        pytest.param(
            "tests/test_performance.py::PerformanceTestCase::test_average_cumulative_return_by_quantile#05",
            id="tests/test_performance.py::PerformanceTestCase::test_average_cumulative_return_by_quantile#05",
            marks=pytest.mark.alphalens_upstream_case(
                "tests/test_performance.py::PerformanceTestCase::test_average_cumulative_return_by_quantile#05",
            ),
        ),
    ],
)
def test_average_cumulative_return_by_quantile_upstream_case(source_case_id: str) -> None:
    """Check every ordinary-universe event row has the full quantile/mean/std table."""

    ordinal = int(source_case_id.rsplit("#", 1)[1])
    before, after, demeaned, quantiles = (
        (1, 2, False, 4),
        (1, 2, True, 4),
        (3, 0, False, 4),
        (0, 3, True, 4),
        (3, 3, False, 2),
        (3, 3, True, 2),
    )[ordinal]
    source, returns = _event_factor(quantiles, varying_universe=False)
    original_source = source.copy(deep=True)
    original_returns = returns.copy(deep=True)
    actual = average_cumulative_return_by_quantile(source, returns, before, after, demeaned)
    expected_index = pd.MultiIndex.from_product(
        (range(1, quantiles + 1), ["mean", "std"]), names=("factor_quantile", None)
    )
    pd.testing.assert_index_equal(actual.index, expected_index)
    pd.testing.assert_index_equal(actual.columns, pd.Index(range(-before, after + 1)))
    assert np.isfinite(actual.xs("mean", level=1).to_numpy(dtype=float, copy=False)).all()
    pd.testing.assert_frame_equal(source, original_source)
    pd.testing.assert_frame_equal(returns, original_returns)


@pytest.mark.parametrize(
    "source_case_id",
    [
        pytest.param(
            "tests/test_performance.py::PerformanceTestCase::test_average_cumulative_return_by_quantile_2#00",
            id="tests/test_performance.py::PerformanceTestCase::test_average_cumulative_return_by_quantile_2#00",
            marks=pytest.mark.alphalens_upstream_case(
                "tests/test_performance.py::PerformanceTestCase::test_average_cumulative_return_by_quantile_2#00",
            ),
        ),
        pytest.param(
            "tests/test_performance.py::PerformanceTestCase::test_average_cumulative_return_by_quantile_2#01",
            id="tests/test_performance.py::PerformanceTestCase::test_average_cumulative_return_by_quantile_2#01",
            marks=pytest.mark.alphalens_upstream_case(
                "tests/test_performance.py::PerformanceTestCase::test_average_cumulative_return_by_quantile_2#01",
            ),
        ),
        pytest.param(
            "tests/test_performance.py::PerformanceTestCase::test_average_cumulative_return_by_quantile_2#02",
            id="tests/test_performance.py::PerformanceTestCase::test_average_cumulative_return_by_quantile_2#02",
            marks=pytest.mark.alphalens_upstream_case(
                "tests/test_performance.py::PerformanceTestCase::test_average_cumulative_return_by_quantile_2#02",
            ),
        ),
        pytest.param(
            "tests/test_performance.py::PerformanceTestCase::test_average_cumulative_return_by_quantile_2#03",
            id="tests/test_performance.py::PerformanceTestCase::test_average_cumulative_return_by_quantile_2#03",
            marks=pytest.mark.alphalens_upstream_case(
                "tests/test_performance.py::PerformanceTestCase::test_average_cumulative_return_by_quantile_2#03",
            ),
        ),
    ],
)
def test_average_cumulative_return_by_quantile_2_upstream_case(source_case_id: str) -> None:
    """Keep the source generated-name collision and varying-universe rows live."""

    ordinal = int(source_case_id.rsplit("#", 1)[1])
    before, after, demeaned, quantiles = (
        (0, 2, False, 4),
        (0, 3, True, 4),
        (0, 3, False, 2),
        (0, 3, True, 2),
    )[ordinal]
    source, returns = _event_factor(quantiles, varying_universe=True)
    original_source = source.copy(deep=True)
    original_returns = returns.copy(deep=True)
    actual = average_cumulative_return_by_quantile(source, returns, before, after, demeaned)
    expected_index = pd.MultiIndex.from_product(
        (range(1, quantiles + 1), ["mean", "std"]), names=("factor_quantile", None)
    )
    pd.testing.assert_index_equal(actual.index, expected_index)
    pd.testing.assert_index_equal(actual.columns, pd.Index(range(-before, after + 1)))
    assert np.isfinite(actual.xs("mean", level=1).to_numpy(dtype=float, copy=False)).all()
    pd.testing.assert_frame_equal(source, original_source)
    pd.testing.assert_frame_equal(returns, original_returns)


def test_average_event_returns_by_group_enhanced_contract() -> None:
    """Enhanced event output has a group level when group-wise analytics are requested."""

    source, returns = _event_factor(2, varying_universe=True)
    source["group"] = pd.Categorical(["g1", "g2", "g2", "g1"])
    actual = average_cumulative_return_by_quantile(source, returns, 1, 1, by_group=True)
    assert actual.index.nlevels == 3
    pd.testing.assert_index_equal(actual.columns, pd.Index([-1, 0, 1]))
