"""Task 4 RED skeleton for the strict Alphalens performance facade."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from pandas.tseries.offsets import BDay, CustomBusinessDay, Day

from fincore.alphalens.performance import (
    compute_mean_returns_spread,
    cumulative_returns,
    mean_return_by_quantile,
)


@pytest.mark.parametrize(
    "source_case_id",
    [
        pytest.param(
            "tests/test_performance.py::PerformanceTestCase::test_cumulative_returns#00",
            id="tests/test_performance.py::PerformanceTestCase::test_cumulative_returns#00",
            marks=pytest.mark.alphalens_upstream_case(
                "tests/test_performance.py::PerformanceTestCase::test_cumulative_returns#00",
            ),
        ),
        pytest.param(
            "tests/test_performance.py::PerformanceTestCase::test_cumulative_returns#01",
            id="tests/test_performance.py::PerformanceTestCase::test_cumulative_returns#01",
            marks=pytest.mark.alphalens_upstream_case(
                "tests/test_performance.py::PerformanceTestCase::test_cumulative_returns#01",
            ),
        ),
        pytest.param(
            "tests/test_performance.py::PerformanceTestCase::test_cumulative_returns#02",
            id="tests/test_performance.py::PerformanceTestCase::test_cumulative_returns#02",
            marks=pytest.mark.alphalens_upstream_case(
                "tests/test_performance.py::PerformanceTestCase::test_cumulative_returns#02",
            ),
        ),
        pytest.param(
            "tests/test_performance.py::PerformanceTestCase::test_cumulative_returns#03",
            id="tests/test_performance.py::PerformanceTestCase::test_cumulative_returns#03",
            marks=pytest.mark.alphalens_upstream_case(
                "tests/test_performance.py::PerformanceTestCase::test_cumulative_returns#03",
            ),
        ),
        pytest.param(
            "tests/test_performance.py::PerformanceTestCase::test_cumulative_returns#04",
            id="tests/test_performance.py::PerformanceTestCase::test_cumulative_returns#04",
            marks=pytest.mark.alphalens_upstream_case(
                "tests/test_performance.py::PerformanceTestCase::test_cumulative_returns#04",
            ),
        ),
        pytest.param(
            "tests/test_performance.py::PerformanceTestCase::test_cumulative_returns#05",
            id="tests/test_performance.py::PerformanceTestCase::test_cumulative_returns#05",
            marks=pytest.mark.alphalens_upstream_case(
                "tests/test_performance.py::PerformanceTestCase::test_cumulative_returns#05",
            ),
        ),
        pytest.param(
            "tests/test_performance.py::PerformanceTestCase::test_cumulative_returns#06",
            id="tests/test_performance.py::PerformanceTestCase::test_cumulative_returns#06",
            marks=pytest.mark.alphalens_upstream_case(
                "tests/test_performance.py::PerformanceTestCase::test_cumulative_returns#06",
            ),
        ),
        pytest.param(
            "tests/test_performance.py::PerformanceTestCase::test_cumulative_returns#07",
            id="tests/test_performance.py::PerformanceTestCase::test_cumulative_returns#07",
            marks=pytest.mark.alphalens_upstream_case(
                "tests/test_performance.py::PerformanceTestCase::test_cumulative_returns#07",
            ),
        ),
        pytest.param(
            "tests/test_performance.py::PerformanceTestCase::test_cumulative_returns#08",
            id="tests/test_performance.py::PerformanceTestCase::test_cumulative_returns#08",
            marks=pytest.mark.alphalens_upstream_case(
                "tests/test_performance.py::PerformanceTestCase::test_cumulative_returns#08",
            ),
        ),
    ],
)
def test_cumulative_returns_upstream_case(source_case_id: str) -> None:
    """Rebuild all 1D/BDay/custom-business-day source compounding rows."""

    ordinal = int(source_case_id.rsplit("#", 1)[1])
    returns, expected_values = (
        ([1.0, 0.5, 1.0, 0.5, 0.5], [2.0, 3.0, 6.0, 9.0, 13.5]),
        ([0.1, 0.1, 0.1, 0.1, 0.1], [1.1, 1.21, 1.331, 1.4641, 1.61051]),
        ([-0.1, -0.1, -0.1, -0.1, -0.1], [0.9, 0.81, 0.729, 0.6561, 0.59049]),
    )[ordinal % 3]
    frequency = (Day(), BDay(), CustomBusinessDay(weekmask="Tue Wed Thu Fri Sun"))[ordinal // 3]
    index = pd.date_range("1999-01-01", periods=len(returns), freq=frequency)
    source = pd.Series(returns, index=index)
    original = source.copy(deep=True)
    actual = cumulative_returns(source)
    expected = pd.Series(expected_values, index=index)
    pd.testing.assert_series_equal(actual, expected, rtol=1e-12, atol=1e-12)
    pd.testing.assert_series_equal(source, original)


def _mean_quantile_source() -> pd.DataFrame:
    """Create a fresh pre-cleaned source-shaped factor table for all eight rows."""

    dates = pd.date_range("2015-01-11", periods=3, freq="D", name="date")
    assets = pd.Index(["A", "B", "C", "D", "E", "F"], name="asset")
    index = pd.MultiIndex.from_product((dates, assets), names=("date", "asset"))
    frame = pd.DataFrame(index=index)
    frame["factor"] = np.tile([1.1, 1.2, 1.1, 1.2, 1.1, 1.2], len(dates))
    frame["factor_quantile"] = np.tile([1, 2, 1, 2, 1, 2], len(dates))
    frame["group"] = pd.Categorical(np.tile([1, 1, 1, 2, 2, 2], len(dates)))
    frame["1D"] = np.tile([0.1, 0.2, 0.1, 0.2, 0.1, 0.2], len(dates))
    return frame


@pytest.mark.parametrize(
    "source_case_id",
    [
        pytest.param(
            "tests/test_performance.py::PerformanceTestCase::test_mean_return_by_quantile#00",
            id="tests/test_performance.py::PerformanceTestCase::test_mean_return_by_quantile#00",
            marks=pytest.mark.alphalens_upstream_case(
                "tests/test_performance.py::PerformanceTestCase::test_mean_return_by_quantile#00",
            ),
        ),
        pytest.param(
            "tests/test_performance.py::PerformanceTestCase::test_mean_return_by_quantile#01",
            id="tests/test_performance.py::PerformanceTestCase::test_mean_return_by_quantile#01",
            marks=pytest.mark.alphalens_upstream_case(
                "tests/test_performance.py::PerformanceTestCase::test_mean_return_by_quantile#01",
            ),
        ),
        pytest.param(
            "tests/test_performance.py::PerformanceTestCase::test_mean_return_by_quantile#02",
            id="tests/test_performance.py::PerformanceTestCase::test_mean_return_by_quantile#02",
            marks=pytest.mark.alphalens_upstream_case(
                "tests/test_performance.py::PerformanceTestCase::test_mean_return_by_quantile#02",
            ),
        ),
        pytest.param(
            "tests/test_performance.py::PerformanceTestCase::test_mean_return_by_quantile#03",
            id="tests/test_performance.py::PerformanceTestCase::test_mean_return_by_quantile#03",
            marks=pytest.mark.alphalens_upstream_case(
                "tests/test_performance.py::PerformanceTestCase::test_mean_return_by_quantile#03",
            ),
        ),
        pytest.param(
            "tests/test_performance.py::PerformanceTestCase::test_mean_return_by_quantile#04",
            id="tests/test_performance.py::PerformanceTestCase::test_mean_return_by_quantile#04",
            marks=pytest.mark.alphalens_upstream_case(
                "tests/test_performance.py::PerformanceTestCase::test_mean_return_by_quantile#04",
            ),
        ),
        pytest.param(
            "tests/test_performance.py::PerformanceTestCase::test_mean_return_by_quantile#05",
            id="tests/test_performance.py::PerformanceTestCase::test_mean_return_by_quantile#05",
            marks=pytest.mark.alphalens_upstream_case(
                "tests/test_performance.py::PerformanceTestCase::test_mean_return_by_quantile#05",
            ),
        ),
        pytest.param(
            "tests/test_performance.py::PerformanceTestCase::test_mean_return_by_quantile#06",
            id="tests/test_performance.py::PerformanceTestCase::test_mean_return_by_quantile#06",
            marks=pytest.mark.alphalens_upstream_case(
                "tests/test_performance.py::PerformanceTestCase::test_mean_return_by_quantile#06",
            ),
        ),
        pytest.param(
            "tests/test_performance.py::PerformanceTestCase::test_mean_return_by_quantile#07",
            id="tests/test_performance.py::PerformanceTestCase::test_mean_return_by_quantile#07",
            marks=pytest.mark.alphalens_upstream_case(
                "tests/test_performance.py::PerformanceTestCase::test_mean_return_by_quantile#07",
            ),
        ),
    ],
)
def test_mean_return_by_quantile_upstream_case(source_case_id: str) -> None:
    """Exercise every mapped source row with mean and standard-error frames."""

    ordinal = int(source_case_id.rsplit("#", 1)[1])
    source = _mean_quantile_source()
    original = source.copy(deep=True)
    by_group = ordinal in {1, 3, 7}
    actual, standard_error = mean_return_by_quantile(
        source,
        by_date=False,
        by_group=by_group,
        demeaned=False,
        group_adjust=False,
    )
    if by_group:
        expected_index: pd.Index = pd.MultiIndex.from_product(
            (
                pd.Index([1, 2], name="factor_quantile"),
                pd.CategoricalIndex([1, 2], categories=[1, 2], name="group"),
            )
        )
        expected_values = [0.1, 0.1, 0.2, 0.2]
    else:
        expected_index = pd.Index([1, 2], name="factor_quantile")
        expected_values = [0.1, 0.2]
    expected = pd.DataFrame({"1D": expected_values}, index=expected_index)
    pd.testing.assert_frame_equal(actual, expected, rtol=1e-12, atol=1e-12)
    pd.testing.assert_index_equal(standard_error.index, expected.index)
    pd.testing.assert_index_equal(standard_error.columns, expected.columns)
    pd.testing.assert_frame_equal(source, original)


def test_mean_quantile_spread_and_by_date_enhanced_contract() -> None:
    """Cover the unmapped spread helper plus date-level enhanced output shape."""

    source = _mean_quantile_source()
    mean_by_date, standard_error = mean_return_by_quantile(source, by_date=True, demeaned=False)
    spread, spread_error = compute_mean_returns_spread(mean_by_date, 2, 1, standard_error)
    expected = pd.Series(0.1, index=spread.index, name="1D")
    pd.testing.assert_series_equal(spread["1D"], expected)
    assert spread_error is not None
    assert (spread_error >= 0).all().all()


def _strict_facade_factor_data() -> pd.DataFrame:
    """Small independent factor table used to compare every Task 4 delegate."""

    dates = pd.date_range("2024-01-02", periods=3, freq="D", name="date")
    assets = pd.Index(["A", "B", "C", "D"], name="asset")
    index = pd.MultiIndex.from_product((dates, assets), names=("date", "asset"))
    return pd.DataFrame(
        {
            "factor": [1.0, 2.0, 3.0, 4.0, 4.0, 3.0, 2.0, 1.0, 2.0, 4.0, 1.0, 3.0],
            "factor_quantile": [1, 2, 3, 4] * 3,
            "group": ["one", "one", "two", "two"] * 3,
            "1D": [0.01, 0.02, -0.01, 0.03] * 3,
        },
        index=index,
    )


def test_strict_facade_matches_every_task4_enhanced_kernel() -> None:
    """Strict performance symbols delegate to the profile-free Task 4 kernel."""

    from fincore.alphalens import performance as strict
    from fincore.factor_analysis import performance as enhanced

    factor_data = _strict_facade_factor_data()
    quantiles = factor_data["factor_quantile"]
    event_returns = pd.DataFrame(
        {
            "A": [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07],
            "B": [0.02, 0.01, 0.00, -0.01, 0.02, 0.03, 0.04],
            "C": [-0.01, 0.01, 0.02, 0.00, -0.02, 0.01, 0.02],
            "D": [0.03, 0.02, 0.01, 0.00, 0.01, 0.02, 0.03],
        },
        index=pd.date_range("2024-01-01", periods=7, freq="D", name="date"),
    )
    event_factor = factor_data.loc[
        factor_data.index.get_level_values("date") == pd.Timestamp("2024-01-03"), ["factor_quantile"]
    ]

    pd.testing.assert_frame_equal(
        strict.factor_information_coefficient(factor_data),
        enhanced.factor_information_coefficient(factor_data),
    )
    pd.testing.assert_series_equal(
        strict.mean_information_coefficient(factor_data),
        enhanced.mean_information_coefficient(factor_data),
    )
    pd.testing.assert_series_equal(strict.factor_weights(factor_data), enhanced.factor_weights(factor_data))
    pd.testing.assert_frame_equal(strict.factor_returns(factor_data), enhanced.factor_returns(factor_data))
    pd.testing.assert_frame_equal(strict.factor_alpha_beta(factor_data), enhanced.factor_alpha_beta(factor_data))
    pd.testing.assert_series_equal(
        strict.cumulative_returns(pd.Series([0.1, -0.1])),
        enhanced.cumulative_returns(pd.Series([0.1, -0.1])),
    )
    strict_mean, strict_error = strict.mean_return_by_quantile(factor_data)
    enhanced_mean, enhanced_error = enhanced.mean_return_by_quantile(factor_data)
    pd.testing.assert_frame_equal(strict_mean, enhanced_mean)
    pd.testing.assert_frame_equal(strict_error, enhanced_error)
    strict_spread, strict_spread_error = strict.compute_mean_returns_spread(strict_mean, 4, 1, strict_error)
    enhanced_spread, enhanced_spread_error = enhanced.compute_mean_returns_spread(enhanced_mean, 4, 1, enhanced_error)
    pd.testing.assert_series_equal(strict_spread, enhanced_spread)
    assert strict_spread_error is not None
    assert enhanced_spread_error is not None
    pd.testing.assert_series_equal(strict_spread_error, enhanced_spread_error)
    pd.testing.assert_series_equal(strict.quantile_turnover(quantiles, 1), enhanced.quantile_turnover(quantiles, 1))
    pd.testing.assert_series_equal(
        strict.factor_rank_autocorrelation(factor_data),
        enhanced.factor_rank_autocorrelation(factor_data),
    )
    pd.testing.assert_frame_equal(
        strict.common_start_returns(event_factor, event_returns, 1, 2, cumulative=True),
        enhanced.common_start_returns(event_factor, event_returns, 1, 2, cumulative=True),
    )
    pd.testing.assert_frame_equal(
        strict.average_cumulative_return_by_quantile(event_factor, event_returns, 1, 2),
        enhanced.average_cumulative_return_by_quantile(event_factor, event_returns, 1, 2),
    )


def test_strict_alpha_beta_projects_missing_optional_dependency(monkeypatch: pytest.MonkeyPatch) -> None:
    """Statsmodels remains a call-time strict-facade dependency boundary."""

    from fincore.alphalens import performance as strict
    from fincore.exceptions import DependencyError

    def _missing(_: str) -> None:
        raise ModuleNotFoundError("no statsmodels")

    monkeypatch.setattr(strict.importlib, "import_module", _missing)
    with pytest.raises(DependencyError, match="factor-analysis"):
        strict.factor_alpha_beta(_strict_facade_factor_data())
