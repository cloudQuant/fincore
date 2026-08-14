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
from fincore.alphalens.utils import get_clean_factor_and_forward_returns


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


def test_cumulative_returns_projects_nan_as_a_zero_return() -> None:
    """Strict Alphalens compounding treats NaN as zero and clears Series.name."""

    from fincore.factor_analysis.performance import cumulative_returns as enhanced_cumulative_returns

    index = pd.date_range("2024-01-02", periods=4, freq="D", name="date")
    source = pd.Series([0.10, np.nan, -0.10, 0.20], index=index, name="factor_return")
    original = source.copy(deep=True)
    strict_expected = pd.Series([1.10, 1.10, 0.99, 1.188], index=index)
    enhanced_expected = pd.Series([1.10, 1.10, 0.99, 1.188], index=index, name="factor_return")

    pd.testing.assert_series_equal(cumulative_returns(source), strict_expected)
    pd.testing.assert_series_equal(enhanced_cumulative_returns(source), enhanced_expected)
    pd.testing.assert_series_equal(source, original)


def test_strict_cumulative_returns_preserves_an_empty_named_series() -> None:
    """Pinned empyrical returns its copy unchanged before compounding empties."""

    source = pd.Series(
        [],
        index=pd.DatetimeIndex([], name="date"),
        dtype=float,
        name="factor_return",
    )
    original = source.copy(deep=True)

    pd.testing.assert_series_equal(cumulative_returns(source), source)
    pd.testing.assert_series_equal(source, original)


_MEAN_RETURN_BY_QUANTILE_CASES = (
    # daily returns, literal factor matrix, bins, by_group, source means/errors
    (
        [1.1, 1.2, 1.1, 1.2, 1.1, 1.2],
        [[1, 2, 1, 2, 1, 2], [1, 2, 1, 2, 1, 2], [1, 2, 1, 2, 1, 2]],
        2,
        False,
        [0.1, 0.2],
        [0.0, 0.0],
    ),
    (
        [1.1, 1.2, 1.1, 1.2, 1.1, 1.2],
        [[1, 2, 1, 2, 1, 2], [1, 2, 1, 2, 1, 2], [1, 2, 1, 2, 1, 2]],
        2,
        True,
        [0.1, 0.1, 0.2, 0.2],
        [0.0, 0.0, 0.0, 0.0],
    ),
    (
        [1.1, 1.1, 1.1, 1.2, 1.2, 1.2],
        [[1, 2, 3, 1, 2, 3], [1, 2, 3, 1, 2, 3], [1, 2, 3, 1, 2, 3]],
        3,
        False,
        [0.15, 0.15, 0.15],
        [0.0, 0.0, 0.0],
    ),
    (
        [1.1, 1.1, 1.1, 1.2, 1.2, 1.2],
        [[1, 2, 3, 1, 2, 3], [1, 2, 3, 1, 2, 3], [1, 2, 3, 1, 2, 3]],
        3,
        True,
        [0.1, 0.2, 0.1, 0.2, 0.1, 0.2],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    ),
    (
        [1.5, 1.5, 1.2, 1.0, 1.0, 1.0],
        [[1, 1, 2, 2, 2, 2], [2, 2, 1, 2, 2, 2], [2, 2, 1, 2, 2, 2]],
        2,
        False,
        [0.3, 0.15],
        [0.1, 0.05],
    ),
    (
        [1.5, 1.5, 1.2, 1.0, 1.0, 1.0],
        [[1, 1, 3, 2, 2, 2], [3, 3, 1, 2, 2, 2], [3, 3, 1, 2, 2, 2]],
        3,
        False,
        [0.3, 0.0, 0.4],
        [0.1, 0.0, 0.1],
    ),
    (
        [1.6, 1.6, 1.0, 1.0, 1.0, 1.0],
        [[1, 1, 2, 2, 2, 2], [2, 2, 1, 1, 1, 1], [2, 2, 1, 1, 1, 1]],
        2,
        False,
        [0.2, 0.4],
        [0.2, 0.2],
    ),
    (
        [1.6, 1.6, 1.0, 1.6, 1.6, 1.0],
        [[1, 1, 2, 1, 1, 2], [2, 2, 1, 2, 2, 1], [2, 2, 1, 2, 2, 1]],
        2,
        True,
        [0.2, 0.2, 0.4, 0.4],
        [0.2, 0.2, 0.2, 0.2],
    ),
)


def _mean_quantile_source(daily_returns: list[float], factor_values: list[list[int]]) -> tuple[pd.Series, pd.DataFrame]:
    """Reconstruct the pinned prices/factor/group inputs for one source row."""

    dates = pd.date_range("2015-01-11", periods=4, freq="D", name="date")
    assets = pd.Index(["A", "B", "C", "D", "E", "F"], name="asset")
    prices = pd.DataFrame(
        [[daily_return**power for daily_return in daily_returns] for power in range(1, 5)],
        index=dates,
        columns=assets,
    )
    factor_dates = dates[:-1]
    factor_index = pd.MultiIndex.from_product((factor_dates, assets), names=("date", "asset"))
    factor = pd.Series(
        [value for row in factor_values for value in row],
        index=factor_index,
        name="factor",
        dtype=float,
    )
    return factor, prices


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
    """Reconstruct each pinned row and assert complete mean/error tables."""

    ordinal = int(source_case_id.rsplit("#", 1)[1])
    daily_returns, factor_values, bins, by_group, expected_values, expected_errors = _MEAN_RETURN_BY_QUANTILE_CASES[
        ordinal
    ]
    factor, prices = _mean_quantile_source(daily_returns, factor_values)
    original_factor = factor.copy(deep=True)
    original_prices = prices.copy(deep=True)
    source = get_clean_factor_and_forward_returns(
        factor,
        prices,
        groupby={"A": 1, "B": 1, "C": 1, "D": 2, "E": 2, "F": 2},
        quantiles=None,
        bins=bins,
        periods=(1,),
    )
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
                pd.Index(range(1, bins + 1), name="factor_quantile"),
                pd.CategoricalIndex([1, 2], categories=[1, 2], name="group"),
            )
        )
    else:
        expected_index = pd.Index(range(1, bins + 1), name="factor_quantile")
    expected = pd.DataFrame({"1D": expected_values}, index=expected_index)
    expected_error = pd.DataFrame({"1D": expected_errors}, index=expected_index)
    pd.testing.assert_frame_equal(actual, expected, rtol=1e-12, atol=1e-12)
    pd.testing.assert_frame_equal(standard_error, expected_error, rtol=1e-12, atol=1e-12)
    pd.testing.assert_series_equal(factor, original_factor)
    pd.testing.assert_frame_equal(prices, original_prices)


def test_mean_quantile_spread_and_by_date_enhanced_contract() -> None:
    """Cover the unmapped spread helper plus date-level enhanced output shape."""

    factor, prices = _mean_quantile_source(*_MEAN_RETURN_BY_QUANTILE_CASES[0][:2])
    source = get_clean_factor_and_forward_returns(
        factor,
        prices,
        groupby={"A": 1, "B": 1, "C": 1, "D": 2, "E": 2, "F": 2},
        quantiles=None,
        bins=2,
        periods=(1,),
    )
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
            "1D": [0.01, 0.02, -0.01, 0.03, 0.02, 0.03, 0.00, 0.04, 0.03, 0.04, 0.01, 0.05],
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
    source_assets = list(set(event_factor.index.get_level_values("asset")))
    strict_event_expected = event_returns.iloc[1:5].loc[:, source_assets].copy()
    strict_event_expected.index = pd.RangeIndex(-1, 3)
    strict_event = strict.common_start_returns(event_factor, event_returns, 1, 2, cumulative=True)
    enhanced_event = enhanced.common_start_returns(event_factor, event_returns, 1, 2, cumulative=True)
    pd.testing.assert_frame_equal(strict_event, strict_event_expected)
    enhanced_event_expected = strict_event_expected.loc[:, ["A", "B", "C", "D"]].copy()
    enhanced_event_expected.columns = enhanced_event_expected.columns.rename("asset")
    pd.testing.assert_frame_equal(enhanced_event, enhanced_event_expected)
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


def _alpha_beta_projection_inputs(*, constant_market: bool, missing_return: bool) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create literal OLS edge cases without reusing the production projection."""

    dates = pd.date_range("2024-02-01", periods=3, freq="D", name="date")
    index = pd.MultiIndex.from_product((dates, ["A", "B"]), names=("date", "asset"))
    market_rows = [1.0, 1.0] * 3 if constant_market else [0.01, 0.01, 0.02, 0.02, 0.03, 0.03]
    factor_data = pd.DataFrame(
        {"factor": [1.0, 2.0] * 3, "1D": market_rows},
        index=index,
    )
    portfolio = pd.DataFrame({"1D": [0.10, np.nan if missing_return else 0.20, 0.30]}, index=dates)
    return factor_data, portfolio


def test_strict_alpha_beta_preserves_ols_constant_market_projection() -> None:
    """A one-parameter source OLS fit projects both strict rows to NaN."""

    from fincore.alphalens import performance as strict

    factor_data, portfolio = _alpha_beta_projection_inputs(constant_market=True, missing_return=False)
    expected = pd.DataFrame({"1D": [np.nan, np.nan]}, index=pd.Index(["Ann. alpha", "beta"]))

    pd.testing.assert_frame_equal(strict.factor_alpha_beta(factor_data, returns=portfolio), expected)


def test_strict_alpha_beta_propagates_nan_while_enhanced_kernel_remains_profile_free() -> None:
    """Strict OLS propagates NaN inputs; enhanced least squares has its own policy."""

    from fincore.alphalens import performance as strict
    from fincore.factor_analysis import performance as enhanced

    factor_data, portfolio = _alpha_beta_projection_inputs(constant_market=False, missing_return=True)
    strict_expected = pd.DataFrame({"1D": [np.nan, np.nan]}, index=pd.Index(["Ann. alpha", "beta"]))

    pd.testing.assert_frame_equal(strict.factor_alpha_beta(factor_data, returns=portfolio), strict_expected)
    enhanced_actual = enhanced.factor_alpha_beta(factor_data, returns=portfolio)
    assert np.isfinite(enhanced_actual.to_numpy(dtype=float, copy=False)).all()


def test_strict_alpha_beta_accepts_a_series_for_the_first_of_multiple_periods() -> None:
    """Pinned source renames a Series to the first universe-return period."""

    from statsmodels.regression.linear_model import OLS
    from statsmodels.tools.tools import add_constant

    from fincore.alphalens import performance as strict

    factor_data, _ = _alpha_beta_projection_inputs(constant_market=False, missing_return=False)
    factor_data["5D"] = factor_data["1D"] * 2.0
    returns = pd.Series([0.10, 0.20, 0.30], index=pd.date_range("2024-02-01", periods=3, freq="D"))
    universe = factor_data.groupby(level="date", observed=False, sort=True)["1D"].mean()
    alpha, beta = OLS(returns.to_numpy(), add_constant(universe.to_numpy())).fit().params
    expected = pd.DataFrame(
        {"1D": [(1.0 + alpha) ** (pd.Timedelta("252Days") / pd.Timedelta("1D")) - 1.0, beta]},
        index=pd.Index(["Ann. alpha", "beta"]),
    )

    pd.testing.assert_frame_equal(strict.factor_alpha_beta(factor_data, returns=returns), expected)


def test_strict_alpha_beta_implicit_returns_use_the_strict_factor_returns_projection() -> None:
    """Implicit OLS inputs retain the strict all-missing portfolio row semantics."""

    from statsmodels.tools.sm_exceptions import MissingDataError

    from fincore.alphalens import performance as strict

    dates = pd.date_range("2024-04-01", periods=3, freq="D", name="date")
    index = pd.MultiIndex.from_product((dates, ["A", "B"]), names=("date", "asset"))
    factor_data = pd.DataFrame({"factor": [1.0, 2.0] * 3, "1D": [np.nan] * 6}, index=index)
    expected_returns = pd.DataFrame({"1D": [0.0, 0.0, 0.0]}, index=pd.DatetimeIndex(dates.to_numpy(), name="date"))

    pd.testing.assert_frame_equal(strict.factor_returns(factor_data), expected_returns)
    with pytest.raises(MissingDataError, match="exog contains inf or nans"):
        strict.factor_alpha_beta(factor_data)


@pytest.mark.parametrize("empty_boundary", ["missing-forward-column", "empty-explicit-returns"])
def test_strict_alpha_beta_projects_empty_periods_to_the_pinned_empty_frame(
    empty_boundary: str,
) -> None:
    """Pinned source adds no alpha/beta rows when no forward period is present."""

    from fincore.alphalens import performance as strict

    dates = pd.date_range("2024-05-01", periods=2, freq="D", name="date")
    index = pd.MultiIndex.from_product((dates, ["A", "B"]), names=("date", "asset"))
    factor_data = pd.DataFrame({"factor": [1.0, 2.0, 1.0, 2.0]}, index=index)
    if empty_boundary == "empty-explicit-returns":
        factor_data["1D"] = [0.01, 0.02, 0.03, 0.04]
        returns: pd.DataFrame | None = pd.DataFrame(index=dates)
    else:
        returns = None

    pd.testing.assert_frame_equal(
        strict.factor_alpha_beta(factor_data, returns=returns),
        pd.DataFrame(),
    )


@pytest.mark.parametrize("returns_kind", ["dataframe", "series"])
def test_strict_alpha_beta_does_not_swallow_explicit_returns_without_a_factor_period(
    returns_kind: str,
) -> None:
    """Pinned source preserves its downstream lookup errors for explicit returns."""

    from fincore.alphalens import performance as strict

    dates = pd.date_range("2024-05-01", periods=2, freq="D", name="date")
    index = pd.MultiIndex.from_product((dates, ["A", "B"]), names=("date", "asset"))
    factor_data = pd.DataFrame({"factor": [1.0, 2.0, 1.0, 2.0]}, index=index)
    if returns_kind == "dataframe":
        returns: pd.DataFrame | pd.Series = pd.DataFrame({"1D": [0.01, 0.02]}, index=dates)
        expected_error: type[Exception] = KeyError
    else:
        returns = pd.Series([0.01, 0.02], index=dates)
        expected_error = IndexError

    with pytest.raises(expected_error):
        strict.factor_alpha_beta(factor_data, returns=returns)


def test_strict_alpha_beta_aligns_zero_column_returns_before_empty_projection() -> None:
    """Pinned source reaches ``.loc`` before returning a genuine empty frame."""

    from fincore.alphalens import performance as strict

    dates = pd.date_range("2024-05-01", periods=2, freq="D", name="date")
    index = pd.MultiIndex.from_product((dates, ["A", "B"]), names=("date", "asset"))
    factor_data = pd.DataFrame(
        {"factor": [1.0, 2.0, 1.0, 2.0], "1D": [0.01, 0.02, 0.03, 0.04]},
        index=index,
    )
    out_of_universe_returns = pd.DataFrame(index=pd.DatetimeIndex([pd.Timestamp("2024-06-01")], name="date"))

    with pytest.raises(KeyError):
        strict.factor_alpha_beta(factor_data, returns=out_of_universe_returns)


@pytest.mark.parametrize("entrypoint", ["factor_information_coefficient", "mean_information_coefficient"])
def test_strict_group_adjusted_empty_factor_data_projects_the_source_concat_error(entrypoint: str) -> None:
    """Pinned strict group adjustment errors instead of inventing an empty IC result."""

    from fincore.alphalens import performance as strict

    empty_index = pd.MultiIndex.from_arrays(
        [pd.DatetimeIndex([], name="date"), pd.Index([], name="asset")],
        names=("date", "asset"),
    )
    factor_data = pd.DataFrame(
        {
            "factor": pd.Series([], dtype=float, index=empty_index),
            "group": pd.Categorical([], categories=["one", "two"]),
            "1D": pd.Series([], dtype=float, index=empty_index),
        },
        index=empty_index,
    )

    with pytest.raises(ValueError, match="No objects to concatenate"):
        getattr(strict, entrypoint)(factor_data, group_adjust=True)


@pytest.mark.parametrize("entrypoint", ["factor_information_coefficient", "mean_information_coefficient"])
def test_strict_group_adjusted_empty_factor_data_without_group_preserves_source_priority(entrypoint: str) -> None:
    """A missing group column wins over the pinned empty-concatenation error."""

    from fincore.alphalens import performance as strict

    empty_index = pd.MultiIndex.from_arrays(
        [pd.DatetimeIndex([], name="date"), pd.Index([], name="asset")],
        names=("date", "asset"),
    )
    factor_data = pd.DataFrame(
        {
            "factor": pd.Series([], dtype=float, index=empty_index),
            "1D": pd.Series([], dtype=float, index=empty_index),
        },
        index=empty_index,
    )

    with pytest.raises(KeyError, match="group"):
        getattr(strict, entrypoint)(factor_data, group_adjust=True)


@pytest.mark.parametrize("entrypoint", ["factor_information_coefficient", "mean_information_coefficient"])
def test_strict_by_group_without_group_projects_the_source_keyerror(entrypoint: str) -> None:
    """Pinned strict grouping looks up ``group`` instead of raising a custom error."""

    from fincore.alphalens import performance as strict

    dates = pd.date_range("2024-05-01", periods=1, freq="D", name="date")
    index = pd.MultiIndex.from_product((dates, ["A", "B"]), names=("date", "asset"))
    factor_data = pd.DataFrame({"factor": [1.0, 2.0], "1D": [0.01, 0.02]}, index=index)

    with pytest.raises(KeyError, match="group"):
        getattr(strict, entrypoint)(factor_data, by_group=True)


def test_strict_common_start_returns_rejects_an_absent_event_calendar() -> None:
    """Pinned ``pd.concat([])`` projection remains distinct from enhanced empty output."""

    from fincore.alphalens import performance as strict
    from fincore.factor_analysis import performance as enhanced

    factor_index = pd.MultiIndex.from_product(([pd.Timestamp("2024-03-04")], ["A"]), names=("date", "asset"))
    factor = pd.DataFrame({"factor_quantile": [1]}, index=factor_index)
    returns = pd.DataFrame({"A": [1.0, 1.1]}, index=pd.date_range("2024-03-01", periods=2, freq="D"))

    with pytest.raises(ValueError, match="No objects to concatenate"):
        strict.common_start_returns(factor, returns, before=1, after=1, cumulative=True)
    pd.testing.assert_frame_equal(
        enhanced.common_start_returns(factor, returns, before=1, after=1, cumulative=True),
        pd.DataFrame(index=pd.Index([-1, 0, 1])),
    )


def test_strict_common_start_returns_emits_one_pinned_slice_per_resolved_event(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The strict facade retains the source's visible per-event ``series`` prints."""

    from fincore.alphalens import performance as strict
    from fincore.factor_analysis import performance as enhanced

    dates = pd.date_range("2024-06-03", periods=4, freq="D", name="date")
    factor = pd.Series(
        [1.0, 1.0],
        index=pd.MultiIndex.from_tuples([(dates[1], "A"), (dates[2], "A")], names=("date", "asset")),
    )
    returns = pd.DataFrame({"A": [0.01, 0.02, 0.03, 0.04]}, index=dates)
    expected_windows: list[pd.DataFrame] = []
    for day_zero in (1, 2):
        window = returns.iloc[day_zero - 1 : day_zero + 2].loc[:, ["A"]].copy()
        window.index = pd.RangeIndex(-1, 2)
        expected_windows.append(window)
    expected_stdout = "".join(f"series =  {window}\n" for window in expected_windows)

    strict_actual = strict.common_start_returns(factor, returns, before=1, after=1, cumulative=True)
    strict_stdout = capsys.readouterr().out
    enhanced_actual = enhanced.common_start_returns(factor, returns, before=1, after=1, cumulative=True)
    enhanced_stdout = capsys.readouterr().out

    strict_expected = pd.concat(expected_windows, axis=1)
    pd.testing.assert_frame_equal(strict_actual, strict_expected)
    pd.testing.assert_frame_equal(enhanced_actual.rename_axis(columns=None), strict_expected)
    assert strict_stdout == expected_stdout
    assert strict_stdout.count("series = ") == 2
    assert enhanced_stdout == ""


def test_strict_common_start_returns_checks_demean_event_before_stdout(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Pinned ``demean_by.loc[timestamp]`` fails before a source slice is printed."""

    from fincore.alphalens import performance as strict

    dates = pd.date_range("2024-06-03", periods=3, freq="D", name="date")
    factor = pd.Series(
        [1.0],
        index=pd.MultiIndex.from_tuples([(dates[1], "A")], names=("date", "asset")),
    )
    demean_by = pd.Series(
        [1.0],
        index=pd.MultiIndex.from_tuples([(dates[0], "A")], names=("date", "asset")),
    )
    returns = pd.DataFrame({"A": [0.01, 0.02, 0.03]}, index=dates)

    with pytest.raises(KeyError) as error:
        strict.common_start_returns(factor, returns, before=1, after=1, cumulative=True, demean_by=demean_by)

    assert error.value.args == (dates[1],)
    assert capsys.readouterr().out == ""


def test_strict_common_start_returns_prints_source_set_slice_for_demean_assets(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Source stdout selects factor and demean assets through an unnamed set list."""

    from fincore.alphalens import performance as strict

    dates = pd.date_range("2024-06-03", periods=3, freq="D", name="date")
    factor = pd.Series(
        [1.0],
        index=pd.MultiIndex.from_tuples([(dates[1], "A")], names=("date", "asset")),
    )
    demean_by = pd.Series(
        [1.0],
        index=pd.MultiIndex.from_tuples([(dates[1], "B")], names=("date", "asset")),
    )
    returns = pd.DataFrame({"A": [0.01, 0.02, 0.03], "B": [0.04, 0.05, 0.06]}, index=dates)
    source_assets = list({"A"} | {"B"})
    expected_slice = returns.iloc[0:3].loc[:, source_assets].copy()
    expected_slice.index = pd.RangeIndex(-1, 2)

    strict.common_start_returns(factor, returns, before=1, after=1, cumulative=True, demean_by=demean_by)

    assert capsys.readouterr().out == f"series =  {expected_slice}\n"


@pytest.mark.parametrize(
    ("before", "after", "expected_offsets"),
    [(-1, 1, pd.RangeIndex(1, 2)), (1, -1, pd.RangeIndex(-1, 0))],
    ids=("negative-before", "negative-after"),
)
def test_strict_common_start_returns_preserves_pinned_signed_windows(
    before: int,
    after: int,
    expected_offsets: pd.RangeIndex,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Strict source compatibility permits signed windows while enhanced rejects them."""

    from fincore.alphalens import performance as strict
    from fincore.factor_analysis import performance as enhanced

    dates = pd.date_range("2024-06-03", periods=3, freq="D", name="date")
    factor = pd.Series(
        [1.0],
        index=pd.MultiIndex.from_tuples([(dates[1], "A")], names=("date", "asset")),
    )
    returns = pd.DataFrame({"A": [0.01, 0.02, 0.03]}, index=dates)
    start = max(1 - before, 0)
    stop = min(1 + after + 1, len(returns.index))
    expected = returns.iloc[start:stop].loc[:, ["A"]].copy()
    expected.index = expected_offsets

    actual = strict.common_start_returns(factor, returns, before=before, after=after, cumulative=True)
    stdout = capsys.readouterr().out

    pd.testing.assert_frame_equal(actual, expected)
    assert stdout == f"series =  {expected}\n"
    with pytest.raises(ValueError, match="non-negative"):
        enhanced.common_start_returns(factor, returns, before=before, after=after, cumulative=True)


def test_strict_common_start_returns_preserves_caller_columns_name_without_demean(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Pinned source selection retains the caller-provided columns metadata."""

    from fincore.alphalens import performance as strict

    dates = pd.date_range("2024-06-03", periods=3, freq="D", name="date")
    factor = pd.Series(
        [1.0],
        index=pd.MultiIndex.from_tuples([(dates[1], "A")], names=("date", "asset")),
    )
    returns = pd.DataFrame({"A": [0.01, 0.02, 0.03]}, index=dates)
    returns.columns = returns.columns.rename("ticker")
    expected = returns.iloc[0:3].loc[:, ["A"]].copy()
    expected.index = pd.RangeIndex(-1, 2)

    actual = strict.common_start_returns(factor, returns, before=1, after=1, cumulative=True)

    pd.testing.assert_frame_equal(actual, expected)
    assert capsys.readouterr().out == f"series =  {expected}\n"


@pytest.mark.parametrize("columns_name", ["asset", "ticker"])
def test_strict_common_start_returns_preserves_columns_name_after_compounding(
    columns_name: str,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Pinned ``returns.apply`` keeps named columns when ``cumulative`` is false."""

    from fincore.alphalens import performance as strict
    from fincore.factor_analysis import performance as enhanced

    dates = pd.date_range("2024-06-03", periods=3, freq="D", name="date")
    factor = pd.Series(
        [1.0],
        index=pd.MultiIndex.from_tuples([(dates[1], "A")], names=("date", "asset")),
    )
    returns = pd.DataFrame({"A": [0.10, 0.20, -0.10]}, index=dates)
    returns.columns = returns.columns.rename(columns_name)
    expected_returns = pd.DataFrame(
        {column: enhanced.cumulative_returns(returns[column]) for column in returns.columns},
        index=returns.index,
    )
    expected_returns.columns = returns.columns
    expected = expected_returns.iloc[0:3].loc[:, ["A"]].copy()
    expected.index = pd.RangeIndex(-1, 2)

    actual = strict.common_start_returns(factor, returns, before=1, after=1, cumulative=False)

    pd.testing.assert_frame_equal(actual, expected)
    assert capsys.readouterr().out == f"series =  {expected}\n"


def test_strict_common_start_returns_compounds_duplicate_columns_positionally(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Pinned ``DataFrame.apply`` compounds duplicate asset labels independently."""

    from fincore.alphalens import performance as strict
    from fincore.factor_analysis import performance as enhanced

    dates = pd.date_range("2024-06-03", periods=3, freq="D", name="date")
    factor = pd.Series(
        [1.0],
        index=pd.MultiIndex.from_tuples([(dates[1], "A")], names=("date", "asset")),
    )
    returns = pd.DataFrame(
        [[0.10, 0.20], [0.20, -0.10], [-0.10, 0.30]],
        index=dates,
        columns=pd.Index(["A", "A"], name="ticker"),
    )
    expected_returns = returns.apply(enhanced.cumulative_returns, axis=0)
    expected = expected_returns.iloc[0:3].loc[:, ["A"]].copy()
    expected.index = pd.RangeIndex(-1, 2)

    actual = strict.common_start_returns(factor, returns, before=1, after=1, cumulative=False)

    pd.testing.assert_frame_equal(actual, expected)
    assert capsys.readouterr().out == f"series =  {expected}\n"


@pytest.mark.parametrize("before", [1, np.int64(1)], ids=("python-int", "numpy-int64"))
def test_strict_common_start_returns_preserves_raw_concat_index_order(
    before: int | np.integer[object],
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Pinned ``pd.concat`` retains the first truncated event window's row order."""

    from fincore.alphalens import performance as strict

    dates = pd.date_range("2024-06-03", periods=4, freq="D", name="date")
    factor = pd.Series(
        [1.0, 1.0],
        index=pd.MultiIndex.from_tuples(
            [(dates[0], "A"), (dates[2], "A")],
            names=("date", "asset"),
        ),
    )
    returns = pd.DataFrame({"A": [0.01, 0.02, 0.03, 0.04]}, index=dates)
    first = returns.iloc[0:2].loc[:, ["A"]].copy()
    first.index = pd.RangeIndex(0, 2)
    second = returns.iloc[1:4].loc[:, ["A"]].copy()
    second.index = pd.RangeIndex(-1, 2)
    expected = pd.concat([first, second], axis=1)
    pd.testing.assert_index_equal(expected.index, pd.Index([0, 1, -1]))

    actual = strict.common_start_returns(factor, returns, before=before, after=1, cumulative=True)

    pd.testing.assert_frame_equal(actual, expected)
    assert capsys.readouterr().out == f"series =  {first}\nseries =  {second}\n"


@pytest.mark.parametrize("cumulative", [True, False], ids=("cumulative", "compound-first"))
def test_strict_common_start_returns_returns_source_set_column_order(
    cumulative: bool,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Normal strict results use the pinned source slices, not enhanced asset ordering."""

    from fincore.alphalens import performance as strict

    dates = pd.date_range("2024-06-03", periods=4, freq="D", name="date")
    factor = pd.Series(
        [1.0, 1.0, 1.0, 1.0],
        index=pd.MultiIndex.from_tuples(
            [(dates[1], "A"), (dates[1], "B"), (dates[2], "A"), (dates[2], "B")],
            names=("date", "asset"),
        ),
    )
    returns = pd.DataFrame({"A": [0.10, 0.20, 0.30, 0.40], "B": [-0.10, 0.10, 0.30, 0.40]}, index=dates)
    source_returns = returns if cumulative else (returns + 1.0).cumprod()
    source_assets = list({"A", "B"})
    expected_windows: list[pd.DataFrame] = []
    for day_zero in (1, 2):
        window = source_returns.iloc[day_zero - 1 : day_zero + 2].loc[:, source_assets].copy()
        window.index = pd.RangeIndex(-1, 2)
        expected_windows.append(window)
    expected = pd.concat(expected_windows, axis=1)

    actual = strict.common_start_returns(factor, returns, before=1, after=1, cumulative=cumulative)

    pd.testing.assert_frame_equal(actual, expected)
    assert capsys.readouterr().out == "".join(f"series =  {window}\n" for window in expected_windows)


@pytest.mark.parametrize(
    ("before", "after", "expected_offsets"),
    [(np.int64(1), np.int64(1), pd.RangeIndex(-1, 2)), (np.int64(-1), np.int64(1), pd.RangeIndex(1, 2))],
    ids=("positive-numpy-integrals", "negative-numpy-integral"),
)
def test_strict_common_start_returns_accepts_numpy_integral_windows(
    before: np.integer[object],
    after: np.integer[object],
    expected_offsets: pd.RangeIndex,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Pinned source accepts NumPy integral windows regardless of sign."""

    from fincore.alphalens import performance as strict
    from fincore.factor_analysis import performance as enhanced

    dates = pd.date_range("2024-06-03", periods=3, freq="D", name="date")
    factor = pd.Series(
        [1.0],
        index=pd.MultiIndex.from_tuples([(dates[1], "A")], names=("date", "asset")),
    )
    returns = pd.DataFrame({"A": [0.01, 0.02, 0.03]}, index=dates)
    start = max(1 - int(before), 0)
    stop = min(1 + int(after) + 1, len(returns.index))
    expected = returns.iloc[start:stop].loc[:, ["A"]].copy()
    expected.index = expected_offsets

    actual = strict.common_start_returns(factor, returns, before=before, after=after, cumulative=True)

    pd.testing.assert_frame_equal(actual, expected)
    assert capsys.readouterr().out == f"series =  {expected}\n"
    with pytest.raises(ValueError, match="non-negative"):
        enhanced.common_start_returns(factor, returns, before=before, after=after, cumulative=True)
