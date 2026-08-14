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
    """Pinned Alphalens compounding treats a missing daily return as zero."""

    from fincore.factor_analysis.performance import cumulative_returns as enhanced_cumulative_returns

    index = pd.date_range("2024-01-02", periods=4, freq="D", name="date")
    source = pd.Series([0.10, np.nan, -0.10, 0.20], index=index, name="factor_return")
    original = source.copy(deep=True)
    expected = pd.Series([1.10, 1.10, 0.99, 1.188], index=index, name="factor_return")

    pd.testing.assert_series_equal(cumulative_returns(source), expected)
    pd.testing.assert_series_equal(enhanced_cumulative_returns(source), expected)
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
