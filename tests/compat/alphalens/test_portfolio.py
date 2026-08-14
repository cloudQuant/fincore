"""Strict facade regressions for the Task 5 portfolio APIs."""

from __future__ import annotations

import inspect
import re

import numpy as np
import pandas as pd
import pytest
from pandas.tseries.offsets import BDay

from fincore.alphalens import performance as strict_performance
from fincore.factor_analysis.portfolio import create_pyfolio_input as enhanced_create_pyfolio_input
from fincore.factor_analysis.portfolio import factor_cumulative_returns as enhanced_factor_cumulative_returns
from fincore.factor_analysis.portfolio import factor_positions as enhanced_factor_positions
from fincore.factor_analysis.portfolio import positions as enhanced_positions


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


def _strict_weights(names: tuple[object, object]) -> pd.Series:
    dates = pd.bdate_range("2024-01-02", periods=2)
    assets = pd.Index(["A", "B"])
    index = pd.MultiIndex.from_product((dates, assets), names=names)
    return pd.Series([0.75, -0.25, -0.25, 0.75], index=index, name="factor")


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


@pytest.mark.parametrize("names", [(None, None), ("when", "symbol"), ("asset", "date")])
def test_strict_positions_preserves_source_level_names_and_object_dtypes(names: tuple[object, object]) -> None:
    """Pinned ``unstack`` retains caller level metadata on the strict surface."""

    weights = _strict_weights(names)
    expected = enhanced_positions(weights, "1D", freq=BDay()).astype(object)
    expected.index = expected.index.rename(names[0])
    expected.columns = expected.columns.rename(names[1])

    actual = strict_performance.positions(weights, "1D", freq=BDay())

    pd.testing.assert_frame_equal(actual, expected)


def test_strict_factor_position_and_pyfolio_position_frames_project_object_dtypes() -> None:
    """The enhanced float kernel must not leak its dtype through strict APIs."""

    source = _strict_factor_data()
    expected_factor_positions = enhanced_factor_positions(source, "5D", equal_weight=True).astype(object)
    strict_factor_positions = strict_performance.factor_positions(source, "5D", equal_weight=True)
    pd.testing.assert_frame_equal(strict_factor_positions, expected_factor_positions)

    enhanced_pyfolio = enhanced_create_pyfolio_input(
        source,
        "1D",
        capital=100_000,
        long_short=False,
        equal_weight=True,
        benchmark_period="missing",
    )
    expected_pyfolio = enhanced_pyfolio.positions.copy(deep=True)
    expected_pyfolio.loc[~expected_pyfolio.index.isin(enhanced_pyfolio.returns.index), :] = np.nan
    expected_pyfolio = expected_pyfolio.astype(object)
    _, strict_pyfolio_positions, _ = strict_performance.create_pyfolio_input(
        source,
        "1D",
        capital=100_000,
        long_short=False,
        equal_weight=True,
        benchmark_period="missing",
    )
    pd.testing.assert_frame_equal(strict_pyfolio_positions, expected_pyfolio)


def test_strict_pyfolio_capital_projection_retains_pinned_trailing_nan_rows() -> None:
    """Enhanced capital alignment is intentionally projected back at strict edge."""

    source = _strict_factor_data()
    returns, positions, _ = strict_performance.create_pyfolio_input(
        source,
        "5D",
        capital=100_000,
        long_short=False,
        equal_weight=True,
        benchmark_period="missing",
    )

    trailing = positions.loc[~positions.index.isin(returns.index)]
    assert not trailing.empty
    assert trailing.isna().all().all()
    assert positions.dtypes.astype(str).eq("object").all()


@pytest.mark.parametrize(("filter_name", "filter_values"), [("quantiles", [99]), ("groups", ["missing"])])
def test_strict_empty_portfolio_filters_project_the_pinned_unstack_error(
    filter_name: str,
    filter_values: list[object],
) -> None:
    """The strict source reaches ``Series.unstack`` after an empty filter."""

    source = _strict_factor_data()
    kwargs = {filter_name: filter_values}
    expected_message = "index must be a MultiIndex to unstack, <class 'pandas.RangeIndex'> was passed"

    assert enhanced_factor_positions(source, "1D", **kwargs).empty
    with pytest.raises(ValueError, match=re.escape(expected_message)):
        strict_performance.factor_positions(source, "1D", **kwargs)
    with pytest.raises(ValueError, match=re.escape(expected_message)):
        strict_performance.create_pyfolio_input(source, "1D", **kwargs)


@pytest.mark.parametrize(
    "function_name",
    ["factor_cumulative_returns", "factor_positions", "create_pyfolio_input"],
)
def test_strict_group_neutral_missing_group_preserves_source_key_error(function_name: str) -> None:
    """Pinned factor-weights indexing raises KeyError before enhanced validation."""

    source = _strict_factor_data().drop(columns="group")
    enhanced = {
        "factor_cumulative_returns": enhanced_factor_cumulative_returns,
        "factor_positions": enhanced_factor_positions,
        "create_pyfolio_input": enhanced_create_pyfolio_input,
    }[function_name]
    strict = getattr(strict_performance, function_name)

    with pytest.raises(ValueError, match="group"):
        enhanced(source, "1D", group_neutral=True)
    with pytest.raises(KeyError, match="group"):
        strict(source, "1D", group_neutral=True)


@pytest.mark.parametrize(
    "function_name",
    ["factor_cumulative_returns", "factor_positions", "create_pyfolio_input"],
)
def test_strict_period_validation_precedes_missing_group_and_empty_filter_projections(function_name: str) -> None:
    """Pinned APIs reject an unknown period before any downstream validation."""

    strict = getattr(strict_performance, function_name)
    expected_message = "Period 'missing' not found"

    with pytest.raises(ValueError, match=re.escape(expected_message)):
        strict(_strict_factor_data().drop(columns="group"), "missing", group_neutral=True)
    with pytest.raises(ValueError, match=re.escape(expected_message)):
        strict(_strict_factor_data(), "missing", quantiles=[99])


def test_strict_duplicate_forward_periods_project_key_error_but_keep_position_values() -> None:
    """Only the strict return-selection paths retain the source duplicate-label error."""

    source = _strict_factor_data()
    source.insert(2, "1D", source["1D"].to_numpy(), allow_duplicates=True)

    with pytest.raises(ValueError, match="exactly one"):
        enhanced_factor_cumulative_returns(source, "1D")
    with pytest.raises(KeyError, match=re.escape("'1D'")):
        strict_performance.factor_cumulative_returns(source, "1D")
    with pytest.raises(KeyError, match=re.escape("'1D'")):
        strict_performance.create_pyfolio_input(source, "1D")

    expected_positions = enhanced_factor_positions(source, "1D").astype(object)
    actual_positions = strict_performance.factor_positions(source, "1D")
    pd.testing.assert_frame_equal(actual_positions, expected_positions)


def test_strict_empty_canonical_factor_data_keeps_source_positions_error_only() -> None:
    """Source cumulative returns is empty, while its position path reaches unstack."""

    source = _strict_factor_data().iloc[:0]
    expected_message = "index must be a MultiIndex to unstack, <class 'pandas.RangeIndex'> was passed"

    assert enhanced_factor_cumulative_returns(source, "1D").empty
    assert strict_performance.factor_cumulative_returns(source, "1D").empty
    with pytest.raises(ValueError, match=re.escape(expected_message)):
        strict_performance.factor_positions(source, "1D")
    with pytest.raises(ValueError, match=re.escape(expected_message)):
        strict_performance.create_pyfolio_input(source, "1D")


def test_strict_duplicate_benchmark_period_projects_key_error_after_main_period_validation() -> None:
    """A duplicate benchmark label reaches its own pinned factor-return selection."""

    source = _strict_factor_data()
    source.insert(3, "5D", source["5D"].to_numpy(), allow_duplicates=True)

    with pytest.raises(ValueError, match="exactly one"):
        enhanced_create_pyfolio_input(source, "1D", benchmark_period="5D")
    with pytest.raises(KeyError, match=re.escape("'5D'")):
        strict_performance.create_pyfolio_input(source, "1D", benchmark_period="5D")
    with pytest.raises(ValueError, match=re.escape("Period 'missing' not found")):
        strict_performance.create_pyfolio_input(source, "missing", benchmark_period="5D")


@pytest.mark.parametrize("index_names", [(None, None), ("when", "symbol")])
@pytest.mark.parametrize(
    "function_name",
    ["factor_cumulative_returns", "factor_positions", "create_pyfolio_input"],
)
def test_strict_factor_portfolio_apis_require_the_pinned_date_level_name(
    index_names: tuple[object, object],
    function_name: str,
) -> None:
    """Strict factor-data APIs reject names that source ``get_level_values`` rejects."""

    source = _strict_factor_data()
    source.index = source.index.set_names(index_names)

    with pytest.raises(KeyError, match=re.escape("Level date not found")):
        getattr(strict_performance, function_name)(source, "1D")


@pytest.mark.parametrize("asset_name", ["ticker", None])
def test_strict_factor_portfolio_positions_preserve_the_source_asset_level_name(asset_name: object) -> None:
    """Only a source-named first level is required; the asset level is retained."""

    source = _strict_factor_data()
    source.index = source.index.set_names(("date", asset_name))

    factor_positions = strict_performance.factor_positions(source, "1D")
    _, pyfolio_positions, _ = strict_performance.create_pyfolio_input(source, "1D")

    assert factor_positions.columns.name == asset_name
    assert pyfolio_positions.columns.name == asset_name


@pytest.mark.parametrize(
    "function_name",
    ["factor_cumulative_returns", "factor_positions", "create_pyfolio_input"],
)
def test_strict_factor_date_level_validation_keeps_period_and_filter_priority(function_name: str) -> None:
    """The new date-name gate must not preempt earlier pinned source accesses."""

    strict = getattr(strict_performance, function_name)
    source = _strict_factor_data()
    source.index = source.index.set_names(("when", "symbol"))

    with pytest.raises(ValueError, match=re.escape("Period 'missing' not found")):
        strict(source, "missing")

    missing_quantile = source.drop(columns="factor_quantile")
    with pytest.raises(KeyError, match=re.escape("'factor_quantile'")):
        strict(missing_quantile, "1D", quantiles=[1])


@pytest.mark.parametrize(
    "function_name",
    ["factor_cumulative_returns", "factor_positions", "create_pyfolio_input"],
)
def test_strict_source_validation_orders_filters_and_group_before_duplicate_main_period(function_name: str) -> None:
    """Strict prechecks must follow the source call path rather than global rules."""

    strict = getattr(strict_performance, function_name)

    duplicate_missing_group = _strict_factor_data().drop(columns="group")
    duplicate_missing_group.insert(2, "1D", duplicate_missing_group["1D"].to_numpy(), allow_duplicates=True)
    with pytest.raises(KeyError, match=re.escape("'group'")):
        strict(duplicate_missing_group, "1D", group_neutral=True)

    duplicate_missing_quantile = _strict_factor_data().drop(columns="factor_quantile")
    duplicate_missing_quantile.insert(2, "1D", duplicate_missing_quantile["1D"].to_numpy(), allow_duplicates=True)
    with pytest.raises(KeyError, match=re.escape("'factor_quantile'")):
        strict(duplicate_missing_quantile, "1D", quantiles=[1])

    missing_filter_columns = _strict_factor_data().drop(columns=["factor_quantile", "group"])
    with pytest.raises(KeyError, match=re.escape("'factor_quantile'")):
        strict(missing_filter_columns, "1D", quantiles=[1], group_neutral=True)
