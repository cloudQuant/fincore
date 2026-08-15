"""Pandas 3 compatibility budget for the enhanced factor-analysis surface."""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd


def _clean_factor_data() -> pd.DataFrame:
    dates = pd.date_range("2024-01-30", periods=4, name="date")
    assets = pd.CategoricalIndex(("A", "B"), categories=("A", "B", "UNOBSERVED"), name="asset")
    index = pd.MultiIndex.from_product((dates, assets), names=("date", "asset"))
    return pd.DataFrame(
        {
            "factor": np.tile((-1.0, 1.0), len(dates)),
            "factor_quantile": pd.Categorical(np.tile((1, 2), len(dates)), categories=(1, 2, 3)),
            "group": pd.Categorical(np.tile(("g1", "g2"), len(dates)), categories=("g1", "g2", "unused")),
            "1D": np.linspace(-0.02, 0.03, len(index)),
        },
        index=index,
    )


def test_stack_missing_value_policy_is_explicit_in_event_model() -> None:
    from fincore.factor_analysis.analysis import analyze_factor

    data = _clean_factor_data()
    returns = data["1D"].unstack("asset")
    returns.iloc[0, 0] = np.nan
    model = analyze_factor(
        data,
        include_pyfolio=False,
        event_returns=returns,
        event_before=0,
        event_after=1,
    )

    assert model.event_returns is not None
    assert not model.event_returns.return_distribution.isna().any()
    assert model.event_returns.return_distribution.index.nlevels == 2


def test_groupby_apply_preserves_quantize_index_and_order() -> None:
    from fincore.factor_analysis.data import quantize_factor

    data = _clean_factor_data().iloc[[3, 0, 2, 1, 7, 4, 6, 5]]
    actual = quantize_factor(data[["factor", "group"]], quantiles=2, by_group=False)

    assert actual.index.equals(data.index)
    assert actual.index.names == ["date", "asset"]


def test_categorical_grouping_declares_observed_policy() -> None:
    from fincore.factor_analysis.performance import mean_return_by_quantile

    mean, error = mean_return_by_quantile(_clean_factor_data(), by_group=True)

    assert error is not None
    assert set(mean.index.get_level_values("group")) == {"g1", "g2", "unused"}
    assert set(mean.index.get_level_values("factor_quantile")) == {1, 2, 3}


def test_dateoffset_is_on_offset_calendar_path() -> None:
    from fincore.factor_analysis.calendar import infer_trading_calendar

    weekdays = pd.date_range("2024-01-01", periods=10, freq="B")
    calendar = infer_trading_calendar(weekdays, weekdays)

    assert calendar.is_on_offset(pd.Timestamp("2024-01-08"))
    assert not calendar.is_on_offset(pd.Timestamp("2024-01-07"))


def test_forward_fill_uses_supported_api_without_warning() -> None:
    from fincore.factor_analysis.portfolio import positions

    data = _clean_factor_data()
    weights = data["factor"].groupby(level="date").transform(lambda values: values / values.abs().sum())
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = positions(weights, "1D", freq=pd.offsets.BDay())

    assert not result.isna().any().any()
    assert not [item for item in caught if issubclass(item.category, FutureWarning)]


def test_multiindex_codes_round_trip_in_serialization() -> None:
    from fincore.factor_analysis.models import deserialize_serializable_value, serializable_value

    source = _clean_factor_data()
    restored = deserialize_serializable_value(serializable_value(source))

    assert isinstance(restored, pd.DataFrame)
    assert all(
        np.array_equal(left, right) for left, right in zip(source.index.codes, restored.index.codes, strict=True)
    )


def test_monthly_alias_is_normalized_without_warning() -> None:
    from fincore.factor_analysis.performance import mean_information_coefficient

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        monthly = mean_information_coefficient(_clean_factor_data(), by_time="M")

    assert isinstance(monthly.index, pd.DatetimeIndex)
    assert monthly.index.freqstr == "ME"
    assert not [item for item in caught if issubclass(item.category, FutureWarning)]


def test_enhanced_kernel_warning_budget_is_zero_future_warnings() -> None:
    from fincore.factor_analysis.analysis import analyze_factor
    from fincore.factor_analysis.performance import factor_information_coefficient, factor_weights

    data = _clean_factor_data()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        factor_information_coefficient(data)
        factor_weights(data)
        analyze_factor(data, include_pyfolio=False)

    assert not [item for item in caught if issubclass(item.category, FutureWarning)]
