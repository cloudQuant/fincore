"""Exact regression and call-count contracts for quantile statistics reuse."""

from __future__ import annotations

from itertools import product

import numpy as np
import pandas as pd
import pytest

from fincore.factor_analysis import analysis, data, performance, research


def _factor_data() -> pd.DataFrame:
    dates = pd.bdate_range("2024-01-02", periods=5, tz="UTC")
    assets = tuple(f"A{index:02d}" for index in range(8))
    index = pd.MultiIndex.from_product((dates, assets), names=("date", "asset"))
    row = np.tile(np.arange(len(assets), dtype=float), len(dates))
    session = np.repeat(np.arange(len(dates), dtype=float), len(assets))
    quantiles = np.tile(np.repeat((1, 2, 3, 4), 2), len(dates))
    quantile_values = pd.Categorical(quantiles, categories=(1, 2, 3, 4, 5), ordered=True)
    first = 0.002 * row + 0.001 * session
    second = -0.001 * row + 0.0005 * session
    first[[2, 17, 31]] = np.nan
    second[[7, 18, 38]] = np.nan
    groups = np.tile(np.array(("left", "left", "left", "left", "right", "right", "right", "right")), len(dates))
    return pd.DataFrame(
        {
            "factor": row - 3.5 + session / 10.0,
            "1D": first,
            "5D": second,
            "factor_quantile": quantile_values,
            "group": groups,
        },
        index=index,
    )


def _legacy_mean_return_by_quantile(
    factor_data: pd.DataFrame,
    *,
    by_date: bool,
    by_group: bool,
    demeaned: bool,
    group_adjust: bool,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Frozen pre-reuse aggregation body, independent of the new helpers."""
    copied = performance._copy_factor_data(factor_data)
    if "factor_quantile" not in copied.columns:
        raise ValueError("factor_data must contain a 'factor_quantile' column")
    if by_group and "group" not in copied.columns:
        raise ValueError("factor_data must contain a 'group' column when by_group=True")
    if group_adjust:
        copied = performance._demean_forward_returns(copied, by_group=True)
    elif demeaned:
        copied = performance._demean_forward_returns(copied, by_group=False)
    columns = performance._forward_columns(copied)
    dates = pd.DatetimeIndex(copied.index.get_level_values(0), name="date")
    groupers: list[object] = [copied["factor_quantile"], dates]
    if by_group:
        groupers.append(copied["group"])
    per_date = copied.groupby(groupers, observed=False, sort=True)[list(columns)].agg(["mean", "std", "count"])
    per_date_mean = per_date.xs("mean", axis=1, level=1)
    if by_date:
        return (
            per_date_mean,
            per_date.xs("std", axis=1, level=1) / np.sqrt(per_date.xs("count", axis=1, level=1)),
        )
    aggregate_keys: list[object] = [per_date_mean.index.get_level_values("factor_quantile")]
    if by_group:
        aggregate_keys.append(per_date_mean.index.get_level_values("group"))
    aggregate = per_date_mean.groupby(aggregate_keys, observed=False, sort=True).agg(["mean", "std", "count"])
    return (
        aggregate.xs("mean", axis=1, level=1),
        aggregate.xs("std", axis=1, level=1) / np.sqrt(aggregate.xs("count", axis=1, level=1)),
    )


@pytest.mark.parametrize(
    ("by_date", "by_group", "demeaned", "group_adjust"),
    list(product((False, True), repeat=4)),
)
def test_public_quantile_statistics_match_frozen_two_stage_oracle(
    by_date: bool,
    by_group: bool,
    demeaned: bool,
    group_adjust: bool,
) -> None:
    factor_data = _factor_data()
    expected = _legacy_mean_return_by_quantile(
        factor_data,
        by_date=by_date,
        by_group=by_group,
        demeaned=demeaned,
        group_adjust=group_adjust,
    )

    actual = performance.mean_return_by_quantile(
        factor_data,
        by_date=by_date,
        by_group=by_group,
        demeaned=demeaned,
        group_adjust=group_adjust,
    )

    for actual_frame, expected_frame in zip(actual, expected, strict=True):
        pd.testing.assert_frame_equal(actual_frame, expected_frame, check_exact=True, check_dtype=True)


@pytest.mark.parametrize("by_group", (False, True))
def test_report_views_reuse_one_per_date_aggregate_per_grouping(
    monkeypatch: pytest.MonkeyPatch,
    by_group: bool,
) -> None:
    factor_data = _factor_data()
    selected_data = factor_data.drop(columns=["5D"])
    original = performance._mean_return_by_quantile_views
    calls: list[tuple[object, ...]] = []

    def counted(
        *args: object,
        **kwargs: object,
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        calls.append(tuple(kwargs[key] for key in ("by_group", "demeaned", "group_adjust")))
        return original(*args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(performance, "_mean_return_by_quantile_views", counted)
    actual = analysis.analyze_factor(
        factor_data,
        long_short=False,
        group_neutral=False,
        by_group=by_group,
        periods=("1D",),
        turnover_periods=(1,),
        time_aggregation=(),
        include_portfolio_inputs=False,
    )

    expected_group = _legacy_mean_return_by_quantile(
        selected_data,
        by_date=False,
        by_group=by_group,
        demeaned=False,
        group_adjust=False,
    )
    expected_group_by_date = _legacy_mean_return_by_quantile(
        selected_data,
        by_date=True,
        by_group=by_group,
        demeaned=False,
        group_adjust=False,
    )
    expected_aggregate = _legacy_mean_return_by_quantile(
        selected_data,
        by_date=False,
        by_group=False,
        demeaned=False,
        group_adjust=False,
    )
    expected_aggregate_by_date = _legacy_mean_return_by_quantile(
        selected_data,
        by_date=True,
        by_group=False,
        demeaned=False,
        group_adjust=False,
    )
    assert len(calls) == (2 if by_group else 1)
    assert calls == ([(True, False, False), (False, False, False)] if by_group else [(False, False, False)])
    pd.testing.assert_frame_equal(actual.mean_returns_by_quantile, expected_group[0], check_exact=True)
    pd.testing.assert_frame_equal(actual.std_error_by_quantile, expected_group[1], check_exact=True)
    pd.testing.assert_frame_equal(actual.mean_returns_by_date, expected_group_by_date[0], check_exact=True)
    pd.testing.assert_frame_equal(actual.std_error_by_date, expected_group_by_date[1], check_exact=True)
    pd.testing.assert_frame_equal(
        actual.aggregate_mean_returns_by_quantile,
        expected_aggregate[0],
        check_exact=True,
    )
    pd.testing.assert_frame_equal(
        actual.aggregate_std_error_by_quantile,
        expected_aggregate[1],
        check_exact=True,
    )
    pd.testing.assert_frame_equal(
        actual.aggregate_mean_returns_by_date,
        expected_aggregate_by_date[0],
        check_exact=True,
    )
    pd.testing.assert_frame_equal(
        actual.aggregate_std_error_by_date,
        expected_aggregate_by_date[1],
        check_exact=True,
    )


def test_one_report_view_helper_builds_one_per_date_aggregate(monkeypatch: pytest.MonkeyPatch) -> None:
    original = performance._mean_return_by_quantile_per_date
    calls: list[tuple[object, ...]] = []

    def counted(*args: object, **kwargs: object) -> pd.DataFrame:
        calls.append(tuple(kwargs[key] for key in ("by_group", "demeaned", "group_adjust")))
        return original(*args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(performance, "_mean_return_by_quantile_per_date", counted)
    performance._mean_return_by_quantile_views(
        _factor_data(),
        by_group=False,
        demeaned=False,
        group_adjust=False,
    )
    assert calls == [(False, False, False)]


@pytest.mark.parametrize("min_quantile_assets", (1, 4))
def test_research_reuses_per_date_mean_and_count_grouping(
    monkeypatch: pytest.MonkeyPatch,
    min_quantile_assets: int,
) -> None:
    dates = pd.bdate_range("2024-01-02", periods=4, tz="UTC")
    assets = tuple(f"A{index:02d}" for index in range(14))
    index = pd.MultiIndex.from_product((dates, assets), names=("time", "asset"))
    values = np.tile(np.linspace(-0.05, 0.05, len(assets)), len(dates))
    factors = pd.DataFrame({"momentum": values}, index=index)
    returns = pd.DataFrame({"1D": values * 0.1}, index=index)
    grid = pd.DataFrame({"open_time": dates}, index=dates)
    original = research._mean_return_by_quantile_per_date
    calls: list[tuple[object, ...]] = []

    def counted(*args: object, **kwargs: object) -> pd.DataFrame:
        calls.append(tuple(kwargs[key] for key in ("by_group", "demeaned", "group_adjust")))
        return original(*args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(research, "_mean_return_by_quantile_per_date", counted)
    result = research.evaluate_factor_research(
        factors,
        grid=grid,
        prices=None,
        universe=None,
        groups=None,
        precomputed_returns=returns,
        spec={
            "targets": assets,
            "start": dates[0],
            "end": dates[-1],
            "periods": (1,),
            "label": "precomputed",
            "preprocess": "none",
            "min_assets": 1,
            "min_quantile_assets": min_quantile_assets,
            "quantiles": 4,
            "stability_lags": (),
        },
    )

    assert result.status == "OK"
    assert calls == [(False, False, False)]
    expected = data.prepare_factor_data_from_forward_returns(
        pd.Series(values, index=index, name="factor"),
        returns,
        groupby=dict.fromkeys(assets, "default"),
        quantiles=4,
        max_loss=1.0,
    ).data
    expected_means = _legacy_mean_return_by_quantile(
        expected,
        by_date=True,
        by_group=False,
        demeaned=False,
        group_adjust=False,
    )[0]
    expected_counts = expected.groupby(
        [expected["factor_quantile"], expected.index.get_level_values("date")],
        observed=False,
        sort=True,
    )["1D"].count()
    expected_means.index = expected_means.index.set_names(("quantile", "time"))
    expected_counts.index = expected_counts.index.set_names(("quantile", "time"))
    actual = result.quantile_returns.set_index(["quantile", "time"])
    expected_table = expected_means.rename(columns={"1D": "mean_return"})
    expected_table["n_assets"] = expected_counts
    unfiltered_rows = len(expected_table)
    expected_table = expected_table.loc[expected_table["n_assets"] >= min_quantile_assets]
    if min_quantile_assets == 4:
        assert not expected_table.empty
        assert len(expected_table) < unfiltered_rows
        assert "min_quantile_assets" in set(result.diagnostics["code"])
    pd.testing.assert_frame_equal(
        actual[["mean_return", "n_assets"]].sort_index(),
        expected_table[["mean_return", "n_assets"]].sort_index(),
        check_exact=True,
        check_dtype=True,
    )
