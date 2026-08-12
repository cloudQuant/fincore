from __future__ import annotations

import warnings

import matplotlib.pyplot as plt
import pandas as pd
import pytest

from fincore.metrics.drawdown import gen_drawdown_table, get_top_drawdowns


def test_drawdown_table_preserves_padded_compatibility_shape(short_drawdown_returns: pd.Series) -> None:
    table = gen_drawdown_table(short_drawdown_returns, top=10)
    assert table.shape == (10, 5)
    assert table["Peak date"].notna().sum() == 1


def test_drawdown_plot_skips_padding_rows(short_drawdown_returns: pd.Series) -> None:
    from fincore.pyfolio import Pyfolio

    original_index = short_drawdown_returns.index.copy()
    figure, (reference_ax, ax) = plt.subplots(2, 1, sharex=True)
    short_drawdown_returns.plot(ax=reference_ax)
    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        result = Pyfolio().plot_drawdown_periods(short_drawdown_returns, top=10, ax=ax)
    assert result is ax
    assert ax.get_title() == "Top 10 drawdown periods"
    assert len(ax.collections) == 1
    assert short_drawdown_returns.index.equals(original_index)
    assert all(getattr(value, "tz", None) is None for value in ax.lines[0].get_xdata())
    plt.close(figure)


def test_drawdown_plot_preserves_exact_instants_across_nonexistent_dst_midnight() -> None:
    from fincore.pyfolio import Pyfolio

    index = pd.date_range("2018-11-04 02:00", periods=4, freq="h", tz="UTC").tz_convert("America/Sao_Paulo")
    returns = pd.Series([0.01, 0.02, -0.20, 0.25], index=index, name="returns")
    original = returns.copy()

    figure, ax = plt.subplots()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = Pyfolio().plot_drawdown_periods(returns, top=10, ax=ax)

    assert result is ax
    assert len(ax.collections) == 1
    shaded_x = ax.collections[0].get_paths()[0].vertices[:, 0]
    assert shaded_x.min() == pytest.approx(ax.convert_xunits(pd.Timestamp("2018-11-04 03:00")))
    assert shaded_x.max() == pytest.approx(ax.convert_xunits(pd.Timestamp("2018-11-04 05:00")))
    assert caught == []
    pd.testing.assert_series_equal(returns, original)
    plt.close(figure)


def test_drawdown_plot_handles_empty_datetime_series_without_shading() -> None:
    from fincore.pyfolio import Pyfolio

    returns = pd.Series([], index=pd.DatetimeIndex([], name="date"), dtype=float, name="returns")
    original = returns.copy()

    figure, ax = plt.subplots()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = Pyfolio().plot_drawdown_periods(returns, top=10, ax=ax)

    assert result is ax
    assert ax.get_title() == "Top 10 drawdown periods"
    assert ax.get_ylabel() == "Cumulative returns"
    assert ax.get_xlabel() == ""
    assert len(ax.collections) == 0
    assert caught == []
    pd.testing.assert_series_equal(returns, original)
    plt.close(figure)


def test_range_index_top_drawdowns_removes_recovered_period_once() -> None:
    returns = pd.Series([0.10, -0.10, 0.20], index=pd.RangeIndex(10, 13), name="returns")
    original = returns.copy()

    drawdowns = get_top_drawdowns(returns, top=10)

    assert drawdowns == [(10, 11, 12)]
    pd.testing.assert_series_equal(returns, original)


def test_range_index_drawdown_plot_shades_recovered_period_once_without_warnings() -> None:
    from fincore.pyfolio import Pyfolio

    returns = pd.Series([0.10, -0.10, 0.20], index=pd.RangeIndex(10, 13), name="returns")
    original = returns.copy()

    figure, ax = plt.subplots()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = Pyfolio().plot_drawdown_periods(returns, top=10, ax=ax)

    assert result is ax
    assert len(ax.collections) == 1
    shaded_x = ax.collections[0].get_paths()[0].vertices[:, 0]
    assert shaded_x.min() == pytest.approx(10)
    assert shaded_x.max() == pytest.approx(12)
    assert caught == []
    pd.testing.assert_series_equal(returns, original)
    plt.close(figure)


def test_unique_nonmonotonic_object_index_top_drawdowns_uses_positions() -> None:
    index = pd.Index(["z", "a", "m"], dtype=object)
    returns = pd.Series([0.10, -0.10, 0.20], index=index, name="returns")
    original = returns.copy()

    assert get_top_drawdowns(returns, top=10) == [("z", "a", "m")]
    pd.testing.assert_series_equal(returns, original)


def test_duplicate_object_index_top_drawdowns_uses_positions_without_repeating() -> None:
    index = pd.Index(["point", "point", "recovery"], dtype=object)
    returns = pd.Series([0.10, -0.10, 0.20], index=index, name="returns")
    original = returns.copy()

    assert get_top_drawdowns(returns, top=10) == [("point", "point", "recovery")]
    pd.testing.assert_series_equal(returns, original)
