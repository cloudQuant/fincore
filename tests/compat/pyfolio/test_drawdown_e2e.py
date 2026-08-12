from __future__ import annotations

import warnings

import matplotlib.pyplot as plt
import pandas as pd

from fincore.metrics.drawdown import gen_drawdown_table


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
    assert caught == []
    pd.testing.assert_series_equal(returns, original)
    plt.close(figure)
