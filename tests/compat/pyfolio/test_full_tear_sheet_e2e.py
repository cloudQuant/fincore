from __future__ import annotations

import warnings

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _html_text(value: object) -> str:
    return str(getattr(value, "data", value))


def test_returns_workflow_returns_figure_with_expected_axes_and_tables(
    workflow_returns: pd.Series, monkeypatch
) -> None:
    from fincore import pyfolio
    from fincore.utils import common_utils

    displayed: list[object] = []
    monkeypatch.setattr(common_utils, "display", lambda value: displayed.append(value))
    backend = matplotlib.get_backend()

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        figure = pyfolio.create_returns_tear_sheet(workflow_returns, run_flask_app=True)

    assert isinstance(figure, matplotlib.figure.Figure)
    assert len(figure.axes) == 12
    assert figure.axes[0].get_title() == "Cumulative returns"
    assert any("Worst drawdown periods" in _html_text(value) for value in displayed)
    assert not [
        item
        for item in caught
        if isinstance(item.message, (DeprecationWarning, FutureWarning, PendingDeprecationWarning))
        or "timezone" in str(item.message).lower()
    ]
    assert matplotlib.get_backend() == backend
    plt.close(figure)


def test_strict_full_workflow_runs_real_subsheets_and_preserves_none_projection(
    workflow_returns: pd.Series, monkeypatch
) -> None:
    from fincore import pyfolio
    from fincore.utils import common_utils

    displayed: list[object] = []
    monkeypatch.setattr(common_utils, "display", lambda value: displayed.append(value))
    plt.close("all")

    naive_returns = workflow_returns.copy()
    naive_returns.index = naive_returns.index.tz_localize(None)
    result = pyfolio.create_full_tear_sheet(naive_returns, set_context=False)
    figures = [plt.figure(number) for number in plt.get_fignums()]

    assert result is None
    assert sorted(len(figure.axes) for figure in figures) == [1, 12]
    html = "\n".join(_html_text(value) for value in displayed)
    assert "Worst drawdown periods" in html
    assert "Stress Events" in html
    plt.close("all")


def test_functional_and_class_facades_share_the_same_workflow(workflow_returns: pd.Series, monkeypatch) -> None:
    from fincore import pyfolio

    sentinel = object()
    calls: list[pd.Series] = []

    def fake_returns(self, returns, **_kwargs):
        calls.append(returns)
        return sentinel

    monkeypatch.setattr(pyfolio.Pyfolio, "create_returns_tear_sheet", fake_returns)
    assert pyfolio.create_returns_tear_sheet(workflow_returns, run_flask_app=True) is sentinel
    assert calls == [workflow_returns]


def test_stored_state_pyfolio_perf_plot_uses_explicit_attribution_series() -> None:
    from fincore.pyfolio import Pyfolio

    index = pd.date_range("2024-01-02", periods=5, freq="B")
    stored_returns = pd.Series(0.5, index=index)
    total = pd.Series([0.01, -0.02, 0.03, 0.0, 0.01], index=index)
    common = total / 4
    specific = total - common
    data = pd.DataFrame(
        {
            "total_returns": total,
            "common_returns": common,
            "specific_returns": specific,
        }
    )

    figure, ax = plt.subplots()
    result = Pyfolio(returns=stored_returns).plot_perf_attrib_returns(data, ax=ax)

    assert result is ax
    np.testing.assert_allclose(ax.lines[1].get_ydata(), (1 + specific).cumprod() - 1)
    plt.close(figure)


def test_stored_state_pyfolio_slippage_plots_use_explicit_adjusted_returns() -> None:
    from fincore.pyfolio import Pyfolio

    index = pd.date_range("2024-01-02", periods=5, freq="B", tz="UTC")
    stored_returns = pd.Series(0.5, index=index)
    local_returns = pd.Series([0.01, -0.02, 0.03, 0.0, 0.01], index=index)
    positions = pd.DataFrame({"AAA": 50.0, "cash": 50.0}, index=index)
    transactions = pd.DataFrame(
        {"amount": [1.0], "price": [10.0], "symbol": ["AAA"]},
        index=index[:1],
    )
    instance = Pyfolio(returns=stored_returns)

    figure, (sweep_ax, sensitivity_ax) = plt.subplots(2, 1)
    sweep = instance.plot_slippage_sweep(
        local_returns,
        positions,
        transactions,
        slippage_params=(3,),
        ax=sweep_ax,
    )
    sensitivity = instance.plot_slippage_sensitivity(
        local_returns,
        positions,
        transactions,
        ax=sensitivity_ax,
    )

    assert sweep is sweep_ax
    assert sensitivity is sensitivity_ax
    assert len(sweep.lines) == 1
    assert len(sensitivity.lines) == 1
    assert np.isfinite(sweep.lines[0].get_ydata()).all()
    assert np.isfinite(sensitivity.lines[0].get_ydata()).all()
    plt.close(figure)
