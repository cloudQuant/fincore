"""Branch-completion tests for factor_analysis.render_matplotlib validation paths."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pandas as pd
import pytest

import fincore.factor_analysis.render_matplotlib as rm


def _quantile_frame() -> pd.DataFrame:
    idx = pd.MultiIndex.from_tuples(
        [(1, pd.Timestamp("2024-01-01")), (2, pd.Timestamp("2024-01-01")), (1, pd.Timestamp("2024-01-02"))],
        names=("factor_quantile", "date"),
    )
    return pd.DataFrame({"1D": [0.01, -0.02, 0.03]}, index=idx)


# ---------------------------------------------------------------------------
# input helpers
# ---------------------------------------------------------------------------


def test_as_series_rejects_non_series() -> None:
    with pytest.raises(TypeError, match="Series"):
        rm._as_series([1.0, 2.0], "x")  # type: ignore[arg-type]


def test_as_frame_series_becomes_frame() -> None:
    result = rm._as_frame(pd.Series([1.0, 2.0], name="col"), "x")
    assert isinstance(result, pd.DataFrame)
    assert result.columns == ["col"]


def test_normalize_axes_rejects_insufficient_axes() -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 1)
    try:
        with pytest.raises(ValueError, match="expected at least"):
            rm._normalize_axes([ax], 2, plt, figsize=(6, 6))
    finally:
        plt.close(fig)


# ---------------------------------------------------------------------------
# table builders
# ---------------------------------------------------------------------------


def test_build_quantile_statistics_table_rejects_non_dataframe() -> None:
    with pytest.raises(TypeError, match="DataFrame"):
        rm.build_quantile_statistics_table(pd.Series([1.0]))  # type: ignore[arg-type]


def test_build_quantile_statistics_table_rejects_missing_columns() -> None:
    with pytest.raises(ValueError, match="factor_quantile"):
        rm.build_quantile_statistics_table(pd.DataFrame({"factor": [1.0]}))


def test_build_turnover_tables_rejects_non_dataframe() -> None:
    with pytest.raises(TypeError, match="DataFrames"):
        rm.build_turnover_tables({}, {1: pd.Series([1.0])})  # type: ignore[dict-item]


# ---------------------------------------------------------------------------
# plot functions — validation branches
# ---------------------------------------------------------------------------


def test_plot_ic_qq_with_explicit_distribution() -> None:
    from scipy import stats

    ic = pd.DataFrame({"1D": np.random.default_rng(1).normal(0, 0.1, 50)})
    result = rm.plot_ic_qq(ic, theoretical_dist=stats.t)
    assert result is not None


def test_plot_quantile_returns_bar_by_group_requires_group() -> None:
    frame = pd.DataFrame(
        {"1D": [0.01, -0.02]},
        index=pd.Index([1, 2], name="factor_quantile"),
    )
    with pytest.raises(ValueError, match="group"):
        rm.plot_quantile_returns_bar(frame, by_group=True)


def test_plot_quantile_returns_violin_requires_quantile_index() -> None:
    frame = pd.DataFrame({"1D": [0.01, -0.02]})
    with pytest.raises(ValueError, match="factor_quantile"):
        rm.plot_quantile_returns_violin(frame)


def test_plot_mean_quantile_returns_spread_rejects_bad_type() -> None:
    with pytest.raises(TypeError, match="Series"):
        rm.plot_mean_quantile_returns_spread_time_series([1.0, 2.0])  # type: ignore[arg-type]


def test_plot_top_bottom_quantile_turnover_rejects_empty() -> None:
    with pytest.raises(ValueError, match="at least one quantile"):
        rm.plot_top_bottom_quantile_turnover(pd.DataFrame())


def test_plot_cumulative_returns_by_quantile_requires_quantile_index() -> None:
    frame = pd.DataFrame({"1": [1.0]})
    with pytest.raises(ValueError, match="factor_quantile"):
        rm.plot_cumulative_returns_by_quantile(frame, "1D")


def test_plot_quantile_average_cumulative_return_requires_quantile_index() -> None:
    frame = pd.DataFrame({"0": [1.0]})
    with pytest.raises(ValueError, match="factor_quantile"):
        rm.plot_quantile_average_cumulative_return(frame)


def test_plot_events_distribution_requires_date_level() -> None:
    with pytest.raises(ValueError, match="date"):
        rm.plot_events_distribution(pd.Series([1.0, 2.0]))


def test_plot_events_distribution_rejects_bad_num_bars() -> None:
    idx = pd.MultiIndex.from_product([["2024-01-01"], ["A"]], names=("date", "asset"))
    events = pd.Series([1.0], index=idx)
    with pytest.raises(ValueError, match="num_bars"):
        rm.plot_events_distribution(events, num_bars=0)


def test_plot_events_distribution_single_date() -> None:
    idx = pd.MultiIndex.from_product([["2024-01-01"], ["A", "B"]], names=("date", "asset"))
    events = pd.Series([1.0, 2.0], index=idx)
    result = rm.plot_events_distribution(events)
    assert result is not None


def test_plot_monthly_ic_heatmap_rejects_non_datetime_index() -> None:
    frame = pd.DataFrame({"1D": [0.01]}, index=[1])
    with pytest.raises(TypeError, match="year and month"):
        rm.plot_monthly_ic_heatmap(frame)


def test_plot_dependencies_missing_matplotlib(monkeypatch) -> None:
    import importlib

    from fincore.exceptions import DependencyError

    real_import = importlib.import_module

    def fake_import(name, *args, **kwargs):
        if name == "matplotlib.pyplot":
            raise ModuleNotFoundError("no matplotlib")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(importlib, "import_module", fake_import)
    with pytest.raises(DependencyError, match="alphalens"):
        rm._plot_dependencies()


def test_build_information_table_missing_scipy(monkeypatch) -> None:
    import importlib

    from fincore.exceptions import DependencyError

    real_import = importlib.import_module

    def fake_import(name, *args, **kwargs):
        if name == "scipy.stats":
            raise ModuleNotFoundError("no scipy")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(importlib, "import_module", fake_import)
    with pytest.raises(DependencyError, match="scipy"):
        rm.build_information_table(pd.DataFrame({"1D": [0.01, -0.02]}))


def test_plot_ic_qq_missing_scipy(monkeypatch) -> None:
    import importlib

    from fincore.exceptions import DependencyError

    real_import = importlib.import_module

    def fake_import(name, *args, **kwargs):
        if name == "scipy.stats":
            raise ModuleNotFoundError("no scipy")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(importlib, "import_module", fake_import)
    ic = pd.DataFrame({"1D": np.random.default_rng(1).normal(0, 0.1, 20)})
    with pytest.raises(DependencyError, match="scipy"):
        rm.plot_ic_qq(ic)
