"""Branch-completion tests for factor_analysis.tears helpers and workflows."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pandas as pd
import pytest

from fincore.factor_analysis.analysis import analyze_factor
from fincore.factor_analysis.tears import (
    GridFigure,
    _event_data,
    _group_mean_returns,
    _overall_information,
    _overall_quantile_frame,
    _quantile_slice,
    _rate_of_return,
    _std_conversion,
    close_owned_figures,
    create_event_returns_tear_sheet,
    create_event_study_tear_sheet,
    create_information_tear_sheet,
    create_returns_tear_sheet,
    create_summary_tear_sheet,
    create_turnover_tear_sheet,
)


def _factor_data(with_group: bool = False) -> pd.DataFrame:
    dates = pd.bdate_range("2024-01-02", periods=30)
    assets = ["A", "B", "C", "D"]
    index = pd.MultiIndex.from_product((dates, assets), names=("date", "asset"))
    data: dict[str, object] = {
        "factor": np.random.default_rng(1).normal(0, 1, len(index)),
        "factor_quantile": [i % 4 + 1 for i in range(len(index))],
        "1D": np.random.default_rng(2).normal(0, 0.01, len(index)),
        "5D": np.random.default_rng(3).normal(0, 0.02, len(index)),
    }
    if with_group:
        data["group"] = ["g1", "g2"] * (len(index) // 2)
    return pd.DataFrame(data, index=index)


# ---------------------------------------------------------------------------
# GridFigure
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("rows", [0, -1, True])
def test_grid_figure_rejects_invalid_rows(rows: object) -> None:
    with pytest.raises(ValueError, match="rows"):
        GridFigure(rows, 2)  # type: ignore[arg-type]


@pytest.mark.parametrize("cols", [0, -1, False])
def test_grid_figure_rejects_invalid_cols(cols: object) -> None:
    with pytest.raises(ValueError, match="cols"):
        GridFigure(2, cols)  # type: ignore[arg-type]


def test_grid_figure_closed_raises_on_create_new_figure() -> None:
    grid = GridFigure(2, 1)
    grid.close()
    with pytest.raises(RuntimeError, match="closed"):
        grid.create_new_figure()
    with pytest.raises(RuntimeError, match="closed"):
        grid.next_row()
    with pytest.raises(RuntimeError, match="closed"):
        grid.next_cell()


def test_grid_figure_close_is_idempotent() -> None:
    grid = GridFigure(2, 1)
    grid.close()
    grid.close()  # no-op, no error


# ---------------------------------------------------------------------------
# helper functions
# ---------------------------------------------------------------------------


def test_overall_quantile_frame_collapses_group() -> None:
    idx = pd.MultiIndex.from_tuples([(1, "g1"), (1, "g2"), (2, "g1")], names=("factor_quantile", "group"))
    frame = pd.DataFrame({"p": [1.0, 2.0, 3.0]}, index=idx)
    result = _overall_quantile_frame(frame)
    assert "group" not in result.index.names


def test_overall_information_collapses_group() -> None:
    idx = pd.MultiIndex.from_tuples(
        [(pd.Timestamp("2024-01-01"), "g1"), (pd.Timestamp("2024-01-01"), "g2")],
        names=("date", "group"),
    )
    frame = pd.DataFrame({"p": [1.0, 2.0]}, index=idx)
    result = _overall_information(frame)
    assert "group" not in result.index.names


def test_rate_of_return_empty() -> None:
    empty = pd.DataFrame()
    result = _rate_of_return(empty)
    assert result.empty


def test_std_conversion_empty() -> None:
    empty = pd.DataFrame()
    result = _std_conversion(empty)
    assert result.empty


def test_quantile_slice_non_multiindex() -> None:
    frame = pd.DataFrame({"p": [1.0, 2.0]}, index=pd.Index([1, 2], name="factor_quantile"))
    result = _quantile_slice(frame, 1)
    assert result.shape == (1, 1)


# ---------------------------------------------------------------------------
# model-dependent helpers
# ---------------------------------------------------------------------------


def test_event_data_raises_without_event() -> None:
    model = analyze_factor(_factor_data(), periods=("1D",))
    with pytest.raises(ValueError, match="event"):
        _event_data(model)


def test_group_mean_returns_none_without_groups() -> None:
    model = analyze_factor(_factor_data(), periods=("1D",))
    assert _group_mean_returns(model) is None


# ---------------------------------------------------------------------------
# tear sheet workflows (legacy projection and empty/group branches)
# ---------------------------------------------------------------------------


def test_summary_tear_sheet_legacy_projection() -> None:
    model = analyze_factor(_factor_data(), periods=("1D",))
    artifacts = create_summary_tear_sheet(model, legacy_projection=True)
    assert artifacts is not None
    close_owned_figures(artifacts)


def test_returns_tear_sheet_legacy_group() -> None:
    model = analyze_factor(_factor_data(with_group=True), periods=("1D",), by_group=True)
    artifacts = create_returns_tear_sheet(model, by_group=True, legacy_projection=True)
    assert artifacts is not None
    close_owned_figures(artifacts)


def test_information_tear_sheet_legacy_projection() -> None:
    model = analyze_factor(_factor_data(), periods=("1D",))
    artifacts = create_information_tear_sheet(model, legacy_projection=True)
    assert artifacts is not None
    close_owned_figures(artifacts)


def test_turnover_tear_sheet() -> None:
    model = analyze_factor(_factor_data(), periods=("1D",))
    artifacts = create_turnover_tear_sheet(model)
    assert artifacts is not None
    close_owned_figures(artifacts)


def test_event_returns_tear_sheet_requires_event() -> None:
    model = analyze_factor(_factor_data(), periods=("1D",))
    with pytest.raises(ValueError, match="event"):
        create_event_returns_tear_sheet(model)


def test_event_returns_tear_sheet_with_event() -> None:
    dates = pd.bdate_range("2024-01-02", periods=30)
    assets = ["A", "B", "C", "D"]
    event_returns = pd.DataFrame(
        {asset: np.random.default_rng(20).normal(0, 0.01, len(dates)) for asset in assets},
        index=dates,
    )
    model = analyze_factor(
        _factor_data(),
        periods=("1D",),
        event_returns=event_returns,
        event_before=1,
        event_after=2,
    )
    artifacts = create_event_returns_tear_sheet(model)
    assert artifacts is not None
    close_owned_figures(artifacts)


def test_event_study_tear_sheet_without_event() -> None:
    model = analyze_factor(_factor_data(), periods=("1D",))
    artifacts = create_event_study_tear_sheet(model)
    assert artifacts is not None
    close_owned_figures(artifacts)


def test_overall_quantile_frame_non_multiindex() -> None:
    frame = pd.DataFrame({"p": [1.0, 2.0]}, index=pd.Index([1, 2], name="factor_quantile"))
    result = _overall_quantile_frame(frame)
    assert list(result.columns) == ["p"]


def test_overall_information_non_multiindex() -> None:
    frame = pd.DataFrame({"p": [1.0, 2.0]}, index=pd.Index([1, 2], name="date"))
    result = _overall_information(frame)
    assert list(result.columns) == ["p"]


def test_overall_monthly_without_month_key() -> None:
    from fincore.factor_analysis.tears import _aggregate_monthly, _overall_monthly

    model = analyze_factor(_factor_data(), periods=("1D",), time_aggregation=("W",))
    assert _overall_monthly(model) is not None
    assert _aggregate_monthly(model) is not None


def test_returns_group_section_empty_model() -> None:
    from fincore.factor_analysis.tears import _returns_group_section

    model = analyze_factor(_factor_data(), periods=("1D",))
    artifacts = _returns_group_section(model, legacy_projection=False, plotter=None)
    assert artifacts.figures == ()
