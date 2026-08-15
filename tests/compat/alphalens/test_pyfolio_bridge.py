"""Real C4 bridge from strict Alphalens portfolio output into fincore Pyfolio."""

from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib
import matplotlib.pyplot as plt

if TYPE_CHECKING:
    import pandas as pd
    import pytest

matplotlib.use("Agg", force=True)


def test_factor_output_runs_real_fincore_pyfolio_workflow(
    clean_factor_data: pd.DataFrame,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Use the local Pyfolio workflow, not a fake or external pyfolio package."""

    import fincore.pyfolio as pyfolio
    from fincore.alphalens.performance import create_pyfolio_input
    from fincore.utils import common_utils

    returns, positions, benchmark = create_pyfolio_input(
        clean_factor_data,
        "1D",
        capital=1_000_000,
    )
    displayed: list[object] = []
    monkeypatch.setattr(common_utils, "display", lambda value: displayed.append(value))

    figure = pyfolio.create_returns_tear_sheet(
        returns,
        positions=positions,
        benchmark_rets=benchmark,
        run_flask_app=True,
    )

    assert isinstance(figure, matplotlib.figure.Figure)
    assert figure.axes
    assert displayed
    plt.close(figure)

    shown: list[bool] = []
    monkeypatch.setattr("matplotlib.pyplot.show", lambda: shown.append(True))
    result = pyfolio.create_returns_tear_sheet(
        returns,
        positions=positions,
        benchmark_rets=benchmark,
    )
    assert result is None
    assert shown == []
    plt.close("all")
