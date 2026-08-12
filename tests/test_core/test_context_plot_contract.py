from __future__ import annotations

import pandas as pd

from fincore.core.context import AnalysisContext


def test_plot_returns_artifacts_not_backend() -> None:
    returns = pd.Series([0.01, -0.02, 0.03], index=pd.date_range("2024-01-01", periods=3))

    result = AnalysisContext(returns).plot(backend="matplotlib")

    assert result.backend == "matplotlib"
    assert len(result.figures) == 2
    assert all(getattr(item, "figure", None) is not None for item in result.figures)


def test_report_artifacts_close_closes_matplotlib_figures() -> None:
    import matplotlib.pyplot as plt

    returns = pd.Series([0.01, -0.02, 0.03], index=pd.date_range("2024-01-01", periods=3))
    result = AnalysisContext(returns).plot(backend="matplotlib")
    numbers = [item.figure.number for item in result.figures]

    result.close()

    assert all(not plt.fignum_exists(number) for number in numbers)
