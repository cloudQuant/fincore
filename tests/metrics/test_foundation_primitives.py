"""Contracts for canonical metrics-owned frequency and numeric primitives."""

from __future__ import annotations

import numpy as np
import pytest


def test_frequency_annualization_and_pandas_frequency_are_owned_by_metrics() -> None:
    from fincore.metrics.frequencies import annualization_factor, pandas_frequency

    assert annualization_factor("daily") == 252.0
    assert annualization_factor("monthly") == 12.0
    assert annualization_factor("daily", annualization=365) == 365.0
    assert pandas_frequency("daily") == "D"
    with pytest.raises(ValueError, match="unknown frequency"):
        annualization_factor("hourly")


def test_numeric_and_rolling_primitives_keep_nan_and_window_semantics_explicit() -> None:
    from fincore.metrics._numeric import nanmean
    from fincore.metrics._rolling import rolling_window

    assert nanmean(np.array([1.0, np.nan, 3.0])) == pytest.approx(2.0)
    windows = rolling_window(np.array([1.0, 2.0, 3.0]), 2)

    assert windows.tolist() == [[1.0, 2.0], [2.0, 3.0]]
    with pytest.raises(ValueError, match="positive"):
        rolling_window(np.array([1.0]), 0)
