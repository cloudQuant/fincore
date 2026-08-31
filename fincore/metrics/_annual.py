"""Foundational annual-return kernel shared by metric domains."""

from __future__ import annotations

from typing import cast

import numpy as np
import pandas as pd

from fincore.metrics.basic import annualization_factor
from fincore.metrics.frequencies import DAILY
from fincore.metrics.returns import cum_returns_final


def annual_return(
    returns: pd.Series | pd.DataFrame | np.ndarray,
    period: str = DAILY,
    annualization: float | None = None,
) -> float | np.ndarray | pd.Series:
    """Return compound annual growth for non-cumulative period returns."""
    normalized_returns = np.asarray(returns) if isinstance(returns, list) else returns
    if len(normalized_returns) < 1:
        return np.nan

    ann_factor = annualization_factor(period, annualization)
    num_years = len(normalized_returns) / ann_factor
    ending_value = cum_returns_final(normalized_returns, starting_value=1)
    if isinstance(ending_value, (pd.Series, np.ndarray)):
        result = np.asarray(ending_value, dtype=float).copy()
        mask = result <= 0
        result[mask] = -1.0
        result[~mask] = result[~mask] ** (1 / num_years) - 1
        if isinstance(ending_value, pd.Series):
            return pd.Series(result, index=ending_value.index)
        return result
    if ending_value <= 0:
        return -1.0
    return cast("float", ending_value ** (1 / num_years)) - 1
