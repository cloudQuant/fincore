"""Behavior probes for the strict compatibility façades (C0–C4).

Each probe records the observable behavior of a frozen callable — success
shape/dtype, exception type on invalid input, and NaN handling — so a change
to the strict façade's behavior is caught even when the static snapshot is
unchanged.  Probes use minimal fixtures and never touch the enhanced profiles.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from fincore import empyrical


def _returns(values: list[float]) -> pd.Series:
    return pd.Series(np.asarray(values, dtype=float))


def test_empyrical_sharpe_ratio_success_shape() -> None:
    result = empyrical.sharpe_ratio(_returns([0.01, -0.005, 0.002, 0.004]))
    assert isinstance(result, float)
    assert np.isfinite(result)


def test_empyrical_sharpe_ratio_empty_is_nan() -> None:
    result = empyrical.sharpe_ratio(pd.Series([], dtype=float))
    assert np.isnan(result)


def test_empyrical_max_drawdown_success_shape() -> None:
    result = empyrical.max_drawdown(_returns([0.01, -0.02, 0.03, -0.01]))
    assert isinstance(result, float)
    assert result <= 0.0


def test_empyrical_cum_returns_shape() -> None:
    result = empyrical.cum_returns(_returns([0.01, -0.005, 0.002]))
    assert isinstance(result, pd.Series)
    assert len(result) == 3


def test_empyrical_annual_volatility_scalar() -> None:
    result = empyrical.annual_volatility(_returns([0.01, -0.005, 0.002, 0.004]))
    assert isinstance(result, float)
    assert result >= 0.0


def test_empyrical_calmar_ratio_success() -> None:
    result = empyrical.calmar_ratio(_returns([0.01, 0.02, -0.01, 0.03]))
    assert isinstance(result, float)


def test_empyrical_public_symbols_present() -> None:
    expected = {"sharpe_ratio", "max_drawdown", "cum_returns", "annual_return", "annual_volatility"}
    assert expected.issubset(set(empyrical.__all__))


def test_empyrical_does_not_expose_enhanced_stateful_class() -> None:
    assert not hasattr(empyrical, "AnalysisContext")
    assert not hasattr(empyrical, "RollingEngine")
