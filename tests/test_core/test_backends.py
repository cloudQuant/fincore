"""Numerical backend tests."""

from __future__ import annotations

import numpy as np

from fincore.backends import NumPyBackend, get_backend


def test_numpy_backend_cum_returns_matches_reference() -> None:
    returns = np.array([0.01, -0.02, 0.03])
    backend = NumPyBackend()
    expected = np.cumprod(1.0 + returns) - 1.0
    np.testing.assert_allclose(backend.cum_returns(returns), expected)


def test_numpy_backend_max_drawdown_is_non_positive() -> None:
    backend = NumPyBackend()
    returns = np.array([0.01, -0.05, 0.02, -0.03])
    dd = backend.max_drawdown(returns)
    assert dd <= 0.0


def test_numpy_backend_sharpe_matches_formula() -> None:
    backend = NumPyBackend()
    returns = np.array([0.001, -0.002, 0.003, 0.001])
    std = np.std(returns, ddof=1)
    expected = np.mean(returns) / std * np.sqrt(252)
    assert np.isclose(backend.sharpe_ratio(returns), expected, rtol=1e-9)


def test_get_backend_falls_back_to_reference() -> None:
    assert get_backend("numpy").name == "numpy"
    assert get_backend("unknown").name == "numpy"
