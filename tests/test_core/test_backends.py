"""Numerical backend tests."""

from __future__ import annotations

import numpy as np
import pytest

from fincore.backends import NumPyBackend, get_backend
from fincore.backends.numpy_backend import NumPyBackend as ModuleNumPyBackend
from fincore.exceptions import NumericalError
from fincore.metrics.drawdown import max_drawdown


def test_numpy_backend_cum_returns_matches_reference() -> None:
    returns = np.array([0.01, -0.02, 0.03])
    backend = NumPyBackend()
    expected = np.cumprod(1.0 + returns) - 1.0
    np.testing.assert_allclose(backend.cum_returns(returns), expected)


@pytest.mark.parametrize(
    "returns",
    [
        pytest.param(np.array([-0.10, 0.05, -0.02]), id="first-period-loss"),
        pytest.param(np.array([0.01, 0.02, 0.03]), id="all-positive"),
        pytest.param(np.array([0.10, -1.0, 0.50]), id="ruin"),
        pytest.param(np.array([0.10, -1.50, 0.20]), id="below-wealth-floor"),
    ],
)
def test_numpy_backend_max_drawdown_matches_canonical_for_finite_returns(returns: np.ndarray) -> None:
    backend = NumPyBackend()
    expected = max_drawdown(returns)

    actual = backend.max_drawdown(returns)

    assert actual == pytest.approx(expected)


def test_numpy_backend_max_drawdown_matches_canonical_nan_rejection() -> None:
    backend = NumPyBackend()
    returns = np.array([0.05, np.nan, -0.10, 0.02])

    with pytest.raises(NumericalError) as canonical_error:
        max_drawdown(returns)

    with pytest.raises(NumericalError) as backend_error:
        backend.max_drawdown(returns)

    assert str(backend_error.value) == str(canonical_error.value)


def test_numpy_backend_max_drawdown_matches_canonical_for_multiple_assets() -> None:
    backend = NumPyBackend()
    returns = np.array(
        [
            [-0.10, 0.02],
            [0.05, -1.00],
            [-0.02, 0.50],
        ]
    )
    expected = max_drawdown(returns)

    actual = backend.max_drawdown(returns)

    assert isinstance(actual, np.ndarray)
    np.testing.assert_allclose(actual, expected)


def test_numpy_backend_sharpe_matches_formula() -> None:
    backend = NumPyBackend()
    returns = np.array([0.001, -0.002, 0.003, 0.001])
    std = np.std(returns, ddof=1)
    expected = np.mean(returns) / std * np.sqrt(252)
    assert np.isclose(backend.sharpe_ratio(returns), expected, rtol=1e-9)


def test_get_backend_falls_back_to_reference() -> None:
    assert get_backend("numpy").name == "numpy"
    assert get_backend("unknown").name == "numpy"


def test_public_numpy_backend_is_the_explicit_reference_implementation() -> None:
    assert NumPyBackend is ModuleNumPyBackend
