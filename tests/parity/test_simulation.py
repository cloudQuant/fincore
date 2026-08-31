"""Canonical Monte Carlo scenario for source and installed-wheel evidence."""

from __future__ import annotations

import numpy as np

from fincore.simulation import MonteCarlo


def test_monte_carlo() -> None:
    """A fixed seed makes the domain simulation reproducible and finite."""
    historical_returns = np.linspace(-0.012, 0.018, 60)
    model = MonteCarlo(historical_returns)

    first = model.simulate(n_paths=16, horizon=5, seed=17)
    second = model.simulate(n_paths=16, horizon=5, seed=17)

    assert first.paths.shape == (16, 5)
    assert np.isfinite(first.paths).all()
    np.testing.assert_array_equal(first.paths, second.paths)
