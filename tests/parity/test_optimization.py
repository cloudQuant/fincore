"""Canonical optimization scenarios for source and installed-wheel evidence."""

from __future__ import annotations

import numpy as np
import pandas as pd

from fincore.optimization.exceptions import OptimizationError
from fincore.optimization.objectives import optimize
from fincore.optimization.risk_parity import risk_parity


def _returns() -> pd.DataFrame:
    rng = np.random.default_rng(20260831)
    return pd.DataFrame(
        rng.multivariate_normal(
            mean=np.array([0.0005, 0.0007, 0.0003]),
            cov=np.array(
                [
                    [0.00010, 0.00003, 0.00002],
                    [0.00003, 0.00016, 0.00002],
                    [0.00002, 0.00002, 0.00008],
                ]
            ),
            size=300,
        ),
        columns=["alpha", "beta", "gamma"],
    )


def test_optimization_error() -> None:
    """The domain error preserves structured solver diagnostics."""
    error = OptimizationError("solver did not converge", status=9, solver_message="iteration limit")

    assert str(error) == "solver did not converge"
    assert error.status == 9
    assert error.solver_message == "iteration limit"


def test_optimize() -> None:
    """A constrained optimization keeps the allocation and result contract coherent."""
    result = optimize(_returns(), objective="max_sharpe")

    assert result["objective"] == "max_sharpe"
    assert result["asset_names"] == ["alpha", "beta", "gamma"]
    assert np.isclose(result["weights"].sum(), 1.0)
    assert np.all(result["weights"] >= 0.0)
    assert np.isfinite(result["volatility"])
    assert np.isfinite(result["sharpe"])


def test_risk_parity() -> None:
    """Equal-budget risk parity returns a normalized positive allocation."""
    result = risk_parity(_returns())
    contribution_share = result["risk_contributions"] / result["risk_contributions"].sum()

    assert result["asset_names"] == ["alpha", "beta", "gamma"]
    assert np.isclose(result["weights"].sum(), 1.0)
    assert np.all(result["weights"] > 0.0)
    np.testing.assert_allclose(contribution_share, np.full(3, 1.0 / 3.0), atol=0.05)
