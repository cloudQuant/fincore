"""Optimization feasibility tests."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from fincore.optimization import efficient_frontier
from fincore.optimization._utils import OptimizationError, check_feasibility, make_positive_semidefinite


def test_check_feasibility_accepts_psd() -> None:
    cov = np.array([[1.0, 0.2], [0.2, 1.0]])
    check_feasibility(cov)


def test_check_feasibility_rejects_indefinite() -> None:
    cov = np.array([[1.0, 2.0], [2.0, 1.0]])
    with pytest.raises(OptimizationError, match="not positive semidefinite"):
        check_feasibility(cov)


def test_check_feasibility_rejects_non_symmetric() -> None:
    cov = np.array([[1.0, 0.9], [0.1, 1.0]])
    with pytest.raises(OptimizationError, match="not symmetric"):
        check_feasibility(cov)


def test_check_feasibility_rejects_nan() -> None:
    cov = np.array([[1.0, np.nan], [np.nan, 1.0]])
    with pytest.raises(OptimizationError, match="NaN"):
        check_feasibility(cov)


def test_make_psd_projects_indefinite() -> None:
    cov = np.array([[1.0, 2.0], [2.0, 1.0]])
    psd = make_positive_semidefinite(cov)
    check_feasibility(psd)


def test_make_psd_preserves_psd() -> None:
    cov = np.array([[1.0, 0.2], [0.2, 1.0]])
    psd = make_positive_semidefinite(cov)
    assert np.allclose(psd, cov, atol=1e-6)


def _returns() -> pd.DataFrame:
    rng = np.random.default_rng(20260821)
    return pd.DataFrame(rng.normal(0.0004, 0.01, size=(300, 3)), columns=["A", "B", "C"])


def test_frontier_fails_closed_for_infeasible_weight_bounds() -> None:
    with pytest.raises(OptimizationError, match="weight bounds are infeasible"):
        efficient_frontier(_returns(), max_weight=0.3)


def test_frontier_has_finite_weights_and_explicit_constraint_residuals() -> None:
    result = efficient_frontier(_returns(), n_points=9, max_weight=0.7)

    assert np.isfinite(result["frontier_weights"]).all()
    assert np.isfinite(result["frontier_returns"]).all()
    assert np.isfinite(result["frontier_volatilities"]).all()
    np.testing.assert_allclose(result["frontier_weights"].sum(axis=1), 1.0, atol=1e-8)
    for diagnostics in result["solver_diagnostics"]["frontier"]:
        assert diagnostics["sum_residual"] <= 1e-8
        assert diagnostics["target_return_residual"] <= 1e-8


def test_frontier_annualization_is_explicit_not_hard_coded() -> None:
    returns = _returns()
    monthly = efficient_frontier(returns, n_points=5, periods_per_year=12)
    daily = efficient_frontier(returns, n_points=5, periods_per_year=252)

    assert monthly["periods_per_year"] == 12.0
    assert daily["periods_per_year"] == 252.0
    np.testing.assert_allclose(monthly["frontier_returns"], daily["frontier_returns"] * (12.0 / 252.0), atol=1e-8)
    np.testing.assert_allclose(
        monthly["frontier_volatilities"],
        daily["frontier_volatilities"] * np.sqrt(12.0 / 252.0),
        atol=1e-8,
    )
