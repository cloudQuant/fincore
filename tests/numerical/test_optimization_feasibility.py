"""Optimization feasibility tests."""

from __future__ import annotations

import numpy as np
import pytest

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
