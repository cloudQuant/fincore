"""Utility functions for optimization module.

Provides common error handling and result validation for scipy.optimize
results across the optimization module.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from fincore.optimization.exceptions import OptimizationError

if TYPE_CHECKING:
    from numpy.typing import NDArray
    from scipy import optimize


def validate_result(
    res: optimize.OptimizeResult,
    context: str,
    allow_nan: bool = False,
) -> NDArray[np.float64]:
    """Validate scipy.optimize result and return weights.

    Parameters
    ----------
    res : OptimizeResult
        Result from scipy.optimize.minimize.
    context : str
        Description of optimization context (e.g., "max_sharpe", "risk_parity").
    allow_nan : bool, default False
        Whether to allow NaN/inf weights (useful for frontier computation).

    Returns
    -------
    NDArray[np.float64]
        Validated weight array.

    Raises
    ------
    OptimizationError
        If optimization failed or returned invalid weights.
    """
    if not res.success:
        msg = f"Optimization failed for {context}: status={res.status}, message={res.message!r}"
        raise OptimizationError(
            msg,
            status=res.status,
            solver_message=str(res.message),
        )

    weights: NDArray[np.float64] = res.x

    # Check for NaN/inf
    if not allow_nan and np.any(~np.isfinite(weights)):
        msg = f"Optimization for {context} returned invalid weights (NaN/inf detected): {weights}"
        raise OptimizationError(
            msg,
            status=res.status,
            solver_message=str(res.message),
        )

    return weights


def normalize_weights(
    weights: NDArray[np.float64],
    epsilon: float = 1e-12,
) -> NDArray[np.float64]:
    """Normalize weights to sum to 1, handling near-zero cases.

    Parameters
    ----------
    weights : NDArray[np.float64]
        Raw weight array.
    epsilon : float, default 1e-12
        Threshold for treating sum as zero.

    Returns
    -------
    NDArray[np.float64]
        Normalized weights summing to 1.

    Raises
    ------
    OptimizationError
        If weights sum is too small or negative.
    """
    total = float(weights.sum())
    if abs(total) < epsilon:
        raise OptimizationError(f"Cannot normalize weights: sum ({total}) is too close to zero")
    if total < 0:
        raise OptimizationError(f"Cannot normalize weights: sum ({total}) is negative")
    return weights / total


def check_feasibility(cov: NDArray[np.float64]) -> None:
    """Pre-flight feasibility check: covariance must be square, symmetric, finite PSD.

    An infeasible frontier must fail deterministically here, not silently fill
    NaN downstream.
    """
    matrix = np.asarray(cov, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise OptimizationError(f"covariance must be square, got shape {matrix.shape}")
    if np.any(~np.isfinite(matrix)):
        raise OptimizationError("covariance matrix contains NaN/inf")
    if not np.allclose(matrix, matrix.T, rtol=1e-10, atol=1e-12):
        raise OptimizationError("covariance matrix is not symmetric")
    if np.any(np.linalg.eigvalsh(matrix) < -1e-10):
        raise OptimizationError("covariance matrix is not positive semidefinite")


def make_positive_semidefinite(
    cov: NDArray[np.float64],
    epsilon: float = 1e-8,
) -> NDArray[np.float64]:
    """Project a symmetric matrix onto the PSD cone via an eigenvalue floor.

    Used as shrinkage/conditioning so a slightly indefinite sample covariance
    does not abort optimization.
    """
    matrix = np.asarray(cov, dtype=np.float64)
    if not np.allclose(matrix, matrix.T, rtol=1e-10, atol=1e-12):
        raise OptimizationError("covariance matrix is not symmetric")
    eigvals, eigvecs = np.linalg.eigh(matrix)
    eigvals = np.maximum(eigvals, epsilon)
    return np.asarray(eigvecs @ np.diag(eigvals) @ eigvecs.T, dtype=np.float64)
