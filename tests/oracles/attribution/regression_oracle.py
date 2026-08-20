"""Independent regression oracles via statsmodels.

``statsmodels`` is the accepted statistical oracle for OLS/WLS with
Newey-West (HAC) sandwich covariance.  These references never import
``fincore`` and compute the true per-coefficient HAC standard errors that a
scalar "residual-autocorrelation adjustment factor" cannot reproduce.

Formulas
--------
Sandwich covariance: ``(X'X)^{-1} S (X'X)^{-1}`` where ``S`` is the
Newey-West weighted autocovariance matrix with Bartlett kernel weights
``w_j = 1 - j/(L+1)``.
"""

from __future__ import annotations

import numpy as np

__all__ = ["ols_hac_reference", "ols_reference", "wls_reference"]


def _statsmodels():
    import statsmodels.api as sm

    return sm


def ols_reference(y: np.ndarray, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Plain OLS coefficients and classical standard errors (statsmodels)."""
    sm = _statsmodels()
    model = sm.OLS(y, X).fit()
    return np.asarray(model.params), np.asarray(model.bse)


def ols_hac_reference(y: np.ndarray, X: np.ndarray, nlags: int = 1) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """OLS with Newey-West HAC standard errors.

    Returns ``(params, hac_bse, pvalues)``.
    """
    sm = _statsmodels()
    model = sm.OLS(y, X).fit(cov_type="HAC", cov_kwds={"maxlags": nlags})
    return np.asarray(model.params), np.asarray(model.bse), np.asarray(model.pvalues)


def wls_reference(y: np.ndarray, X: np.ndarray, weights: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Weighted least squares coefficients and standard errors (statsmodels)."""
    sm = _statsmodels()
    model = sm.WLS(y, X, weights=weights).fit()
    return np.asarray(model.params), np.asarray(model.bse)
