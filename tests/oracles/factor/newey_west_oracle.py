"""Independent statsmodels oracle for Fama-MacBeth Newey-West means.

Fama-MacBeth produces one intercept and one exposure coefficient per fitted
cross-section.  HAC inference is therefore an intercept-only regression over
each chronological coefficient sequence.  This oracle deliberately imports
statsmodels and never imports ``fincore``.
"""

from __future__ import annotations

import numpy as np

__all__ = ["newey_west_mean_reference"]


def newey_west_mean_reference(values: np.ndarray, nlags: int) -> tuple[float, float]:
    """Return ``(mean, HAC standard error)`` for a one-dimensional sequence."""

    import statsmodels.api as sm

    observations = np.asarray(values, dtype=float)
    result = sm.OLS(observations, np.ones((len(observations), 1))).fit(
        cov_type="HAC",
        cov_kwds={"maxlags": int(nlags)},
    )
    return float(result.params[0]), float(result.bse[0])
