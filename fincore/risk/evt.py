"""Extreme Value Theory (EVT) models for tail risk estimation.

EVT provides better estimates of extreme losses than normal
distribution assumptions, particularly for:

- Tail index estimation (Hill estimator)
- Peaks-over-threshold (POT) with Generalized Pareto Distribution (GPD)
- Block maxima with Generalized Extreme Value (GEV) distribution

References
----------
Embrechts, P., Klüppelberg, C., & Mikosch, T. (1997).
Modelling Extremal Events for Insurance and Finance.
McNeil, A. J., Frey, R., & Embrechts, P. (2015).
Quantitative Risk Management.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Union

import numpy as np
import pandas as pd
from scipy import optimize, stats

__all__ = ["evt_cvar", "evt_var", "extreme_risk", "gev_fit", "gpd_fit", "hill_estimator"]


if TYPE_CHECKING:
    ArrayLike = Union[np.ndarray, pd.Series, pd.DataFrame]


def hill_estimator(
    data: ArrayLike,
    threshold: float | None = None,
    tail: str = "upper",
) -> tuple[float, np.ndarray]:
    """Estimate tail index using Hill estimator.

    The Hill estimator is a popular method for estimating the
    tail index (extreme value index) of heavy-tailed distributions.

    Parameters
    ----------
    data : array-like
        Input data (returns or losses).
    threshold : float, optional
        Positive tail-magnitude threshold. If None, uses the 90th percentile
        of the selected positive tail magnitudes.
    tail : str, default 'upper'
        Which tail to estimate: 'upper' (right/gains) or 'lower' (left/losses).

    Returns
    -------
    xi : float
        Estimated tail index (shape parameter).
        The Hill estimator is defined for regularly varying heavy tails, so
        finite-sample estimates are non-negative. Values close to zero can
        indicate a light/near-exponential tail; use a GPD/GEV model rather
        than Hill for a bounded-tail (negative-shape) conclusion.
    tail_observations : ndarray
        Positive tail magnitudes strictly above ``threshold``. For a lower
        return tail these are reflected loss magnitudes.

    Examples
    --------
    >>> returns = np.random.standard_t(3, 10000)
    >>> xi, tail_observations = hill_estimator(returns, tail="lower")
    >>> print(f"Tail index: {xi:.3f}")
    """
    data_arr: np.ndarray = np.asarray(data, dtype=float).flatten()
    data_arr = data_arr[~np.isnan(data_arr)]

    if tail == "upper":
        tail_data = data_arr[data_arr > 0.0]
    elif tail == "lower":
        tail_data = -data_arr[data_arr < 0.0]
    else:
        raise ValueError("tail must be 'upper' or 'lower'")

    if threshold is None:
        threshold = float(np.percentile(tail_data, 90))
    else:
        threshold = float(threshold)
        if not np.isfinite(threshold) or threshold <= 0.0:
            raise ValueError("threshold must be finite and positive")

    tail_observations = tail_data[tail_data > threshold]

    if len(tail_observations) < 10:
        raise ValueError("Not enough exceedances for Hill estimation (need >= 10)")

    # Threshold form of Hill's tail-index estimator.  The ratio uses observed
    # tail magnitudes, not excesses (x - u): E[log(X / u) | X > u].
    xi = float(np.mean(np.log(tail_observations / threshold)))

    return xi, tail_observations


def gpd_fit(
    data: ArrayLike,
    threshold: float | None = None,
    method: str = "mle",
    tail: str = "lower",
) -> dict[str, float]:
    """Fit Generalized Pareto Distribution (GPD) to exceedances.

    GPD is used in Peaks-Over-Threshold (POT) method for modeling
    tail exceedances above a threshold.

    Parameters
    ----------
    data : array-like
        Input data (returns or losses).
    threshold : float, optional
        Threshold for POT. If None, uses 90th percentile.
    method : str, default 'mle'
        Estimation method: 'mle' (maximum likelihood) or 'pwm' (probability weighted moments).

    Returns
    -------
    dict
        Fitted parameters:
        - 'xi' (shape): Tail index
        - 'beta' (scale): Scale parameter
        - 'threshold': Fitted threshold
        - 'n_exceed': Number of exceedances

    Examples
    --------
    >>> returns = np.random.standard_t(4, 10000)
    >>> params = gpd_fit(returns, tail="lower")
    >>> print(f"xi={params['xi']:.3f}, beta={params['beta']:.3f}")
    """
    data_arr: np.ndarray = np.asarray(data, dtype=float).flatten()
    data_arr = data_arr[~np.isnan(data_arr)]

    if tail == "lower":
        # Losses: positive tail of negated negative returns.
        tail_data = -data_arr[data_arr < 0]
        if len(tail_data) == 0:
            raise ValueError("No negative returns in data; need at least some losses for GPD fitting")
    elif tail == "upper":
        # Gains: positive tail of positive returns.
        tail_data = data_arr[data_arr > 0]
        if len(tail_data) == 0:
            raise ValueError("No positive returns in data; need at least some gains for GPD fitting")
    else:
        raise ValueError("tail must be 'lower' or 'upper'")

    # Set threshold
    if threshold is None:
        threshold = float(np.percentile(tail_data, 90))

    # Get exceedances
    excesses = tail_data[tail_data > threshold] - threshold

    if len(excesses) < 10:
        raise ValueError("Not enough exceedances for GPD fitting (need >= 10)")

    if method == "mle":
        # Maximum likelihood estimation
        def neg_loglik(params):
            xi, beta = params
            beta = np.abs(beta)

            # Avoid invalid parameter combinations
            if beta <= 0:
                return 1e10  # pragma: no cover -- Edge case for optimization

            z = 1 + xi * excesses / beta

            if np.any(z <= 0):
                return 1e10

            # Log-likelihood for GPD
            if np.abs(xi) < 1e-10:
                # Exponential case (xi -> 0)
                ll = np.sum(np.log(beta) + excesses / beta)  # pragma: no cover -- Rare edge case
            else:
                ll = np.sum(np.log(beta) + (1 + 1 / xi) * np.log(z))

            return ll

        # Optimize
        result = optimize.minimize(
            neg_loglik,
            x0=[0.1, np.std(excesses)],
            bounds=[(-0.5, 1.0), (1e-6, None)],
            method="L-BFGS-B",
        )

        xi, beta = result.x
        beta = np.abs(beta)

    elif method == "pwm":
        # Probability weighted moments
        n = len(excesses)
        excesses_sorted = np.sort(excesses)

        # L-moments estimators
        m1 = np.mean(excesses)
        m2 = 2 / n * np.sum([(i + 1) * excesses_sorted[i] / n for i in range(n)])

        # PWM estimators
        xi = (m1 / (m1 - 2 * m2) - 2) / 3
        beta = (2 * m1 * m2) / (m1 - 2 * m2)

    else:
        raise ValueError(f"Unknown method: {method}")

    return {
        "xi": xi,
        "beta": beta,
        "threshold": threshold,
        "n_exceed": len(excesses),
    }


def gev_fit(
    data: ArrayLike,
    block_size: int | None = None,
    tail: str = "lower",
) -> dict[str, float]:
    """Fit Generalized Extreme Value (GEV) distribution to block maxima.

    GEV is used for modeling maximum values over fixed time blocks
    (e.g., monthly maximum losses).

    Parameters
    ----------
    data : array-like
        Input data (returns or losses).
    block_size : int, optional
        Size of each block for extracting maxima.
        If None, uses sqrt(n) blocks.

    Returns
    -------
    dict
        Fitted parameters:
        - 'xi' (shape): Tail index
        - 'mu' (location): Location parameter
        - 'sigma' (scale): Scale parameter

    Examples
    --------
    >>> returns = np.random.standard_t(4, 10000)
    >>> params = gev_fit(returns, block_size=252)  # Annual maxima
    >>> print(f"xi={params['xi']:.3f}")
    """
    data_arr: np.ndarray = np.asarray(data, dtype=float).flatten()
    data_arr = data_arr[~np.isnan(data_arr)]

    n = len(data_arr)

    if block_size is None:
        block_size = int(np.sqrt(n))

    # Split into blocks and get maxima
    n_blocks = n // block_size
    trimmed_data = data_arr[: n_blocks * block_size]
    block_maxima = trimmed_data.reshape(-1, block_size)

    if tail == "lower":
        block_extreme = np.min(block_maxima, axis=1)
        neg_extreme = -block_extreme  # positive loss magnitude
    elif tail == "upper":
        block_extreme = np.max(block_maxima, axis=1)
        neg_extreme = block_extreme  # positive gain magnitude
    else:
        raise ValueError("tail must be 'lower' or 'upper'")

    # Use scipy's genextreme fit on negated minima (positive loss space).
    # scipy's shape parameter ``c`` has the opposite sign of the standard GEV
    # shape ``xi`` (c > 0 is bounded Weibull, xi = -c > 0 is heavy Fréchet).
    c, loc, scale = stats.genextreme.fit(neg_extreme)

    # Return standard GEV parameters in loss space (positive).
    return {
        "xi": -c,
        "mu": loc,
        "sigma": scale,
        "n_blocks": n_blocks,
    }


def evt_var(
    data: ArrayLike,
    alpha: float = 0.05,
    model: str = "gpd",
    tail: str = "lower",
    threshold: float | None = None,
    block_size: int | None = None,
) -> float:
    """Calculate VaR using Extreme Value Theory.

    EVT-based VaR provides better tail risk estimates than
    normal distribution assumptions.

    Parameters
    ----------
    data : array-like
        Input return data.
    alpha : float, default 0.05
        Significance level (e.g., 0.05 for 95% VaR).
    model : str, default 'gpd'
        EVT model: 'gpd' (POT) or 'gev' (block maxima).
    tail : str, default 'lower'
        Tail to estimate: 'lower' for losses, 'upper' for gains.
    threshold : float, optional
        Threshold for GPD fitting.
    block_size : int, optional
        Block size for GEV fitting.

    Returns
    -------
    float
        EVT-based VaR estimate (negative value for losses).

    Examples
    --------
    >>> returns = np.random.standard_t(4, 1000)
    >>> var_95 = evt_var(returns, alpha=0.05, model="gpd")
    >>> print(f"95% EVT-VaR: {var_95:.2%}")
    """
    data = np.asarray(data).flatten()
    data = data[~np.isnan(data)]

    var: float = np.nan  # Initialize var

    if model == "gpd":
        params = gpd_fit(data, threshold=threshold, tail=tail)

        xi = params["xi"]
        beta = params["beta"]
        u = params["threshold"]

        # Number of exceedances
        n_exceed = params["n_exceed"]
        n_total = len(data)
        exceed_prob = n_exceed / n_total

        # GPD-based VaR (in tail magnitude space, positive)
        ratio = alpha / exceed_prob
        var_mag = u - beta * np.log(ratio) if np.abs(xi) < 1e-10 else u + (beta / xi) * (ratio ** (-xi) - 1)

        # Convert to return-space: negative for lower tail, positive for upper.
        var = -var_mag if tail == "lower" else var_mag

    elif model == "gev":
        # Fit GEV
        params = gev_fit(data, block_size=block_size, tail=tail)

        xi = params["xi"]
        mu = params["mu"]
        sigma = params["sigma"]

        # GEV quantile of the (1 - alpha) upper quantile in tail-magnitude space.
        q = 1.0 - alpha
        if np.abs(xi) < 1e-10:
            var_mag = mu - sigma * np.log(-np.log(q))
        else:
            var_mag = mu + (sigma / xi) * ((-np.log(q)) ** (-xi) - 1)

        var = -var_mag if tail == "lower" else var_mag
    else:
        raise ValueError(f"Unknown model: {model}")  # pragma: no cover -- Invalid input

    return var


def evt_cvar(
    data: ArrayLike,
    alpha: float = 0.05,
    model: str = "gpd",
    tail: str = "lower",
    threshold: float | None = None,
    block_size: int | None = None,
) -> float:
    """Calculate CVaR (Expected Shortfall) using EVT.

    EVT-based CVaR provides better average tail loss estimates.

    Parameters
    ----------
    data : array-like
        Input return data.
    alpha : float, default 0.05
        Significance level.
    model : str, default 'gpd'
        EVT model: 'gpd' or 'gev'.
    tail : str, default 'lower'
        Tail to estimate.
    threshold : float, optional
        Threshold for GPD fitting.
    block_size : int, optional
        Block size for GEV fitting.

    Returns
    -------
    float
        EVT-based CVaR estimate (negative value for losses).

    Examples
    --------
    >>> returns = np.random.standard_t(4, 1000)
    >>> cvar_95 = evt_cvar(returns, alpha=0.05, model="gpd")
    >>> print(f"95% EVT-CVaR: {cvar_95:.2%}")
    """
    data = np.asarray(data).flatten()
    data = data[~np.isnan(data)]

    var = evt_var(data, alpha, model, tail, threshold=threshold, block_size=block_size)

    if model == "gpd":
        params = gpd_fit(data, threshold=threshold, tail=tail)
        xi = params["xi"]
        beta = params["beta"]
        u = params["threshold"]

        var_mag = -var if tail == "lower" else var

        if np.abs(xi) < 1e-10:
            cvar_mag = var_mag + beta
        elif xi < 1:
            cvar_mag = var_mag + (beta + xi * (var_mag - u)) / (1 - xi)
        else:
            raise ValueError("CVaR infinite for xi >= 1")

        cvar = -cvar_mag if tail == "lower" else cvar_mag

    elif model == "gev":
        params = gev_fit(data, block_size=block_size, tail=tail)
        xi = params["xi"]
        sigma = params["sigma"]

        var_mag = -var if tail == "lower" else var
        t = -np.log(1.0 - alpha)

        if np.abs(xi) < 1e-10:
            cvar_mag = var_mag + sigma * (1 + np.euler_gamma)
        elif xi < 1:
            cvar_mag = var_mag + (sigma / xi) * (1 - 1 / (1 - xi) + t ** (-xi) / (1 - xi))
        else:
            raise ValueError("CVaR infinite for xi >= 1")

        cvar = -cvar_mag if tail == "lower" else cvar_mag
    else:
        raise ValueError(f"Unknown model: {model}")  # pragma: no cover -- Invalid input

    return cvar


def extreme_risk(
    returns: pd.Series,
    alpha: float = 0.05,
    tail: str = "lower",
    model: str = "gpd",
    threshold: float | None = None,
    block_size: int | None = None,
) -> pd.DataFrame:
    """Calculate comprehensive EVT-based risk measures.

    Parameters
    ----------
    returns : pd.Series
        Return series.
    alpha : float, default 0.05
        Significance level.
    tail : str, default 'lower'
        Tail to estimate.
    model : str, default 'gpd'
        EVT model.
    threshold : float, optional
        Threshold for GPD fitting.
    block_size : int, optional
        Block size for GEV fitting.

    Returns
    -------
    pd.DataFrame
        Risk measures including VaR, CVaR, tail index,
        and threshold.

    Examples
    --------
    >>> returns = pd.Series(np.random.standard_t(4, 1000))
    >>> risk = extreme_risk(returns, alpha=0.05)
    >>> print(risk)
    """
    data = returns.to_numpy(dtype=float)

    # Fit model
    if model == "gpd":
        params = gpd_fit(data, threshold=threshold, tail=tail)

        var = evt_var(data, alpha, model, tail, threshold=params["threshold"])
        cvar = evt_cvar(data, alpha, model, tail, threshold=params["threshold"])

        return pd.DataFrame(
            {
                "VaR": [var],
                "CVaR": [cvar],
                "tail_index": [params["xi"]],
                "threshold": [params["threshold"]],
                "n_exceedances": [params["n_exceed"]],
            },
            index=[alpha],
        )

    if model == "gev":
        params = gev_fit(data, block_size=block_size, tail=tail)
        var = evt_var(data, alpha, model, tail, block_size=block_size)
        cvar = evt_cvar(data, alpha, model, tail, block_size=block_size)

        return pd.DataFrame(
            {
                "VaR": [var],
                "CVaR": [cvar],
                "tail_index": [params["xi"]],
                "location": [params["mu"]],
                "scale": [params["sigma"]],
            },
            index=[alpha],
        )
    raise ValueError(f"Unknown model: {model}")  # pragma: no cover -- Invalid input
