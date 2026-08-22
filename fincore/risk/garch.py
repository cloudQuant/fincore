"""GARCH models for conditional volatility estimation.

Provides GARCH family models for time-varying volatility estimation:
- GARCH(p, q): Generalized Autoregressive Conditional Heteroskedasticity
- EGARCH: Exponential GARCH (asymmetric effects)
- GJR-GARCH: Glosten-Jagannathan-Runkle GARCH (leverage effect)

References
----------
Engle, R. F. (1982). Autoregressive Conditional Heteroscedasticity.
Bollerslev, T. (1986). Generalized Autoregressive Conditional Heteroscedasticity.
Nelson, D. B. (1991). Conditional Heteroskedasticity in Asset Returns.
Glosten, L. R., Jagannathan, R., & Runkle, D. E. (1993). On the Relation
Between the Expected Value and the Volatility of the Nominal Excess Return on Stocks.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from scipy import optimize, stats

if TYPE_CHECKING:
    import pandas as pd

__all__ = [
    "EGARCH",
    "GARCH",
    "GJRGARCH",
    "GARCHResult",
    "conditional_es",
    "conditional_var",
    "forecast_volatility",
]


@dataclass
class GARCHResult:
    """Result of GARCH model fitting.

    Attributes
    ----------
    params : dict
        Fitted parameters (omega, alpha, beta, etc.).
    conditional_var : ndarray
        Fitted conditional variances.
    residuals : ndarray
        Standardized residuals.
    log_likelihood : float
        Maximized log-likelihood value.
    model_type : str
        One of ``garch``, ``egarch`` or ``gjrgarch``; controls the forecast
        recursion.
    converged : bool
        Whether the optimizer reported success and the fitted parameters pass
        the finite/stationarity checks for the selected GARCH family.
    """

    params: dict[str, float]
    conditional_var: np.ndarray
    residuals: np.ndarray
    log_likelihood: float
    model_type: str = "garch"
    converged: bool = True

    def forecast(self, horizon: int = 1) -> np.ndarray:
        """Forecast future conditional variances.

        Parameters
        ----------
        horizon : int, default 1
            Number of steps ahead to forecast.

        Returns
        -------
        ndarray
            Forecasted variances.
        """
        if horizon < 1:
            raise ValueError("horizon must be >= 1")

        forecasts = np.zeros(horizon)
        omega = self.params["omega"]
        alpha = self.params.get("alpha", 0.0)
        beta = self.params.get("beta", 0.0)
        gamma = self.params.get("gamma", 0.0)

        last_var = float(self.conditional_var[-1])
        last_resid = float(self.residuals[-1])

        if self.model_type == "garch":
            last_eps = last_resid * np.sqrt(last_var)
            persistence = alpha + beta
            forecasts[0] = omega + alpha * last_eps**2 + beta * last_var
            for h in range(1, horizon):
                forecasts[h] = omega + persistence * forecasts[h - 1]
        elif self.model_type == "gjrgarch":
            last_eps = last_resid * np.sqrt(last_var)
            indicator = 1.0 if last_eps < 0 else 0.0
            persistence = alpha + 0.5 * gamma + beta
            forecasts[0] = omega + (alpha + gamma * indicator) * last_eps**2 + beta * last_var
            for h in range(1, horizon):
                forecasts[h] = omega + persistence * forecasts[h - 1]
        elif self.model_type == "egarch":
            last_z = last_resid
            last_log_s2 = np.log(last_var) if last_var > 0 else 0.0
            drift = alpha * np.sqrt(2.0 / np.pi)
            log_f0 = omega + alpha * np.abs(last_z) + gamma * last_z + beta * last_log_s2
            forecasts[0] = np.exp(log_f0)
            log_prev = log_f0
            for h in range(1, horizon):
                log_prev = omega + drift + beta * log_prev
                forecasts[h] = np.exp(log_prev)
        else:  # pragma: no cover - model_type is validated at construction
            raise ValueError(f"unknown model_type: {self.model_type}")

        return forecasts


class GARCH:
    """GARCH(p, q) model for conditional volatility.

    The standard GARCH(p, q) model:
    sigma_t^2 = omega + sum(alpha_i * epsilon_{t-i}^2) + sum(beta_j * sigma_{t-j}^2)

    Parameters
    ----------
    p : int, default 1
        Order of ARCH terms (past squared shocks).
    q : int, default 1
        Order of GARCH terms (past conditional variances).
    mean_model : str, default 'zero'
        Mean model: 'zero' (zero mean), 'constant' (constant mean).

    Examples
    --------
    >>> returns = pd.Series(np.random.randn(1000) * 0.02)
    >>> model = GARCH(p=1, q=1)
    >>> result = model.fit(returns)
    >>> forecasts = result.forecast(horizon=10)
    """

    def __init__(
        self,
        p: int = 1,
        q: int = 1,
        mean_model: str = "zero",
    ):
        if p != 1 or q != 1:
            raise ValueError(
                f"GARCH currently implements only the (1, 1) order; got (p={p}, q={q}). "
                "Higher-order models are not yet supported."
            )
        if mean_model not in ("zero", "constant"):
            raise ValueError(f"mean_model must be 'zero' or 'constant'; got {mean_model!r}")
        self.p = p
        self.q = q
        self.mean_model = mean_model

    def fit(
        self,
        returns: pd.Series | np.ndarray,
        method: str = "MLE",
    ) -> GARCHResult:
        """Fit GARCH model to returns.

        Parameters
        ----------
        returns : Series or ndarray
            Return series (not prices!).
        method : str, default 'MLE'
            Estimation method: 'MLE' (max likelihood) or 'OLS'.

        Returns
        -------
        GARCHResult
            Fitted model result.
        """
        y = np.asarray(returns).flatten()
        y = y[~np.isnan(y)]
        T = len(y)

        if max(self.p, self.q) + 10 > T:
            raise ValueError("Insufficient data for GARCH estimation")

        # Initialize parameters
        omega_init = np.var(y) * 0.1
        alpha_init = 0.1
        beta_init = 0.85

        if self.mean_model == "constant":
            mu_init = np.mean(y)
            init_params = [mu_init, omega_init, alpha_init, beta_init]
        else:
            init_params = [omega_init, alpha_init, beta_init]

        # Define bounds
        bounds = [
            (None, None),  # mu or omega
            (1e-6, None),  # omega
            (1e-6, 1.0),  # alpha (sum <= 1 for stability)
            (1e-6, 1.0),  # beta (sum <= 1 for stability)
        ]

        if self.mean_model == "zero":
            bounds = bounds[1:]

        # Optimize log-likelihood
        result = optimize.minimize(
            self._neg_log_likelihood,
            init_params,
            args=(y,),
            bounds=bounds,
            method="L-BFGS-B",
        )

        # Extract parameters
        params_opt = result.x

        if self.mean_model == "constant":
            mu, omega, alpha, beta = params_opt
        else:
            mu = 0
            omega, alpha, beta = params_opt

        # Compute conditional variances
        cond_var = self._compute_conditional_var(y - mu, omega, alpha, beta)

        # Standardized residuals
        residuals = (y - mu) / np.sqrt(cond_var)

        is_stationary = bool(
            np.all(np.isfinite((omega, alpha, beta)))
            and omega > 0.0
            and alpha >= 0.0
            and beta >= 0.0
            and alpha + beta < 1.0
        )
        return GARCHResult(
            params={"mu": mu, "omega": omega, "alpha": alpha, "beta": beta},
            conditional_var=cond_var,
            residuals=residuals,
            log_likelihood=-result.fun,
            model_type="garch",
            converged=bool(result.success) and is_stationary,
        )

    def _neg_log_likelihood(
        self,
        params: np.ndarray,
        y: np.ndarray,
    ) -> float:
        """Negative log-likelihood function."""
        if self.mean_model == "constant":
            mu, omega, alpha, beta = params
            eps = y - mu
        else:
            mu = 0
            omega, alpha, beta = params
            eps = y

        # Compute conditional variances
        sigma2 = self._compute_conditional_var(eps, omega, alpha, beta)

        # Remove first p+q values (burn-in)
        burn = max(self.p, self.q)
        sigma2_valid = sigma2[burn:]
        eps_valid = eps[burn:]

        # Log-likelihood (assuming normal innovations)
        loglik = -0.5 * np.sum(np.log(2 * np.pi * sigma2_valid) + eps_valid**2 / sigma2_valid)

        return float(-loglik)

    def _compute_conditional_var(
        self,
        eps: np.ndarray,
        omega: float,
        alpha: float,
        beta: float,
    ) -> np.ndarray:
        """Compute conditional variances."""
        T = len(eps)
        sigma2 = np.ones(T) * float(np.var(eps))

        for t in range(1, T):
            # GARCH(1,1): sigma2[t] = omega + alpha*eps[t-1]^2 + beta*sigma2[t-1]
            sigma2[t] = omega + alpha * eps[t - 1] ** 2 + beta * sigma2[t - 1]

        return sigma2


class EGARCH:
    """Exponential GARCH model for asymmetric volatility.

    EGARCH models the log of variance, allowing for:
    - Leverage effects (negative shocks have different impact)
    - Guaranteed positive variance

    The EGARCH(1,1) model:
    log(sigma_t^2) = omega + alpha * |z_{t-1}| + gamma * z_{t-1} + beta * log(sigma_{t-1}^2)

    where z_t = epsilon_t / sigma_t are standardized shocks.

    Parameters
    ----------
    p : int, default 1
        Order of asymmetric terms.
    q : int, default 1
        Order of GARCH terms.

    Examples
    --------
    >>> returns = pd.Series(np.random.randn(1000) * 0.02)
    >>> model = EGARCH()
    >>> result = model.fit(returns)
    >>> forecasts = result.forecast(horizon=10)
    """

    def __init__(self, p: int = 1, q: int = 1):
        if p != 1 or q != 1:
            raise ValueError(
                f"EGARCH currently implements only the (1, 1) order; got (p={p}, q={q}). "
                "Higher-order models are not yet supported."
            )
        self.p = p
        self.q = q

    def fit(
        self,
        returns: pd.Series | np.ndarray,
    ) -> GARCHResult:
        """Fit EGARCH model to returns.

        Parameters
        ----------
        returns : Series or ndarray
            Return series with at least ten finite observations and strictly
            positive finite sample variance.

        Returns
        -------
        GARCHResult
            Fitted model result.
        """
        y = np.asarray(returns).flatten()
        y = y[~np.isnan(y)]
        T = len(y)

        if T < 10:
            raise ValueError("Insufficient data for EGARCH estimation")
        with np.errstate(over="ignore", invalid="ignore"):
            initial_variance = float(np.var(y))
        if not np.all(np.isfinite(y)) or not np.isfinite(initial_variance) or initial_variance <= 0.0:
            raise ValueError("EGARCH estimation requires finite returns with positive variance")

        # Initialize EGARCH params: omega, alpha, gamma, beta
        init_params = [0.01, 0.1, -0.1, 0.95]

        bounds = [
            (None, None),  # omega
            (1e-6, None),  # alpha
            (-1.0, 1.0),  # gamma (asymmetry)
            (1e-6, 1.0),  # beta
        ]

        result = optimize.minimize(
            self._neg_log_likelihood,
            init_params,
            args=(y,),
            bounds=bounds,
            method="L-BFGS-B",
        )

        omega, alpha, gamma, beta = result.x

        # Compute the same conditional-innovation recursion used by the
        # likelihood.  Forecasting consumes these standardized residuals, so
        # substituting a whole-sample standard deviation here would make the
        # fitted variance path and forecast model inconsistent.
        cond_var = self._compute_conditional_var(y, omega, alpha, gamma, beta)
        residuals = y / np.sqrt(cond_var)

        is_stationary = bool(np.all(np.isfinite((omega, alpha, gamma, beta))) and alpha >= 0.0 and abs(beta) < 1.0)
        return GARCHResult(
            params={
                "omega": omega,
                "alpha": alpha,
                "gamma": gamma,
                "beta": beta,
            },
            conditional_var=cond_var,
            residuals=residuals,
            log_likelihood=-result.fun,
            model_type="egarch",
            converged=bool(result.success) and is_stationary,
        )

    def _neg_log_likelihood(
        self,
        params: np.ndarray,
        y: np.ndarray,
    ) -> float:
        """Negative log-likelihood for EGARCH."""
        omega, alpha, gamma, beta = params
        with np.errstate(over="ignore", invalid="ignore"):
            initial_variance = float(np.var(y))
        if not np.all(np.isfinite(y)) or not np.isfinite(initial_variance) or initial_variance <= 0.0:
            return 1e100
        sigma2 = self._compute_conditional_var(y, omega, alpha, gamma, beta)
        if not np.all(np.isfinite(sigma2)) or np.any(sigma2 <= 0.0):
            # L-BFGS-B probes parameters outside the finite likelihood domain.
            # Return a finite penalty so its numerical derivative remains
            # defined instead of leaking an overflow warning to callers.
            return 1e100

        # Suppress numerical warnings during optimization
        # These warnings are expected during early iterations and don't affect final results
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            eps_valid = y[1:] / np.sqrt(sigma2[1:])
            sigma2_valid = sigma2[1:]
            loglik = -0.5 * np.sum(np.log(2 * np.pi * sigma2_valid) + eps_valid**2)

        if not np.isfinite(loglik):
            return 1e100
        return float(-loglik)

    @staticmethod
    def _compute_conditional_var(
        y: np.ndarray,
        omega: float,
        alpha: float,
        gamma: float,
        beta: float,
    ) -> np.ndarray:
        """Return the EGARCH(1,1) variance path with conditional shocks.

        ``z[t-1]`` is standardized by its own preceding conditional variance,
        matching the model equation and the residuals exposed in
        :class:`GARCHResult`.
        """
        t = len(y)
        sigma2 = np.empty(t, dtype=float)
        with np.errstate(over="ignore", invalid="ignore"):
            initial_variance = float(np.var(y))
        if not np.isfinite(initial_variance) or initial_variance <= 0.0:
            return np.full(t, np.nan, dtype=float)
        sigma2[0] = initial_variance
        log_variance = np.log(initial_variance)

        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            for index in range(1, t):
                z_previous = y[index - 1] / np.sqrt(sigma2[index - 1])
                log_variance = omega + alpha * np.abs(z_previous) + gamma * z_previous + beta * log_variance
                sigma2[index] = np.exp(log_variance)

        return sigma2


class GJRGARCH:
    """GJR-GARCH model with leverage effect.

    GJR-GARCH adds a term to capture asymmetric response to shocks:
    - Negative shocks (bad news) increase volatility more than positive shocks

    The GJR-GARCH(1,1) model:
    sigma_t^2 = omega + alpha * epsilon_{t-1}^2 + gamma * I_{t-1} * epsilon_{t-1}^2 + beta * sigma_{t-1}^2

    where I_t = 1 if epsilon_t < 0 else 0.

    Parameters
    ----------
    p : int, default 1
        Order of ARCH terms.
    q : int, default 1
        Order of GARCH terms.

    Examples
    --------
    >>> returns = pd.Series(np.random.randn(1000) * 0.02)
    >>> model = GJRGARCH()
    >>> result = model.fit(returns)
    >>> print(f"Leverage gamma: {result.params['gamma']:.3f}")
    """

    def __init__(self, p: int = 1, q: int = 1):
        if p != 1 or q != 1:
            raise ValueError(
                f"GJRGARCH currently implements only the (1, 1) order; got (p={p}, q={q}). "
                "Higher-order models are not yet supported."
            )
        self.p = p
        self.q = q

    def fit(
        self,
        returns: pd.Series | np.ndarray,
    ) -> GARCHResult:
        """Fit GJR-GARCH model to returns.

        Parameters
        ----------
        returns : Series or ndarray
            Return series.

        Returns
        -------
        GARCHResult
            Fitted model result including leverage parameter gamma.
        """
        y = np.asarray(returns).flatten()
        y = y[~np.isnan(y)]
        T = len(y)

        if T < 10:
            raise ValueError("Insufficient data for GJR-GARCH estimation")

        # Initialize parameters: omega, alpha, gamma (leverage), beta
        init_params = [0.01, 0.05, 0.05, 0.9]

        bounds = [
            (1e-6, None),  # omega
            (1e-6, 1.0),  # alpha
            (0.0, 1.0),  # gamma (leverage, >= 0)
            (1e-6, 1.0),  # beta
        ]

        result = optimize.minimize(
            self._neg_log_likelihood,
            init_params,
            args=(y,),
            bounds=bounds,
            method="L-BFGS-B",
        )

        omega, alpha, gamma, beta = result.x

        # Compute conditional variances
        cond_var = self._compute_conditional_var(y, omega, alpha, gamma, beta)

        residuals = y / np.sqrt(cond_var)

        is_stationary = bool(
            np.all(np.isfinite((omega, alpha, gamma, beta)))
            and omega > 0.0
            and alpha >= 0.0
            and gamma >= 0.0
            and beta >= 0.0
            and alpha + 0.5 * gamma + beta < 1.0
        )
        return GARCHResult(
            params={
                "omega": omega,
                "alpha": alpha,
                "gamma": gamma,  # Leverage effect
                "beta": beta,
            },
            conditional_var=cond_var,
            residuals=residuals,
            log_likelihood=-result.fun,
            model_type="gjrgarch",
            converged=bool(result.success) and is_stationary,
        )

    def _neg_log_likelihood(
        self,
        params: np.ndarray,
        y: np.ndarray,
    ) -> float:
        """Negative log-likelihood for GJR-GARCH."""
        omega, alpha, gamma, beta = params

        cond_var = self._compute_conditional_var(y, omega, alpha, gamma, beta)

        # Skip initial values
        burn = 1
        var_valid = cond_var[burn:]
        y_valid = y[burn:]

        loglik = -0.5 * np.sum(np.log(2 * np.pi * var_valid) + y_valid**2 / var_valid)

        return float(-loglik)

    def _compute_conditional_var(
        self,
        y: np.ndarray,
        omega: float,
        alpha: float,
        gamma: float,
        beta: float,
    ) -> np.ndarray:
        """Compute conditional variances with leverage effect."""
        T = len(y)
        sigma2 = np.ones(T) * float(np.var(y))

        for t in range(1, T):
            # Indicator for negative shock
            indicator = 1 if y[t - 1] < 0 else 0

            # GJR-GARCH variance recursion
            sigma2[t] = omega + alpha * y[t - 1] ** 2 + gamma * indicator * y[t - 1] ** 2 + beta * sigma2[t - 1]

        return sigma2


def forecast_volatility(
    returns: pd.Series | np.ndarray,
    model: str = "GARCH",
    horizon: int = 1,
    **kwargs,
) -> tuple[np.ndarray, GARCHResult]:
    """Forecast future volatility using GARCH models.

    Convenience function for volatility forecasting.

    Parameters
    ----------
    returns : Series or ndarray
        Historical returns.
    model : str, default 'GARCH'
        Model type: 'GARCH', 'EGARCH', 'GJRGARCH'.
    horizon : int, default 1
        Forecast horizon.
    **kwargs
        Additional model parameters (p, q, etc.).

    Returns
    -------
    forecasts : ndarray
        Forecasted conditional variances.
    result : GARCHResult
        Fitted model result.

    Examples
    --------
    >>> returns = pd.Series(np.random.randn(1000) * 0.02)
    >>> forecasts, result = forecast_volatility(returns, model="EGARCH", horizon=5)
    >>> print(f"5-day volatility forecast: {np.sqrt(forecasts)}")
    """
    models = {
        "GARCH": GARCH,
        "EGARCH": EGARCH,
        "GJRGARCH": GJRGARCH,
    }

    if model not in models:
        raise ValueError(f"Unknown model: {model}. Available: {list(models.keys())}")

    model_class = models[model]
    model_instance = model_class(**kwargs)
    result = model_instance.fit(returns)
    forecasts = result.forecast(horizon=horizon)

    return forecasts, result


def conditional_var(
    returns: pd.Series | np.ndarray,
    model: str = "GARCH",
    alpha: float = 0.05,
    horizon: int = 1,
    **kwargs,
) -> dict[str, float | np.ndarray | GARCHResult]:
    """Calculate conditional VaR using GARCH models.

    Parameters
    ----------
    returns : Series or ndarray
        Historical returns.
    model : str, default 'GARCH'
        Model type.
    alpha : float, default 0.05
        Significance level.
    horizon : int, default 1
        Forecast horizon; multi-horizon VaR aggregates the forecast variances
        over the horizon (square-root-of-sum under independent increments).
    **kwargs
        Additional model parameters.

    Returns
    -------
    dict
        Contains 'var' (VaR estimate), 'cond_var' (forecast variances over the
        horizon), 'result' (full model fit) and 'converged' (optimizer status).
    """
    forecasts, result = forecast_volatility(returns, model, horizon=horizon, **kwargs)

    total_var = float(np.sum(forecasts))
    z_alpha = stats.norm.ppf(alpha)
    var = z_alpha * np.sqrt(total_var)

    return {
        "var": var,
        "cond_var": forecasts,
        "result": result,
        "converged": result.converged,
    }


def conditional_es(
    returns: pd.Series | np.ndarray,
    model: str = "GARCH",
    alpha: float = 0.05,
    horizon: int = 1,
    **kwargs,
) -> dict[str, float | np.ndarray | GARCHResult]:
    """Calculate conditional Expected Shortfall using GARCH models.

    Under the normal-innovation assumption used by the GARCH family here, the
    Expected Shortfall at tail probability ``alpha`` is::

        ES = -sqrt(sigma^2) * phi(z_alpha) / alpha

    which is strictly more extreme than the VaR ``z_alpha * sqrt(sigma^2)``.
    ``sigma^2`` is the horizon-aggregated forecast variance.

    Returns
    -------
    dict
        Contains 'es', 'cond_var' (forecast variances), 'result' and
        'converged'.
    """
    forecasts, result = forecast_volatility(returns, model, horizon=horizon, **kwargs)

    total_var = float(np.sum(forecasts))
    z_alpha = stats.norm.ppf(alpha)
    es = -np.sqrt(total_var) * stats.norm.pdf(z_alpha) / alpha

    return {
        "es": es,
        "cond_var": forecasts,
        "result": result,
        "converged": result.converged,
    }
