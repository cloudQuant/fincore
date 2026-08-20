"""Independent Geometric-Brownian-Motion oracle.

GBM closed form: for ``dS = mu*S*dt + sigma*S*dW`` over horizon ``T`` (in
years), the terminal log-price is normal::

    log(S_T / S0) ~ N((mu - 0.5*sigma^2)*T, sigma^2 * T)

so the terminal log-return standard deviation is ``sigma * sqrt(T)``.  A 20%
annualized volatility over one year must produce a ~20% terminal log-return
std, not ~1.26%.  This oracle exposes the analytic moments and Monte Carlo
confidence-interval checks without importing ``fincore``.
"""

from __future__ import annotations

__all__ = [
    "gbm_terminal_log_mean",
    "gbm_terminal_log_std",
    "gbm_terminal_log_std_ci",
    "gbm_terminal_mean",
    "gbm_terminal_mean_ci",
    "gbm_terminal_std",
]


def gbm_terminal_log_mean(mu: float, sigma: float, T: float) -> float:
    """Analytic mean of ``log(S_T / S0)``."""
    return float((mu - 0.5 * sigma**2) * T)


def gbm_terminal_log_std(sigma: float, T: float) -> float:
    """Analytic standard deviation of ``log(S_T / S0)``."""
    return float(sigma * (T**0.5))


def gbm_terminal_mean(S0: float, mu: float, T: float) -> float:
    """Analytic expected terminal price ``S0 * exp(mu*T)``."""
    import math

    return float(S0 * math.exp(mu * T))


def gbm_terminal_std(S0: float, mu: float, sigma: float, T: float) -> float:
    """Analytic standard deviation of terminal price."""
    import math

    return float(S0 * math.exp(mu * T) * math.sqrt(math.expm1(sigma**2 * T)))


def gbm_terminal_mean_ci(
    S0: float,
    mu: float,
    sigma: float,
    T: float,
    n_paths: int,
    *,
    z: float = 2.5758293035489004,
) -> tuple[float, float]:
    """Analytic 99% Monte-Carlo confidence interval for the terminal-price mean.

    The Monte Carlo estimator of ``E[S_T]`` has standard error
    ``std(S_T) / sqrt(n_paths)``.  Returns ``(lo, hi)`` for the given ``z``
    (default is the 99% two-sided normal quantile).
    """
    mean = gbm_terminal_mean(S0, mu, T)
    std = gbm_terminal_std(S0, mu, sigma, T)
    se = std / (n_paths**0.5)
    return mean - z * se, mean + z * se


def gbm_terminal_log_std_ci(
    sigma: float,
    T: float,
    n_paths: int,
    *,
    z: float = 2.5758293035489004,
) -> tuple[float, float]:
    """Analytic CI for the *sample* std of terminal log-returns.

    The sample standard deviation of a normal sample has standard error
    ``sigma / sqrt(2 * (n-1))``.
    """
    std = gbm_terminal_log_std(sigma, T)
    se = std / (2.0 * (n_paths - 1)) ** 0.5
    return std - z * se, std + z * se
