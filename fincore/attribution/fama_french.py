"""Fama-French multi-factor model implementation.

Provides factor model estimation and attribution using the Fama-French
multi-factor framework:
- 3-Factor model: Market (MKT), Size (SMB), Value (HML)
- 5-Factor model: MKT, SMB, HML, Profitability (RMW), Investment (CMA)
- Momentum: Fama-French-Carhart 4-factor model
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from scipy import stats


# Standard Fama-French factor definitions
FF3_FACTORS = ["MKT", "SMB", "HML"]
FF5_FACTORS = ["MKT", "SMB", "HML", "RMW", "CMA"]
FF4MOM_FACTORS = ["MKT", "SMB", "HML", "MOM"]


class FamaFrenchModel:
    """Fama-French multi-factor model estimator.

    Supports OLS regression with Newey-West standard errors
    and various factor model specifications.
    """

    def __init__(
        self,
        model_type: str = "5factor",
        risk_free_rate: float = 0.0,
    ):
        """Initialize Fama-French model.

        Parameters
        ----------
        model_type : str, default "5factor"
            Factor model specification.
            Options: '3factor', '5factor', '4factor_mom'
        risk_free_rate : float, default 0.0
            Risk-free rate for excess returns calculation.
        """
        self.model_type = model_type
        self.risk_free_rate = risk_free_rate
        self._set_factors()

    def _set_factors(self) -> None:
        """Set factor list based on model type."""
        if self.model_type == "3factor":
            self.factors = FF3_FACTORS
        elif self.model_type == "5factor":
            self.factors = FF5_FACTORS
        elif self.model_type == "4factor_mom":
            self.factors = FF4MOM_FACTORS
        else:
            raise ValueError(
                f"Unknown model_type: {self.model_type}. "
                f"Use: '3factor', '5factor', '4factor_mom'"
            )

    def fit(
        self,
        returns: Union[pd.Series, pd.DataFrame],
        factor_data: pd.DataFrame,
        method: str = "ols",
        newey_west_lags: int = 1,
    ) -> Dict[str, Union[float, np.ndarray]]:
        """Estimate factor model using OLS regression.

        Parameters
        ----------
        returns : pd.Series or pd.DataFrame
            Asset or portfolio returns to explain.
        factor_data : pd.DataFrame
            Factor returns with columns matching factor definitions.
        method : str, default "ols"
            Estimation method. Options: 'ols', 'wls', 'gls'.
        newey_west_lags : int, default 1
            Number of lags for Newey-West standard errors.

        Returns
        -------
        dict
            Dictionary containing:
            - 'alpha': Intercept (alpha)
            - 'betas': Factor loadings
            - 'r_squared': R-squared of regression
            - 'std_errors': Standard errors (Newey-West if lags > 0)
            - 'p_values': P-values for coefficients
            - 'residuals': Regression residuals
        """
        # Prepare data
        if isinstance(returns, pd.Series):
            returns = returns.to_frame()

        # Align data
        y = returns.values.flatten()
        X = factor_data[self.factors].values

        # Add constant for intercept
        X_with_const = np.column_stack([np.ones(X.shape[0]), X.T]).T

        # OLS regression
        if method == "ols":
            beta_coeffs, residuals, _, _ = np.linalg.lstsq(X_with_const, y)

            # Calculate R-squared
            y_pred = X_with_const @ beta_coeffs
            ss_tot = np.sum((y - np.mean(y)) ** 2)
            ss_res = np.sum(residuals ** 2)
            r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 0

        elif method == "wls":
            # Weighted least squares (if we had weights)
            beta_coeffs, residuals, _, _ = np.linalg.lstsq(X_with_const, y)
            y_pred = X_with_const @ beta_coeffs
            ss_tot = np.sum((y - np.mean(y)) ** 2)
            ss_res = np.sum(residuals ** 2)
            r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 0

        else:
            raise ValueError(f"Unknown method: {method}")

        # Extract alpha and betas
        alpha = float(beta_coeffs[0])
        betas = dict(zip(self.factors, beta_coeffs[1:].tolist()))

        # Calculate standard errors
        n = len(y)
        k = len(beta_coeffs)

        # Newey-West standard errors
        if newey_west_lags > 0:
            # Calculate autocorrelation of residuals
            acorr = []
            for lag in range(1, newey_west_lags + 1):
                if len(residuals) > lag:
                    corr = np.corr(residuals[:-lag], residuals[lag:])
                    acorr.append(corr)
                else:
                    acorr.append(0)

            # Newey-West adjustment
            # Start with variance estimate
            var_beta = np.var(residuals) / n

            # Newey-West HAC estimator
            for j, rho_j in enumerate(acorr):
                if j == 0:
                    var_beta = var_beta * (1 + rho_j) / 2
                else:
                    var_beta = var_beta * (1 + rho_j)

            std_errors = np.sqrt(var_beta)
        else:
            # Simple OLS standard errors
            resid_var = np.var(residuals)
            std_errors = np.sqrt(np.diag(resid_var / (n - k)) * np.ones(k))

        # Calculate t-statistics and p-values
        std_err_total = np.sqrt(np.var(residuals) / n)
        t_stats = beta_coeffs / std_err_total
        from scipy import stats as stats
        p_values = 2 * (1 - stats.t.cdf(np.abs(t_stats), n - k - 1))

        return {
            "alpha": alpha,
            "betas": betas,
            "r_squared": r_squared,
            "std_errors": std_errors,
            "p_values": p_values,
            "residuals": residuals,
        }

    def predict(
        self,
        factor_data: pd.DataFrame,
    ) -> np.ndarray:
        """Predict returns using estimated model.

        Parameters
        ----------
        factor_data : pd.DataFrame
            Factor returns with columns matching factor definitions.

        Returns
        -------
        np.ndarray
            Predicted returns.
        """
        coeffs = self._get_regression_coeffs()
        X = factor_data[self.factors].values

        # Add constant
        X_with_const = np.column_stack([np.ones(X.shape[0]), X.T]).T
        return X_with_const @ coeffs

    def _get_regression_coeffs(self) -> np.ndarray:
        """Get concatenated alpha and betas from last fit."""
        if not hasattr(self, "_alpha"):
            raise RuntimeError("Model must be fit before prediction")

        coeffs = np.concatenate([[self._alpha], list(self._betas.values())])
        return coeffs

    def get_factor_exposures(
        self,
        returns: pd.DataFrame,
        factor_data: pd.DataFrame,
        rolling_window: Optional[int] = None,
    ) -> pd.DataFrame:
        """Calculate rolling factor exposures (betas).

        Parameters
        ----------
        returns : pd.DataFrame
            Asset returns (T x N).
        factor_data : pd.DataFrame
            Factor returns (T x K factors).
        rolling_window : int, optional
            Rolling window size. If None, full sample.

        Returns
        -------
        pd.DataFrame
            Rolling factor exposures (T x K).
        """
        exposures = []

        for t in range(len(returns)):
            if rolling_window is None:
                # Use full history up to time t
                start = 0
                end = t + 1
            else:
                # Use rolling window
                start = max(0, t - rolling_window + 1)
                end = min(t + 1, len(returns))

            window_returns = returns.iloc[start:end]
            window_factors = factor_data.iloc[start:end]

            if end > start and len(window_returns) > len(self.factors):
                # Fit model on window (first column as proxy)
                result = self.fit(
                    window_returns.iloc[:, 0],
                    window_factors,
                )
                exposures.append([result["alpha"]] + list(result["betas"].values()))
            else:
                # Not enough data - use NaN
                exposures.append([np.nan] * (len(self.factors) + 1))

        # Create column names
        column_names = ["alpha"] + self.factors
        df = pd.DataFrame(exposures, index=returns.index, columns=column_names)

        return df

    def attribution_decomposition(
        self,
        returns: pd.Series,
        factor_data: pd.DataFrame,
    ) -> Dict[str, float]:
        """Decompose returns using factor model.

        Parameters
        ----------
        returns : pd.Series
            Portfolio returns to attribute.
        factor_data : pd.DataFrame
            Factor returns.

        Returns
        -------
        dict
            Factor attribution breakdown.
        """
        # Fit model
        result = self.fit(returns, factor_data)

        # Calculate average factor returns
        avg_factor_returns = factor_data[self.factors].mean()

        # Factor contributions
        factor_contribs = {}
        for factor, beta in result["betas"].items():
            factor_contribs[factor] = beta * avg_factor_returns[factor]

        # Calculate contributions
        total_return = float(np.mean(returns)) - self.risk_free_rate
        alpha_contrib = result["alpha"]
        specific_contrib = sum(factor_contribs.values())

        return {
            "alpha": alpha_contrib,
            **{f"{factor}_attribution": contrib for factor, contrib in factor_contribs.items()},
            "specific_return": specific_contrib,
            "common_return": total_return,
            "unexplained": total_return - alpha_contrib - specific_contrib,
        }


def fetch_ff_factors(
    start: str,
    end: str,
    library: str = "french",
) -> pd.DataFrame:
    """Fetch Fama-French factors (placeholder).

    This is a placeholder function for future implementation.
    When implemented, will fetch data from:
    - French library (Dartmouth)
    - Ken French data library
    - Chinese A-share market factors

    Parameters
    ----------
    start : str
        Start date (YYYY-MM-DD).
    end : str
        End date (YYYY-MM-DD).
    library : str, default "french"
        Data source. Options: 'french', 'chinese'.

    Returns
    -------
    pd.DataFrame
        DataFrame with factor returns. Columns depend on library.
    """
    # TODO: Implement actual data fetching
    # For now, return empty DataFrame with expected structure
    if library == "5factor":
        columns = ["MKT", "SMB", "HML", "RF"]
    elif library == "3factor":
        columns = ["MKT", "SMB", "HML", "RF"]
    elif library == "4factor_mom":
        columns = ["MKT", "SMB", "HML", "MOM", "RF"]
    else:
        columns = ["MKT", "SMB", "HML", "RF"]

    # Return empty DataFrame with correct structure
    date_index = pd.date_range(start, end, freq="B")[1:]
    return pd.DataFrame(np.nan, index=date_index, columns=columns)


def calculate_idiosyncratic_risk(
    returns: pd.DataFrame,
    factor_data: pd.DataFrame,
    model: Optional[FamaFrenchModel] = None,
) -> pd.Series:
    """Calculate idiosyncratic volatility (asset-specific risk).

    Measures risk that cannot be diversified away:
    VAR(asset) - cov(asset, market) * var(market) / cov(market, market)

    Parameters
    ----------
    returns : pd.DataFrame
        Asset returns (N assets x T).
    factor_data : pd.DataFrame
        Market factor returns (T x K).
    model : FamaFrenchModel, optional
        Pre-fitted model. If None, fits new model.

    Returns
    -------
    pd.Series
        Idiosyncratic volatility for each asset.
    """
    n_assets = returns.shape[1]

    if model is None:
        model = FamaFrenchModel()
        model.fit(returns.iloc[:, 0], factor_data)

    # Calculate systematic and specific returns
    market_returns = factor_data["MKT"].values
    rf = model.risk_free_rate

    specific_volatilities = []

    for i in range(n_assets):
        asset_returns = returns.iloc[:, i].values

        if hasattr(model, "_betas"):
            beta = model._betas.get("MKT", 1.0)  # Default to market beta
            specific_return = asset_returns - rf - beta * (market_returns - rf)
            specific_var = np.var(specific_return)

            # Total return variance
            total_return = asset_returns - rf
            total_var = np.var(total_return)

            # Systematic variance
            systematic_var = (beta ** 2) * np.var(market_returns - rf)

            # Idiosyncratic variance
            idio_var = total_var - systematic_var

            specific_volatilities.append(np.sqrt(idio_var) if idio_var > 0 else 0)

    return pd.Series(specific_volatilities, index=returns.columns)
