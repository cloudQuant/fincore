"""Style analysis for portfolio returns.

Analyzes portfolio returns by style characteristics:
- Size (large-cap, mid-cap, small-cap)
- Value (value, growth)
    Momentum (winner, loser)
    Volatility (high, low)
"""

from __future__ import annotations

from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd


class StyleResult:
    """Container for style analysis results."""

    def __init__(
        self,
        exposures: pd.DataFrame,
        returns_by_style: pd.DataFrame,
        overall_returns: pd.Series,
    ):
        """Initialize style result.

        Parameters
        ----------
        exposures : pd.DataFrame
            Factor exposures by style (T x styles).
        returns_by_style : pd.DataFrame
            Returns for each style (styles x 1).
        overall_returns : pd.Series
            Overall portfolio returns.
        """
        self.exposures = exposures
        self.returns_by_style = returns_by_style
        self.overall_returns = overall_returns

    @property
    def style_summary(self) -> Dict[str, float]:
        """Get summary of style returns."""
        summary = {}

        for style in self.returns_by_style.index:
            summary[style] = float(self.returns_by_style[style].iloc[0])

        return summary

    def to_dict(self) -> Dict:
        """Convert results to dictionary."""
        return {
            "exposures": self.exposures.to_dict(),
            "returns_by_style": self.returns_by_style.to_dict(),
            "overall_returns": self.overall_returns.to_dict(),
        }


def style_analysis(
    returns: pd.DataFrame,
    market_caps: Optional[pd.Series] = None,
    book_to_price: Optional[pd.Series] = None,
    momentum_window: int = 252,
    size_quantiles: List[float] = [0.5, 0.5],
    value_scores: Optional[pd.Series] = None,
) -> StyleResult:
    """Perform comprehensive style analysis on portfolio returns.

    Parameters
    ----------
    returns : pd.DataFrame
        Asset returns (T x N). Columns should be asset names.
    market_caps : pd.Series, optional
        Market capitalizations for size classification.
        If None, market_cap weighted returns used.
    book_to_price : pd.Series, optional
        Book-to-price ratios for value classification.
        If None, assumes all equity.
    momentum_window : int, default 252
        Lookback period for momentum calculation.
    size_quantiles : list of float, default [0.5, 0.5]
        Quantiles for size classification.
    value_scores : pd.Series, optional
        Fundamental value scores for value classification.
        If provided, used instead of book_to_price.

    Returns
    -------
    StyleResult
        Style analysis results with exposures and returns by style.
    """
    n_assets, n_periods = returns.shape

    # 1. Size Classification
    if market_caps is not None:
        size_exposure = _calculate_size_exposure(market_caps, size_quantiles)
    else:
        # Use market cap weights as size exposure
        total_cap = market_caps.sum()
        size_exposure = market_caps / total_cap
        size_exposure = size_exposure.to_frame().T

    # 2. Momentum Classification
    momentum_returns = _calculate_momentum(returns, momentum_window)
    momentum_exposure = _exposure_from_lookback(
            momentum_returns, periods=1, direction="positive"
        )

    # 3. Value Classification
    if value_scores is not None:
        value_exposure = _value_from_scores(value_scores)
    elif book_to_price is not None:
        # Use B/P as value proxy
        # Low B/P = value, High B/P = growth
        bp_scores = book_to_price.where(book_to_price < 1, 1, 0).fillna(0)
        value_exposure = bp_scores.to_frame().T
    else:
        # Equal weight value exposure
        value_exposure = pd.DataFrame(
            np.ones(n_assets) / n_assets,
            index=returns.columns
        ).T

    # 4. Volatility Classification
    # Use rolling standard deviation
    rolling_vol = returns.rolling(window=60, min_periods=20).std(ddof=0)
    vol_median = rolling_vol.median()
    # Volatility is high if above median
    vol_exposure = (rolling_vol > vol_median).astype(int).to_frame().T

    # Combine exposures
    all_exposures = pd.concat([
        size_exposure,
        momentum_exposure,
        value_exposure,
        vol_exposure,
    ], axis=0)

    # Calculate returns by style
    returns_by_style = {}

    # Size returns
    size_long = (all_exposures["large"] * returns).sum(axis=1)
    size_small = (all_exposures["small"] * returns).sum(axis=1)
    returns_by_style["large"] = size_long.mean()
    returns_by_style["small"] = size_small.mean()

    # Momentum returns
    mom_winner = (all_exposures["winner"] * returns).sum(axis=1)
    mom_loser = (all_exposures["loser"] * returns).sum(axis=1)
    returns_by_style["winner"] = mom_winner.mean()
    returns_by_style["loser"] = mom_loser.mean()

    # Value returns
    if "value" in all_exposures.index:
        value_growth = (all_exposures["growth"] * returns).sum(axis=1)
        value_value = (all_exposures["value"] * returns).sum(axis=1)
        returns_by_style["growth"] = value_growth.mean()
        returns_by_style["value"] = value_value.mean()

    # Volatility returns
    vol_high = (all_exposures["high"] * returns).sum(axis=1)
    vol_low = (all_exposures["low"] * returns).sum(axis=1)
    returns_by_style["high"] = vol_high.mean()
    returns_by_style["low"] = vol_low.mean()

    # Overall portfolio returns (equal-weighted)
    equal_weights = np.ones(n_assets) / n_assets
    overall_returns = returns.mul(equal_weights, axis=0).sum(axis=1)

    return StyleResult(
        exposures=all_exposures,
        returns_by_style=pd.DataFrame(returns_by_style),
        overall_returns=overall_returns,
    )


def _calculate_size_exposure(
    market_caps: pd.Series,
    quantiles: List[float],
) -> pd.DataFrame:
    """Calculate size factor exposures based on market cap quantiles.

    Parameters
    ----------
    market_caps : pd.Series
        Market capitalizations (N assets).
    quantiles : list of float
        Quantile thresholds [e.g., 0.3, 0.7].

    Returns
    -------
    pd.DataFrame
        Binary exposure matrix (N x categories).
        Columns: small, mid, large (using first two quantiles).
    """
    exposures = pd.DataFrame(index=market_caps.index)

    exposures["large"] = (market_caps >= np.percentile(market_caps, quantiles[1] * 100)).astype(int)
    exposures["mid"] = (
        (market_caps >= np.percentile(market_caps, quantiles[0] * 100))
        & (market_caps < np.percentile(market_caps, quantiles[1] * 100))
    ).astype(int)
    exposures["small"] = (market_caps < np.percentile(market_caps, quantiles[0] * 100)).astype(int)

    return exposures


def _calculate_momentum(
    returns: pd.DataFrame,
    window: int,
) -> pd.DataFrame:
    """Calculate momentum signal for each asset.

    Parameters
    ----------
    returns : pd.DataFrame
        Asset returns (T x N).
    window : int
        Lookback window.

    Returns
    -------
    pd.DataFrame
        Momentum scores (1 if positive, 0 if negative).
    """
    # Cumulative returns over window
    cum_returns = (1 + returns).cumsum()

    # Momentum: position relative to window periods ago
    momentum = cum_returns.shift(window) / cum_returns.shift(window) - 1

    # Current momentum signal
    current_momentum = momentum.iloc[-1]

    return (current_momentum > 0).astype(int).to_frame().T


def _exposure_from_lookback(
    data: pd.DataFrame,
    periods: int,
    direction: str = "positive",
) -> pd.DataFrame:
    """Calculate exposure weights based on past performance.

    Parameters
    ----------
    data : pd.DataFrame
        Performance data (T x N).
    periods : int
        Number of lookback periods.
    direction : str, default "positive"
        'positive': high performers get 1, low get 0
        'negative': low performers get 1, high get 0

    Returns
    -------
    pd.DataFrame
        Binary exposure matrix.
    """
    # Calculate cumulative performance over periods
    cum_data = (1 + data).cumsum()

    # Performance over lookback periods
    if direction == "positive":
        # Top performers get 1
        lookback_perf = cum_data.shift(periods).iloc[-1]
        exposure = (cum_data.iloc[-1] >= lookback_perf).astype(int).to_frame().T
    else:
        # Bottom performers get 1
        lookback_perf = cum_data.shift(periods).iloc[-1]
        exposure = (cum_data.iloc[-1] <= lookback_perf).astype(int).to_frame().T

    return exposure


def _value_from_scores(book_to_price: pd.Series) -> pd.DataFrame:
    """Calculate value exposure from B/P scores.

    Parameters
    ----------
    book_to_price : pd.Series
        Book-to-price ratios for each asset.

    Returns
    -------
    pd.DataFrame
        Binary exposure matrix (N x 2): value, growth.
    """
    # Value: B/P < 1, Growth: B/P > 1
    value = (book_to_price < 1).astype(int).to_frame().T
    growth = (book_to_price > 1).astype(int).to_frame().T

    return pd.concat([value, growth], axis=1)


def calculate_style_tilts(
    returns: pd.DataFrame,
    factor_returns: Optional[pd.DataFrame] = None,
    window: int = 252,
) -> pd.DataFrame:
    """Calculate rolling style exposures (size, value, momentum).

    Parameters
    ----------
    returns : pd.DataFrame
        Asset returns (T x N).
    factor_returns : pd.DataFrame, optional
        Factor returns for calculating style relative to factor.
        If None, uses cross-sectional ranking.
    window : int, default 252
        Rolling window size.

    Returns
    -------
    pd.DataFrame
        Time series of style exposures for each asset.
    """
    n_periods, n_assets = returns.shape

    # Storage for rolling exposures
    all_exposures = []

    for t in range(n_periods):
        if t < window:
            continue  # Not enough data

        # Get historical returns
        hist_returns = returns.iloc[:t]

        # Size: use market cap ranking
        market_caps = hist_returns.iloc[-1]  # Use latest as proxy
        size_ranks = market_caps.rank(ascending=True)
        size_exposure = _size_rank_to_exposure(size_ranks)

        # Momentum: use past returns
        mom_returns = hist_returns.tail(window)
        momentum = _calculate_momentum(mom_returns, window)
        mom_exposure = (momentum.iloc[-1] > 0).astype(int).to_frame().T

        # Value: use B/P if available, else ranking
        # For now, use fundamental scores if provided
        # Otherwise use rank as proxy
        # This is simplified - full implementation would need B/P data
        value_exposure = pd.DataFrame(
            np.ones(n_assets),  # Placeholder - equal weight
            index=returns.columns
        ).T

        # Volatility: use rolling z-score
        vol_returns = hist_returns.tail(60)
        vol_z_score = ((vol_returns - vol_returns.mean()) / vol_returns.std()).iloc[-1]
        vol_exposure = (vol_z_score > 0).astype(int).to_frame().T

        # Combine for this period
        period_exposures = pd.concat([
            size_exposure,
            mom_exposure,
            value_exposure,
            vol_exposure,
        ], axis=0)

        all_exposures.append(period_exposures)

    # Build result DataFrame
    # MultiIndex: (date, style)
    # We'll simplify to just style names as columns
    style_columns = {
        "large": size_exposure["large"],
        "small": size_exposure["small"],
        "winner": mom_exposure["winner"],
        "loser": mom_exposure["loser"],
        "value": value_exposure["value"],
        "growth": value_exposure["growth"],
        "high_vol": vol_exposure["high"],
        "low_vol": vol_exposure["low"],
    }

    result_df = pd.DataFrame()

    for style_name, exposure in style_columns.items():
        # exposure is (T, N) - transpose to (N, T) for appending
        for t_idx in range(exposure.shape[0]):
            result_df[f"{t_idx}_{style_name}"] = exposure.iloc[:, t_idx]

    return result_df


def _size_rank_to_exposure(ranks: pd.Series) -> pd.DataFrame:
    """Convert size ranks to binary exposures."""
    exposure = pd.DataFrame(index=ranks.index)

    exposure["large"] = (ranks <= 2 / 3).astype(int)  # Top 33%
    exposure["small"] = (ranks > 1 / 3).astype(int) & (ranks <= 2 / 3).astype(int)  # Middle 34%

    return exposure.T


def calculate_regression_attribution(
    portfolio_returns: pd.Series,
    style_returns: pd.DataFrame,
    style_exposures: pd.DataFrame,
) -> Dict[str, float]:
    """Attribute portfolio returns using style exposures.

    Performs regression: R_p = a_s * S_s + sum(s_i * e_i)

    Parameters
    ----------
    portfolio_returns : pd.Series
        Portfolio returns.
    style_returns : pd.DataFrame
        Returns for each style factor.
    style_exposures : pd.DataFrame
        Current style exposures (T x S).

    Returns
    -------
    dict
        Attribution by style factor.
    """
    portfolio_return = float(np.mean(portfolio_returns))

    attributions = {}

    for style in style_returns.columns:
        style_return = float(style_returns[style].iloc[0])
        style_beta = float(np.corr(
            portfolio_returns.values,
            style_returns[style].values,
        ))
        style_contribution = style_beta * style_exposures[style].mean()  # Average exposure

        # Alpha = return - sum(beta * exposure)
        other_styles = sum(
            float(v * style_exposures[s].mean())
            for s, v in style_exposures.items() if s != style
        )

        alpha = portfolio_return - style_contribution - other_styles

        attributions[style] = alpha + style_contribution

    # Residual (specific to these styles)
    attributions["residual"] = portfolio_return - sum(attributions.values())

    return attributions


def analyze_performance_by_style(
    returns: pd.DataFrame,
    style_exposures: pd.DataFrame,
) -> pd.DataFrame:
    """Analyze performance metrics grouped by style.

    Parameters
    ----------
    returns : pd.DataFrame
        Asset returns (T x N).
    style_exposures : pd.DataFrame
        Style exposures (T x S) for each period.

    Returns
    -------
    pd.DataFrame
        Performance metrics by style.
    """
    results = []

    for t in range(1, len(style_exposures)):  # Skip initial period
        exposures_t = style_exposures.iloc[t - 1]

        # Calculate returns for each style at time t
        t_returns = returns.iloc[t]

        # Metrics by style
        row_data = {"Period": t}

        for style in exposures_t.columns:
            # Get assets with this style
            style_assets = exposures_t[exposures_t[style] == 1].index

            if len(style_assets) > 0:
                style_returns = t_returns[style_assets].mean()
                row_data[f"{style}_return"] = style_returns
            else:
                row_data[f"{style}_return"] = 0.0

        results.append(row_data)

    df = pd.DataFrame(results)
    df.set_index("Period")

    return df


def fetch_style_factors(
    tickers: List[str],
    factors: List[str] = ["size", "value", "momentum"],
    library: str = "us",
) -> pd.DataFrame:
    """Fetch style factor data (placeholder).

    This is a placeholder function for future implementation.
    When implemented, will fetch data from:
    - Compustat (for US stocks)
    - CSI (for Chinese A-shares)
    - Custom data source

    Parameters
    ----------
    tickers : list of str
        Asset tickers.
    factors : list of str, default ["size", "value", "momentum"]
        Style factors to fetch.
    library : str, default "us"
        Data source library. Options: 'us', 'chinese'.

    Returns
    -------
    pd.DataFrame
        Factor data with MultiIndex (date, factor).
    """
    # TODO: Implement actual data fetching
    # For now, return empty DataFrame with expected structure

    if library == "us":
        index = pd.date_range("2020-01-01", periods=12, freq="M")
    else:
        index = pd.date_range("2020-01-01", periods=12, freq="M")

    # Create placeholder data
    data = {}
    for factor in factors:
        # Create random data for demonstration
        n = len(tickers)
        data[factor] = np.random.randn(len(index), n) * 0.10

    return pd.DataFrame(data, index=index)
