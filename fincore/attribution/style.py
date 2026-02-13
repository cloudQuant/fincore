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
    def style_summary(self) -> dict[str, float]:
        """Get summary of style returns."""
        summary = {}

        # Handle both Series and DataFrame for returns_by_style
        if isinstance(self.returns_by_style, pd.Series):
            # Series: index contains styles
            for style in self.returns_by_style.index:
                summary[style] = float(self.returns_by_style[style])
        else:
            # DataFrame: either 'style' column or index contains styles
            if 'style' in self.returns_by_style.columns:
                # Style is a column
                for _, row in self.returns_by_style.iterrows():
                    summary[row['style']] = float(row['return'])
            else:
                # Index contains styles
                for style in self.returns_by_style.index:
                    val = self.returns_by_style.loc[style]
                    # Handle both scalar and Series values
                    if isinstance(val, pd.Series):
                        summary[style] = float(val.iloc[0]) if len(val) > 0 else 0.0
                    else:
                        summary[style] = float(val)

        return summary

    def to_dict(self) -> dict:
        """Convert results to dictionary."""
        return {
            "exposures": self.exposures.to_dict(),
            "returns_by_style": self.returns_by_style.to_dict(),
            "overall_returns": self.overall_returns.to_dict(),
        }


def style_analysis(
    returns: pd.DataFrame,
    market_caps: pd.Series | None = None,
    book_to_price: pd.Series | None = None,
    momentum_window: int = 252,
    size_quantiles: list[float] | None = None,
    value_scores: pd.Series | None = None,
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
    if size_quantiles is None:
        size_quantiles = [0.5, 0.5]

    n_periods, n_assets = returns.shape

    # Initialize exposures dict
    exposures_data = {}

    # 1. Size Classification
    if market_caps is not None:
        size_exposure = _calculate_size_exposure(market_caps, size_quantiles)
        exposures_data.update(size_exposure.to_dict(orient='index'))
    else:
        # Equal weight size exposure
        exposures_data["equal_weight"] = {col: 1.0 / n_assets for col in returns.columns}

    # 2. Momentum Classification - simplified
    # Use recent returns to classify momentum
    recent_returns = returns.tail(min(momentum_window, len(returns)))
    cumulative_returns = recent_returns.sum()
    mom_threshold = cumulative_returns.median()
    exposures_data["winner"] = {
        col: 1.0 if cumulative_returns[col] >= mom_threshold else 0.0
        for col in returns.columns
    }
    exposures_data["loser"] = {
        col: 1.0 if cumulative_returns[col] < mom_threshold else 0.0
        for col in returns.columns
    }

    # 3. Value Classification
    if value_scores is not None:
        value_exposure = _value_from_scores(value_scores)
        exposures_data.update(value_exposure.to_dict(orient='index'))
    elif book_to_price is not None:
        # Use B/P as value proxy
        for col in returns.columns:
            if col in book_to_price.index:
                bp = book_to_price[col]
                if bp < 1:
                    exposures_data.setdefault("value", {})[col] = 1.0
                    exposures_data.setdefault("growth", {})[col] = 0.0
                else:
                    exposures_data.setdefault("value", {})[col] = 0.0
                    exposures_data.setdefault("growth", {})[col] = 1.0
    else:
        # Equal weight value exposure
        for col in returns.columns:
            exposures_data.setdefault("value", {})[col] = 1.0 / n_assets
            exposures_data.setdefault("growth", {})[col] = 1.0 / n_assets

    # 4. Volatility Classification
    rolling_vol = returns.rolling(window=60, min_periods=20).std(ddof=0)
    vol_median = rolling_vol.median()
    vol_at_end = rolling_vol.iloc[-1]
    for col in returns.columns:
        if vol_at_end[col] >= vol_median[col]:
            exposures_data.setdefault("high", {})[col] = 1.0
            exposures_data.setdefault("low", {})[col] = 0.0
        else:
            exposures_data.setdefault("high", {})[col] = 0.0
            exposures_data.setdefault("low", {})[col] = 1.0

    # Build exposures DataFrame
    all_exposures = pd.DataFrame.from_dict(exposures_data, orient='index').T

    # Calculate returns by style
    returns_by_style = {}

    for style in all_exposures.columns:
        # Calculate weighted return for each time period, then average
        style_exposure = all_exposures[style].reindex(returns.columns, fill_value=0)
        weighted_returns = (returns.T * style_exposure).T
        style_returns = weighted_returns.sum(axis=1).mean()
        returns_by_style[style] = style_returns

    # Overall portfolio returns (equal-weighted)
    equal_weights = pd.Series(np.ones(n_assets) / n_assets, index=returns.columns)
    overall_returns = returns.mul(equal_weights, axis=1).sum(axis=1)

    # Convert returns_by_style dict to DataFrame
    returns_by_style_df = pd.DataFrame(
        list(returns_by_style.items()),
        columns=['style', 'return']
    )
    returns_by_style_df = returns_by_style_df.set_index('style')

    return StyleResult(
        exposures=all_exposures,
        returns_by_style=returns_by_style_df,
        overall_returns=overall_returns,
    )


def _calculate_size_exposure(
    market_caps: pd.Series,
    quantiles: list[float],
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
    factor_returns: pd.DataFrame | None = None,
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
        Index: dates, Columns: {asset}_{style} format.
    """
    n_periods, n_assets = returns.shape

    # Build result DataFrame
    result_df = pd.DataFrame(index=returns.index[window:])

    for t in range(window, n_periods):
        # Get historical returns
        hist_returns = returns.iloc[:t]

        # Size: use returns ranking as proxy for market cap
        latest_returns = hist_returns.iloc[-1]
        size_ranks = latest_returns.rank(ascending=True)
        # Top third = large, bottom third = small
        for asset in returns.columns:
            rank = size_ranks[asset]
            result_df.loc[returns.index[t], f"{asset}_large"] = 1 if rank <= n_assets / 3 else 0
            result_df.loc[returns.index[t], f"{asset}_small"] = 1 if rank > 2 * n_assets / 3 else 0

        # Momentum: use cumulative returns over window
        mom_returns = hist_returns.tail(window)
        cum_returns = (1 + mom_returns).prod() - 1
        for asset in returns.columns:
            result_df.loc[returns.index[t], f"{asset}_winner"] = 1 if cum_returns[asset] > 0 else 0
            result_df.loc[returns.index[t], f"{asset}_loser"] = 1 if cum_returns[asset] <= 0 else 0

        # Value: use ranking as proxy (higher returns = value outperformers)
        value_ranks = cum_returns.rank(ascending=True)
        for asset in returns.columns:
            result_df.loc[returns.index[t], f"{asset}_value"] = 1 if value_ranks[asset] <= n_assets / 2 else 0
            result_df.loc[returns.index[t], f"{asset}_growth"] = 1 if value_ranks[asset] > n_assets / 2 else 0

        # Volatility: use rolling std
        vol_returns = hist_returns.tail(60)
        vol_std = vol_returns.std()
        vol_median = vol_std.median()
        for asset in returns.columns:
            result_df.loc[returns.index[t], f"{asset}_high_vol"] = 1 if vol_std[asset] >= vol_median else 0
            result_df.loc[returns.index[t], f"{asset}_low_vol"] = 1 if vol_std[asset] < vol_median else 0

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
) -> dict[str, float]:
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

    # Calculate contribution for each style
    for style in style_returns.columns:
        if style not in style_exposures.columns:
            continue

        style_return = float(style_returns[style].iloc[0])
        # np.corrcoef returns 2x2 matrix, extract [0,1] element
        corr_matrix = np.corrcoef(
            portfolio_returns.values,
            style_returns[style].values,
        )
        style_beta = float(corr_matrix[0, 1]) if not np.isnan(corr_matrix[0, 1]) else 0.0

        # Average exposure for this style
        avg_exposure = float(style_exposures[style].mean())
        style_contribution = style_beta * avg_exposure

        attributions[style] = style_contribution

    # Residual (specific to these styles)
    total_attributed = sum(attributions.values())
    attributions["residual"] = portfolio_return - total_attributed

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
        row_data: dict[str, int | float] = {"Period": int(t)}

        # exposures_t is a Series if style_exposures is DataFrame
        if isinstance(exposures_t, pd.Series):
            # If it's a Series, treat it as single style exposure
            style_assets = exposures_t[exposures_t == 1].index
            if len(style_assets) > 0:
                style_ret = t_returns[style_assets].mean()
                row_data["style_return"] = style_ret
            else:
                row_data["style_return"] = 0.0
        else:
            # It's a DataFrame, iterate over columns
            for style in exposures_t.columns:
                # Get assets with this style
                style_assets = exposures_t.index[exposures_t[style] == 1]

                if len(style_assets) > 0:
                    style_returns = t_returns[style_assets].mean()
                    row_data[f"{style}_return"] = style_returns
                else:
                    row_data[f"{style}_return"] = 0.0

        results.append(row_data)

    if results:
        df = pd.DataFrame(results)
        df = df.set_index("Period")
        return df
    else:
        return pd.DataFrame()


def fetch_style_factors(
    tickers: list[str],
    factors: list[str] | None = None,
    library: str = "us",
) -> pd.DataFrame:
    """Fetch style factor data.

    .. note::

       A concrete data provider must be configured before calling this
       function.  Pass pre-fetched factor data directly to
       :func:`style_analysis` or :func:`calculate_style_tilts`.

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

    Raises
    ------
    NotImplementedError
        Always raised until a data provider is configured.
    """
    import logging

    logger = logging.getLogger(__name__)
    logger.error(
        "fetch_style_factors called without a configured data provider. "
        "Pass pre-fetched factor data directly to style_analysis() or "
        "calculate_style_tilts()."
    )
    raise NotImplementedError(
        "No style factor data provider is configured. "
        "Please pass pre-fetched factor data directly to "
        "style_analysis() or calculate_style_tilts()."
    )
