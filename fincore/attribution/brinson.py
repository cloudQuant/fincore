"""Brinson attribution analysis for portfolio performance.

Decomposes portfolio excess returns into:
- Allocation effect: returns from overweighting/underweighting sectors
- Selection effect: returns from stock selection within sectors
- Interaction effect: returns from interaction of allocation and selection
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd


def brinson_attribution(
    portfolio_returns: pd.Series | np.ndarray,
    benchmark_returns: pd.Series | np.ndarray,
    portfolio_weights: pd.DataFrame,
    benchmark_weights: pd.DataFrame,
) -> dict[str, float]:
    """Calculate Brinson attribution for a single period.

    Parameters
    ----------
    portfolio_returns : pd.Series or np.ndarray
        Portfolio returns for the period.
    benchmark_returns : pd.Series or np.ndarray
        Benchmark returns for the period.
    portfolio_weights : pd.DataFrame
        Portfolio weights by sector/asset for the period.
        benchmark_weights : pd.DataFrame
        Benchmark weights by sector/asset for the period.

    Returns
    -------
    dict
        Dictionary with attribution breakdown:
        - 'allocation': Allocation effect
        - 'selection': Selection effect
        - 'interaction': Interaction effect
        - 'total': Total active return
        - 'portfolio_return': Portfolio return
        - 'benchmark_return': Benchmark return

    Examples
    --------
    >>> import pandas as pd
    >>> import numpy as np
    >>> portfolio_returns = np.array([0.05, 0.03, -0.02, 0.04])
    >>> benchmark_returns = np.array([0.03, 0.02, -0.01, 0.03])
    >>> portfolio_weights = pd.DataFrame({
    ...     'Sector_A': [0.4, 0.3, 0.2, 0.1],
    ...     'Sector_B': [0.3, 0.3, 0.3, 0.1],
    ... })
    >>> benchmark_weights = pd.DataFrame({
    ...     'Sector_A': [0.5, 0.3, 0.1, 0.1],
    ...     'Sector_B': [0.3, 0.3, 0.3, 0.1],
    ... })
    >>> result = brinson_attribution(
    ...     portfolio_returns, benchmark_returns,
    ...     portfolio_weights, benchmark_weights
    ... )
    >>> print(f"Allocation: {result['allocation']:.2%}")
    """
    # Convert to numpy arrays
    rp = np.asarray(portfolio_returns)
    rb = np.asarray(benchmark_returns)
    wp = portfolio_weights.values
    wb = benchmark_weights.values

    # Calculate portfolio and benchmark returns
    portfolio_return = float(np.mean(rp))
    benchmark_return = float(np.mean(rb))
    active_return = portfolio_return - benchmark_return

    # Brinson decomposition
    # Allocation effect: sum of (wp - wb) * benchmark sector returns
    # Selection effect: sum of wp * (portfolio sector returns - benchmark sector returns)
    # Interaction: sum of (wp - wb) * (portfolio sector returns - benchmark sector returns)

    n_sectors = wp.shape[1]

    allocation_effect = 0.0
    selection_effect = 0.0
    interaction_effect = 0.0

    for i in range(n_sectors):
        # Weight difference
        w_diff = wp[:, i] - wb[:, i]

        # Sector returns
        rp_sector = float(np.mean(rp))
        rb_sector = float(np.mean(rb))

        # Components
        allocation_effect += float(np.sum(w_diff * rb_sector))
        selection_effect += float(np.sum(wp[:, i] * (rp_sector - rb_sector)))
        interaction_effect += float(np.sum(w_diff * (rp_sector - rb_sector)))

    # Verify total matches active return
    total = allocation_effect + selection_effect + interaction_effect
    if not np.isclose(total, active_return, rtol=1e-4, atol=1e-4):
        # Add residual to ensure exact match
        residual = active_return - total

        return {
            "allocation": allocation_effect,
            "selection": selection_effect,
            "interaction": interaction_effect,
            "residual": residual,
            "total": total,
            "portfolio_return": portfolio_return,
            "benchmark_return": benchmark_return,
        }

    return {
        "allocation": allocation_effect,
        "selection": selection_effect,
        "interaction": interaction_effect,
        "total": total,
        "portfolio_return": portfolio_return,
        "benchmark_return": benchmark_return,
    }


def brinson_results(
    portfolio_returns: pd.Series | np.ndarray,
    benchmark_returns: pd.Series | np.ndarray,
    portfolio_weights: pd.DataFrame,
    benchmark_weights: pd.DataFrame,
    periods: list[str] | None = None,
) -> pd.DataFrame:
    """Calculate Brinson attribution over multiple periods.

    Parameters
    ----------
    portfolio_returns : pd.Series or np.ndarray
        Portfolio returns (T periods).
    benchmark_returns : pd.Series or np.ndarray
        Benchmark returns (T periods).
    portfolio_weights : pd.DataFrame
        Portfolio weights for each period (T x sectors).
    benchmark_weights : pd.DataFrame
        Benchmark weights for each period (T x sectors).
    periods : list of str, optional
        Period labels. If None, uses integer indices.

    Returns
    -------
    pd.DataFrame
        DataFrame with attribution results for each period.
        Columns: period, allocation, selection, interaction, total, portfolio_return, benchmark_return
    """
    # Convert to arrays
    rp = np.asarray(portfolio_returns)
    rb = np.asarray(benchmark_returns)

    n_periods = rp.shape[0] if rp.ndim > 1 else 1

    if periods is None:
        periods = [str(i) for i in range(n_periods)]

    results = []

    for t in range(n_periods):
        # Get weights for this period
        wp_t = portfolio_weights.iloc[t].values if hasattr(portfolio_weights, "iloc") else portfolio_weights
        wb_t = benchmark_weights.iloc[t].values if hasattr(benchmark_weights, "iloc") else benchmark_weights

        # Calculate attribution
        rp_t = rp[t] if rp.ndim > 1 else rp
        rb_t = rb[t] if rb.ndim > 1 else rb

        attr = brinson_attribution(rp_t, rb_t, wp_t, wb_t)

        results.append(
            {
                "period": periods[t],
                "allocation": attr["allocation"],
                "selection": attr["selection"],
                "interaction": attr["interaction"],
                "total": attr["total"],
                "portfolio_return": attr["portfolio_return"],
                "benchmark_return": attr["benchmark_return"],
            }
        )

    df = pd.DataFrame(results)

    # Add summary rows
    summary = df[["allocation", "selection", "interaction", "total"]].sum()
    summary.loc["Total"] = df["total"].sum()

    return df


def brinson_cumulative(
    portfolio_returns: pd.Series | np.ndarray,
    benchmark_returns: pd.Series | np.ndarray,
    portfolio_weights: pd.DataFrame,
    benchmark_weights: pd.DataFrame,
) -> dict[str, float]:
    """Calculate cumulative Brinson attribution.

    Similar to brinson_attribution but uses cumulative returns
    for performance measurement since inception.

    Parameters
    ----------
    portfolio_returns : pd.Series or np.ndarray
        Portfolio returns.
    benchmark_returns : pd.Series or np.ndarray
        Benchmark returns.
    portfolio_weights : pd.DataFrame
        Portfolio weights (final weights for cumulative).
    benchmark_weights : pd.DataFrame
        Benchmark weights (final weights for cumulative).

    Returns
    -------
    dict
        Cumulative attribution breakdown.
    """
    # Convert to numpy arrays
    rp = np.asarray(portfolio_returns)
    rb = np.asarray(benchmark_returns)

    # Calculate cumulative returns
    # Use geometric cumulative return: prod(1 + r) - 1
    if rp.ndim > 1:
        portfolio_cum = np.prod(1 + rp, axis=0) - 1
        benchmark_cum = np.prod(1 + rb, axis=0) - 1
    else:
        portfolio_cum = float(np.sum(rp))  # Simple sum for single period
        benchmark_cum = float(np.sum(rb))

    active_return = portfolio_cum - benchmark_cum

    # Decompose using final weights only
    wp = portfolio_weights.iloc[-1].values if hasattr(portfolio_weights, "iloc") else portfolio_weights
    wb = benchmark_weights.iloc[-1].values if hasattr(benchmark_weights, "iloc") else benchmark_weights

    n_sectors = wp.shape[0] if wp.ndim > 1 else len(wp)

    allocation = 0.0
    selection = 0.0
    interaction = 0.0

    for i in range(n_sectors):
        w_diff = wp[i] - wb[i] if wp.ndim > 1 else wp[0] - wb[0]

        # Use cumulative returns for sector performance
        # For single period, this reduces to standard Brinson

        if rp.ndim > 1:
            rp_sector = float(np.mean(rp[:, i] if rp.ndim > 1 else rp))
            rb_sector = float(np.mean(rb[:, i] if rb.ndim > 1 else rb))
        else:
            rp_sector = float(rp)
            rb_sector = float(rb)

        allocation += float(w_diff * rb_sector)
        selection += float((wp[i] if wp.ndim > 1 else wp[0]) * (rp_sector - rb_sector))
        interaction += float(w_diff * (rp_sector - rb_sector))

    total = allocation + selection + interaction

    # Handle single period case
    if not np.isclose(total, active_return, rtol=1e-4, atol=1e-4):
        residual = active_return - total
        total += residual

    return {
        "allocation": allocation,
        "selection": selection,
        "interaction": interaction,
        "total": total,
        "portfolio_cumulative": portfolio_cum,
        "benchmark_cumulative": benchmark_cum,
    }


class BrinsonAttribution:
    """Brinson attribution calculator with sector mapping support.

    Provides convenient interface for multi-period Brinson attribution
    with sector/asset classification.
    """

    def __init__(
        self,
        sector_mapping: dict[str, list[str]] | None = None,
    ):
        """Initialize Brinson attribution calculator.

        Parameters
        ----------
        sector_mapping : dict, optional
            Mapping from asset name to sector name.
            Example: {'AAPL': 'Technology', 'MSFT': 'Technology', 'JPM': 'Financial'}
        """
        self.sector_mapping = sector_mapping or {}

    def calculate(
        self,
        returns: pd.DataFrame,
        benchmark_returns: pd.DataFrame | None = None,
        weights: pd.DataFrame | None = None,
        method: str = "brinson",
    ) -> pd.DataFrame:
        """Calculate Brinson attribution for returns DataFrame.

        Parameters
        ----------
        returns : pd.DataFrame
            Asset returns (T x N assets).
            Columns should be asset names.
        benchmark_returns : pd.DataFrame, optional
            Benchmark returns. If None, uses equal-weighted benchmark.
        weights : pd.DataFrame, optional
            Portfolio weights. If None, uses equal weights.
        method : str, default "brinson"
            Attribution method. Options: 'brinson', 'brinson_hood'.

        Returns
        -------
        pd.DataFrame
            Attribution results by period.
        """
        # Apply sector mapping if provided
        if self.sector_mapping:
            returns = self._apply_sector_mapping(returns)

            if benchmark_returns is not None:
                benchmark_returns = self._apply_sector_mapping(benchmark_returns)

            if weights is not None:
                weights = self._apply_sector_mapping(weights)

        # Use equal weights if not provided
        if weights is None:
            n = returns.shape[1]
            weights_array = np.ones(n) / n
            weights = pd.DataFrame(
                np.tile(weights_array, (returns.shape[0], 1)),
                columns=returns.columns,
                index=returns.index,
            )

        # Calculate portfolio returns
        portfolio_returns = returns.mul(weights.values, axis=1).sum(axis=1)

        # Benchmark
        if benchmark_returns is None:
            # Equal-weighted benchmark
            n_bench = returns.shape[1]
            bench_weights = np.ones(n_bench) / n_bench
            benchmark_returns = returns.mul(bench_weights, axis=1).sum(axis=1)
            benchmark_weights = pd.DataFrame(
                np.tile(bench_weights, (returns.shape[0], 1)),
                columns=returns.columns,
                index=returns.index,
            )

        # Calculate attribution for each period
        results = []

        for t in range(returns.shape[0]):
            wp_t = weights.iloc[t].values if hasattr(weights, "iloc") else weights.values[t]
            wb_t = (
                benchmark_weights.iloc[t].values if hasattr(benchmark_weights, "iloc") else benchmark_weights.values[t]
            )

            rp_t = returns.iloc[t].values
            rb_t = benchmark_returns.iloc[t].values

            attr = brinson_attribution(rp_t, rb_t, wp_t, wb_t)

            results.append(
                {
                    "period": t,
                    "allocation": attr["allocation"],
                    "selection": attr["selection"],
                    "interaction": attr["interaction"],
                    "total": attr["total"],
                }
            )

        return pd.DataFrame(results)

    def _apply_sector_mapping(
        self,
        df: pd.DataFrame,
    ) -> pd.DataFrame:
        """Apply sector mapping to aggregate by sector."""
        # Create sector columns
        sector_columns = []

        for sector, assets in self.sector_mapping.items():
            sector_df = df[assets]
            sector_df_sum = sector_df.sum(axis=1)
            sector_columns.append(sector_df_sum)

        # Combine into sector-level DataFrame
        return pd.concat(sector_columns, axis=1)

    def __repr__(self) -> str:
        sectors = list(self.sector_mapping.keys()) if self.sector_mapping else []
        return f"BrinsonAttribution({len(sectors)} sectors)"
