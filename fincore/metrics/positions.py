#
# Copyright 2016 Quantopian, Inc.
# Copyright 2025 CloudQuant Team
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Position and holdings metrics."""

from __future__ import annotations

from typing import Any, cast

import numpy as np
import pandas as pd

from fincore.constants import CAP_BUCKETS, SECTORS
from fincore.contracts.portfolio import ExposureBundle, VolumeExposureBundle
from fincore.contracts.time_series import align_time_series
from fincore.exceptions import ValidationError

__all__ = [
    "compute_cap_exposures",
    "compute_sector_exposures",
    "compute_style_factor_exposures",
    "compute_volume_exposures",
    "extract_pos",
    "get_long_short_notional",
    "get_long_short_pos",
    "get_max_median_position_concentration",
    "get_percent_alloc",
    "get_sector_exposures",
    "get_top_long_short_abs",
    "gross_lev",
    "stack_positions",
]


def get_percent_alloc(values: pd.DataFrame) -> pd.DataFrame:
    """Determine a portfolio's allocations.

    Parameters
    ----------
    values : pd.DataFrame
        Contains position values or amounts.

    Returns
    -------
    pd.DataFrame
        Positions and their allocations.
    """
    result = values.divide(values.sum(axis="columns"), axis="index")
    return result.replace([np.inf, -np.inf], np.nan)


def get_top_long_short_abs(
    positions: pd.DataFrame,
    top: int = 10,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Find the top long, short, and absolute positions.

    Parameters
    ----------
    positions : pd.DataFrame
        The positions that the strategy takes over time.
    top : int, optional
        How many of each to find (default 10).

    Returns
    -------
    df_top_long : pd.DataFrame
        Top long positions.
    df_top_short : pd.DataFrame
        Top short positions.
    df_top_abs : pd.DataFrame
        Top absolute positions.
    """
    positions = positions.drop("cash", axis="columns")
    df_max = positions.max()
    df_min = positions.min()
    df_abs_max = positions.abs().max()
    df_top_long = df_max[df_max > 0].nlargest(top)
    df_top_short = df_min[df_min < 0].nsmallest(top)
    df_top_abs = df_abs_max.nlargest(top)
    return df_top_long, df_top_short, df_top_abs


def get_max_median_position_concentration(positions: pd.DataFrame) -> pd.DataFrame:
    """Find the max and median long and short position concentrations.

    Parameters
    ----------
    positions : pd.DataFrame
        The positions that the strategy takes over time.

    Returns
    -------
    pd.DataFrame
        Columns are the max long, max short, median long, and median short
        position concentrations. Rows are time periods.
    """
    expos = get_percent_alloc(positions)
    expos = expos.drop("cash", axis=1)

    longs = expos.where(expos > 0)
    shorts = expos.where(expos < 0)

    alloc_summary = pd.DataFrame()
    alloc_summary["max_long"] = longs.max(axis=1)
    alloc_summary["median_long"] = longs.median(axis=1)
    alloc_summary["median_short"] = shorts.median(axis=1)
    alloc_summary["max_short"] = shorts.min(axis=1)

    return alloc_summary


def extract_pos(positions: pd.DataFrame, cash: pd.Series) -> pd.DataFrame:
    """Extract position values from get_backtest() output.

    Convert the backtest object's positions and cash series into a
    DataFrame of daily net position values (one column per symbol).

    Parameters
    ----------
    positions : pd.DataFrame
        timeseries containing one row per symbol (and potentially
        duplicate datetime indices) and columns for amount and
        last_sale_price.
    cash : pd.Series
        timeseries containing cash in the portfolio.

    Returns
    -------
    pd.DataFrame
        Daily net position values.
    """
    positions = positions.copy()
    positions["values"] = positions.amount * positions.last_sale_price

    cash.name = "cash"

    values = positions.reset_index().pivot_table(
        index="index",
        columns="sid",
        values="values",
    )

    values = values.join(cash).fillna(0)

    # NOTE: Set the name of DataFrame.columns to sid, to match the behavior
    # of DataFrame.join in earlier versions of pandas.
    values.columns.name = "sid"

    return values


def get_sector_exposures(
    positions: pd.DataFrame,
    symbol_sector_map: dict[Any, Any] | pd.Series,
) -> pd.DataFrame:
    """Sum position exposures by sector.

    Parameters
    ----------
    positions : pd.DataFrame
        Contains position values or amounts.
    symbol_sector_map : dict or pd.Series
        Security identifier to sector mapping.

    Returns
    -------
    pd.DataFrame
        Sectors and their allocations.
    """
    import warnings

    cash = positions["cash"]
    positions = positions.drop("cash", axis=1)

    unmapped_pos = np.setdiff1d(
        positions.columns.values,
        list(symbol_sector_map.keys()),
    )
    if len(unmapped_pos) > 0:
        warn_message = (
            "Warning: Symbols {} have no sector mapping. They will not be included in sector allocations"
        ).format(", ".join(map(str, unmapped_pos)))
        warnings.warn(warn_message, UserWarning, stacklevel=2)

    sector_exp = positions.T.groupby(by=symbol_sector_map).sum().T

    sector_exp["cash"] = cash

    return sector_exp


def get_long_short_notional(positions: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """Determine the long and short notionals in a portfolio.

    Parameters
    ----------
    positions : pd.DataFrame
        The positions that the strategy takes over time.

    Returns
    -------
    tuple of pd.Series
        Positive long notional and absolute short notional.
    """
    positions = positions.copy()

    if "cash" in positions.columns:
        positions = positions.drop("cash", axis=1)

    longs = positions.where(positions > 0, 0).sum(axis=1)
    shorts = positions.where(positions < 0, 0).abs().sum(axis=1)

    return longs, shorts


def get_long_short_pos(positions: pd.DataFrame) -> pd.DataFrame:
    """Return pyfolio-compatible long, short, and net portfolio exposure.

    Exposures are normalized by net liquidation value. Short exposure retains
    its negative sign, and the cash column participates only in the
    denominator. Zero liquidation rows return finite zero exposures.
    """

    values = positions.copy()
    assets = values.drop(columns="cash", errors="ignore")
    longs = assets.where(assets > 0).sum(axis="columns").fillna(0.0)
    shorts = assets.where(assets < 0).sum(axis="columns").fillna(0.0)
    cash = values["cash"] if "cash" in values else pd.Series(0.0, index=values.index)
    net_liquidation = longs + shorts + cash
    denominator = net_liquidation.replace(0, np.nan)

    result = pd.DataFrame(
        {
            "long": longs.divide(denominator),
            "short": shorts.divide(denominator),
        },
        index=values.index,
    ).fillna(0.0)
    result["net exposure"] = result["long"] + result["short"]
    return result


def _align_portfolio_panels(
    left: pd.DataFrame,
    right: pd.DataFrame,
    *,
    left_name: str,
    right_name: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Validate asset labels and align two portfolio panels by date."""

    for name, panel in ((left_name, left), (right_name, right)):
        if not panel.columns.is_unique:
            duplicates = panel.columns[panel.columns.duplicated()].tolist()
            raise ValidationError(
                "duplicate asset columns are ambiguous",
                param_name=name,
                value=duplicates,
            )
    aligned = align_time_series(left, right, policy="inner")
    return cast("pd.DataFrame", aligned[0]), cast("pd.DataFrame", aligned[1])


def compute_style_factor_exposures(
    positions: pd.DataFrame,
    risk_factor: pd.DataFrame,
) -> pd.Series:
    """Return style factor exposure of an algorithm's positions.

    Parameters
    ----------
    positions : pd.DataFrame
        Daily equity positions of algorithm, in dollars.
    risk_factor : pd.DataFrame
        Daily risk factor per asset.

    Returns
    -------
    pd.Series
        Total style factor exposure over time.
    """
    positions, risk_factor = _align_portfolio_panels(
        positions,
        risk_factor,
        left_name="positions",
        right_name="risk_factor",
    )
    assets = positions.drop(columns="cash", errors="ignore")
    factors = risk_factor.drop(columns="cash", errors="ignore").reindex(columns=assets.columns)

    gross = assets.abs().sum(axis="columns").replace(0, np.nan)
    exposure = assets.multiply(factors).sum(axis="columns", skipna=True).divide(gross)
    return cast("pd.Series", exposure.replace([np.inf, -np.inf], np.nan).fillna(0.0))


def compute_sector_exposures(
    positions: pd.DataFrame,
    sectors: pd.DataFrame,
    sector_dict: dict[Any, str] | None = None,
) -> ExposureBundle:
    """Return sector exposures of an algorithm's positions.

    Parameters
    ----------
    positions : pd.DataFrame
        Daily equity positions of algorithm, in dollars.
    sectors : pd.DataFrame
        Daily sector identifiers per asset.
    sector_dict : dict, optional
        Dictionary mapping security identifiers to sectors.

    Returns
    -------
    ExposureBundle
        Named exposure tables in the frozen sector order.
    """
    sector_map = dict(SECTORS if sector_dict is None else sector_dict)
    sector_names = list(sector_map.values())
    if len(sector_names) != len(set(sector_names)):
        raise ValidationError(
            "duplicate sector display names are ambiguous",
            param_name="sector_dict",
            value=sector_names,
        )
    positions, sector_panel = _align_portfolio_panels(
        positions,
        sectors,
        left_name="positions",
        right_name="sectors",
    )
    assets = positions.drop(columns="cash", errors="ignore")
    sector_panel = sector_panel.reindex(columns=assets.columns)

    total_long = assets.where(assets > 0).sum(axis="columns").replace(0, np.nan)
    total_short = assets.where(assets < 0).abs().sum(axis="columns").replace(0, np.nan)
    total_gross = assets.abs().sum(axis="columns").replace(0, np.nan)

    long: dict[str, pd.Series] = {}
    short: dict[str, pd.Series] = {}
    gross: dict[str, pd.Series] = {}
    net: dict[str, pd.Series] = {}
    for sector_id, sector_name in sector_map.items():
        in_sector = assets.where(sector_panel == sector_id)
        long[sector_name] = in_sector.where(in_sector > 0).sum(axis="columns").divide(total_long)
        short[sector_name] = in_sector.where(in_sector < 0).sum(axis="columns").divide(total_short)
        gross[sector_name] = in_sector.abs().sum(axis="columns").divide(total_gross)
        net[sector_name] = long[sector_name] - short[sector_name]

    def frame(values: dict[str, pd.Series]) -> pd.DataFrame:
        return pd.DataFrame(values, index=assets.index).replace([np.inf, -np.inf], np.nan).fillna(0.0)

    return ExposureBundle(
        long=frame(long),
        short=frame(short),
        gross=frame(gross),
        net=frame(net),
    )


def compute_cap_exposures(
    positions: pd.DataFrame,
    caps: pd.DataFrame,
) -> ExposureBundle:
    """Compute market capitalization exposures.

    Parameters
    ----------
    positions : pd.DataFrame
        Daily equity positions of algorithm, in dollars.
    caps : pd.DataFrame
        Daily market capitalization per asset.

    Returns
    -------
    ExposureBundle
        Named exposure tables in the frozen cap-bucket order.
    """
    positions, cap_panel = _align_portfolio_panels(
        positions,
        caps,
        left_name="positions",
        right_name="caps",
    )
    assets = positions.drop(columns="cash", errors="ignore")
    cap_panel = cap_panel.reindex(columns=assets.columns)

    total_long = assets.where(assets > 0).sum(axis="columns").replace(0, np.nan)
    total_short = assets.where(assets < 0).abs().sum(axis="columns").replace(0, np.nan)
    total_gross = assets.abs().sum(axis="columns").replace(0, np.nan)

    long: dict[str, pd.Series] = {}
    short: dict[str, pd.Series] = {}
    gross: dict[str, pd.Series] = {}
    net: dict[str, pd.Series] = {}
    for bucket_name, (lower, upper) in CAP_BUCKETS.items():
        in_bucket = assets.where((cap_panel >= lower) & (cap_panel <= upper))
        long[bucket_name] = in_bucket.where(in_bucket > 0).sum(axis="columns").divide(total_long)
        short[bucket_name] = in_bucket.where(in_bucket < 0).sum(axis="columns").divide(total_short)
        gross[bucket_name] = in_bucket.abs().sum(axis="columns").divide(total_gross)
        net[bucket_name] = long[bucket_name] - short[bucket_name]

    def frame(values: dict[str, pd.Series]) -> pd.DataFrame:
        return pd.DataFrame(values, index=assets.index).replace([np.inf, -np.inf], np.nan).fillna(0.0)

    return ExposureBundle(
        long=frame(long),
        short=frame(short),
        gross=frame(gross),
        net=frame(net),
    )


def compute_volume_exposures(
    shares_held: pd.DataFrame,
    volumes: pd.DataFrame,
    percentile: float,
) -> VolumeExposureBundle:
    """Compute volume-based liquidity exposures.

    Parameters
    ----------
    shares_held : pd.DataFrame
        Number of shares held per security.
    volumes : pd.DataFrame
        Daily trading volumes per security. Every nonzero held share must have
        a matching finite, positive volume observation.
    percentile : float
        Threshold percentile for days-to-liquidate.

    Returns
    -------
    VolumeExposureBundle
        Long, short, and gross percentile exposure series.
    """
    shares, aligned_volumes = _align_portfolio_panels(
        shares_held,
        volumes,
        left_name="shares_held",
        right_name="volumes",
    )

    active = shares_held.notna() & shares_held.ne(0)
    volume_lookup = volumes.reindex(index=shares_held.index, columns=shares_held.columns)
    numeric_lookup = volume_lookup.apply(pd.to_numeric, errors="coerce")
    valid_volume = numeric_lookup.notna() & numeric_lookup.gt(0) & np.isfinite(numeric_lookup)
    invalid = active & ~valid_volume
    if invalid.to_numpy().any():
        rows, columns = np.where(invalid.to_numpy())
        locations = [
            (shares_held.index[row], shares_held.columns[column]) for row, column in zip(rows, columns, strict=True)
        ]
        raise ValidationError(
            "active nonzero shares require matching finite positive volume observations",
            param_name="volumes",
            value=locations,
        )

    common_columns = shares.columns.intersection(aligned_volumes.columns, sort=False)
    shares = shares.loc[:, common_columns].replace(0, np.nan)
    aligned_volumes = aligned_volumes.loc[:, common_columns].apply(pd.to_numeric, errors="coerce").replace(0, np.nan)

    def percentile_exposure(values: pd.DataFrame) -> pd.Series:
        if values.empty:
            return pd.Series(0.0, index=values.index, dtype=float)
        fraction = values.divide(aligned_volumes)
        result = 100.0 * fraction.quantile(percentile, axis="columns")
        return result.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    long = percentile_exposure(shares.where(shares > 0))
    short = percentile_exposure(-shares.where(shares < 0))
    gross = percentile_exposure(shares.abs())
    return VolumeExposureBundle(long=long, short=short, gross=gross)


def gross_lev(positions: pd.DataFrame) -> pd.Series:
    """Calculate the gross leverage of a strategy.

    Parameters
    ----------
    positions : pd.DataFrame
        Daily net position values.

    Returns
    -------
    pd.Series
        Gross leverage.
    """
    exposure = positions.drop("cash", axis=1).abs().sum(axis=1)
    total = positions.sum(axis=1)
    result = exposure / total
    return result.replace([np.inf, -np.inf], np.nan)


def stack_positions(
    positions: pd.DataFrame,
    pos_in_dollars: bool = True,
) -> pd.Series:
    """Stack positions into a multi-index Series.

    Parameters
    ----------
    positions : pd.DataFrame
        Daily positions with tickers as columns.
    pos_in_dollars : bool, optional
        Whether positions are in dollars (default True).

    Returns
    -------
    pd.Series
        Stacked positions with MultiIndex (dt, ticker).
    """
    positions = positions.copy()

    if "cash" in positions.columns:
        positions = positions.drop("cash", axis=1)

    stacked = cast("pd.Series", positions.stack())
    stacked.index.names = ["dt", "ticker"]

    return stacked
