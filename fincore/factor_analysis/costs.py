"""Auditable transaction-cost and capacity accounting for enhanced factor portfolios.

The strict Alphalens facade deliberately does not import this module.  These
functions operate on explicitly supplied, gross-normalized enhanced weights,
simple return periods, and dollar-volume observations; they never infer an
execution policy or silently fill missing liquidity/borrow inputs.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, ClassVar, Hashable, cast

import numpy as np
import pandas as pd

__all__ = [
    "FactorCapacityResult",
    "FactorCostModel",
    "FactorCostResult",
    "apply_factor_costs",
    "estimate_factor_capacity",
]


_GROSS_WEIGHT_TOLERANCE = 1.0e-12


@dataclass(frozen=True, slots=True)
class FactorCostModel:
    """Explicit simple-return cost assumptions for one enhanced rebalance period.

    ``half_spread_bps`` is paid once for each side of the absolute weight
    change.  ``impact_coefficient * participation ** impact_exponent`` is an
    additional temporary-impact rate on each traded weight.  Both are simple
    return costs, not estimates of realised execution quality.  Borrow rates
    are supplied separately as per-period simple costs for short exposure.
    """

    half_spread_bps: float
    impact_coefficient: float
    max_participation: float
    impact_exponent: float = 0.5

    def __post_init__(self) -> None:
        for name in ("half_spread_bps", "impact_coefficient", "impact_exponent", "max_participation"):
            value = getattr(self, name)
            if not isinstance(value, (int, float, np.number)) or isinstance(value, bool) or not np.isfinite(value):
                raise ValueError(f"{name} must be a finite number")
            object.__setattr__(self, name, float(value))
        if self.half_spread_bps < 0.0:
            raise ValueError("half_spread_bps must be non-negative")
        if self.impact_coefficient < 0.0:
            raise ValueError("impact_coefficient must be non-negative")
        if self.impact_exponent <= 0.0:
            raise ValueError("impact_exponent must be positive")
        if not 0.0 < self.max_participation <= 1.0:
            raise ValueError("max_participation must be in (0, 1]")


@dataclass(frozen=True, slots=True)
class FactorCapacityResult:
    """Capacity bound implied by one-way trades and per-date dollar volume."""

    maximum_portfolio_value: float
    maximum_portfolio_value_by_date: pd.Series
    trade_weights: pd.DataFrame
    max_participation: float
    binding_date: pd.Timestamp
    binding_asset: Hashable

    _DATA_FIELDS: ClassVar[frozenset[str]] = frozenset({"maximum_portfolio_value_by_date", "trade_weights"})

    def __post_init__(self) -> None:
        for name in self._DATA_FIELDS:
            value = object.__getattribute__(self, name)
            object.__setattr__(self, name, value.copy(deep=True))
        object.__setattr__(self, "maximum_portfolio_value", float(self.maximum_portfolio_value))
        object.__setattr__(self, "max_participation", float(self.max_participation))
        object.__setattr__(self, "binding_date", pd.Timestamp(self.binding_date))

    def __getattribute__(self, name: str) -> Any:
        value = object.__getattribute__(self, name)
        if name in object.__getattribute__(self, "_DATA_FIELDS"):
            return value.copy(deep=True)
        return value


@dataclass(frozen=True, slots=True)
class FactorCostResult:
    """A reconcilable gross-to-net enhanced factor portfolio ledger."""

    model: FactorCostModel
    portfolio_value: float
    gross_returns: pd.Series
    net_returns: pd.Series
    turnover: pd.Series
    spread_cost: pd.Series
    impact_cost: pd.Series
    borrow_cost: pd.Series
    total_cost: pd.Series
    trade_weights: pd.DataFrame
    participation: pd.DataFrame
    capacity: FactorCapacityResult

    _DATA_FIELDS: ClassVar[frozenset[str]] = frozenset(
        {
            "gross_returns",
            "net_returns",
            "turnover",
            "spread_cost",
            "impact_cost",
            "borrow_cost",
            "total_cost",
            "trade_weights",
            "participation",
        }
    )

    def __post_init__(self) -> None:
        if not isinstance(self.model, FactorCostModel):
            raise TypeError("model must be a FactorCostModel")
        if not isinstance(self.portfolio_value, (int, float, np.number)) or isinstance(self.portfolio_value, bool):
            raise ValueError("portfolio_value must be a positive finite number")
        if not np.isfinite(self.portfolio_value) or self.portfolio_value <= 0.0:
            raise ValueError("portfolio_value must be a positive finite number")
        for name in self._DATA_FIELDS:
            value = object.__getattribute__(self, name)
            object.__setattr__(self, name, value.copy(deep=True))
        object.__setattr__(self, "portfolio_value", float(self.portfolio_value))

    def __getattribute__(self, name: str) -> Any:
        value = object.__getattribute__(self, name)
        if name in object.__getattribute__(self, "_DATA_FIELDS"):
            return value.copy(deep=True)
        return value


def _normalise_weights(weights: pd.Series) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Validate weights and return wide positions/trades with sparse entries as zero.

    A factor portfolio Series commonly omits an asset once it leaves that
    date's eligible universe.  For turnover and capacity accounting, that is
    a zero position rather than an unknown position: entering and leaving the
    sparse ledger must create an explicit trade, never an unpriced ``NaN``.
    """

    if not isinstance(weights, pd.Series):
        raise TypeError("weights must be a pandas Series")
    if not isinstance(weights.index, pd.MultiIndex) or weights.index.nlevels != 2:
        raise ValueError("weights must use a two-level (date, asset) MultiIndex")
    if weights.index.has_duplicates:
        raise ValueError("weights index must be unique")
    copied = weights.copy(deep=True)
    try:
        dates = pd.DatetimeIndex(copied.index.get_level_values(0), name="date")
        values = pd.to_numeric(copied, errors="raise").astype(float)
    except (TypeError, ValueError) as error:
        raise ValueError("weights must use datetime dates and numeric values") from error
    if dates.hasnans or not np.isfinite(values.to_numpy(dtype=float, copy=False)).all():
        raise ValueError("weights must use valid dates and finite values")
    copied = pd.Series(
        values.to_numpy(dtype=float, copy=True),
        index=pd.MultiIndex.from_arrays((dates, copied.index.get_level_values(1)), names=("date", "asset")),
        name="weight",
        dtype=float,
    ).sort_index()
    positions = copied.unstack("asset").sort_index().fillna(0.0)
    gross = positions.abs().sum(axis=1)
    if not np.all(np.isclose(gross.to_numpy(dtype=float), 1.0, rtol=0.0, atol=_GROSS_WEIGHT_TOLERANCE)):
        raise ValueError("weights must be gross-normalized to one on every date")
    trades = positions.diff().abs()
    trades.iloc[0] = positions.iloc[0].abs()
    return positions, trades.astype(float)


def _aligned_numeric_panel(
    value: pd.DataFrame,
    *,
    dates: pd.DatetimeIndex,
    assets: pd.Index,
    name: str,
    strictly_positive: bool,
) -> pd.DataFrame:
    """Copy a labelled numeric panel and require full factor-portfolio coverage."""

    if not isinstance(value, pd.DataFrame):
        raise TypeError(f"{name} must be a pandas DataFrame")
    if value.index.has_duplicates or value.columns.has_duplicates:
        raise ValueError(f"{name} index and columns must be unique")
    try:
        copied = value.copy(deep=True)
        copied.index = pd.DatetimeIndex(copied.index, name="date")
        copied = copied.apply(pd.to_numeric, errors="raise").astype(float)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must use datetime dates and numeric values") from error
    missing_dates = dates.difference(copied.index)
    missing_assets = assets.difference(copied.columns)
    if not missing_dates.empty or not missing_assets.empty:
        raise ValueError(f"{name} must cover every weight date and asset")
    aligned = copied.reindex(index=dates, columns=assets)
    values = aligned.to_numpy(dtype=float, copy=False)
    if not np.isfinite(values).all():
        raise ValueError(f"{name} must be finite")
    if strictly_positive and np.any(values <= 0.0):
        raise ValueError(f"{name} must be strictly positive")
    if not strictly_positive and np.any(values < 0.0):
        raise ValueError(f"{name} must be non-negative")
    return cast("pd.DataFrame", aligned)


def _aligned_borrow_availability(
    value: pd.DataFrame,
    *,
    dates: pd.DatetimeIndex,
    assets: pd.Index,
) -> pd.DataFrame:
    """Require a labelled boolean borrow-availability ledger."""

    if not isinstance(value, pd.DataFrame):
        raise TypeError("borrow_available must be a pandas DataFrame")
    if value.index.has_duplicates or value.columns.has_duplicates:
        raise ValueError("borrow_available index and columns must be unique")
    try:
        copied = value.copy(deep=True)
        copied.index = pd.DatetimeIndex(copied.index, name="date")
    except (TypeError, ValueError) as error:
        raise ValueError("borrow_available must use datetime dates") from error
    missing_dates = dates.difference(copied.index)
    missing_assets = assets.difference(copied.columns)
    if not missing_dates.empty or not missing_assets.empty:
        raise ValueError("borrow_available must cover every weight date and asset")
    aligned = copied.reindex(index=dates, columns=assets)
    for item in aligned.to_numpy(dtype=object, copy=False).ravel():
        if not isinstance(item, (bool, np.bool_)):
            raise ValueError("borrow_available must contain only boolean values")
    return aligned.astype(bool)


def _normalise_gross_returns(gross_returns: pd.Series, dates: pd.DatetimeIndex) -> pd.Series:
    """Label-align finite simple returns to the rebalance dates."""

    if not isinstance(gross_returns, pd.Series):
        raise TypeError("gross_returns must be a pandas Series")
    if gross_returns.index.has_duplicates:
        raise ValueError("gross_returns index must be unique")
    try:
        copied = gross_returns.copy(deep=True)
        copied.index = pd.DatetimeIndex(copied.index, name="date")
        copied = pd.to_numeric(copied, errors="raise").astype(float).sort_index()
    except (TypeError, ValueError) as error:
        raise ValueError("gross_returns must use datetime dates and numeric values") from error
    if not copied.index.equals(dates):
        raise ValueError("gross_returns dates must exactly match weights dates")
    if not np.isfinite(copied.to_numpy(dtype=float, copy=False)).all():
        raise ValueError("gross_returns must be finite")
    return copied.rename("gross_return")


def _require_finite_ledger_values(**values: pd.Series | pd.DataFrame) -> None:
    """Reject an arithmetic overflow before exposing an unverifiable ledger."""

    for name, value in values.items():
        numeric = value.to_numpy(dtype=float, copy=False)
        if not np.isfinite(numeric).all():
            raise ValueError(f"cost ledger {name} must remain finite")


def _capacity_from_trade_ledger(
    trades: pd.DataFrame,
    dollar_volume: pd.DataFrame,
    *,
    max_participation: float,
) -> FactorCapacityResult:
    """Return the binding liquidity inequality for every nonzero trade weight."""

    limits = (float(max_participation) * dollar_volume).div(trades.where(trades > 0.0))
    limits = limits.where(trades > 0.0, np.inf)
    by_date = limits.min(axis=1).rename("maximum_portfolio_value")
    if not np.isfinite(by_date.to_numpy(dtype=float, copy=False)).any():
        raise ValueError("weights must contain at least one nonzero trade")
    maximum = float(by_date.min())
    binding_date = pd.Timestamp(by_date.idxmin())
    binding_asset = cast("Hashable", limits.loc[binding_date].idxmin())
    return FactorCapacityResult(
        maximum_portfolio_value=maximum,
        maximum_portfolio_value_by_date=by_date,
        trade_weights=trades,
        max_participation=float(max_participation),
        binding_date=binding_date,
        binding_asset=binding_asset,
    )


def estimate_factor_capacity(
    weights: pd.Series,
    dollar_volume: pd.DataFrame,
    *,
    max_participation: float,
) -> FactorCapacityResult:
    """Estimate the largest portfolio value satisfying every trade participation cap.

    ``dollar_volume`` must use the same currency as the desired portfolio value.
    For each rebalance/asset trade ``|Δw|``, the bound is
    ``max_participation * ADV / |Δw|``.  The result is the minimum bound over
    all traded assets and dates; a zero trade does not create a spurious bound.
    """

    if not isinstance(max_participation, (int, float, np.number)) or isinstance(max_participation, bool):
        raise ValueError("max_participation must be a finite number in (0, 1]")
    if not np.isfinite(max_participation) or not 0.0 < float(max_participation) <= 1.0:
        raise ValueError("max_participation must be a finite number in (0, 1]")
    positions, trades = _normalise_weights(weights)
    volume = _aligned_numeric_panel(
        dollar_volume,
        dates=pd.DatetimeIndex(positions.index, name="date"),
        assets=pd.Index(positions.columns),
        name="dollar_volume",
        strictly_positive=True,
    )
    return _capacity_from_trade_ledger(trades, volume, max_participation=float(max_participation))


def apply_factor_costs(
    gross_returns: pd.Series,
    weights: pd.Series,
    dollar_volume: pd.DataFrame,
    *,
    portfolio_value: float,
    model: FactorCostModel,
    borrow_rates: pd.DataFrame | None = None,
    borrow_available: pd.DataFrame | None = None,
) -> FactorCostResult:
    """Apply a fully labelled, capacity-checked cost ledger to simple returns.

    The first row is an entry from zero holdings.  Later trade weights are
    ``abs(w_t - w_{t-1})``; one-way turnover is half their row sum.  The output
    obeys ``net_return = gross_return - spread - impact - borrow`` exactly.
    Short exposure requires *both* an explicit per-period borrow-rate panel and
    a boolean availability panel.  Missing or unavailable borrow, missing ADV,
    and portfolios above the configured capacity fail closed.
    """

    if not isinstance(model, FactorCostModel):
        raise TypeError("model must be a FactorCostModel")
    if not isinstance(portfolio_value, (int, float, np.number)) or isinstance(portfolio_value, bool):
        raise ValueError("portfolio_value must be a positive finite number")
    portfolio_value = float(portfolio_value)
    if not np.isfinite(portfolio_value) or portfolio_value <= 0.0:
        raise ValueError("portfolio_value must be a positive finite number")

    positions, trades = _normalise_weights(weights)
    dates = pd.DatetimeIndex(positions.index, name="date")
    assets = pd.Index(positions.columns)
    returns = _normalise_gross_returns(gross_returns, dates)
    volume = _aligned_numeric_panel(
        dollar_volume,
        dates=dates,
        assets=assets,
        name="dollar_volume",
        strictly_positive=True,
    )

    short_exposure = (-positions.clip(upper=0.0)).astype(float)
    if bool((short_exposure > 0.0).to_numpy().any()):
        if borrow_rates is None or borrow_available is None:
            raise ValueError("short positions require both borrow_rates and borrow_available")
        rates = _aligned_numeric_panel(
            borrow_rates,
            dates=dates,
            assets=assets,
            name="borrow_rates",
            strictly_positive=False,
        )
        availability = _aligned_borrow_availability(borrow_available, dates=dates, assets=assets)
        if bool(((short_exposure > 0.0) & ~availability).to_numpy().any()):
            raise ValueError("borrow is unavailable for one or more short positions")
        borrow_cost = (short_exposure * rates).sum(axis=1).rename("borrow_cost")
    else:
        if borrow_rates is not None or borrow_available is not None:
            raise ValueError("borrow inputs are only valid when weights contain short exposure")
        borrow_cost = pd.Series(0.0, index=dates, name="borrow_cost")

    capacity = _capacity_from_trade_ledger(trades, volume, max_participation=model.max_participation)
    if portfolio_value > capacity.maximum_portfolio_value:
        raise ValueError(
            f"portfolio_value exceeds maximum capacity {capacity.maximum_portfolio_value:.12g} "
            f"at max_participation={model.max_participation:.6g}"
        )

    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        participation = (trades * portfolio_value / volume).astype(float)
        turnover = trades.sum(axis=1).mul(0.5).rename("turnover")
        spread_cost = trades.sum(axis=1).mul(model.half_spread_bps * 1.0e-4).rename("spread_cost")
        impact_cost = (
            (trades * model.impact_coefficient * participation.pow(model.impact_exponent))
            .sum(axis=1)
            .rename("impact_cost")
        )
        total_cost = (spread_cost + impact_cost + borrow_cost).rename("total_cost")
        net_returns = (returns - total_cost).rename("net_return")
    _require_finite_ledger_values(
        participation=participation,
        turnover=turnover,
        spread_cost=spread_cost,
        impact_cost=impact_cost,
        borrow_cost=borrow_cost,
        total_cost=total_cost,
        net_returns=net_returns,
    )
    return FactorCostResult(
        model=model,
        portfolio_value=portfolio_value,
        gross_returns=returns,
        net_returns=net_returns,
        turnover=turnover,
        spread_cost=spread_cost,
        impact_cost=impact_cost,
        borrow_cost=borrow_cost,
        total_cost=total_cost,
        trade_weights=trades,
        participation=participation,
        capacity=capacity,
    )
