"""Portfolio liquidity and capacity workflows built from direct kernels."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pandas as pd

from .transactions import (
    days_to_liquidate_positions,
    get_low_liquidity_transactions,
    get_max_days_to_liquidate_by_ticker,
)

__all__ = ["CapacityAssessment", "CapacityConfig", "assess_liquidity"]


@dataclass(frozen=True, slots=True)
class CapacityConfig:
    """Explicit assumptions used by a liquidity capacity assessment."""

    max_bar_consumption: float = 0.2
    capital_base: float = 1_000_000.0
    mean_volume_window: int = 5
    last_n_days: int | None = None

    def __post_init__(self) -> None:
        if not 0 < self.max_bar_consumption <= 1:
            raise ValueError("max_bar_consumption must be in (0, 1]")
        if self.capital_base <= 0:
            raise ValueError("capital_base must be positive")
        if self.mean_volume_window < 1:
            raise ValueError("mean_volume_window must be positive")
        if self.last_n_days is not None and self.last_n_days < 1:
            raise ValueError("last_n_days must be positive when provided")


@dataclass(frozen=True, slots=True)
class CapacityAssessment:
    """Named results of one portfolio liquidity assessment."""

    liquidation_days: pd.DataFrame
    ticker_maximums: pd.DataFrame
    low_liquidity_transactions: pd.DataFrame


_DEFAULT_CONFIG = CapacityConfig()


def assess_liquidity(
    positions: pd.DataFrame,
    transactions: pd.DataFrame,
    market_data: dict[str, pd.DataFrame],
    config: CapacityConfig | None = None,
) -> CapacityAssessment:
    """Compute the full capacity model without invoking a renderer or façade."""
    active_config = config or _DEFAULT_CONFIG
    return CapacityAssessment(
        liquidation_days=days_to_liquidate_positions(
            positions,
            market_data,
            max_bar_consumption=active_config.max_bar_consumption,
            capital_base=active_config.capital_base,
            mean_volume_window=active_config.mean_volume_window,
        ),
        ticker_maximums=get_max_days_to_liquidate_by_ticker(
            positions,
            market_data,
            max_bar_consumption=active_config.max_bar_consumption,
            capital_base=active_config.capital_base,
            mean_volume_window=active_config.mean_volume_window,
            last_n_days=active_config.last_n_days,
        ),
        low_liquidity_transactions=get_low_liquidity_transactions(
            transactions,
            market_data,
            last_n_days=active_config.last_n_days,
        ),
    )
