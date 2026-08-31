"""Explicit canonical operation declarations for portfolio analytics."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from fincore.runtime.specs import OperationSpec, make_operations_provider

from .capacity import assess_liquidity
from .positions import (
    compute_cap_exposures,
    compute_sector_exposures,
    compute_style_factor_exposures,
    compute_volume_exposures,
    extract_pos,
    get_long_short_notional,
    get_long_short_pos,
    get_max_median_position_concentration,
    get_percent_alloc,
    get_sector_exposures,
    get_top_long_short_abs,
    gross_lev,
    stack_positions,
)
from .round_trips import (
    add_closing_transactions,
    agg_all_long_short,
    apply_sector_mappings_to_round_trips,
    extract_round_trips,
    gen_round_trip_stats,
    groupby_consecutive,
)
from .transactions import (
    adjust_returns_for_slippage,
    apply_slippage_penalty,
    daily_txns_with_bar_data,
    days_to_liquidate_positions,
    get_low_liquidity_transactions,
    get_max_days_to_liquidate_by_ticker,
    get_turnover,
    get_txn_vol,
    make_transaction_frame,
    map_transaction,
)

if TYPE_CHECKING:
    from collections.abc import Callable

__all__ = ["operations"]

_BINDINGS: tuple[tuple[str, Callable[..., Any]], ...] = (
    ("portfolio.capacity.assess_liquidity", assess_liquidity),
    ("portfolio.positions.compute_cap_exposures", compute_cap_exposures),
    ("portfolio.positions.compute_sector_exposures", compute_sector_exposures),
    ("portfolio.positions.compute_style_factor_exposures", compute_style_factor_exposures),
    ("portfolio.positions.compute_volume_exposures", compute_volume_exposures),
    ("portfolio.positions.extract_pos", extract_pos),
    ("portfolio.positions.get_long_short_notional", get_long_short_notional),
    ("portfolio.positions.get_long_short_pos", get_long_short_pos),
    ("portfolio.positions.get_max_median_position_concentration", get_max_median_position_concentration),
    ("portfolio.positions.get_percent_alloc", get_percent_alloc),
    ("portfolio.positions.get_sector_exposures", get_sector_exposures),
    ("portfolio.positions.get_top_long_short_abs", get_top_long_short_abs),
    ("portfolio.positions.gross_lev", gross_lev),
    ("portfolio.positions.stack_positions", stack_positions),
    ("portfolio.round_trips.add_closing_transactions", add_closing_transactions),
    ("portfolio.round_trips.agg_all_long_short", agg_all_long_short),
    ("portfolio.round_trips.apply_sector_mappings_to_round_trips", apply_sector_mappings_to_round_trips),
    ("portfolio.round_trips.extract_round_trips", extract_round_trips),
    ("portfolio.round_trips.gen_round_trip_stats", gen_round_trip_stats),
    ("portfolio.round_trips.groupby_consecutive", groupby_consecutive),
    ("portfolio.transactions.adjust_returns_for_slippage", adjust_returns_for_slippage),
    ("portfolio.transactions.apply_slippage_penalty", apply_slippage_penalty),
    ("portfolio.transactions.daily_txns_with_bar_data", daily_txns_with_bar_data),
    ("portfolio.transactions.days_to_liquidate_positions", days_to_liquidate_positions),
    ("portfolio.transactions.get_low_liquidity_transactions", get_low_liquidity_transactions),
    ("portfolio.transactions.get_max_days_to_liquidate_by_ticker", get_max_days_to_liquidate_by_ticker),
    ("portfolio.transactions.get_turnover", get_turnover),
    ("portfolio.transactions.get_txn_vol", get_txn_vol),
    ("portfolio.transactions.make_transaction_frame", make_transaction_frame),
    ("portfolio.transactions.map_transaction", map_transaction),
)
_OPERATIONS = tuple(
    OperationSpec(
        operation_id=operation_id,
        capability_id=operation_id,
        domain="portfolio",
        callable=callable_,
        provenance={"owner": "portfolio", "kernel_module": callable_.__module__},
    )
    for operation_id, callable_ in _BINDINGS
)


operations = make_operations_provider(_OPERATIONS)
