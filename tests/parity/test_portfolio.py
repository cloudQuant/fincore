"""Capability-level portfolio scenarios independent of the Pyfolio façade."""

from __future__ import annotations

import pandas as pd


def test_position_exposure_contract_preserves_signed_short_and_net_columns() -> None:
    from fincore.portfolio.positions import get_long_short_pos

    positions = pd.DataFrame(
        {"cash": [20.0], "long": [100.0], "short": [-40.0]},
        index=pd.DatetimeIndex(["2024-01-02"], tz="UTC"),
    )

    result = get_long_short_pos(positions)

    assert list(result.columns) == ["long", "short", "net exposure"]
    assert result.loc[positions.index[0], "long"] == 100.0 / 80.0
    assert result.loc[positions.index[0], "short"] == -40.0 / 80.0
    assert result.loc[positions.index[0], "net exposure"] == 60.0 / 80.0


def test_round_trip_capability_matches_fifo_pnl_and_duration_contract() -> None:
    from fincore.portfolio.round_trips import extract_round_trips

    transactions = pd.DataFrame(
        {
            "symbol": ["A", "A"],
            "amount": [2, -2],
            "price": [10.0, 12.0],
        },
        index=pd.DatetimeIndex(["2024-01-02", "2024-01-04"], tz="UTC"),
    )

    result = extract_round_trips(transactions)

    assert result.loc[0, "pnl"] == 4.0
    assert result.loc[0, "rt_returns"] == 0.2
    assert result.loc[0, "duration"] == pd.Timedelta(days=2)
