"""Direct migration-oracle scenarios for display-only portfolio behaviors."""

from __future__ import annotations

import pandas as pd


def test_display_helpers_emit_named_table_contracts(monkeypatch) -> None:
    """Portfolio display helpers preserve their analytical table boundaries.

    This calls the underlying functions rather than the Pyfolio class facade;
    the final report/portfolio APIs will retain these named-table observables
    without retaining the class or its forwarding methods.
    """
    from fincore.tearsheets import returns as returns_tears
    from fincore.tearsheets import round_trips as round_trip_tears

    observed: list[str] = []

    def record_table(_table, *, name=None, **_kwargs) -> None:
        observed.append(name or "")

    monkeypatch.setattr(returns_tears, "print_table", record_table)
    monkeypatch.setattr(round_trip_tears, "print_table", record_table)

    class Metrics:
        def gen_drawdown_table(self, _returns: pd.Series, *, top: int) -> pd.DataFrame:
            assert top == 1
            return pd.DataFrame({"Net drawdown in %": [-12.5]}, index=["period-1"])

        def gen_round_trip_stats(self, _round_trips: pd.DataFrame) -> dict[str, pd.Series | pd.DataFrame]:
            return {
                "summary": pd.Series({"count": 2.0}),
                "pnl": pd.Series({"mean": 1.0}),
                "duration": pd.Series({"mean": 3.0}),
                "returns": pd.Series({"mean": 0.05}),
                "symbols": pd.DataFrame({"A": [0.05]}, index=["mean"]),
            }

    returns = pd.Series([0.01, -0.02], index=pd.date_range("2024-01-02", periods=2, freq="B"))
    round_trips = pd.DataFrame({"symbol": ["A", "B"], "pnl": [3.0, -1.0]})

    returns_tears.show_worst_drawdown_periods(Metrics(), returns, top=1)
    round_trip_tears.print_round_trip_stats(Metrics(), round_trips)
    round_trip_tears.show_profit_attribution(round_trips)

    assert observed == [
        "Worst drawdown periods",
        "Summary stats",
        "PnL stats",
        "Duration stats",
        "Return stats",
        "Symbol stats",
        "Profitability (PnL / PnL total) per name",
    ]
