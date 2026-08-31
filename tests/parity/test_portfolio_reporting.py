"""Direct portfolio-report contracts replacing display-only tear-sheet tests."""

from __future__ import annotations

import pandas as pd


def test_portfolio_report_emits_named_sections_and_tables_without_a_facade() -> None:
    from fincore.report.portfolio.compute import build_portfolio_report

    dates = pd.date_range("2024-01-02", periods=8, freq="B")
    returns = pd.Series([0.01, -0.02, 0.003, 0.002, -0.001, 0.004, 0.0, 0.002], index=dates)
    positions = pd.DataFrame({"AAA": 100.0, "BBB": -30.0, "cash": 80.0}, index=dates)
    transactions = pd.DataFrame(
        {"symbol": ["AAA", "BBB"], "amount": [2.0, -1.0], "price": [10.0, 20.0]},
        index=pd.DatetimeIndex([dates[2], dates[5]]),
    )

    document = build_portfolio_report(returns, positions=positions, transactions=transactions, rolling_window=3)

    assert tuple(section.key for section in document.sections) == ("performance", "portfolio", "transactions")
    performance = document.section("performance")
    assert "drawdowns" in performance.tables
    assert "cumulative_returns" in performance.series
    assert document.section("portfolio").metrics["asset_count"] == 2
    assert document.section("transactions").metrics["transaction_count"] == 2
