"""Offline end-to-end coverage for the canonical 0.5 report workflow."""

from __future__ import annotations

import pandas as pd
import pytest

pytestmark = [pytest.mark.integration, pytest.mark.integration_offline]


def test_canonical_portfolio_report_flows_from_inputs_to_catalogued_html_artifact(tmp_path) -> None:
    """Exercise direct report construction, runtime discovery, and offline HTML output."""

    from fincore.report.operations import operations
    from fincore.report.portfolio.compute import build_portfolio_report
    from fincore.report.renderers.html import write_html
    from fincore.runtime import OperationCatalog

    dates = pd.date_range("2024-01-02", periods=8, freq="B")
    returns = pd.Series([0.01, -0.02, 0.003, 0.002, -0.001, 0.004, 0.0, 0.002], index=dates)
    positions = pd.DataFrame({"AAA": 100.0, "BBB": -30.0, "cash": 80.0}, index=dates)
    transactions = pd.DataFrame(
        {"symbol": ["AAA", "BBB"], "amount": [2.0, -1.0], "price": [10.0, 20.0]},
        index=pd.DatetimeIndex([dates[2], dates[5]]),
    )

    document = build_portfolio_report(returns, positions=positions, transactions=transactions, rolling_window=3)
    catalog = OperationCatalog(operations())
    bundle = write_html(document, tmp_path / "portfolio-report.html")

    assert catalog.resolve("report.portfolio.build_portfolio_report").callable is build_portfolio_report
    assert tuple(section.key for section in document.sections) == ("performance", "portfolio", "transactions")
    assert document.section("portfolio").metrics["asset_count"] == 2
    assert bundle.named_artifacts["file"].is_file()
    assert "portfolio" in bundle.named_artifacts["html"].lower()
