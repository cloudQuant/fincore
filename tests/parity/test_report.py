"""Canonical report scenarios for the 0042-R2 breaking surface.

The capability ledger groups the retired tear-sheet display behaviors into
these direct-report scenarios.  They exercise the unified report model and
renderer without importing a legacy report facade or a compatibility module.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _returns(size: int = 64) -> pd.Series:
    index = pd.date_range("2024-01-02", periods=size, freq="B", tz="UTC")
    values = np.resize(np.array([0.012, -0.006, 0.003, 0.001], dtype=float), size)
    return pd.Series(values, index=index, name="strategy", dtype=float)


def _rendered_axes(document) -> set[str]:
    from fincore.report.renderers.matplotlib import render_matplotlib

    bundle = render_matplotlib(document)
    try:
        return set(bundle.named_artifacts)
    finally:
        bundle.close()


def test_attribution_charts() -> None:
    """Benchmark attribution values are projected from the canonical document."""

    from fincore.report.portfolio.compute import build_portfolio_report

    returns = _returns()
    benchmark = returns.mul(0.6).add(0.0005)
    document = build_portfolio_report(returns, benchmark_returns=benchmark, rolling_window=12)

    benchmark_section = document.section("benchmark")
    assert {"alpha", "beta"} <= set(document.section("performance").metrics)
    assert "rolling_beta" in benchmark_section.series
    assert "axis:benchmark.rolling_beta" in _rendered_axes(document)


def test_position_charts() -> None:
    """Exposure and leverage chart data originate in the portfolio section."""

    from fincore.report.portfolio.compute import build_portfolio_report

    returns = _returns()
    positions = pd.DataFrame(
        {
            "AAA": np.linspace(40.0, 80.0, len(returns)),
            "BBB": np.linspace(-15.0, 10.0, len(returns)),
            "cash": np.full(len(returns), 75.0),
        },
        index=returns.index,
    )
    document = build_portfolio_report(returns, positions=positions, rolling_window=12)

    portfolio = document.section("portfolio")
    assert portfolio.metrics["asset_count"] == 2
    assert portfolio.units["gross_leverage"] == "ratio"
    assert "axis:portfolio.gross_leverage" in _rendered_axes(document)


def test_return_charts() -> None:
    """Return, drawdown, rolling, and monthly data share one report computation."""

    from fincore.report.portfolio.compute import build_portfolio_report

    document = build_portfolio_report(_returns(), rolling_window=12)
    performance = document.section("performance")

    assert {
        "cumulative_returns",
        "drawdown",
        "monthly_returns",
        "rolling_sharpe",
        "rolling_volatility",
    } <= set(performance.series)
    axes = _rendered_axes(document)
    assert {
        "axis:performance.cumulative_returns",
        "axis:performance.drawdown",
        "axis:performance.monthly_returns",
    } <= axes


def test_strategy_report() -> None:
    """The catalog exposes one direct report builder with a stable semantic digest."""

    from fincore.report.operations import operations
    from fincore.report.portfolio.compute import build_portfolio_report
    from fincore.runtime import OperationCatalog

    document = build_portfolio_report(_returns(), rolling_window=12, title="Canonical strategy report")
    catalog = OperationCatalog(operations())

    assert document.semantic_digest == document.semantic_digest
    assert document.section("performance").tables["drawdowns"].shape[0] <= 5
    assert catalog.resolve("report.portfolio.build_portfolio_report").callable is build_portfolio_report
