"""Canonical report-model contracts for the breaking 0042-R2 surface."""

from __future__ import annotations

import numpy as np
import pandas as pd


def _returns(size: int = 80) -> pd.Series:
    index = pd.date_range("2024-01-02", periods=size, freq="B", tz="UTC")
    values = np.where(np.arange(size) % 3 == 0, -0.002, 0.003)
    return pd.Series(values, index=index, name="strategy", dtype=float)


def test_portfolio_report_is_a_compute_once_document_with_canonical_metric_data() -> None:
    from fincore.report.portfolio.compute import build_portfolio_report

    document = build_portfolio_report(_returns(), rolling_window=20, title="Canonical Portfolio")

    performance = document.section("performance")
    assert document.domain == "portfolio"
    assert document.title == "Canonical Portfolio"
    assert "annual_return" in performance.metrics
    assert "cumulative_returns" in performance.series
    assert performance.units["cumulative_returns"] == "growth_multiple"
    assert performance.legends["cumulative_returns"] == "Strategy"
    assert "drawdowns" in performance.tables
    assert document.semantic_digest == document.semantic_digest


def test_portfolio_report_reuses_direct_portfolio_inputs_without_a_facade_context() -> None:
    from fincore.report.portfolio.compute import build_portfolio_report

    returns = _returns()
    positions = pd.DataFrame(
        {"AAA": 100.0, "BBB": -30.0, "cash": 80.0},
        index=returns.index,
    )
    transactions = pd.DataFrame(
        {"amount": [2.0, -1.0], "price": [10.0, 20.0], "symbol": ["AAA", "BBB"]},
        index=pd.DatetimeIndex([returns.index[5] + pd.Timedelta(hours=9), returns.index[7] + pd.Timedelta(hours=10)]),
    )

    document = build_portfolio_report(returns, positions=positions, transactions=transactions, rolling_window=20)

    portfolio = document.section("portfolio")
    assert "gross_leverage" in portfolio.series
    assert "turnover" in portfolio.series
    assert portfolio.metrics["asset_count"] == 2


def test_portfolio_report_keeps_undefined_zero_return_ratios_as_nan() -> None:
    from fincore.report.portfolio.compute import build_portfolio_report

    returns = pd.Series(0.0, index=pd.date_range("2024-01-02", periods=8, freq="B"))

    document = build_portfolio_report(returns, rolling_window=2)

    metrics = document.section("performance").metrics
    assert metrics["annual_return"] == 0.0
    assert metrics["cumulative_return"] == 0.0
    assert np.isnan(metrics["calmar_ratio"])
    assert np.isnan(metrics["omega_ratio"])


def test_factor_and_risk_report_builders_only_project_precomputed_domain_models() -> None:
    from fincore.factor_analysis.analysis import analyze_factor
    from fincore.report.factor.compute import build_factor_report
    from fincore.report.risk import build_risk_report
    from fincore.risk.report import RiskValidationReport

    dates = pd.bdate_range("2024-01-02", periods=8)
    assets = ["A", "B", "C", "D"]
    index = pd.MultiIndex.from_product((dates, assets), names=("date", "asset"))
    factor_data = pd.DataFrame(
        {
            "factor": np.tile(np.arange(1.0, 5.0), len(dates)),
            "factor_quantile": np.tile([1, 2, 3, 4], len(dates)),
            "1D": np.tile([0.01, -0.01, 0.015, -0.005], len(dates)),
        },
        index=index,
    )
    factor_document = build_factor_report(analyze_factor(factor_data, periods=("1D",), include_portfolio_inputs=False))

    risk_document = build_risk_report(
        RiskValidationReport(
            status="success",
            inputs_digest="digest",
            specification={"confidence_level": 0.95},
            forecast_events=(),
            refits=(),
            diagnostics={"observations": 0},
            backtest=None,
        )
    )

    assert factor_document.section("factor_summary").metrics["forward_period_count"] == 1
    assert "quantile_statistics" in factor_document.section("factor_summary").tables
    assert risk_document.section("risk_validation").metrics["status"] == "success"


def test_report_operations_resolve_to_the_one_direct_compute_functions() -> None:
    from fincore.report.factor.compute import build_factor_report
    from fincore.report.operations import operations
    from fincore.report.portfolio.compute import build_portfolio_report
    from fincore.report.risk import build_risk_report
    from fincore.runtime import OperationCatalog

    catalog = OperationCatalog(operations())

    assert catalog.resolve("report.portfolio.build_portfolio_report").callable is build_portfolio_report
    assert catalog.resolve("report.factor.build_factor_report").callable is build_factor_report
    assert catalog.resolve("report.risk.build_risk_report").callable is build_risk_report
