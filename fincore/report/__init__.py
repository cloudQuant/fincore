"""
Strategy report generator: build an HTML or PDF strategy report from the data you provide.

The more inputs you pass, the more sections the report will include:

- **returns** (required): core performance metrics + return charts
- **+ benchmark_rets**: alpha/beta, information ratio, tracking error, rolling beta
- **+ positions**: holdings analysis, long/short exposure, leverage, concentration
- **+ transactions**: turnover, volume analysis, trading time distribution
- **+ trades**: trade statistics (win rate, payoff ratio, long/short breakdown, holding time distribution)

Usage::

    from fincore.report import create_strategy_report

    # Minimal: returns only
    create_strategy_report(returns, output="report.html")

    # Full: pass everything you have
    create_strategy_report(
        returns,
        benchmark_rets=benchmark,
        positions=positions,
        transactions=transactions,
        trades=closed_trades_df,
        title="My Strategy",
        output="report.pdf",
    )

Modular structure
-----------------
- ``compute``     – statistics computation engine
- ``format``      – CSS styles and HTML formatting helpers
- ``render_html`` – HTML body assembly + ECharts JavaScript
- ``render_pdf``  – PDF rendering via Playwright + PyPDF2 bookmarks
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, cast

from fincore.report.artifacts import ReportArtifacts

if TYPE_CHECKING:
    import pandas as pd

    from fincore.performance.disclosures import DisclosureContext
    from fincore.report.model import ReportModel


def create_strategy_report(
    returns: pd.Series,
    *,
    benchmark_rets: pd.Series | None = None,
    positions: pd.DataFrame | None = None,
    transactions: pd.DataFrame | None = None,
    trades: pd.DataFrame | None = None,
    title: str = "Strategy Report",
    output: str = "report.html",
    rolling_window: int = 63,
    period: str = "daily",
    return_result: bool = False,
    audit_manifest: bool = False,
    disclosure_context: DisclosureContext | None = None,
) -> str | ReportArtifacts:
    """Generate a strategy report (HTML or PDF) based on the inputs you provide.

    Parameters
    ----------
    returns : pd.Series
        Daily return series (required). Must be indexed by a DatetimeIndex.
    benchmark_rets : pd.Series, optional
        Benchmark return series. Enables alpha/beta, tracking error, rolling beta, etc.
    positions : pd.DataFrame, optional
        Daily positions DataFrame (columns = asset symbols plus a ``cash`` column). Enables positions analysis.
    transactions : pd.DataFrame, optional
        Transactions DataFrame (must include ``amount``, ``price``, ``symbol``). Enables transaction analysis.
    trades : pd.DataFrame, optional
        Closed trades DataFrame (must include ``pnlcomm``; optional ``long``, ``barlen``, ``commission``).
        Enables trade statistics (win rate, payoff ratio, etc.).
    title : str
        Report title.
    output : str
        Output path. Use ``.html`` for HTML and ``.pdf`` for PDF.
    rolling_window : int
        Rolling window size (trading days). Default is 63 (about 3 months).
    period : str
        Returns period. Default is "daily".
    return_result : bool, default False
        Enhanced interface only.  When True, compute the report model once
        and return a :class:`ReportArtifacts` (owned files, HTML content, and
        the precomputed model) instead of just the output path.  The default
        path-based return behavior is unchanged.
    audit_manifest : bool, default False
        Enhanced interface only.  When True together with ``return_result``,
        write a sidecar JSON audit manifest (input shapes/hashes, code commit,
        dependency versions) beside the report and expose it as
        ``ReportArtifacts.manifest_path``.  The manifest never contains raw
        returns, positions, transactions, credentials, or absolute local paths.
    disclosure_context : DisclosureContext, optional
        Enhanced-interface calculation context for the displayed performance
        figures.  Its established defaults are complete caller declarations
        (TWR, gross-of-fees, no cashflows, annualized), so pass it only when
        all declarations are supported by the calculation record.  Without a
        context, the report uses conservative source-derived language: it
        explicitly states that it received a caller-supplied simple return
        series and did not perform cashflow adjustment.

    Returns
    -------
    str
        The path to the generated report (``return_result=False``).
    ReportArtifacts
        The generated file plus the computed model (``return_result=True``).
    """
    model: ReportModel | None = None
    if return_result:
        from fincore.report.compute import compute_sections

        model = compute_sections(
            returns,
            benchmark_rets,
            positions,
            transactions,
            trades,
            rolling_window,
            period=period,
            disclosure_context=disclosure_context,
        )

    if output.lower().endswith(".pdf"):
        from fincore.report.render_pdf import generate_pdf

        path = cast(
            "str",
            generate_pdf(
                returns,
                benchmark_rets=benchmark_rets,
                positions=positions,
                transactions=transactions,
                trades=trades,
                title=title,
                output=output,
                rolling_window=rolling_window,
                period=period,
                model=model,
                disclosure_context=disclosure_context if model is None else None,
            ),
        )
        backend = "pdf"
        html: str | None = None
    else:
        from fincore.report.render_html import generate_html

        path = cast(
            "str",
            generate_html(
                returns,
                benchmark_rets=benchmark_rets,
                positions=positions,
                transactions=transactions,
                trades=trades,
                title=title,
                output=output,
                rolling_window=rolling_window,
                period=period,
                model=model,
                disclosure_context=disclosure_context if model is None else None,
            ),
        )
        backend = "html"
        html = Path(path).read_text(encoding="utf-8")

    if return_result:
        artifacts = ReportArtifacts(backend=backend, files=[Path(path)], html=html, model=model)
        if audit_manifest:
            from fincore import __version__ as fincore_version
            from fincore.report.provenance import ReportProvenance

            assert model is not None
            provenance = ReportProvenance.build(
                code_version=fincore_version,
                configuration={
                    "title": title,
                    "rolling_window": rolling_window,
                    "period": period,
                    "backend": backend,
                    "performance_disclosure": model["performance_disclosure"],
                },
                inputs={
                    "returns": returns,
                    "benchmark_rets": benchmark_rets,
                    "positions": positions,
                    "transactions": transactions,
                    "trades": trades,
                },
            )
            manifest_path = Path(path).with_suffix(".manifest.json")
            provenance.write(manifest_path)
            artifacts.manifest_path = manifest_path
        return artifacts
    return path


__all__ = ["ReportArtifacts", "create_strategy_report"]
