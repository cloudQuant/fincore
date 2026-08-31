"""Direct metrics and canonical portfolio-report example."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from fincore.metrics.drawdown import max_drawdown
from fincore.metrics.ratios import sharpe_ratio
from fincore.metrics.yearly import annual_return
from fincore.report.portfolio.compute import build_portfolio_report
from fincore.report.renderers.html import write_html


def build_inputs() -> tuple[pd.Series, pd.DataFrame]:
    """Return deterministic returns and aligned positions for this example."""

    index = pd.date_range("2024-01-02", periods=8, freq="B")
    returns = pd.Series([0.01, -0.005, 0.002, 0.004, -0.001, 0.003, 0.0, 0.002], index=index)
    positions = pd.DataFrame({"AAA": 100.0, "BBB": -30.0, "cash": 80.0}, index=index)
    return returns, positions


def main(output: Path = Path("portfolio-report.html")) -> None:
    """Compute metrics once, build a report model once, and render HTML."""

    returns, positions = build_inputs()
    print(f"Sharpe ratio: {sharpe_ratio(returns):.6f}")
    print(f"Maximum drawdown: {max_drawdown(returns):.6f}")
    print(f"Annual return: {annual_return(returns):.6f}")

    document = build_portfolio_report(returns, positions=positions, rolling_window=3)
    artifact = write_html(document, output)
    print(f"Report: {artifact.named_artifacts['file']}")


if __name__ == "__main__":
    main()
