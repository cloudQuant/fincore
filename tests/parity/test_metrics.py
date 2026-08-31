"""Domain-native metric scenarios reused by source and installed-wheel gates.

These cases exercise the canonical ``fincore.metrics`` implementations.  They
intentionally do not import any retired Empyrical compatibility surface.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from fincore.metrics import drawdown
from fincore.portfolio import positions


def test_gross_leverage() -> None:
    """Gross leverage uses absolute non-cash exposure and has no zero-NAV infinity."""
    portfolio = pd.DataFrame(
        {
            "A": [100.0, -50.0],
            "B": [-20.0, 30.0],
            "cash": [20.0, 20.0],
        }
    )

    actual = positions.gross_lev(portfolio)

    assert actual.iloc[0] == 1.2
    assert np.isnan(actual.iloc[1])


def test_second_max_drawdown() -> None:
    """The second drawdown is the second-most-negative distinct episode."""
    returns = pd.Series([0.0, -0.20, 0.25, -0.10, 0.12, -0.05, 0.06])

    assert drawdown.second_max_drawdown(returns) == -0.10


def test_third_max_drawdown() -> None:
    """The third drawdown remains ordered after the two more-severe episodes."""
    returns = pd.Series([0.0, -0.20, 0.25, -0.10, 0.12, -0.05, 0.06])

    np.testing.assert_allclose(drawdown.third_max_drawdown(returns), -0.05, rtol=0.0, atol=1e-12)


def test_metrics_operation_manifest_exposes_direct_kernel_paths_without_legacy_aliases() -> None:
    from fincore.metrics.operations import operations
    from fincore.runtime import OperationCatalog

    catalog = OperationCatalog(operations())

    assert {
        "metrics.returns.simple_returns",
        "metrics.returns.cum_returns",
        "metrics.drawdown.max_drawdown",
        "metrics.ratios.sharpe_ratio",
        "metrics.risk.annual_volatility",
        "metrics.alpha_beta.alpha_beta",
        "metrics.rolling.roll_sharpe_ratio",
        "metrics.stats.skewness",
        "metrics.yearly.annual_return",
    }.issubset(catalog.operation_ids)
    assert "metrics.cagr" not in catalog.operation_ids
    assert "metrics.cumulative_returns" not in catalog.operation_ids
    assert catalog.resolve("metrics.ratios.sharpe_ratio").callable.__module__ == "fincore.metrics.ratios"
