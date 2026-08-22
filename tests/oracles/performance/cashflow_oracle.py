"""Small hand-calculation oracle for cashflow-adjusted performance returns.

This fixture deliberately contains only the two published valuation identities
used by the tests.  It does not import fincore or share implementation helpers
with the production module.
"""

from __future__ import annotations


def cashflow_adjusted_period_return(
    opening_value: float,
    closing_value: float,
    external_cashflow: float,
    fee: float,
    *,
    timing: str,
    fee_treatment: str,
) -> float:
    """Calculate one period's TWR return from the published identity."""

    gross_closing_value = closing_value + fee if fee_treatment == "gross" else closing_value
    if timing == "end":
        return (gross_closing_value - external_cashflow) / opening_value - 1.0
    if timing == "start":
        return gross_closing_value / (opening_value + external_cashflow) - 1.0
    raise ValueError(f"unsupported timing: {timing}")
