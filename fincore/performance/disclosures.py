"""GIPS-aware disclosure templates.

fincore provides calculation and disclosure *support*; it does not certify GIPS
compliance.  These helpers assemble the disclosure context a report must show:
calculation convention, sample period, data-quality notes, fees, cashflows and
benchmark/risk-free units.
"""

from __future__ import annotations

from dataclasses import dataclass

__all__ = ["DisclosureContext", "render_disclosure"]


@dataclass(frozen=True)
class DisclosureContext:
    """Caller assertions displayed alongside enhanced performance numbers.

    This public template retains its established defaults: constructing it
    asserts a TWR, gross-of-fees, no-cashflow, annualized calculation.  Pass
    it to a report only when the calculation record supports every such
    declaration.  To receive the report's conservative source-derived
    disclosure, omit ``disclosure_context`` entirely.
    """

    convention: str = "TWR"
    sample_period: str = ""
    data_quality: str = ""
    fees: str = "gross-of-fees"
    cashflows: str = "none"
    benchmark: str = ""
    risk_free: str = ""
    annualized: bool = True
    notes: tuple[str, ...] = ()
    # Appended so positional construction of the original context remains
    # compatible.  A report must state the representation of the input as
    # well as the calculation convention.
    return_type: str = ""
    units: str = ""
    frequency: str = ""


def render_disclosure(context: DisclosureContext) -> str:
    """Render a single disclosure block for a report."""
    lines = [
        f"Convention: {context.convention or 'unspecified'}",
        f"Return type: {context.return_type or 'unspecified'}",
        f"Units: {context.units or 'unspecified'}",
        f"Frequency: {context.frequency or 'unspecified'}",
        f"Sample period: {context.sample_period or 'unspecified'}",
        f"Data quality: {context.data_quality or 'unspecified'}",
        f"Fees: {context.fees or 'unspecified'}",
        f"Cashflows: {context.cashflows or 'unspecified'}",
        f"Benchmark: {context.benchmark or 'none'}",
        f"Risk-free: {context.risk_free or 'unspecified'}",
        f"Annualized: {'yes' if context.annualized else 'no'}",
    ]
    if context.notes:
        lines.append("Notes: " + "; ".join(context.notes))
    return "\n".join(lines)
