"""GIPS-aware disclosure templates.

fincore provides calculation and disclosure *support*; it does not certify GIPS
compliance.  These helpers assemble the disclosure context a report must show:
calculation convention, sample period, data-quality notes, fees, cashflows and
benchmark/risk-free units.
"""

from __future__ import annotations

from dataclasses import dataclass, field

__all__ = ["DisclosureContext", "render_disclosure"]


@dataclass(frozen=True)
class DisclosureContext:
    """The context a report must disclose alongside any performance number."""

    convention: str = "TWR"
    sample_period: str = ""
    data_quality: str = ""
    fees: str = "gross-of-fees"
    cashflows: str = "none"
    benchmark: str = ""
    risk_free: str = ""
    annualized: bool = True
    notes: tuple[str, ...] = ()


def render_disclosure(context: DisclosureContext) -> str:
    """Render a single disclosure block for a report."""
    lines = [
        f"Convention: {context.convention}",
        f"Sample period: {context.sample_period or 'unspecified'}",
        f"Data quality: {context.data_quality or 'unspecified'}",
        f"Fees: {context.fees}",
        f"Cashflows: {context.cashflows}",
        f"Benchmark: {context.benchmark or 'none'}",
        f"Risk-free: {context.risk_free or 'unspecified'}",
        f"Annualized: {'yes' if context.annualized else 'no'}",
    ]
    if context.notes:
        lines.append("Notes: " + "; ".join(context.notes))
    return "\n".join(lines)
