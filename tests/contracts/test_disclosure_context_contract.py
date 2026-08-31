"""Public constructor contract for enhanced performance disclosures."""

from __future__ import annotations

import inspect
from dataclasses import fields

from fincore.performance.disclosures import DisclosureContext, render_disclosure


def test_disclosure_context_fields_and_constructor_order_are_versioned() -> None:
    expected = [
        "convention",
        "sample_period",
        "data_quality",
        "fees",
        "cashflows",
        "benchmark",
        "risk_free",
        "annualized",
        "notes",
        "return_type",
        "units",
        "frequency",
    ]

    assert [field.name for field in fields(DisclosureContext)] == expected
    assert list(inspect.signature(DisclosureContext).parameters) == expected
    assert DisclosureContext().convention == "TWR"
    assert DisclosureContext().fees == "gross-of-fees"
    assert DisclosureContext().cashflows == "none"
    assert DisclosureContext().annualized is True
    assert DisclosureContext().return_type == ""
    assert DisclosureContext().units == ""
    assert DisclosureContext().frequency == ""


def test_render_disclosure_states_return_representation_and_frequency() -> None:
    disclosure = render_disclosure(DisclosureContext(return_type="log", units="decimal log return", frequency="weekly"))

    assert "Return type: log" in disclosure
    assert "Units: decimal log return" in disclosure
    assert "Frequency: weekly" in disclosure


def test_empty_disclosure_context_preserves_its_legacy_caller_assertions() -> None:
    disclosure = render_disclosure(DisclosureContext())

    assert "Convention: TWR" in disclosure
    assert "Fees: gross-of-fees" in disclosure
    assert "Cashflows: none" in disclosure
    assert "Benchmark: none" in disclosure
    assert "Annualized: yes" in disclosure
