"""Capability registry contract tests."""

from __future__ import annotations

from fincore.capabilities import STATUSES, get_capability, list_capabilities


def test_public_capabilities_have_unique_ids_and_actionable_statuses() -> None:
    rows = list_capabilities()
    assert {row.status for row in rows} <= set(STATUSES)
    assert len({row.id for row in rows}) == len(rows)
    assert all(row.docs_path for row in rows)
    assert all(row.public_path for row in rows)
    assert all(row.domain for row in rows)


def test_brinson_hood_is_not_implemented() -> None:
    cap = get_capability("attribution.brinson_hood")
    assert cap.status == "not_implemented"


def test_provider_entry_points_are_provider_required() -> None:
    for cap_id in (
        "data.yahoo",
        "data.alphavantage",
        "data.tushare",
        "data.akshare",
        "attribution.ff_factor_provider",
        "attribution.style_factor_provider",
    ):
        assert get_capability(cap_id).status == "provider_required"


def test_strict_facades_are_stable() -> None:
    assert get_capability("compat.empyrical").status == "stable"
    assert get_capability("compat.pyfolio").status == "stable"


def test_pit_factor_preparation_is_discoverable_as_experimental() -> None:
    capability = get_capability("factor_analysis.pit_prepare")

    assert capability.status == "experimental"
    assert capability.public_path == "fincore.factor_analysis.prepare_pit_factor_data"
    assert capability.docs_path == "concepts/factor-research-protocol.md"


def test_factor_model_inference_is_discoverable_as_experimental() -> None:
    capability = get_capability("factor_analysis.inference")

    assert capability.status == "experimental"
    assert capability.public_path == "fincore.factor_analysis.factor_model_inference"
    assert capability.docs_path == "concepts/factor-research-protocol.md"


def test_get_capability_raises_for_unknown_id() -> None:
    import pytest

    with pytest.raises(KeyError):
        get_capability("does.not.exist")
