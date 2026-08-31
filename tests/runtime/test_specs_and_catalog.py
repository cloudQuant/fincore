"""Behavioral contracts for the 0.5 runtime operation model."""

from __future__ import annotations

from dataclasses import fields

import pytest


def _sum_values(*, values: tuple[float, ...]) -> float:
    return sum(values)


def _mean_values(*, values: tuple[float, ...]) -> float:
    return sum(values) / len(values)


def test_operation_spec_keeps_one_direct_canonical_callable_without_legacy_surface_fields() -> None:
    from fincore.runtime.specs import OperationSpec

    spec = OperationSpec(
        operation_id="metrics.sum",
        capability_id="metrics.sum",
        domain="metrics",
        callable=_sum_values,
        input_schema={"values": "tuple[float, ...]"},
        output_schema="float",
        provenance={"formula": "sum(values)"},
    )

    assert spec.callable is _sum_values
    assert spec.implementation_fingerprint == f"{_sum_values.__module__}:{_sum_values.__qualname__}"
    assert spec.optional_extra is None
    assert {field.name for field in fields(OperationSpec)}.isdisjoint(
        {"adapter", "adapter_ref", "profile", "projection", "public_path", "signature"}
    )


def test_catalog_resolves_the_registered_operation_to_its_original_callable() -> None:
    from fincore.runtime.catalog import OperationCatalog
    from fincore.runtime.specs import OperationSpec

    spec = OperationSpec(
        operation_id="metrics.sum",
        capability_id="metrics.sum",
        domain="metrics",
        callable=_sum_values,
    )

    catalog = OperationCatalog((spec,))

    assert catalog.resolve("metrics.sum") is spec
    assert catalog.resolve("metrics.sum").callable(values=(1.0, 2.0)) == 3.0
    assert catalog.operation_ids == ("metrics.sum",)


def test_catalog_rejects_two_implementations_for_one_leaf_capability() -> None:
    from fincore.runtime.catalog import OperationCatalog
    from fincore.runtime.specs import OperationSpec

    total = OperationSpec(
        operation_id="metrics.total",
        capability_id="metrics.aggregate",
        domain="metrics",
        callable=_sum_values,
    )
    mean = OperationSpec(
        operation_id="metrics.mean",
        capability_id="metrics.aggregate",
        domain="metrics",
        callable=_mean_values,
    )

    with pytest.raises(ValueError, match="canonical implementation"):
        OperationCatalog((total, mean))
