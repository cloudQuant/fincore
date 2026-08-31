"""Capability-level checks for the unified attribution operation surface."""

from __future__ import annotations


def test_attribution_operation_ids_are_unique_and_owned_by_the_attribution_domain() -> None:
    from fincore.attribution.operations import operations

    declared = operations()

    assert declared
    assert len({operation.operation_id for operation in declared}) == len(declared)
    assert len({operation.capability_id for operation in declared}) == len(declared)
    assert {operation.domain for operation in declared} == {"attribution"}
    assert all(operation.callable.__module__.startswith("fincore.attribution.") for operation in declared)
