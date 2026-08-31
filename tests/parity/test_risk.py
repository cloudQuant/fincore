"""Capability-level checks for the unified risk operation surface."""

from __future__ import annotations


def test_risk_operation_ids_are_unique_and_owned_by_the_risk_domain() -> None:
    from fincore.risk.operations import operations

    declared = operations()

    assert declared
    assert len({operation.operation_id for operation in declared}) == len(declared)
    assert len({operation.capability_id for operation in declared}) == len(declared)
    assert {operation.domain for operation in declared} == {"risk"}
    assert all(operation.callable.__module__.startswith("fincore.risk.") for operation in declared)
