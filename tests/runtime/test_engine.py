"""Contracts for direct-call-preserving runtime execution."""

from __future__ import annotations

import pytest


def test_runtime_run_calls_the_registered_domain_callable_once_and_wraps_its_natural_value() -> None:
    from fincore.runtime import OperationCatalog, OperationSpec, run

    calls: list[tuple[float, ...]] = []

    def annualized_total(*, returns: tuple[float, ...]) -> float:
        calls.append(returns)
        return sum(returns)

    catalog = OperationCatalog(
        (
            OperationSpec(
                operation_id="metrics.annualized_total",
                capability_id="metrics.annualized_total",
                domain="metrics",
                callable=annualized_total,
            ),
        )
    )

    result = run("metrics.annualized_total", {"returns": (0.01, 0.02)}, catalog=catalog)

    assert calls == [(0.01, 0.02)]
    assert result.value == 0.03
    assert result.metadata["operation_id"] == "metrics.annualized_total"
    assert result.metadata["implementation_fingerprint"] == (
        f"{annualized_total.__module__}:{annualized_total.__qualname__}"
    )


def test_runtime_run_does_not_rewrite_a_domain_error_as_an_orchestration_error() -> None:
    from fincore.runtime import OperationCatalog, OperationSpec, run

    class DomainInputError(ValueError):
        pass

    def guarded_metric(*, value: float) -> float:
        if value < 0:
            raise DomainInputError("value must be non-negative")
        return value

    catalog = OperationCatalog(
        (
            OperationSpec(
                operation_id="metrics.guarded",
                capability_id="metrics.guarded",
                domain="metrics",
                callable=guarded_metric,
            ),
        )
    )

    with pytest.raises(DomainInputError, match="non-negative"):
        run("metrics.guarded", {"value": -1.0}, catalog=catalog)


def test_runtime_plan_and_batch_preserve_request_order_and_bind_one_catalog_snapshot() -> None:
    from fincore.runtime import OperationCatalog, OperationRequest, OperationSpec, batch, plan

    def double(*, value: int) -> int:
        return value * 2

    catalog = OperationCatalog(
        (
            OperationSpec(
                operation_id="metrics.double",
                capability_id="metrics.double",
                domain="metrics",
                callable=double,
            ),
        )
    )
    requests = (
        OperationRequest("metrics.double", {"value": 2}),
        OperationRequest("metrics.double", {"value": 3}),
    )

    execution_plan = plan(requests, catalog=catalog)
    results = batch(execution_plan, catalog=catalog)

    assert execution_plan.catalog_digest == catalog.digest
    assert [result.value for result in results] == [4, 6]
    assert [result.metadata["operation_id"] for result in results] == ["metrics.double", "metrics.double"]


def test_runtime_batch_rejects_a_plan_for_another_catalog_snapshot() -> None:
    from fincore.runtime import OperationCatalog, OperationRequest, OperationSpec, batch, plan

    def first(*, value: int) -> int:
        return value

    def second(*, value: int) -> int:
        return value

    first_catalog = OperationCatalog(
        (
            OperationSpec(
                operation_id="metrics.first",
                capability_id="metrics.first",
                domain="metrics",
                callable=first,
            ),
        )
    )
    second_catalog = OperationCatalog(
        (
            OperationSpec(
                operation_id="metrics.second",
                capability_id="metrics.second",
                domain="metrics",
                callable=second,
            ),
        )
    )

    execution_plan = plan((OperationRequest("metrics.first", {"value": 1}),), catalog=first_catalog)

    with pytest.raises(ValueError, match="catalog snapshot"):
        batch(execution_plan, catalog=second_catalog)
