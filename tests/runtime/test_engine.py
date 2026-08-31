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
