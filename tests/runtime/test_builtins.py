"""Contracts for the explicit, non-scanning runtime composition root."""

from __future__ import annotations

import numpy as np


def test_compose_catalog_uses_only_the_explicit_provider_list() -> None:
    from fincore.runtime import OperationSpec
    from fincore.runtime.builtins import compose_catalog

    calls: list[str] = []

    def runtime_operations() -> tuple[OperationSpec, ...]:
        calls.append("runtime")

        def test_operation(*, value: int) -> int:
            return value

        return (
            OperationSpec(
                operation_id="runtime.test_operation",
                capability_id="runtime.test_operation",
                domain="runtime",
                callable=test_operation,
            ),
        )

    catalog = compose_catalog((runtime_operations,))

    assert calls == ["runtime"]
    assert catalog.operation_ids == ("runtime.test_operation",)


def test_builtin_catalog_loads_only_the_domain_providers_explicitly_registered_by_completed_tasks() -> None:
    from fincore.runtime.builtins import builtin_catalog

    catalog = builtin_catalog()

    assert "metrics.ratios.sharpe_ratio" in catalog.operation_ids
    assert "performance.returns.twr" in catalog.operation_ids


def test_builtin_catalog_runtime_execution_invokes_the_domain_kernel_directly() -> None:
    from fincore.metrics.ratios import sharpe_ratio
    from fincore.runtime import run
    from fincore.runtime.builtins import builtin_catalog

    returns = np.array([0.01, 0.02, -0.01])
    result = run("metrics.ratios.sharpe_ratio", {"returns": returns}, catalog=builtin_catalog())

    assert result.value == sharpe_ratio(returns)
    assert result.metadata["implementation_fingerprint"] == "fincore.metrics.ratios:sharpe_ratio"
