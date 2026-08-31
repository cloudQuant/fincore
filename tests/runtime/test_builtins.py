"""Contracts for the explicit, non-scanning runtime composition root."""

from __future__ import annotations


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


def test_builtin_catalog_is_empty_until_domain_tasks_explicitly_register_providers() -> None:
    from fincore.runtime.builtins import builtin_catalog

    assert builtin_catalog().operations == ()
