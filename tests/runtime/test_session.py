"""Contracts for isolated runtime session state and cache behavior."""

from __future__ import annotations

import pytest


def test_session_caches_a_deterministic_operation_by_catalog_input_and_config_digest() -> None:
    from fincore.runtime import AnalysisSession, OperationCatalog, OperationSpec

    calls: list[float] = []

    def expensive_metric(*, value: float) -> float:
        calls.append(value)
        return value * 2

    catalog = OperationCatalog(
        (
            OperationSpec(
                operation_id="metrics.expensive",
                capability_id="metrics.expensive",
                domain="metrics",
                callable=expensive_metric,
            ),
        )
    )
    session = AnalysisSession(catalog)

    first = session.run("metrics.expensive", {"value": 3.0}, {"annualization": 252})
    second = session.run("metrics.expensive", {"value": 3.0}, {"annualization": 252})

    assert calls == [3.0]
    assert first.value == second.value == 6.0
    assert first.metadata["input_digest"] == second.metadata["input_digest"]
    assert first.metadata["config_digest"] == second.metadata["config_digest"]
    assert first.metadata["catalog_digest"] == second.metadata["catalog_digest"] == catalog.digest
    assert first.metadata["cache"] == "miss"
    assert second.metadata["cache"] == "hit"


def test_sessions_are_isolated_and_closed_sessions_cannot_run_operations() -> None:
    from fincore.runtime import AnalysisSession, OperationCatalog, OperationSpec

    calls: list[int] = []

    def counted(*, value: int) -> int:
        calls.append(value)
        return value

    catalog = OperationCatalog(
        (
            OperationSpec(
                operation_id="metrics.counted",
                capability_id="metrics.counted",
                domain="metrics",
                callable=counted,
            ),
        )
    )
    first = AnalysisSession(catalog)
    second = AnalysisSession(catalog)

    first.run("metrics.counted", {"value": 1})
    first.run("metrics.counted", {"value": 1})
    second.run("metrics.counted", {"value": 1})
    first.close()

    assert calls == [1, 1]
    assert first.closed is True
    with pytest.raises(RuntimeError, match="closed"):
        first.run("metrics.counted", {"value": 1})


def test_session_cache_does_not_expose_a_mutable_cached_domain_value() -> None:
    from fincore.runtime import AnalysisSession, OperationCatalog, OperationSpec

    calls: list[int] = []

    def table(*, value: int) -> list[int]:
        calls.append(value)
        return [value]

    catalog = OperationCatalog(
        (
            OperationSpec(
                operation_id="metrics.table",
                capability_id="metrics.table",
                domain="metrics",
                callable=table,
            ),
        )
    )
    session = AnalysisSession(catalog)

    first = session.run("metrics.table", {"value": 1})
    first.value.append(99)
    second = session.run("metrics.table", {"value": 1})

    assert calls == [1]
    assert second.value == [1]
