"""Contracts for portable runtime results and deterministic provenance."""

from __future__ import annotations

import json

import pytest


def test_result_freezes_nested_metadata_and_round_trips_a_versioned_json_payload() -> None:
    from fincore.runtime import Result

    source_metadata = {"operation_id": "metrics.total", "diagnostics": {"samples": [2]}}
    result = Result(value={"total": 0.03}, metadata=source_metadata)
    source_metadata["diagnostics"]["samples"].append(3)

    payload = result.to_payload()
    restored = Result.from_payload(json.loads(result.to_json()))

    assert payload["schema_version"] == "0.5"
    assert result.metadata["diagnostics"]["samples"] == (2,)
    assert restored == result
    with pytest.raises(TypeError):
        result.metadata["operation_id"] = "other"  # type: ignore[index]


def test_result_semantic_digest_excludes_runtime_timing_but_serializer_rejects_private_objects() -> None:
    from fincore.runtime import Result

    first = Result(value=1.0, metadata={"operation_id": "metrics.total", "duration_ns": 10})
    second = Result(value=1.0, metadata={"operation_id": "metrics.total", "duration_ns": 20})

    assert first.semantic_digest == second.semantic_digest
    with pytest.raises(TypeError, match="unsupported result value"):
        Result(value=lambda: None).to_payload()
