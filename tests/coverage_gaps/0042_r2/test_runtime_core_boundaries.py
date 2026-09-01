"""Focused boundary contracts for the 0042-R2 canonical runtime core."""

from __future__ import annotations

from datetime import date, datetime
from decimal import Decimal

import numpy as np
import pandas as pd
import pytest


def _total(*, values: tuple[float, ...]) -> float:
    return float(sum(values))


def _spec(
    operation_id: str = "metrics.total",
    *,
    capability_id: str = "metrics.total",
    semantic_mode: str | None = None,
) -> object:
    from fincore.runtime import OperationSpec

    return OperationSpec(
        operation_id=operation_id,
        capability_id=capability_id,
        domain="metrics",
        callable=_total,
        semantic_mode=semantic_mode,
        mode_approval="ADR-0042-R2-test" if semantic_mode is not None else None,
    )


def test_catalog_rejects_non_specs_before_attempting_deterministic_sorting() -> None:
    from fincore.runtime import OperationCatalog

    with pytest.raises(TypeError, match="OperationSpec"):
        OperationCatalog((object(),))  # type: ignore[arg-type]


def test_catalog_validates_extension_digest_and_semantic_mode_collisions() -> None:
    from fincore.runtime import OperationCatalog

    with pytest.raises(ValueError, match="extension_digest"):
        OperationCatalog((), extension_digest="")

    with pytest.raises(ValueError, match="cannot reuse a semantic mode"):
        OperationCatalog(
            (
                _spec("metrics.total.first", capability_id="metrics.total", semantic_mode="base"),
                _spec("metrics.total.second", capability_id="metrics.total", semantic_mode="base"),
            )
        )


def test_catalog_extension_snapshot_contract_and_immutable_binding() -> None:
    from fincore.extensions.snapshot import ExtensionSnapshot
    from fincore.runtime import OperationCatalog

    catalog = OperationCatalog((_spec(),))
    with pytest.raises(TypeError, match="tuple of OperationSpec"):
        catalog.with_extensions(object())  # type: ignore[arg-type]

    bound = catalog.with_extensions(ExtensionSnapshot())
    assert bound.extension_snapshot is not None
    with pytest.raises(ValueError, match="already bound"):
        bound.with_extensions(ExtensionSnapshot())


def test_result_round_trips_supported_portable_values_and_masks_volatile_metadata() -> None:
    from fincore.runtime import Result

    value = {
        "array": np.array([[1, 2], [3, 4]], dtype=np.int16),
        "scalar": np.float32(1.25),
        "index": pd.Index(["a", "b"], name="asset"),
        "series": pd.Series([1.0, np.nan], index=pd.Index([1, 2], name="row"), name="return"),
        "frame": pd.DataFrame({"a": [1, 2], "b": [True, False]}),
        "bytes": b"fincore",
        "decimal": Decimal("1.20"),
        "datetime": datetime(2026, 8, 30, 12, 0),
        "date": date(2026, 8, 30),
        "nan": float("nan"),
    }
    first = Result(value=value, metadata={"run_id": "first", "artifact_path": "/volatile", "stable": "yes"})
    second = Result(value=value, metadata={"run_id": "second", "artifact_path": "/other", "stable": "yes"})

    restored = Result.from_json(first.to_json())

    assert restored.value["array"].dtype == np.int16
    assert restored.value["series"].name == "return"
    assert restored.value["frame"].dtypes.tolist() == [np.dtype("int64"), np.dtype("bool")]
    assert restored.value["bytes"] == b"fincore"
    assert first.semantic_digest == second.semantic_digest


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        ([], "must be a mapping"),
        ({"schema_version": "other", "value": 1, "metadata": {}}, "unsupported result schema"),
        ({"schema_version": "0.5", "value": 1}, "must contain value and metadata"),
        (
            {"schema_version": "0.5", "value": 1, "metadata": {"$fincore_type": "tuple", "items": "bad"}},
            "serialized tuple items",
        ),
        (
            {"schema_version": "0.5", "value": 1, "metadata": {"$fincore_type": "unknown"}},
            "unknown serialized result type",
        ),
    ],
)
def test_result_rejects_malformed_portable_payloads(payload: object, message: str) -> None:
    from fincore.runtime import Result

    with pytest.raises((TypeError, ValueError), match=message):
        Result.from_payload(payload)  # type: ignore[arg-type]


def test_result_rejects_bad_mapping_keys_and_decoding_errors() -> None:
    from fincore.runtime import Result

    with pytest.raises(TypeError, match="metadata keys"):
        Result(value=1, metadata={1: "bad"})  # type: ignore[dict-item]
    with pytest.raises(ValueError, match="serialized bytes"):
        Result.from_payload({"schema_version": "0.5", "value": 1, "metadata": {"$fincore_type": "bytes", "data": 1}})
    with pytest.raises(ValueError, match="serialized float"):
        Result.from_payload(
            {"schema_version": "0.5", "value": {"$fincore_type": "float", "value": "unsupported"}, "metadata": {}}
        )


def test_analysis_snapshot_copies_inputs_materializes_safely_and_rejects_bad_shapes() -> None:
    from fincore.runtime.data import AnalysisSnapshot

    values = np.array([1.0, 2.0])
    snapshot = AnalysisSnapshot.from_inputs({"values": values, "nested": {"labels": ["a"]}})
    values[0] = 100.0
    materialized = snapshot.materialize()
    materialized["values"][1] = 200.0

    assert snapshot.materialize()["values"].tolist() == [1.0, 2.0]
    with pytest.raises(ValueError, match="must not be empty"):
        AnalysisSnapshot.from_inputs({})
    with pytest.raises(TypeError, match="input names"):
        AnalysisSnapshot.from_inputs({"": 1})
    with pytest.raises(TypeError, match="unsupported snapshot"):
        AnalysisSnapshot.from_inputs({"bad": object()})
    with pytest.raises(TypeError, match="nested snapshot mappings"):
        AnalysisSnapshot.from_inputs({"nested": {1: "bad"}})  # type: ignore[dict-item]


def test_validation_primitives_reject_invalid_mappings_and_arrays(monkeypatch: pytest.MonkeyPatch) -> None:
    from fincore.exceptions import DependencyError
    from fincore.runtime.validation import load_optional_module, validate_finite_array, validate_mapping

    assert validate_mapping({"one": 1}, name="inputs") == {"one": 1}
    for candidate, error in (([], TypeError), ({"": 1}, ValueError), ({1: 1}, ValueError)):
        with pytest.raises(error):
            validate_mapping(candidate, name="inputs")  # type: ignore[arg-type]
    assert validate_finite_array([1, 2], name="values", ndim=1).tolist() == [1.0, 2.0]
    for candidate, kwargs, message in (
        ([1], {"min_size": -1}, "min_size"),
        ([1], {"ndim": -1}, "ndim"),
        (1, {}, "not a scalar"),
        ([1], {"ndim": 2}, "2-dimensional"),
        ([], {"min_size": 1}, "at least"),
        ([float("nan")], {}, "finite"),
        (["not numeric"], {}, "numeric"),
    ):
        with pytest.raises((TypeError, ValueError), match=message):
            validate_finite_array(candidate, name="values", **kwargs)

    def fail_import(_: str) -> object:
        raise ImportError("missing")

    monkeypatch.setattr("fincore.runtime.validation.importlib.import_module", fail_import)
    with pytest.raises(DependencyError, match="custom diagnostic"):
        load_optional_module("not.installed", dependency="optional", extra="viz", message="custom diagnostic")


def test_requests_plans_and_batches_cover_input_boundaries() -> None:
    from fincore.runtime import OperationCatalog, OperationRequest, batch, plan, run

    catalog = OperationCatalog((_spec(),))
    with pytest.raises(ValueError, match="operation_id"):
        OperationRequest(" ", {"values": (1.0,)})
    with pytest.raises(TypeError, match="inputs must be a mapping"):
        OperationRequest("metrics.total", [])  # type: ignore[arg-type]
    request = OperationRequest("metrics.total", {"values": (1.0, 2.0)}, {"nested": {"x": 1}})
    config = request.config
    config["nested"]["x"] = 9
    assert request.config["nested"]["x"] == 1
    with pytest.raises(TypeError, match="OperationRequest"):
        plan((object(),), catalog=catalog)  # type: ignore[arg-type]
    assert batch((), catalog=catalog) == ()
    assert run("metrics.total", {"values": (1.0, 2.0)}, {"sample": 1}, catalog=catalog).value == 3.0


def test_session_context_copy_fallback_and_artifact_lifecycle_error_paths() -> None:
    from fincore.runtime import AnalysisSession, ArtifactBundle, OperationCatalog, OperationSpec

    class CopyFailure:
        def __deepcopy__(self, memo: object) -> object:
            raise RuntimeError("copy rejected")

    def failing_copy(*, values: tuple[float, ...]) -> CopyFailure:
        return CopyFailure()

    catalog = OperationCatalog(
        (
            OperationSpec(
                operation_id="metrics.copy_failure",
                capability_id="metrics.copy_failure",
                domain="metrics",
                callable=failing_copy,
            ),
        )
    )
    with AnalysisSession(catalog) as session:
        result = session.run("metrics.copy_failure", {"values": (1.0,)})
        assert result.metadata["cache"] == "disabled"
        assert session.cache_entries == 0
    assert session.closed is True

    calls: list[str] = []

    class Resource:
        def close(self) -> None:
            calls.append("resource")
            raise RuntimeError("close failed")

    bundle = ArtifactBundle(metadata={"format": "html"})
    resource = Resource()
    bundle.add(resource, owned=True, name="report")
    bundle.add(resource, owned=True)
    with pytest.raises(RuntimeError, match="close failed"):
        bundle.close()
    assert calls == ["resource"]
    with pytest.raises(RuntimeError, match="closed"):
        bundle.add("late", owned=False)
    with pytest.raises(ValueError, match="duplicate artifact name"):
        duplicate_bundle = ArtifactBundle()
        duplicate_bundle.add("one", owned=False, name="same")
        duplicate_bundle.add("two", owned=False, name="same")
