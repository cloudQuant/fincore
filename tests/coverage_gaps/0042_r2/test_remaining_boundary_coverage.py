"""High-value error and boundary contracts left by the unified-core cutover."""

from __future__ import annotations

import importlib
from datetime import date, datetime
from decimal import Decimal
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest


def _total(*, value: float = 1.0) -> float:
    return value


def _extension_spec(
    operation_id: str,
    *,
    capability_id: str | None = None,
    domain: str = "extensions",
) -> object:
    from fincore.runtime import OperationSpec

    return OperationSpec(
        operation_id=operation_id,
        capability_id=capability_id or operation_id,
        domain=domain,
        callable=_total,
    )


def test_result_private_codec_error_contracts_cover_all_portable_boundaries() -> None:
    from fincore.runtime import results

    frozen = results._freeze({"set": {1, 2}, "list": ["x"]})
    assert frozen["set"] == frozenset({1, 2})
    with pytest.raises(TypeError, match="string keys"):
        results._encode_mapping({1: "bad"})  # type: ignore[dict-item]

    assert results._encode(float("-inf"))["value"] == "-inf"
    assert results._encode(pd.NA)["$fincore_type"] == "pandas_na"
    assert results._encode(pd.Timestamp("2025-01-02", tz="UTC"))["$fincore_type"] == "pandas_timestamp"
    assert results._encode(pd.Timedelta(days=2))["$fincore_type"] == "pandas_timedelta"
    assert results._encode(date(2025, 1, 2))["$fincore_type"] == "date"

    assert results._decode({"$fincore_type": "float", "value": "inf"}) == float("inf")
    assert results._decode({"$fincore_type": "float", "value": "-inf"}) == float("-inf")
    assert results._decode({"$fincore_type": "pandas_na"}) is pd.NA
    assert results._decode({"$fincore_type": "pandas_timestamp", "value": "2025-01-02T00:00:00+00:00"}) == pd.Timestamp(
        "2025-01-02", tz="UTC"
    )
    assert results._decode({"$fincore_type": "pandas_timedelta", "value": "P2D"}) == pd.Timedelta(days=2)
    assert results._decode({"$fincore_type": "decimal", "value": "1.20"}) == Decimal("1.20")
    assert results._decode({"$fincore_type": "datetime", "value": "2025-01-02T03:04:05"}) == datetime(
        2025, 1, 2, 3, 4, 5
    )
    assert results._decode({"$fincore_type": "date", "value": "2025-01-02"}) == date(2025, 1, 2)
    assert results._semantic_metadata(({"temporary_path": "/tmp", "stable": 1},)) == ({"stable": 1},)
    assert results._semantic_metadata([{"workspace_path": "/tmp", "stable": 1}]) == [{"stable": 1}]

    with pytest.raises(ValueError, match="mapping items"):
        results._decode({"$fincore_type": "mapping", "items": {}})
    with pytest.raises(ValueError, match="mapping item"):
        results._decode({"$fincore_type": "mapping", "items": [["only-key"]]})
    with pytest.raises(ValueError, match="array shape"):
        results._decode({"$fincore_type": "numpy_ndarray", "dtype": "float64", "shape": ["bad"], "values": []})
    with pytest.raises(ValueError, match="timestamp"):
        results._decode({"$fincore_type": "pandas_timestamp"})
    with pytest.raises(ValueError, match="timedelta"):
        results._decode({"$fincore_type": "pandas_timedelta"})
    with pytest.raises(ValueError, match="series index"):
        results._decode({"$fincore_type": "pandas_series", "index": [], "values": []})
    with pytest.raises(ValueError, match="frame indexes"):
        results._decode({"$fincore_type": "pandas_frame", "columns": [], "index": [], "values": [], "dtypes": []})

    encoded_columns = results._encode(pd.Index(["value"]))
    encoded_index = results._encode(pd.Index([0]))
    with pytest.raises(ValueError, match="frame columns"):
        results._decode(
            {
                "$fincore_type": "pandas_frame",
                "columns": encoded_columns,
                "index": encoded_index,
                "values": [],
                "dtypes": [],
            }
        )
    for kind, message in (("decimal", "decimal value"), ("datetime", "datetime value"), ("date", "date value")):
        with pytest.raises(ValueError, match=message):
            results._decode({"$fincore_type": kind, "value": 1})
    with pytest.raises(ValueError, match="metadata must decode"):
        results.Result.from_payload({"schema_version": "0.5", "value": 1, "metadata": []})
    with pytest.raises(ValueError, match="JSON must contain"):
        results.Result.from_json("[]")


def test_operation_specs_and_extension_snapshots_reject_incomplete_or_ambiguous_contracts() -> None:
    from fincore.extensions.snapshot import (
        ExtensionHook,
        ExtensionSnapshot,
        RendererRegistration,
        _callable_fingerprint,
        _identifier,
    )
    from fincore.runtime.specs import OperationSpec, _freeze_metadata, make_operations_provider

    assert _freeze_metadata(({"x": 1}, {2})) == ({"x": 1}, frozenset({2}))
    with pytest.raises(ValueError, match="non-empty"):
        _identifier(" ", "name")
    with pytest.raises(TypeError, match="immutable tuple"):
        make_operations_provider([])  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="callable"):
        OperationSpec("x", "x", "x", object())  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="optional_extra"):
        OperationSpec("x", "x", "x", _total, optional_extra="")
    with pytest.raises(ValueError, match="provided together"):
        OperationSpec("x", "x", "x", _total, semantic_mode="base")

    class CallableWithoutIdentity:
        __module__ = None

        def __call__(self) -> None:
            return None

    opaque = CallableWithoutIdentity()
    with pytest.raises(TypeError, match="expose __module__"):
        _callable_fingerprint(opaque)
    opaque_spec = OperationSpec("x", "x", "x", opaque)
    with pytest.raises(TypeError, match="expose __module__"):
        _ = opaque_spec.implementation_fingerprint

    with pytest.raises(TypeError, match="OperationSpec"):
        ExtensionSnapshot(operations=(object(),))  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="capability_id"):
        ExtensionSnapshot(operations=(_extension_spec("extension.demo.one", capability_id="metrics.one"),))
    with pytest.raises(ValueError, match="domain"):
        ExtensionSnapshot(operations=(_extension_spec("extension.demo.one", domain="metrics"),))
    duplicate = _extension_spec("extension.demo.duplicate")
    with pytest.raises(ValueError, match="duplicate extension"):
        ExtensionSnapshot(operations=(duplicate, duplicate))  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="hooks"):
        ExtensionSnapshot(hooks=(object(),))  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="renderers"):
        ExtensionSnapshot(renderers=(object(),))  # type: ignore[arg-type]

    hook = ExtensionHook(event="audit", callable=lambda value: value)
    renderer = RendererRegistration(name="text", renderer=lambda value: value)
    snapshot = ExtensionSnapshot()
    assert snapshot.with_operation(_extension_spec("extension.demo.valid")).operations
    assert snapshot.with_hook(hook).hooks == (hook,)
    assert snapshot.with_renderer(renderer).renderer("text") is renderer.renderer
    assert renderer.fingerprint.endswith("<lambda>")


def test_report_model_and_portfolio_compute_validation_branches_have_direct_contracts() -> None:
    from fincore.report import models
    from fincore.report.models import ReportDocument, ReportSection
    from fincore.report.portfolio import compute

    with pytest.raises(TypeError, match="must be a mapping"):
        models._mapping([], field_name="metadata")  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="keys"):
        models._mapping({"": 1}, field_name="metadata")
    with pytest.raises(TypeError, match="pandas Series"):
        models._copy_series([], field_name="series")  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="pandas DataFrame"):
        models._copy_table([], field_name="table")  # type: ignore[arg-type]
    assert models._semantic_value(np.float32(1.5)) == 1.5
    assert models._semantic_value(pd.NA) == {"type": "pandas_na"}
    assert models._semantic_value(pd.NaT) == {"type": "pandas_nat"}
    assert models._semantic_value([date(2025, 1, 2)]) == [{"type": "date", "value": "2025-01-02"}]
    with pytest.raises(TypeError, match="string keys"):
        models._semantic_value({1: "bad"})
    with pytest.raises(TypeError, match="does not support"):
        models._semantic_value(object())
    with pytest.raises(ValueError, match="title"):
        ReportSection(key="section", title="")
    with pytest.raises(TypeError, match="units"):
        ReportSection(key="section", title="Section", metrics={"value": 1}, units={"value": ""})
    with pytest.raises(TypeError, match="legends"):
        ReportSection(key="section", title="Section", metrics={"value": 1}, legends={"value": ""})
    with pytest.raises(ValueError, match="domain"):
        ReportDocument(domain="", title="Report", sections=())
    with pytest.raises(ValueError, match="title"):
        ReportDocument(domain="report", title="", sections=())
    with pytest.raises(TypeError, match="sections"):
        ReportDocument(domain="report", title="Report", sections=[])  # type: ignore[arg-type]

    index = pd.date_range("2025-01-01", periods=2, freq="B", tz="UTC")
    returns = pd.Series([0.01, 0.02], index=index)
    invalid_returns = (
        ([], "pandas Series"),
        (pd.Series([], dtype=float, index=pd.DatetimeIndex([])), "at least one"),
        (pd.Series([0.01]), "DatetimeIndex"),
        (pd.Series([0.01, 0.02], index=pd.DatetimeIndex([index[1], index[0]])), "unique and increasing"),
        (pd.Series(["bad"], index=index[:1]), "numeric values"),
        (pd.Series([float("inf")], index=index[:1]), "finite values"),
    )
    for candidate, message in invalid_returns:
        with pytest.raises(Exception, match=message):
            compute._validated_returns(candidate, parameter="returns")  # type: ignore[arg-type]
    with pytest.raises(Exception, match="pandas DataFrame"):
        compute._validated_positions([], returns=returns)  # type: ignore[arg-type]
    with pytest.raises(Exception, match="unique DatetimeIndex"):
        compute._validated_positions(pd.DataFrame({"cash": [1.0]}), returns=returns)
    with pytest.raises(Exception, match="finite values"):
        compute._validated_positions(pd.DataFrame({"cash": [float("nan"), 1.0]}, index=index), returns=returns)
    with pytest.raises(Exception, match="cover every"):
        compute._validated_positions(pd.DataFrame({"cash": [1.0]}, index=index[:1]), returns=returns)
    with pytest.raises(Exception, match="pandas DataFrame"):
        compute._validated_transactions([])  # type: ignore[arg-type]
    with pytest.raises(Exception, match="DatetimeIndex"):
        compute._validated_transactions(pd.DataFrame({"amount": [1], "price": [1]}))
    with pytest.raises(Exception, match="finite amount"):
        compute._validated_transactions(pd.DataFrame({"amount": [float("inf")], "price": [1]}, index=index[:1]))


def test_runtime_artifacts_data_sessions_and_builtins_cover_remaining_isolation_paths(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from fincore.runtime import AnalysisSession, ArtifactBundle, OperationCatalog, OperationSpec, builtins
    from fincore.runtime.data import AnalysisSnapshot

    with pytest.raises(TypeError, match="metadata must"):
        ArtifactBundle(metadata=[])  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="metadata keys"):
        ArtifactBundle(metadata={"": 1})
    bundle = ArtifactBundle()
    with pytest.raises(TypeError, match="owned"):
        bundle.add("artifact", owned=1)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="closer"):
        bundle.add("artifact", owned=False, closer="close")  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="name"):
        bundle.add("artifact", owned=False, name="")
    with ArtifactBundle() as context_bundle:
        context_bundle.add("artifact", owned=False)
    assert context_bundle.closed

    snapshot = AnalysisSnapshot.from_inputs({"index": pd.Index(["a", "b"]), "as_of": date(2025, 1, 2)})
    assert snapshot.materialize()["index"].tolist() == ["a", "b"]
    with pytest.raises(TypeError, match="inputs must"):
        AnalysisSnapshot.from_inputs([])  # type: ignore[arg-type]

    holder: dict[str, AnalysisSession] = {}

    def close_before_cache(*, value: float) -> float:
        holder["session"].close()
        return value

    catalog = OperationCatalog(
        (
            OperationSpec(
                operation_id="metrics.close_before_cache",
                capability_id="metrics.close_before_cache",
                domain="metrics",
                callable=close_before_cache,
            ),
        )
    )
    session = AnalysisSession(catalog)
    holder["session"] = session
    with pytest.raises(RuntimeError, match="closed"):
        session.run("metrics.close_before_cache", {"value": 1.0})
    session.close()

    with pytest.raises(TypeError, match="provider"):
        builtins.compose_catalog((object(),))  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="invalid builtin"):
        builtins._load_provider("invalid")
    monkeypatch.setattr(builtins, "import_module", lambda _: SimpleNamespace(operations=object()))
    with pytest.raises(TypeError, match="not callable"):
        builtins._load_provider("module:operations")


def test_namespace_imports_are_deliberately_empty_and_reloadable() -> None:
    namespace_modules = (
        "fincore",
        "fincore.attribution",
        "fincore.data",
        "fincore.extensions",
        "fincore.factor_analysis",
        "fincore.metrics",
        "fincore.optimization",
        "fincore.performance",
        "fincore.portfolio",
        "fincore.report",
        "fincore.report.factor",
        "fincore.report.portfolio",
        "fincore.report.renderers",
        "fincore.risk",
        "fincore.runtime",
        "fincore.simulation",
    )
    reloaded = {name: importlib.reload(importlib.import_module(name)) for name in namespace_modules}

    assert reloaded["fincore"].__version__
    assert reloaded["fincore.metrics"].__all__ == []
    assert reloaded["fincore.runtime"].OperationCatalog is not None
