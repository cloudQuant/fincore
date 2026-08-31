"""Contracts for immutable extension snapshots."""

from __future__ import annotations

import importlib
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pytest

from fincore.exceptions import OperationResolutionError
from fincore.runtime import AnalysisSession, OperationCatalog, OperationSpec


def _increment(*, value: int) -> int:
    return value + 1


def _double(*, value: int) -> int:
    return value * 2


def _extension(operation_id: str, callable_: object) -> OperationSpec:
    return OperationSpec(
        operation_id=operation_id,
        capability_id=operation_id,
        domain="extensions",
        callable=callable_,
        provenance={"owner": "test-extension"},
    )


def test_snapshot_is_immutable_content_addressed_and_namespaced() -> None:
    from fincore.extensions.operations import operations
    from fincore.extensions.snapshot import ExtensionSnapshot

    first = ExtensionSnapshot(operations=(_extension("extension.demo.increment", _increment),))
    second = ExtensionSnapshot(operations=(_extension("extension.demo.increment", _increment),))

    assert first.digest == second.digest
    assert first.operations[0].operation_id == "extension.demo.increment"
    with pytest.raises(AttributeError):
        first.operations.append(first.operations[0])  # type: ignore[attr-defined]
    with pytest.raises(ValueError, match="extension namespace"):
        ExtensionSnapshot(operations=(_extension("metrics.override", _increment),))
    module = importlib.import_module("fincore.extensions")
    assert module.__all__ == []
    assert operations() == ()


def test_catalog_composes_extensions_without_mutating_builtins_or_allowing_overrides() -> None:
    from fincore.extensions.snapshot import ExtensionSnapshot

    builtins = OperationCatalog((_extension("extension.reserved.increment", _increment),))
    snapshot = ExtensionSnapshot(operations=(_extension("extension.demo.double", _double),))

    combined = builtins.with_extensions(snapshot)

    assert builtins.operation_ids == ("extension.reserved.increment",)
    assert combined.operation_ids == ("extension.demo.double", "extension.reserved.increment")
    assert combined.extension_snapshot is snapshot
    assert combined.digest != builtins.digest
    assert combined.resolve("extension.demo.double").callable is _double
    with pytest.raises(ValueError, match="duplicate operation_id"):
        builtins.with_extensions(ExtensionSnapshot(operations=(_extension("extension.reserved.increment", _double),)))


def test_sessions_pin_one_extension_snapshot_and_remain_isolated_across_threads() -> None:
    from fincore.extensions.snapshot import ExtensionSnapshot

    base = OperationCatalog(())
    increment = base.with_extensions(
        ExtensionSnapshot(operations=(_extension("extension.demo.increment", _increment),))
    )
    double = base.with_extensions(ExtensionSnapshot(operations=(_extension("extension.demo.double", _double),)))
    increment_session = AnalysisSession(increment)
    double_session = AnalysisSession(double)

    with ThreadPoolExecutor(max_workers=2) as executor:
        increment_future = executor.submit(increment_session.run, "extension.demo.increment", {"value": 3})
        double_future = executor.submit(double_session.run, "extension.demo.double", {"value": 3})

    assert increment_future.result().value == 4
    assert double_future.result().value == 6
    assert increment_session.catalog_digest == increment.digest
    with pytest.raises(OperationResolutionError, match="extension.demo.double"):
        increment_session.run("extension.demo.double", {"value": 3})


def test_hook_only_snapshot_changes_catalog_identity_without_mutating_the_base() -> None:
    from fincore.extensions.snapshot import ExtensionHook, ExtensionSnapshot

    base = OperationCatalog(())
    empty = base.with_extensions(ExtensionSnapshot())
    hooked = base.with_extensions(ExtensionSnapshot(hooks=(ExtensionHook(event="audit", callable=lambda: None),)))

    assert base.extension_digest is None
    assert empty.extension_digest != hooked.extension_digest
    assert empty.digest != hooked.digest


def test_hooks_and_renderers_are_snapshot_local_and_priority_ordered() -> None:
    from fincore.extensions.snapshot import ExtensionHook, ExtensionSnapshot, RendererRegistration

    calls: list[str] = []

    def second(value: int) -> int:
        calls.append("second")
        return value + 1

    def first(value: int) -> int:
        calls.append("first")
        return value * 2

    class Renderer:
        pass

    snapshot = ExtensionSnapshot(
        hooks=(
            ExtensionHook(event="normalize", callable=second, priority=20),
            ExtensionHook(event="normalize", callable=first, priority=10),
        ),
        renderers=(RendererRegistration(name="recording", renderer=Renderer),),
    )

    assert snapshot.execute_hooks("normalize", 3) == 7
    assert calls == ["first", "second"]
    assert snapshot.renderer("recording") is Renderer
    assert snapshot.hooks_for("missing") == ()


def test_discovery_reads_metadata_without_importing_extension_targets(monkeypatch: pytest.MonkeyPatch) -> None:
    from fincore.extensions import discovery

    entry_point = SimpleNamespace(
        name="demo",
        group="fincore.extensions",
        value="sample_package:build_snapshot",
        dist=SimpleNamespace(name="sample-package"),
    )

    class Entries:
        def select(self, *, group: str):
            return (entry_point,) if group == "fincore.extensions" else ()

    monkeypatch.setattr(discovery.importlib.metadata, "entry_points", lambda: Entries())

    assert discovery.discover_extensions() == (
        discovery.DiscoveredExtension(
            name="demo",
            group="fincore.extensions",
            distribution="sample-package",
            value="sample_package:build_snapshot",
        ),
    )
