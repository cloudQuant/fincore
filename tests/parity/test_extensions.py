"""Canonical extension-registry scenarios for source and wheel evidence."""

from __future__ import annotations

import pytest

from fincore.plugin import (
    DEFAULT_METRIC_FAMILY,
    ROLLING_FAMILY,
    DuplicatePolicy,
    DuplicateRegistrationError,
    ExtensionKind,
    Registration,
    Scope,
    clear_registry,
    execute_hooks,
    get_metric,
    get_registry,
    get_viz_backend,
    isolated_registry,
    list_hooks,
    list_metrics,
    list_viz_backends,
    register_hook,
    register_metric,
    register_viz_backend,
)


@pytest.fixture(autouse=True)
def _isolated_registry() -> None:
    """Leave the process-wide extension registry exactly as it was found."""
    with isolated_registry():
        yield


def test_extension_types_and_default_metric_families() -> None:
    """Extension vocabulary is explicit and registrations are immutable data."""
    registration = Registration(
        kind=ExtensionKind.METRIC,
        name="sample",
        target=lambda values: values,
    )

    assert DEFAULT_METRIC_FAMILY == "default"
    assert ROLLING_FAMILY == "rolling"
    assert {kind.value for kind in ExtensionKind} == {"metric", "viz_backend", "hook"}
    assert registration.family == DEFAULT_METRIC_FAMILY
    assert registration.scope is Scope.GLOBAL


def test_metric_registration_lookup_and_duplicate_policy() -> None:
    """Metric registrations preserve default family resolution and collision policy."""

    @register_metric("signal")
    def first(values: list[float]) -> float:
        return float(sum(values))

    registered_first = get_metric("signal")
    assert list_metrics() == {"signal": registered_first}
    assert registered_first is not None
    assert registered_first.__name__ == "first"
    assert get_metric("missing") is None

    with pytest.raises(DuplicateRegistrationError, match="already registered"):

        @register_metric("signal", duplicate=DuplicatePolicy.ERROR)
        def rejected(values: list[float]) -> float:
            return float(len(values))

    @register_metric("signal", duplicate=DuplicatePolicy.IGNORE)
    def ignored(values: list[float]) -> float:
        return float(len(values))

    assert get_metric("signal") is registered_first
    assert registered_first([1.0, 2.0]) == 3.0


def test_viz_backend_registration_lookup_and_listing() -> None:
    """Custom visualization backends are registered as classes, not instances."""

    @register_viz_backend("recording")
    class RecordingBackend:
        def __init__(self, theme: str = "light") -> None:
            self.theme = theme

    assert list_viz_backends() == {"recording": RecordingBackend}
    assert get_viz_backend("recording") is RecordingBackend
    assert RecordingBackend("dark").theme == "dark"


def test_hook_registration_listing_and_execution() -> None:
    """Hooks execute in priority order and transform the first argument."""
    calls: list[str] = []

    @register_hook("normalize", priority=200)
    def second(value: int) -> int:
        calls.append("second")
        return value + 1

    @register_hook("normalize", priority=10)
    def first(value: int) -> int:
        calls.append("first")
        return value * 2

    registered = list_hooks("normalize")["normalize"]
    assert [hook.__name__ for hook in registered] == ["first", "second"]
    assert execute_hooks("normalize", 3) == 7
    assert calls == ["first", "second"]


def test_registry_lookup_isolation_and_clear_policy() -> None:
    """The singleton retains builtins while clear removes ordinary registrations."""
    registry = get_registry()
    assert registry is get_registry()

    registry.register(ExtensionKind.METRIC, "builtin", lambda _values: 1.0, scope=Scope.BUILTIN)

    @register_metric("temporary")
    def temporary(values: list[float]) -> float:
        return float(len(values))

    clear_registry()

    assert registry.get(ExtensionKind.METRIC, "builtin") is not None
    assert registry.get(ExtensionKind.METRIC, "temporary") is None
