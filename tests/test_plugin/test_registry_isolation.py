"""Isolation semantics of the single :class:`ExtensionRegistry`.

Covers:
- Duplicate registration policy (overwrite / error / ignore).
- Registration scope (builtin entries survive ``clear_registry()``; local
  entries are rolled back at the end of an isolation block).
- Thread lock (concurrent registration stays consistent).
- The ``isolated_registry()`` test-isolation context manager (snapshot /
  restore, nesting, exception safety).
"""

from __future__ import annotations

import threading

import pytest

from fincore.plugin import (
    DuplicatePolicy,
    DuplicateRegistrationError,
    clear_registry,
    isolated_registry,
    register_metric,
)
from fincore.plugin.registry import registry
from fincore.plugin.specs import ExtensionKind, Scope


@pytest.fixture(autouse=True)
def _isolate_registry():
    with isolated_registry():
        yield


# =========================================================================
# Duplicate registration policy
# =========================================================================


class TestDuplicatePolicy:
    def test_default_overwrites(self):
        @register_metric("dup")
        def first(r):
            return 1.0

        @register_metric("dup")
        def second(r):
            return 2.0

        assert registry.get(ExtensionKind.METRIC, "dup").target([]) == 2.0

    def test_error_policy_raises(self):
        @register_metric("dup", duplicate=DuplicatePolicy.ERROR)
        def first(r):
            return 1.0

        with pytest.raises(DuplicateRegistrationError, match="already registered"):
            register_metric("dup", duplicate=DuplicatePolicy.ERROR)(lambda r: 2.0)

        assert registry.get(ExtensionKind.METRIC, "dup").target([]) == 1.0

    def test_ignore_policy_keeps_first(self):
        @register_metric("dup", duplicate=DuplicatePolicy.IGNORE)
        def first(r):
            return 1.0

        @register_metric("dup", duplicate=DuplicatePolicy.IGNORE)
        def second(r):
            return 2.0

        assert registry.get(ExtensionKind.METRIC, "dup").target([]) == 1.0


# =========================================================================
# Registration scope
# =========================================================================


class TestScope:
    def test_builtin_survives_clear(self):
        registry.register(
            ExtensionKind.METRIC,
            "builtin_m",
            lambda r: 1.0,
            scope=Scope.BUILTIN,
        )

        @register_metric("user_m")
        def user_m(r):
            return 2.0

        clear_registry()

        assert registry.get(ExtensionKind.METRIC, "builtin_m") is not None
        assert registry.get(ExtensionKind.METRIC, "user_m") is None

    def test_local_scope_rolls_back_after_isolation(self):
        with isolated_registry():
            registry.register(
                ExtensionKind.METRIC,
                "temp_m",
                lambda r: 0.0,
                scope=Scope.LOCAL,
            )
            assert registry.get(ExtensionKind.METRIC, "temp_m") is not None

        assert registry.get(ExtensionKind.METRIC, "temp_m") is None

    def test_clear_can_target_a_scope(self):
        registry.register(ExtensionKind.METRIC, "builtin_m", lambda r: 1.0, scope=Scope.BUILTIN)

        @register_metric("user_m")
        def user_m(r):
            return 2.0

        clear_registry(scope=Scope.LOCAL)
        assert registry.get(ExtensionKind.METRIC, "user_m") is not None
        assert registry.get(ExtensionKind.METRIC, "builtin_m") is not None

        clear_registry(include_builtins=True)
        assert registry.get(ExtensionKind.METRIC, "builtin_m") is None


# =========================================================================
# Thread lock
# =========================================================================


class TestThreadLock:
    def test_concurrent_registration_is_consistent(self):
        def worker(i):
            @register_metric(f"thr_{i}")
            def fn(r):
                return i

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(32)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        names = registry.metric_names()
        assert all(f"thr_{i}" in names for i in range(32))
        assert registry.get(ExtensionKind.METRIC, "thr_7").target([]) == 7

    def test_concurrent_hook_registration_keeps_priority_order(self):
        def worker(priority):
            def hook(data):
                return data

            registry.register(ExtensionKind.HOOK, "stress", hook, priority=priority)

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(16)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        priorities = [entry.priority for entry in registry.hooks("stress")]
        assert priorities == sorted(priorities)
        assert len(priorities) == 16


# =========================================================================
# Isolation context manager
# =========================================================================


class TestIsolationContextManager:
    def test_restores_prior_state(self):
        @register_metric("keep")
        def keep(r):
            return 1.0

        with isolated_registry():

            @register_metric("temp")
            def temp(r):
                return 2.0

            assert registry.get(ExtensionKind.METRIC, "temp") is not None
            assert registry.get(ExtensionKind.METRIC, "keep") is not None

        assert registry.get(ExtensionKind.METRIC, "temp") is None
        assert registry.get(ExtensionKind.METRIC, "keep").target([]) == 1.0

    def test_nested_isolation(self):
        with isolated_registry():

            @register_metric("outer")
            def outer(r):
                return 1

            with isolated_registry():

                @register_metric("inner")
                def inner(r):
                    return 2

                assert registry.get(ExtensionKind.METRIC, "outer") is not None
                assert registry.get(ExtensionKind.METRIC, "inner") is not None

            assert registry.get(ExtensionKind.METRIC, "inner") is None
            assert registry.get(ExtensionKind.METRIC, "outer") is not None

        assert registry.get(ExtensionKind.METRIC, "outer") is None

    def test_exception_in_block_still_restores(self):
        with pytest.raises(RuntimeError, match="boom"), isolated_registry():

            @register_metric("boom")
            def boom(r):
                return 1

            raise RuntimeError("boom")

        assert registry.get(ExtensionKind.METRIC, "boom") is None
