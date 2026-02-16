"""Tests for fincore plugin registry system.

Covers:
- register_metric: registers function (not result), available at decoration time
- register_viz_backend: registers class (not instance), available at decoration time
- register_hook: registers at decoration time, priority ordering, execution order
- Duplicate registration strategy
- clear_registry
"""

from __future__ import annotations

import numpy as np
import pytest

from fincore.plugin.registry import (
    clear_registry,
    execute_hooks,
    get_metric,
    get_viz_backend,
    list_hooks,
    list_metrics,
    list_viz_backends,
    register_hook,
    register_metric,
    register_viz_backend,
)


@pytest.fixture(autouse=True)
def _clean_registries():
    """Clear all registries before and after each test."""
    clear_registry()
    yield
    clear_registry()


# =========================================================================
# register_metric
# =========================================================================


class TestRegisterMetric:
    """Tests for the register_metric decorator."""

    def test_registers_function_not_result(self):
        """Registered value should be the callable, not its return value."""

        @register_metric("my_mean")
        def my_mean(returns):
            return float(np.mean(returns))

        registered = get_metric("my_mean")
        assert callable(registered)
        assert registered([1, 2, 3]) == 2.0

    def test_registers_at_decoration_time(self):
        """Metric should appear in registry immediately after decoration."""

        @register_metric("early_metric")
        def early_metric(returns):
            return 0.0

        assert "early_metric" in list_metrics()

    def test_decorated_function_still_works(self):
        """The decorated function should remain callable and return correct results."""

        @register_metric("sharpe_test")
        def sharpe_test(returns, period=252):
            mean = np.mean(returns)
            std = np.std(returns, ddof=1)
            return mean / std * np.sqrt(period)

        result = sharpe_test([0.01, 0.02, -0.01, 0.03])
        assert isinstance(result, float)

    def test_name_defaults_to_function_name(self):
        """When name is None, function __name__ should be used."""

        @register_metric()
        def auto_named_metric(returns):
            return 0.0

        assert "auto_named_metric" in list_metrics()

    def test_duplicate_registration_overwrites(self):
        """Re-registering the same name should overwrite the previous entry."""

        @register_metric("dup")
        def first(returns):
            return 1.0

        @register_metric("dup")
        def second(returns):
            return 2.0

        assert get_metric("dup")([]) == 2.0


# =========================================================================
# register_viz_backend
# =========================================================================


class TestRegisterVizBackend:
    """Tests for the register_viz_backend decorator."""

    def test_registers_class_not_instance(self):
        """Registered value should be the class, not an instance."""

        @register_viz_backend("test_be")
        class TestBE:
            def __init__(self, theme="light"):
                self.theme = theme

        registered = get_viz_backend("test_be")
        assert isinstance(registered, type)
        assert registered.__name__ == "TestBE"

    def test_registers_at_decoration_time(self):
        """Backend class should appear in registry immediately."""

        @register_viz_backend("early_be")
        class EarlyBE:
            pass

        assert "early_be" in list_viz_backends()

    def test_instantiation_works_after_registration(self):
        """Instantiating the registered class should work normally."""

        @register_viz_backend("inst_be")
        class InstBE:
            def __init__(self, theme="dark"):
                self.theme = theme

            def plot_returns(self, data):
                return data

        cls = get_viz_backend("inst_be")
        inst = cls(theme="ocean")
        assert inst.theme == "ocean"
        assert inst.plot_returns([1, 2]) == [1, 2]


# =========================================================================
# register_hook
# =========================================================================


class TestRegisterHook:
    """Tests for the register_hook decorator."""

    def test_registers_at_decoration_time(self):
        """Hook should appear in registry immediately after decoration."""

        @register_hook("pre_analysis")
        def my_hook(data):
            pass

        hooks = list_hooks("pre_analysis")
        assert len(hooks["pre_analysis"]) == 1

    def test_priority_ordering(self):
        """Hooks should execute in ascending priority order (lowest first)."""
        call_order = []

        @register_hook("pre_analysis", priority=300)
        def low_prio(data):
            call_order.append("low")

        @register_hook("pre_analysis", priority=10)
        def high_prio(data):
            call_order.append("high")

        @register_hook("pre_analysis", priority=100)
        def mid_prio(data):
            call_order.append("mid")

        execute_hooks("pre_analysis", None)
        assert call_order == ["high", "mid", "low"]

    def test_default_priority_is_100(self):
        """Without explicit priority, hooks get priority=100."""
        call_order = []

        @register_hook("pre_compute", priority=50)
        def before_default(data):
            call_order.append("before")

        @register_hook("pre_compute")
        def default_prio(data):
            call_order.append("default")

        @register_hook("pre_compute", priority=200)
        def after_default(data):
            call_order.append("after")

        execute_hooks("pre_compute", None)
        assert call_order == ["before", "default", "after"]

    def test_execute_hooks_unknown_event_does_nothing(self):
        """Executing hooks for a nonexistent event should not error."""
        execute_hooks("nonexistent_event", None)

    def test_decorated_function_still_callable(self):
        """The decorated hook function should remain directly callable."""

        @register_hook("post_analysis")
        def transform(data):
            return data * 2

        assert transform(5) == 10


# =========================================================================
# clear_registry
# =========================================================================


class TestClearRegistry:
    """Tests for clear_registry."""

    def test_clear_all(self):
        @register_metric("x")
        def x(r):
            return 0

        @register_hook("pre_analysis")
        def h(d):
            pass

        @register_viz_backend("b")
        class B:
            pass

        clear_registry()
        assert list_metrics() == {}
        assert list_hooks() == {}
        assert list_viz_backends() == {}

    def test_clear_selective(self):
        @register_metric("x")
        def x(r):
            return 0

        @register_hook("pre_analysis")
        def h(d):
            pass

        clear_registry("metrics")
        assert list_metrics() == {}
        assert len(list_hooks()["pre_analysis"]) == 1
