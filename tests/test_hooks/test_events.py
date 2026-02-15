"""Tests for fincore.hooks.events.

This module is separate from fincore.plugin.registry hooks. Here we validate:
- event registry copy semantics
- hook execution order
- data transformation via hook return value
- context managers update their stored data
"""

from __future__ import annotations

import pytest

from fincore.hooks.events import (
    AnalysisContext,
    ComputeContext,
    OptimizationContext,
    clear_hooks,
    create_analysis_context,
    create_compute_context,
    create_optimization_context,
    execute_hooks,
    get_event_hooks,
    list_events,
    register_event_hook,
)


@pytest.fixture(autouse=True)
def _clean_event_hooks():
    clear_hooks()
    yield
    clear_hooks()


def test_register_event_hook_unknown_event_raises():
    with pytest.raises(ValueError, match="Unknown event"):
        register_event_hook("nope", lambda x: x)


def test_get_event_hooks_returns_copies():
    register_event_hook("pre_analysis", lambda x: x)

    hooks_by_event = get_event_hooks()
    assert "pre_analysis" in hooks_by_event

    hooks_by_event["pre_analysis"].clear()
    assert len(get_event_hooks("pre_analysis")) == 1


def test_execute_hooks_returns_transformed_first_arg():
    register_event_hook("pre_compute", lambda x: x + 1)
    register_event_hook("pre_compute", lambda x: x * 10)

    out = execute_hooks("pre_compute", 2)
    assert out == 30


def test_execute_hooks_with_no_args_returns_none():
    register_event_hook("post_compute", lambda: 1)
    assert execute_hooks("post_compute") is None


def test_clear_hooks_can_clear_single_event():
    register_event_hook("pre_analysis", lambda x: x)
    register_event_hook("post_analysis", lambda x: x)

    clear_hooks("pre_analysis")
    assert get_event_hooks("pre_analysis") == []
    assert len(get_event_hooks("post_analysis")) == 1


def test_context_managers_update_stored_data():
    register_event_hook("pre_analysis", lambda x: x + 1)
    register_event_hook("post_analysis", lambda x: x * 2)
    register_event_hook("pre_compute", lambda x: x.replace("a", "b"))
    register_event_hook("optimization", lambda x: x * 3)

    with AnalysisContext(returns=1) as ctx:
        assert ctx.returns == 2
    assert ctx.returns == 4

    with ComputeContext(data="a") as cctx:
        assert cctx.data == "b"

    with OptimizationContext(returns=2) as octx:
        assert octx.returns == 6


def test_list_events_contains_known_events():
    events = list_events()
    for name in ["pre_analysis", "post_analysis", "pre_compute", "post_compute", "optimization"]:
        assert name in events


def test_factory_functions_create_contexts():
    a = create_analysis_context(returns=1)
    c = create_compute_context(data="x")
    o = create_optimization_context(returns=2)
    assert isinstance(a, AnalysisContext)
    assert isinstance(c, ComputeContext)
    assert isinstance(o, OptimizationContext)
