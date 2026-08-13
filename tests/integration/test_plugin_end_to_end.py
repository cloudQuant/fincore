"""End-to-end tests: extensions registered in the unified registry are used by real consumers.

Covers:
- A registered viz backend is actually used by ``AnalysisContext.plot``.
- A registered metric is actually used by ``AnalysisContext.compute``.
- ``RollingEngine.available_metrics`` is sourced from the registry.
- Hooks registered through ``fincore.plugin`` fire through the hook context managers
  (one registry, one hook pipeline).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from fincore.core.context import AnalysisContext
from fincore.core.engine import RollingEngine
from fincore.plugin import (
    isolated_registry,
    register_hook,
    register_metric,
    register_viz_backend,
)
from fincore.plugin.registry import registry
from fincore.plugin.specs import ROLLING_FAMILY
from fincore.report.artifacts import ReportArtifacts


@pytest.fixture
def returns():
    """Simple synthetic daily returns."""
    np.random.seed(42)
    return pd.Series(
        np.random.randn(504) * 0.01,
        index=pd.bdate_range("2020-01-01", periods=504),
    )


@pytest.fixture(autouse=True)
def _isolate_registry():
    """Every test runs against a pristine registry state and leaves none behind."""
    with isolated_registry():
        yield


def test_registered_backend_is_used_by_context(returns) -> None:
    @register_viz_backend("recording")
    class RecordingBackend:
        def render(self, model, **kwargs):
            return ReportArtifacts(backend="recording", metadata={"rows": len(model.returns)})

    result = AnalysisContext(returns).plot(backend="recording")
    assert result.metadata["rows"] == len(returns)


def test_registered_metric_is_used_by_analysis(returns) -> None:
    @register_metric("positive_rate")
    def positive_rate(values):
        return float((values > 0).mean())

    assert AnalysisContext(returns).compute("positive_rate") == positive_rate(returns)


def test_compute_unknown_metric_raises(returns) -> None:
    ctx = AnalysisContext(returns)
    with pytest.raises(ValueError, match="Unknown metric"):
        ctx.compute("does_not_exist")


def test_engine_available_metrics_come_from_registry(returns) -> None:
    engine = RollingEngine(returns, window=60)

    assert engine.available_metrics == registry.metric_names(family=ROLLING_FAMILY)
    assert {"sharpe", "volatility", "max_drawdown", "beta", "sortino", "mean_return"} <= engine.available_metrics


def test_plugin_registered_hook_fires_via_hook_context() -> None:
    """Hooks registered through the plugin API run in the hooks context managers."""
    from fincore.hooks.events import AnalysisHookContext

    calls = []

    @register_hook("pre_analysis", priority=50)
    def touch_and_transform(data):
        calls.append(data)
        return data + 1

    with AnalysisHookContext(1) as ctx:
        assert ctx.returns == 2

    assert calls == [1]
