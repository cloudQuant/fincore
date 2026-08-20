"""AnalysisSnapshot and computation DAG tests."""

from __future__ import annotations

import pandas as pd

from fincore.core.execution import DAGExecutor
from fincore.core.planner import DAGNode, ExecutionPlan
from fincore.core.snapshot import AnalysisSnapshot


def test_snapshot_is_copy_on_ingest() -> None:
    returns = pd.Series([0.01, -0.02, 0.03])
    snapshot = AnalysisSnapshot.from_data(returns)
    assert snapshot.returns is not returns
    assert list(snapshot.returns) == list(returns)


def test_snapshot_cache_key_changes_with_data() -> None:
    a = AnalysisSnapshot.from_data(pd.Series([0.01, -0.02]))
    b = AnalysisSnapshot.from_data(pd.Series([0.01, -0.03]))
    assert a.cache_key != b.cache_key


def test_snapshot_cache_key_changes_with_profile() -> None:
    returns = pd.Series([0.01, -0.02])
    a = AnalysisSnapshot.from_data(returns, profile="enhanced_v1")
    b = AnalysisSnapshot.from_data(returns, profile="plugin_v1")
    assert a.cache_key != b.cache_key


def test_snapshot_cache_key_changes_with_overlay_generation() -> None:
    returns = pd.Series([0.01, -0.02])
    a = AnalysisSnapshot.from_data(returns, overlay_generation=0)
    b = AnalysisSnapshot.from_data(returns, overlay_generation=1)
    assert a.cache_key != b.cache_key


def test_dag_executes_dependencies_once() -> None:
    calls: dict[str, int] = {}

    def make(node_id: str):
        def compute(**kwargs: object) -> float:
            calls[node_id] = calls.get(node_id, 0) + 1
            return 1.0

        return compute

    plan = ExecutionPlan(
        (
            DAGNode("base", cache_key="k", compute=make("base")),
            DAGNode("mid", dependencies=("base",), cache_key="k", compute=make("mid")),
            DAGNode("top", dependencies=("mid",), cache_key="k", compute=make("top")),
        )
    )
    executor = DAGExecutor(plan)
    executor.execute("top")
    executor.execute("top")

    assert calls == {"base": 1, "mid": 1, "top": 1}


def test_dag_shares_intermediate_values() -> None:
    calls: dict[str, int] = {}

    def make(node_id: str):
        def compute(**kwargs: object) -> float:
            calls[node_id] = calls.get(node_id, 0) + 1
            return 2.0

        return compute

    plan = ExecutionPlan(
        (
            DAGNode("shared", cache_key="k", compute=make("shared")),
            DAGNode("left", dependencies=("shared",), cache_key="k", compute=make("left")),
            DAGNode("right", dependencies=("shared",), cache_key="k", compute=make("right")),
        )
    )
    executor = DAGExecutor(plan)
    executor.execute("left")
    executor.execute("right")

    assert calls["shared"] == 1
    assert executor.computed() == frozenset({"shared", "left", "right"})
