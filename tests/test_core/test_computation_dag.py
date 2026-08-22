"""AnalysisSnapshot and computation DAG tests."""

from __future__ import annotations

from dataclasses import replace

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


def test_snapshot_cache_key_changes_with_every_analysis_input() -> None:
    """A snapshot key must not reuse a result after auxiliary data changes."""
    returns = pd.Series([0.01, -0.02], index=pd.date_range("2024-01-01", periods=2))
    benchmark = pd.Series([0.0, 0.0], index=returns.index)
    positions = pd.DataFrame({"asset": [1.0, 0.5]}, index=returns.index)
    transactions = pd.DataFrame({"amount": [1.0, -1.0]}, index=returns.index)

    baseline = AnalysisSnapshot.from_data(
        returns,
        benchmark=benchmark,
        positions=positions,
        transactions=transactions,
        backend="pandas",
        overlay_digest="overlay-a",
        operation_version="2",
        config={"annualization": 252},
    )

    variants = (
        AnalysisSnapshot.from_data(
            returns,
            benchmark=benchmark + 0.001,
            positions=positions,
            transactions=transactions,
            backend="pandas",
            overlay_digest="overlay-a",
            operation_version="2",
            config={"annualization": 252},
        ),
        AnalysisSnapshot.from_data(
            returns,
            benchmark=benchmark,
            positions=positions.assign(asset=[0.9, 0.5]),
            transactions=transactions,
            backend="pandas",
            overlay_digest="overlay-a",
            operation_version="2",
            config={"annualization": 252},
        ),
        AnalysisSnapshot.from_data(
            returns,
            benchmark=benchmark,
            positions=positions,
            transactions=transactions.assign(amount=[1.0, -2.0]),
            backend="pandas",
            overlay_digest="overlay-a",
            operation_version="2",
            config={"annualization": 252},
        ),
        AnalysisSnapshot.from_data(
            returns,
            benchmark=benchmark,
            positions=positions,
            transactions=transactions,
            backend="numpy",
            overlay_digest="overlay-a",
            operation_version="2",
            config={"annualization": 252},
        ),
        AnalysisSnapshot.from_data(
            returns,
            benchmark=benchmark,
            positions=positions,
            transactions=transactions,
            backend="pandas",
            overlay_digest="overlay-b",
            operation_version="2",
            config={"annualization": 252},
        ),
        AnalysisSnapshot.from_data(
            returns,
            benchmark=benchmark,
            positions=positions,
            transactions=transactions,
            backend="pandas",
            overlay_digest="overlay-a",
            operation_version="3",
            config={"annualization": 252},
        ),
        AnalysisSnapshot.from_data(
            returns,
            benchmark=benchmark,
            positions=positions,
            transactions=transactions,
            backend="pandas",
            overlay_digest="overlay-a",
            operation_version="2",
            config={"annualization": 365},
        ),
    )

    assert all(variant.cache_key != baseline.cache_key for variant in variants)


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


def test_dag_cache_uses_node_semantic_key_not_only_node_id() -> None:
    """Changing a node semantic key must invalidate an existing executor cache."""
    calls: list[str] = []

    def first(**kwargs: object) -> str:
        calls.append("first")
        return "first"

    def second(**kwargs: object) -> str:
        calls.append("second")
        return "second"

    original = ExecutionPlan((DAGNode("metric", cache_key="semantic-v1", compute=first),))
    executor = DAGExecutor(original)
    assert executor.execute("metric") == "first"

    executor.plan = replace(original, nodes=(DAGNode("metric", cache_key="semantic-v2", compute=second),))
    assert executor.execute("metric") == "second"
    assert calls == ["first", "second"]
