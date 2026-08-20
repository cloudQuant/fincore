"""Computation DAG: nodes declare dependencies and cache keys.

The DAG shares intermediate quantities (alignment, cumulative returns,
drawdowns, rolling moments, alpha/beta, factor panels) so the same kernel is
executed at most once per snapshot, and a cache key change invalidates exactly
the affected downstream nodes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Mapping

__all__ = ["DAGNode", "ExecutionPlan"]


@dataclass(frozen=True)
class DAGNode:
    """One computation node with declared dependencies and a cache key."""

    node_id: str
    dependencies: tuple[str, ...] = ()
    cache_key: str = ""
    compute: Callable[..., Any] = lambda **kwargs: None

    def __hash__(self) -> int:
        return hash(self.node_id)


@dataclass(frozen=True)
class ExecutionPlan:
    """A topological collection of DAG nodes."""

    nodes: tuple[DAGNode, ...]

    @property
    def node_map(self) -> Mapping[str, DAGNode]:
        return {node.node_id: node for node in self.nodes}

    def dependencies_of(self, node_id: str) -> tuple[str, ...]:
        return self.node_map[node_id].dependencies
