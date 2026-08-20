"""Deterministic DAG execution with an in-process, snapshot-scoped cache.

Each node runs at most once per snapshot; results are cached by node id so a
node's computed value is shared by every downstream consumer.  The cache is
process-local and keyed by the node's ``cache_key`` (which already embeds data
content, profile, backend, and overlay generation).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from fincore.core.planner import ExecutionPlan

__all__ = ["DAGExecutor"]


@dataclass
class DAGExecutor:
    """Executes an ExecutionPlan, memoizing node results in-process."""

    plan: ExecutionPlan
    _memo: dict[str, Any] = field(default_factory=dict)

    def execute(self, node_id: str, **inputs: Any) -> Any:
        if node_id in self._memo:
            return self._memo[node_id]
        node = self.plan.node_map[node_id]
        resolved = {dep: self.execute(dep, **inputs) for dep in node.dependencies}
        value = node.compute(**resolved, **inputs)
        self._memo[node_id] = value
        return value

    def computed(self) -> frozenset[str]:
        return frozenset(self._memo)
