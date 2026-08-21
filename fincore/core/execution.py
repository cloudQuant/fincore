"""Deterministic DAG execution with an in-process, snapshot-scoped cache.

Each node runs at most once for a particular semantic node key; results are
shared by every downstream consumer.  The cache is process-local and records
both the stable node identity and its semantic ``cache_key``.  Keying only by
``node_id`` would return a stale result after a planner changes the data,
profile, backend, or plugin overlay encoded by that semantic key.
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
    _memo: dict[tuple[str, str], Any] = field(default_factory=dict)

    def execute(self, node_id: str, **inputs: Any) -> Any:
        node = self.plan.node_map[node_id]
        memo_key = (node.node_id, node.cache_key)
        if memo_key in self._memo:
            return self._memo[memo_key]
        resolved = {dep: self.execute(dep, **inputs) for dep in node.dependencies}
        value = node.compute(**resolved, **inputs)
        self._memo[memo_key] = value
        return value

    def computed(self) -> frozenset[str]:
        return frozenset(node_id for node_id, _ in self._memo)
