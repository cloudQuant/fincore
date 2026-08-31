"""Canonical 0.5 execution primitives.

This package deliberately owns orchestration only. Domain packages provide
their direct callable implementations; the runtime records and executes those
same callables without creating public-path adapters or compatibility profiles.
"""

from __future__ import annotations

from .artifacts import ArtifactBundle
from .backends import NumPyBackend
from .catalog import OperationCatalog
from .data import AnalysisSnapshot
from .engine import ExecutionPlan, OperationRequest, batch, plan, run
from .results import Result
from .session import AnalysisSession
from .specs import OperationSpec
from .validation import validate_finite_array, validate_mapping

__all__ = [
    "AnalysisSession",
    "AnalysisSnapshot",
    "ArtifactBundle",
    "ExecutionPlan",
    "NumPyBackend",
    "OperationCatalog",
    "OperationRequest",
    "OperationSpec",
    "Result",
    "batch",
    "plan",
    "run",
    "validate_finite_array",
    "validate_mapping",
]
