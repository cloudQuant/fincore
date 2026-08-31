"""Canonical 0.5 execution primitives.

This package deliberately owns orchestration only. Domain packages provide
their direct callable implementations; the runtime records and executes those
same callables without creating public-path adapters or compatibility profiles.
"""

from __future__ import annotations

from .catalog import OperationCatalog
from .data import AnalysisSnapshot
from .engine import run
from .results import Result
from .session import AnalysisSession
from .specs import OperationSpec

__all__ = ["AnalysisSession", "AnalysisSnapshot", "OperationCatalog", "OperationSpec", "Result", "run"]
