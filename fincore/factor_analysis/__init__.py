"""Future enhanced factor-analysis package boundary.

Task 2 intentionally publishes static contracts only.  Task 3 owns actual
factor-data preparation and exception behavior; no placeholder results are
fabricated here.
"""

from __future__ import annotations

from fincore.contracts.factor_analysis import FactorFunctionSpec
from fincore.contracts.factor_workflows import FactorWorkflowSpec

__all__ = ["FactorFunctionSpec", "FactorWorkflowSpec"]
