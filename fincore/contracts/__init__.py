"""Shared boundary contracts for fincore's enhanced APIs."""

from fincore.contracts.factor_analysis import ALPHALENS_FUNCTION_SPECS, FactorFunctionSpec
from fincore.contracts.factor_workflows import ALPHALENS_WORKFLOW_SPECS, FactorWorkflowSpec
from fincore.contracts.portfolio import ExposureBundle, VolumeExposureBundle
from fincore.contracts.time_series import AlignmentPolicy, align_binary_metric_inputs, align_time_series
from fincore.contracts.validation import (
    ContextInputs,
    ValidationProfile,
    validate_context_inputs,
    validate_factors_schema,
    validate_market_data_schema,
    validate_positions_schema,
    validate_returns_schema,
    validate_transactions_schema,
)
from fincore.contracts.workflows import WorkflowSpec

__all__ = [
    "ALPHALENS_FUNCTION_SPECS",
    "ALPHALENS_WORKFLOW_SPECS",
    "AlignmentPolicy",
    "ContextInputs",
    "ExposureBundle",
    "FactorFunctionSpec",
    "FactorWorkflowSpec",
    "ValidationProfile",
    "VolumeExposureBundle",
    "WorkflowSpec",
    "align_binary_metric_inputs",
    "align_time_series",
    "validate_context_inputs",
    "validate_factors_schema",
    "validate_market_data_schema",
    "validate_positions_schema",
    "validate_returns_schema",
    "validate_transactions_schema",
]
