"""Enhanced factor-analysis contracts and standalone Task 3 data kernel."""

from __future__ import annotations

from fincore.contracts.factor_analysis import FactorFunctionSpec
from fincore.contracts.factor_workflows import FactorWorkflowSpec
from fincore.factor_analysis.calendar import (
    add_custom_calendar_timedelta,
    backshift_returns_series,
    diff_custom_calendar_timedeltas,
    get_forward_returns_columns,
    infer_trading_calendar,
    timedelta_strings_to_integers,
    timedelta_to_string,
)
from fincore.factor_analysis.data import (
    FactorLossReport,
    PreparedFactorData,
    compute_forward_returns,
    prepare_factor_data,
    prepare_factor_data_from_forward_returns,
    quantize_factor,
)
from fincore.factor_analysis.exceptions import (
    EnhancedNonMatchingTimezoneError,
    FactorDataError,
    FactorLossExceededError,
    MaxLossExceededError,
    NonMatchingTimezoneError,
)

__all__ = [
    "EnhancedNonMatchingTimezoneError",
    "FactorDataError",
    "FactorFunctionSpec",
    "FactorLossExceededError",
    "FactorLossReport",
    "FactorWorkflowSpec",
    "MaxLossExceededError",
    "NonMatchingTimezoneError",
    "PreparedFactorData",
    "add_custom_calendar_timedelta",
    "backshift_returns_series",
    "compute_forward_returns",
    "diff_custom_calendar_timedeltas",
    "get_forward_returns_columns",
    "infer_trading_calendar",
    "prepare_factor_data",
    "prepare_factor_data_from_forward_returns",
    "quantize_factor",
    "timedelta_strings_to_integers",
    "timedelta_to_string",
]
