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
from fincore.factor_analysis.performance import (
    average_cumulative_return_by_quantile,
    common_start_returns,
    compute_mean_returns_spread,
    cumulative_returns,
    factor_alpha_beta,
    factor_information_coefficient,
    factor_rank_autocorrelation,
    factor_returns,
    factor_weights,
    mean_information_coefficient,
    mean_return_by_quantile,
    quantile_turnover,
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
    "average_cumulative_return_by_quantile",
    "backshift_returns_series",
    "common_start_returns",
    "compute_forward_returns",
    "compute_mean_returns_spread",
    "cumulative_returns",
    "diff_custom_calendar_timedeltas",
    "factor_alpha_beta",
    "factor_information_coefficient",
    "factor_rank_autocorrelation",
    "factor_returns",
    "factor_weights",
    "get_forward_returns_columns",
    "infer_trading_calendar",
    "mean_information_coefficient",
    "mean_return_by_quantile",
    "prepare_factor_data",
    "prepare_factor_data_from_forward_returns",
    "quantile_turnover",
    "quantize_factor",
    "timedelta_strings_to_integers",
    "timedelta_to_string",
]
