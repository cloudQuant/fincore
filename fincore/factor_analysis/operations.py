"""Explicit canonical operation declarations for factor analysis."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from fincore.runtime.specs import OperationSpec, make_operations_provider

from .analysis import analyze_factor
from .calendar import (
    add_custom_calendar_timedelta,
    backshift_returns_series,
    diff_custom_calendar_timedeltas,
    get_forward_returns_columns,
    infer_trading_calendar,
    timedelta_strings_to_integers,
    timedelta_to_string,
)
from .costs import apply_factor_costs, estimate_factor_capacity
from .data import (
    compute_forward_returns,
    prepare_factor_data,
    prepare_factor_data_by_horizon,
    prepare_factor_data_from_forward_returns,
    prepare_pit_factor_data,
    quantize_factor,
)
from .inference import (
    benjamini_hochberg,
    factor_model_inference,
    fama_macbeth,
    ic_confidence_interval,
    ic_mean,
    ic_t_stat,
    information_coefficient_inference,
)
from .performance import (
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
from .pit import materialize_pit_factor, validate_pit_alignment
from .portfolio import build_factor_portfolio_inputs, factor_cumulative_returns, factor_positions, positions

if TYPE_CHECKING:
    from collections.abc import Callable

__all__ = ["operations"]

_BINDINGS: tuple[tuple[str, Callable[..., Any]], ...] = (
    ("factor_analysis.analysis.analyze_factor", analyze_factor),
    ("factor_analysis.calendar.add_custom_calendar_timedelta", add_custom_calendar_timedelta),
    ("factor_analysis.calendar.backshift_returns_series", backshift_returns_series),
    ("factor_analysis.calendar.diff_custom_calendar_timedeltas", diff_custom_calendar_timedeltas),
    ("factor_analysis.calendar.get_forward_returns_columns", get_forward_returns_columns),
    ("factor_analysis.calendar.infer_trading_calendar", infer_trading_calendar),
    ("factor_analysis.calendar.timedelta_strings_to_integers", timedelta_strings_to_integers),
    ("factor_analysis.calendar.timedelta_to_string", timedelta_to_string),
    ("factor_analysis.costs.apply_factor_costs", apply_factor_costs),
    ("factor_analysis.costs.estimate_factor_capacity", estimate_factor_capacity),
    ("factor_analysis.data.compute_forward_returns", compute_forward_returns),
    ("factor_analysis.data.prepare_factor_data", prepare_factor_data),
    ("factor_analysis.data.prepare_factor_data_by_horizon", prepare_factor_data_by_horizon),
    ("factor_analysis.data.prepare_factor_data_from_forward_returns", prepare_factor_data_from_forward_returns),
    ("factor_analysis.data.prepare_pit_factor_data", prepare_pit_factor_data),
    ("factor_analysis.data.quantize_factor", quantize_factor),
    ("factor_analysis.inference.benjamini_hochberg", benjamini_hochberg),
    ("factor_analysis.inference.factor_model_inference", factor_model_inference),
    ("factor_analysis.inference.fama_macbeth", fama_macbeth),
    ("factor_analysis.inference.ic_confidence_interval", ic_confidence_interval),
    ("factor_analysis.inference.ic_mean", ic_mean),
    ("factor_analysis.inference.ic_t_stat", ic_t_stat),
    ("factor_analysis.inference.information_coefficient_inference", information_coefficient_inference),
    ("factor_analysis.performance.average_cumulative_return_by_quantile", average_cumulative_return_by_quantile),
    ("factor_analysis.performance.common_start_returns", common_start_returns),
    ("factor_analysis.performance.compute_mean_returns_spread", compute_mean_returns_spread),
    ("factor_analysis.performance.cumulative_returns", cumulative_returns),
    ("factor_analysis.performance.factor_alpha_beta", factor_alpha_beta),
    ("factor_analysis.performance.factor_information_coefficient", factor_information_coefficient),
    ("factor_analysis.performance.factor_rank_autocorrelation", factor_rank_autocorrelation),
    ("factor_analysis.performance.factor_returns", factor_returns),
    ("factor_analysis.performance.factor_weights", factor_weights),
    ("factor_analysis.performance.mean_information_coefficient", mean_information_coefficient),
    ("factor_analysis.performance.mean_return_by_quantile", mean_return_by_quantile),
    ("factor_analysis.performance.quantile_turnover", quantile_turnover),
    ("factor_analysis.pit.materialize_pit_factor", materialize_pit_factor),
    ("factor_analysis.pit.validate_pit_alignment", validate_pit_alignment),
    ("factor_analysis.portfolio.build_factor_portfolio_inputs", build_factor_portfolio_inputs),
    ("factor_analysis.portfolio.factor_cumulative_returns", factor_cumulative_returns),
    ("factor_analysis.portfolio.factor_positions", factor_positions),
    ("factor_analysis.portfolio.positions", positions),
)

_OPERATIONS = tuple(
    OperationSpec(
        operation_id=operation_id,
        capability_id=operation_id,
        domain="factor_analysis",
        callable=callable_,
        provenance={"owner": "factor_analysis", "kernel_module": callable_.__module__},
    )
    for operation_id, callable_ in _BINDINGS
)


operations = make_operations_provider(_OPERATIONS)
