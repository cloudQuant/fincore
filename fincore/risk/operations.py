"""Explicit canonical operation declarations for risk analytics."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from fincore.runtime.specs import OperationSpec, make_operations_provider

from .backtesting import backtest_es, backtest_var, christoffersen_lr, kupiec_lr
from .calibration import basel_traffic_light, es_calibration_score, expected_exception_count
from .diagnostics import walk_forward_var
from .evt import evt_cvar, evt_var, extreme_risk, gev_fit, gpd_fit, hill_estimator
from .garch import conditional_es, conditional_var, forecast_volatility
from .models import forecast_es, forecast_var
from .report import build_risk_validation_report

if TYPE_CHECKING:
    from collections.abc import Callable

__all__ = ["operations"]

_BINDINGS: tuple[tuple[str, Callable[..., Any]], ...] = (
    ("risk.backtesting.backtest_es", backtest_es),
    ("risk.backtesting.backtest_var", backtest_var),
    ("risk.backtesting.christoffersen_lr", christoffersen_lr),
    ("risk.backtesting.kupiec_lr", kupiec_lr),
    ("risk.calibration.basel_traffic_light", basel_traffic_light),
    ("risk.calibration.es_calibration_score", es_calibration_score),
    ("risk.calibration.expected_exception_count", expected_exception_count),
    ("risk.diagnostics.walk_forward_var", walk_forward_var),
    ("risk.evt.evt_cvar", evt_cvar),
    ("risk.evt.evt_var", evt_var),
    ("risk.evt.extreme_risk", extreme_risk),
    ("risk.evt.gev_fit", gev_fit),
    ("risk.evt.gpd_fit", gpd_fit),
    ("risk.evt.hill_estimator", hill_estimator),
    ("risk.garch.conditional_es", conditional_es),
    ("risk.garch.conditional_var", conditional_var),
    ("risk.garch.forecast_volatility", forecast_volatility),
    ("risk.models.forecast_es", forecast_es),
    ("risk.models.forecast_var", forecast_var),
    ("risk.report.build_risk_validation_report", build_risk_validation_report),
)

_OPERATIONS = tuple(
    OperationSpec(
        operation_id=operation_id,
        capability_id=operation_id,
        domain="risk",
        callable=callable_,
        provenance={"owner": "risk", "kernel_module": callable_.__module__},
    )
    for operation_id, callable_ in _BINDINGS
)


operations = make_operations_provider(_OPERATIONS)
