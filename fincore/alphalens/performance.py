"""Strict Alphalens performance facade backed by Task 4 kernels.

The module begins with the static C0/C1 deferred registry, then intentionally
overrides only functions whose numerical kernel has been characterized.  The
remaining Task 5 portfolio symbols keep their original deferred boundary.
"""

from __future__ import annotations

import importlib
from typing import Any

import pandas as pd  # noqa: TC002 - public facade annotations are runtime-reflectable.

from fincore.alphalens._compat import export_deferred_functions
from fincore.contracts.factor_analysis import ALPHALENS_FUNCTION_SPECS, FactorFunctionSpec
from fincore.exceptions import DependencyError
from fincore.factor_analysis import performance as _performance

_PERFORMANCE_NAMES = export_deferred_functions(globals(), "performance")


def _spec(name: str) -> FactorFunctionSpec:
    return ALPHALENS_FUNCTION_SPECS[("performance", name)]


def _deferred(name: str) -> None:
    spec = _spec(name)
    raise NotImplementedError(
        f"Legacy Alphalens symbol '{spec.public_name}' is available for C0/C1 compatibility, "
        "but its numerical or rendering kernel is not implemented yet."
    )


def _reject_opaque(name: str, *values: object) -> None:
    """Keep Task 2's opaque C1 grammar at the explicit implementation boundary."""

    if any(type(value) is object for value in values):
        _deferred(name)


def _attach_spec(function: Any, name: str) -> Any:
    spec = _spec(name)
    function.__signature__ = spec.introspection_signature
    function.__fincore_source_signature__ = spec.source_signature
    function.__fincore_factor_spec__ = spec
    return function


def factor_information_coefficient(
    factor_data: pd.DataFrame,
    group_adjust: bool = False,
    by_group: bool = False,
) -> pd.DataFrame:
    _reject_opaque("factor_information_coefficient", factor_data)
    return _performance.factor_information_coefficient(factor_data, group_adjust=group_adjust, by_group=by_group)


def mean_information_coefficient(
    factor_data: pd.DataFrame,
    group_adjust: bool = False,
    by_group: bool = False,
    by_time: str | None = None,
) -> pd.Series | pd.DataFrame:
    _reject_opaque("mean_information_coefficient", factor_data)
    return _performance.mean_information_coefficient(
        factor_data,
        group_adjust=group_adjust,
        by_group=by_group,
        by_time=by_time,
    )


def factor_weights(
    factor_data: pd.DataFrame,
    demeaned: bool = True,
    group_adjust: bool = False,
    equal_weight: bool = False,
) -> pd.Series:
    _reject_opaque("factor_weights", factor_data)
    return _performance.factor_weights(
        factor_data,
        demeaned=demeaned,
        group_adjust=group_adjust,
        equal_weight=equal_weight,
    )


def factor_returns(
    factor_data: pd.DataFrame,
    demeaned: bool = True,
    group_adjust: bool = False,
    equal_weight: bool = False,
    by_asset: bool = False,
) -> pd.DataFrame:
    _reject_opaque("factor_returns", factor_data)
    return _performance.factor_returns(
        factor_data,
        demeaned=demeaned,
        group_adjust=group_adjust,
        equal_weight=equal_weight,
        by_asset=by_asset,
    )


def _require_statsmodels() -> None:
    """Check the optional strict alpha/beta dependency only at call time."""

    try:
        importlib.import_module("statsmodels.regression.linear_model")
    except ModuleNotFoundError as exc:
        raise DependencyError(
            "factor_alpha_beta requires the optional 'factor-analysis' extra. "
            "Install it with:\n    pip install fincore[factor-analysis]",
            dependency="statsmodels",
        ) from exc


def factor_alpha_beta(
    factor_data: pd.DataFrame,
    returns: pd.DataFrame | pd.Series | None = None,
    demeaned: bool = True,
    group_adjust: bool = False,
    equal_weight: bool = False,
) -> pd.DataFrame:
    _reject_opaque("factor_alpha_beta", factor_data, returns)
    _require_statsmodels()
    return _performance.factor_alpha_beta(
        factor_data,
        returns=returns,
        demeaned=demeaned,
        group_adjust=group_adjust,
        equal_weight=equal_weight,
    )


def cumulative_returns(returns: pd.Series) -> pd.Series:
    _reject_opaque("cumulative_returns", returns)
    return _performance.cumulative_returns(returns)  # type: ignore[return-value]


def mean_return_by_quantile(
    factor_data: pd.DataFrame,
    by_date: bool = False,
    by_group: bool = False,
    demeaned: bool = True,
    group_adjust: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    _reject_opaque("mean_return_by_quantile", factor_data)
    return _performance.mean_return_by_quantile(
        factor_data,
        by_date=by_date,
        by_group=by_group,
        demeaned=demeaned,
        group_adjust=group_adjust,
    )


def compute_mean_returns_spread(
    mean_returns: pd.DataFrame,
    upper_quant: int,
    lower_quant: int,
    std_err: pd.DataFrame | None = None,
) -> tuple[pd.Series | pd.DataFrame, pd.Series | pd.DataFrame | None]:
    _reject_opaque("compute_mean_returns_spread", mean_returns)
    return _performance.compute_mean_returns_spread(mean_returns, upper_quant, lower_quant, std_err=std_err)


def quantile_turnover(quantile_factor: pd.Series, quantile: int, period: int = 1) -> pd.Series:
    _reject_opaque("quantile_turnover", quantile_factor)
    return _performance.quantile_turnover(quantile_factor, quantile, period=period)


def factor_rank_autocorrelation(factor_data: pd.DataFrame, period: int = 1) -> pd.Series:
    _reject_opaque("factor_rank_autocorrelation", factor_data)
    return _performance.factor_rank_autocorrelation(factor_data, period=period)


def common_start_returns(
    factor: pd.Series | pd.DataFrame,
    returns: pd.DataFrame,
    before: int,
    after: int,
    cumulative: bool = False,
    mean_by_date: bool = False,
    demean_by: pd.Series | pd.DataFrame | None = None,
) -> pd.DataFrame:
    _reject_opaque("common_start_returns", factor, returns)
    return _performance.common_start_returns(
        factor,
        returns,
        before,
        after,
        cumulative=cumulative,
        mean_by_date=mean_by_date,
        demean_by=demean_by,
    )


def average_cumulative_return_by_quantile(
    factor_data: pd.DataFrame,
    returns: pd.DataFrame,
    periods_before: int = 10,
    periods_after: int = 15,
    demeaned: bool = True,
    group_adjust: bool = False,
    by_group: bool = False,
) -> pd.DataFrame:
    _reject_opaque("average_cumulative_return_by_quantile", factor_data, returns)
    return _performance.average_cumulative_return_by_quantile(
        factor_data,
        returns,
        periods_before=periods_before,
        periods_after=periods_after,
        demeaned=demeaned,
        group_adjust=group_adjust,
        by_group=by_group,
    )


for _name in (
    "average_cumulative_return_by_quantile",
    "common_start_returns",
    "compute_mean_returns_spread",
    "cumulative_returns",
    "factor_alpha_beta",
    "factor_information_coefficient",
    "factor_rank_autocorrelation",
    "factor_returns",
    "factor_weights",
    "mean_information_coefficient",
    "mean_return_by_quantile",
    "quantile_turnover",
):
    _attach_spec(globals()[_name], _name)


__all__ = _PERFORMANCE_NAMES

del export_deferred_functions
