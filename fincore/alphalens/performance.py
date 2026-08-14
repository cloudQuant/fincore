"""Strict Alphalens performance facade backed by Task 4 kernels.

The module begins with the static C0/C1 deferred registry, then intentionally
overrides only functions whose numerical kernel has been characterized.  The
remaining Task 5 portfolio symbols keep their original deferred boundary.
"""

from __future__ import annotations

import importlib
from typing import Any, cast

import numpy as np
import pandas as pd

from fincore.alphalens._compat import export_deferred_functions
from fincore.contracts.factor_analysis import ALPHALENS_FUNCTION_SPECS, FactorFunctionSpec
from fincore.exceptions import DependencyError
from fincore.factor_analysis import performance as _performance
from fincore.factor_analysis.calendar import get_forward_returns_columns

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
    return _strict_factor_information_coefficient(
        factor_data,
        group_adjust=group_adjust,
        by_group=by_group,
    )


def mean_information_coefficient(
    factor_data: pd.DataFrame,
    group_adjust: bool = False,
    by_group: bool = False,
    by_time: str | None = None,
) -> pd.Series | pd.DataFrame:
    _reject_opaque("mean_information_coefficient", factor_data)
    information = _strict_factor_information_coefficient(
        factor_data,
        group_adjust=group_adjust,
        by_group=by_group,
    )
    groupers: list[object] = []
    if by_time is not None:
        groupers.append(pd.Grouper(freq="ME" if by_time == "M" else by_time))
    if by_group:
        groupers.append("group")
    if not groupers:
        return information.mean()
    return information.reset_index().set_index("date").groupby(groupers, observed=False, sort=True).mean()


def _strict_factor_information_coefficient(
    factor_data: pd.DataFrame,
    *,
    group_adjust: bool,
    by_group: bool,
) -> pd.DataFrame:
    """Use SciPy's pinned ``spearmanr`` NaN propagation at the facade boundary.

    The enhanced kernel intentionally uses pandas' pairwise Spearman
    correlation.  Alphalens 0.4.0 called ``scipy.stats.spearmanr`` without a
    ``nan_policy`` override, so an incomplete date/period produces ``NaN``.
    Keeping that projection here leaves the enhanced API profile-free.
    """

    copied = _performance._copy_factor_data(factor_data)
    columns = get_forward_returns_columns(copied.columns)
    if group_adjust:
        # Source grouping resolves the requested column before it encounters
        # the empty concat path, so keep the missing-column priority exact.
        if "group" not in copied.columns:
            raise KeyError("group")
        # Alphalens' source reaches ``pd.concat([])`` while constructing the
        # group-adjusted empty table. This is deliberately a strict-facade
        # error projection; the enhanced kernel remains profile-free.
        if copied.empty:
            raise ValueError("No objects to concatenate")
        copied = _performance._demean_forward_returns(copied, by_group=True)
    if by_group and "group" not in copied.columns:
        raise KeyError("group")

    try:
        stats = importlib.import_module("scipy.stats")
    except ModuleNotFoundError as exc:
        raise DependencyError(
            "factor_information_coefficient requires scipy. Install it with:\n    pip install scipy",
            dependency="scipy",
        ) from exc

    def source_ic(group: pd.DataFrame) -> pd.Series:
        factor = group["factor"]
        return cast("pd.Series", group.loc[:, columns].apply(lambda values: stats.spearmanr(values, factor)[0]))

    groupers: list[object] = [copied.index.get_level_values("date")]
    if by_group:
        groupers.append("group")
    return copied.groupby(groupers, observed=False, sort=True).apply(source_ic)


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


def _require_statsmodels() -> tuple[Any, Any]:
    """Load the pinned strict alpha/beta OLS primitives only at call time."""

    try:
        linear_model = importlib.import_module("statsmodels.regression.linear_model")
        tools = importlib.import_module("statsmodels.tools.tools")
    except ModuleNotFoundError as exc:
        raise DependencyError(
            "factor_alpha_beta requires the optional 'factor-analysis' extra. "
            "Install it with:\n    pip install fincore[factor-analysis]",
            dependency="statsmodels",
        ) from exc
    return linear_model.OLS, tools.add_constant


def _strict_factor_alpha_beta(
    factor_data: pd.DataFrame,
    returns: pd.DataFrame | pd.Series | None,
    demeaned: bool,
    group_adjust: bool,
    equal_weight: bool,
) -> pd.DataFrame:
    """Project the source's statsmodels OLS edge semantics at the strict boundary."""

    if not isinstance(factor_data, pd.DataFrame):
        raise TypeError("factor_data must be a pandas DataFrame")
    forward_columns = get_forward_returns_columns(factor_data.columns)
    returns_frame: pd.DataFrame | pd.Series
    if returns is None:
        # Retain the strict facade composition point: any future legacy
        # projection in factor_returns (including all-NaN aggregation) must
        # also feed the implicit alpha/beta path.
        returns_frame = factor_returns(
            factor_data,
            demeaned=demeaned,
            group_adjust=group_adjust,
            equal_weight=equal_weight,
        )
    elif isinstance(returns, (pd.Series, pd.DataFrame)):
        returns_frame = returns.copy(deep=True)
    else:
        raise TypeError("returns must be a pandas Series, DataFrame, or None")

    universe_returns = factor_data.groupby(level="date", observed=False, sort=True)[list(forward_columns)].mean()
    universe_returns = universe_returns.loc[returns_frame.index]
    # The pinned implementation aligns explicit returns before its period loop.
    # Only a successfully aligned zero-column DataFrame is a genuine 0x0
    # result; do not swallow missing-date or supplied-period lookup errors.
    if isinstance(returns_frame, pd.DataFrame) and not len(returns_frame.columns):
        return pd.DataFrame()
    if isinstance(returns_frame, pd.Series):
        returns_frame.name = universe_returns.columns[0]
        returns_frame = returns_frame.to_frame()

    ols, add_constant = _require_statsmodels()
    result = pd.DataFrame(index=pd.Index(["Ann. alpha", "beta"]))
    for period in returns_frame.columns:
        design = add_constant(universe_returns[period].to_numpy(dtype=float, copy=False))
        fitted = ols(returns_frame[period].to_numpy(dtype=float, copy=False), design).fit()
        try:
            alpha, beta = fitted.params
        except ValueError:
            result.loc["Ann. alpha", period] = float("nan")
            result.loc["beta", period] = float("nan")
            continue
        annualization = pd.Timedelta("252Days") / pd.Timedelta(period)
        result.loc["Ann. alpha", period] = (1.0 + alpha) ** annualization - 1.0
        result.loc["beta", period] = beta
    return result


def factor_alpha_beta(
    factor_data: pd.DataFrame,
    returns: pd.DataFrame | pd.Series | None = None,
    demeaned: bool = True,
    group_adjust: bool = False,
    equal_weight: bool = False,
) -> pd.DataFrame:
    _reject_opaque("factor_alpha_beta", factor_data, returns)
    return _strict_factor_alpha_beta(
        factor_data,
        returns,
        demeaned,
        group_adjust,
        equal_weight,
    )


def cumulative_returns(returns: pd.Series) -> pd.Series:
    _reject_opaque("cumulative_returns", returns)
    result = _performance.cumulative_returns(returns)
    if isinstance(result, pd.Series) and len(result):
        # ``empyrical.cum_returns`` constructed a fresh Series in the pinned
        # strict surface for nonempty inputs, so their input name is not part
        # of the legacy result. Its empty early-return instead preserves it.
        result = result.copy(deep=True)
        result.name = None
    return result  # type: ignore[return-value]


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


def _strict_print_common_start_return_slices(
    factor: pd.Series | pd.DataFrame,
    returns: pd.DataFrame,
    before: int,
    after: int,
    *,
    cumulative: bool,
    mean_by_date: bool,
    demean_by: pd.Series | pd.DataFrame | None,
) -> list[pd.Series | pd.DataFrame]:
    """Reproduce the pinned per-event ``print('series = ', series)`` trace.

    The enhanced core deliberately has no stdout side effects. This narrow
    strict-facade projection emits and returns each successfully resolved
    source slice in sorted event-date order, before optional demeaning.
    """

    factor_copy = _performance._event_factor_copy(factor)
    returns_copy = returns.copy(deep=True)
    returns_copy.index = pd.DatetimeIndex(returns_copy.index, name="date")
    if not cumulative:
        returns_copy = pd.DataFrame(
            {column: _performance.cumulative_returns(returns_copy[column]) for column in returns_copy.columns},
            index=returns_copy.index,
        )
    demean_copy = _performance._event_factor_copy(demean_by) if demean_by is not None else None
    all_returns: list[pd.Series | pd.DataFrame] = []
    factor_index = cast("pd.MultiIndex", factor_copy.index)
    date_values = factor_index.get_level_values("date")
    asset_values = factor_index.get_level_values("asset")

    for timestamp in pd.Index(date_values).unique().sort_values():
        try:
            day_zero = returns_copy.index.get_loc(timestamp)
        except KeyError:
            continue
        if not isinstance(day_zero, (int, np.integer)):
            raise ValueError("returns index must map each factor date to one row")
        start = max(int(day_zero) - before, 0)
        stop = min(int(day_zero) + after + 1, len(returns_copy.index))
        equities = pd.Index(asset_values[date_values == timestamp])
        if demean_copy is None:
            demean_equities = pd.Index([])
        else:
            # Preserve the source's direct lookup before emitting stdout:
            # missing event dates raise ``KeyError(timestamp)`` rather than
            # being converted into the enhanced core's friendly empty error.
            demean_slice = cast("Any", demean_copy.loc[cast("Any", timestamp)])
            demean_index = cast("pd.Index", demean_slice.index)
            demean_equities = demean_index.get_level_values("asset")
        # Source uses a literal set/list union before printing. Preserve that
        # set-derived order while letting pandas retain caller columns metadata.
        equities_slice = list(set(equities) | set(demean_equities))
        series = cast("pd.DataFrame", returns_copy.loc[returns_copy.index[start:stop], equities_slice].copy())
        series.index = pd.RangeIndex(start - int(day_zero), stop - int(day_zero))
        print("series = ", series)
        if demean_copy is not None:
            mean = series.loc[:, demean_equities].mean(axis=1)
            series = series.loc[:, equities].sub(mean, axis=0)
        if mean_by_date:
            all_returns.append(series.mean(axis=1))
        else:
            all_returns.append(series)

    return all_returns


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
    # The pinned source collected one slice per event date and fed the list to
    # ``pd.concat``.  If every event date is absent from ``returns`` that call
    # raises ``ValueError('No objects to concatenate')``.  The enhanced kernel
    # deliberately returns a typed empty event window instead, so retain the
    # legacy projection here rather than teaching the profile-free core about
    # a facade-only error surface.
    if isinstance(factor, (pd.Series, pd.DataFrame)) and isinstance(factor.index, pd.MultiIndex):
        try:
            event_dates = factor.index.get_level_values("date")
        except KeyError:
            event_dates = factor.index.get_level_values(0)
        if not event_dates.isin(returns.index).any():
            raise ValueError("No objects to concatenate")
    source_slices = _strict_print_common_start_return_slices(
        factor,
        returns,
        before,
        after,
        cumulative=cumulative,
        mean_by_date=mean_by_date,
        demean_by=demean_by,
    )
    # The source returns this same concatenation for every successful path.
    # Delegating the usual non-negative case to the enhanced core changed the
    # observable columns: the core canonicalizes asset order and preserves an
    # ``asset`` columns name, while the pinned source retains the literal
    # ``list(set(...))`` selections created above.  Returning the source
    # slices also accepts NumPy integral windows, which the enhanced API
    # deliberately validates more narrowly.
    return pd.concat(source_slices, axis=1)


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
