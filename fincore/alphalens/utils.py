"""Strict Alphalens utility facade backed by the Task 3 factor-data kernel."""

from __future__ import annotations

import importlib
from functools import wraps
from numbers import Real
from typing import Any, Mapping, NoReturn, Sequence, cast

import numpy as np
import pandas as pd

from fincore.contracts.factor_analysis import ALPHALENS_FUNCTION_SPECS, FactorFunctionSpec, function_specs_for_module
from fincore.exceptions import DependencyError
from fincore.factor_analysis import calendar as _calendar
from fincore.factor_analysis import data as _data
from fincore.factor_analysis.exceptions import (
    EnhancedNonMatchingTimezoneError,
    FactorLossExceededError,
    MaxLossExceededError,
    NonMatchingTimezoneError,
)

_UTILITY_NAMES = tuple(spec.public_name for spec in function_specs_for_module("utils"))
_NON_UNIQUE_BIN_EDGES_MESSAGE = """

    An error occurred while computing bins/quantiles on the input provided.
    This usually happens when the input contains too many identical
    values and they span more than one quantile. The quantiles are choosen
    to have the same number of records each, but the same value cannot span
    multiple quantiles. Possible workarounds are:
    1 - Decrease the number of quantiles
    2 - Specify a custom quantiles range, e.g. [0, .50, .75, 1.] to get unequal
        number of records per quantile
    3 - Use 'bins' option instead of 'quantiles', 'bins' chooses the
        buckets to be evenly spaced according to the values themselves, while
        'quantiles' forces the buckets to have the same number of records.
    4 - for factors with discrete values use the 'bins' option with custom
        ranges and create a range for each discrete value
    Please see utils.get_clean_factor_and_forward_returns documentation for
    full documentation of 'bins' and 'quantiles' options.

"""


def _spec(name: str) -> FactorFunctionSpec:
    return ALPHALENS_FUNCTION_SPECS[("utils", name)]


def _attach_spec(function: Any, name: str) -> Any:
    spec = _spec(name)
    # The actual implementation carries annotations for the enhanced kernel,
    # while the strict surface must expose the reviewed legacy signature.
    function.__signature__ = spec.introspection_signature
    function.__fincore_source_signature__ = spec.source_signature
    function.__fincore_factor_spec__ = spec
    return function


def _legacy_loss_line(report: _data.FactorLossReport) -> str:
    return (
        "Dropped %.1f%% entries from factor data: %.1f%% in forward returns computation and %.1f%% in binning phase "
        "(set max_loss=0 to see potentially suppressed Exceptions)."
        % (report.total_loss * 100, report.legacy_forward_returns_loss * 100, report.binning_loss * 100)
    )


def _raise_legacy_bin_edge_error(error: ValueError) -> NoReturn:
    """Apply the pinned decorator's error projection at every strict entrypoint."""

    if "Bin edges must be unique" in str(error):
        rethrow(error, _NON_UNIQUE_BIN_EDGES_MESSAGE)
    raise error


def _strict_all_nan_factor(factor: object) -> bool:
    """Recognize only the numeric all-NaN adapter case; leave core validation intact."""

    if not isinstance(factor, pd.Series) or not isinstance(factor.index, pd.MultiIndex) or factor.empty:
        return False
    try:
        values = factor.to_numpy(dtype=float, copy=False)
    except (TypeError, ValueError):
        return False
    return not bool(np.isfinite(values).any())


def _strict_all_nan_groupby(
    groupby: Mapping[object, object] | pd.Series | None,
    groupby_labels: Mapping[object, object] | None,
    factor_index: pd.MultiIndex,
) -> pd.Series | None:
    """Mirror the pinned group/category operations before its empty-row drop."""

    if groupby is None:
        return None
    if isinstance(groupby, dict):
        # The pinned implementation filters the all-NaN factor first, so its
        # dictionary asset check receives an empty index and its category has
        # no categories.  Keep that narrow strict projection in the adapter.
        groups = pd.Series(index=factor_index[:0], dtype="str")
    elif isinstance(groupby, pd.Series):
        groups = groupby
    else:
        raise TypeError("groupby must be a mapping, Series, or None")

    if groupby_labels is not None:
        missing = set(groups.values) - set(groupby_labels.keys())
        if missing:
            raise KeyError(f"groups {list(missing)!r} not in passed group names")
        labels = pd.Series(dict(groupby_labels))
        groups = pd.Series(index=groups.index, data=labels[cast("Any", groups.values)].values)
    return groups.astype("category")


def _strict_empty_factor_projection(
    factor: pd.Series,
    forward_returns: pd.DataFrame,
    *,
    groupby: Mapping[object, object] | pd.Series | None,
    groupby_labels: Mapping[object, object] | None,
    max_loss: float,
) -> pd.DataFrame | None:
    """Project the pinned all-NaN clean-factor result without changing the kernel."""

    if not _strict_all_nan_factor(factor):
        return None
    if not isinstance(forward_returns, pd.DataFrame) or not isinstance(forward_returns.index, pd.MultiIndex):
        return None
    if not isinstance(max_loss, Real) or not 0 <= float(max_loss) <= 1:
        return None
    groups = _strict_all_nan_groupby(groupby, groupby_labels, factor.index)

    report = _data.FactorLossReport(
        input_count=len(factor),
        finite_factor_count=0,
        forward_returns_count=0,
        binning_count=0,
        factor_input_loss=1.0,
        forward_returns_loss=0.0,
        binning_loss=0.0,
        total_loss=1.0,
    )
    print(_legacy_loss_line(report))
    if report.total_loss > float(max_loss):
        message = f"max_loss ({float(max_loss) * 100:.1f}%) exceeded 100.0%, consider increasing it."
        raise MaxLossExceededError(message, report)

    result = forward_returns.copy(deep=True).iloc[0:0]
    result["factor"] = pd.Series(index=result.index, dtype=float)
    if groups is not None:
        # Assigning a non-empty Series to an empty frame grows its index;
        # source assigns before ``dropna``.  Reindex first to retain only the
        # empty target while preserving CategoricalDtype metadata.
        result["group"] = groups.reindex(result.index)
    result["factor_quantile"] = pd.Series(index=result.index, dtype=float)
    print(f"max_loss is {float(max_loss) * 100:.1f}%, not exceeded: OK!")
    return result


def _strict_prices_for_factor(factor: object, prices: object) -> object:
    """Pad only strict-facade missing assets to mirror ``prices.filter``.

    The enhanced kernel deliberately rejects missing price columns.  Pinned
    Alphalens instead filters the price table and later reindexes back to the
    factor index, leaving each unavailable asset's forward-return rows as
    ``NaN``.  A copied all-NaN column recreates that adapter-only projection
    while keeping caller data and enhanced validation unchanged.
    """

    if not isinstance(factor, pd.Series) or not isinstance(factor.index, pd.MultiIndex):
        return prices
    if factor.index.nlevels != 2 or not isinstance(prices, pd.DataFrame):
        return prices
    missing = pd.Index(factor.index.levels[1]).difference(prices.columns)
    if missing.empty:
        return prices
    padded = prices.copy(deep=True)
    for asset in missing:
        padded[asset] = np.nan
    return padded


def rethrow(exception: BaseException, additional_message: str) -> NoReturn:
    """Re-raise ``exception`` after appending strict-source context to ``args``."""

    if not exception.args:
        exception.args = (additional_message,)
    else:
        # Alphalens passes string-valued ``args[0]`` here. Retain that source
        # grammar (including its natural TypeError for other exception shapes)
        # instead of coercing a foreign exception into a new message.
        exception.args = (cast("str", exception.args[0]) + additional_message, *exception.args[1:])
    raise exception


def non_unique_bin_edges_error(func: Any) -> Any:
    """Decorate a quantization callable with the pinned duplicate-edge guidance."""

    @wraps(func)
    def decorated(*args: Any, **kwargs: Any) -> Any:
        try:
            return func(*args, **kwargs)
        except ValueError as error:
            if "Bin edges must be unique" in str(error):
                rethrow(error, _NON_UNIQUE_BIN_EDGES_MESSAGE)
            raise

    return decorated


def demean_forward_returns(factor_data: pd.DataFrame, grouper: object = None) -> pd.DataFrame:
    """Return copied forward returns demeaned by date or the supplied grouping."""

    copied = factor_data.copy()
    if grouper is None or (isinstance(grouper, list) and not grouper):
        grouper = copied.index.get_level_values("date")
    columns = _calendar.get_forward_returns_columns(copied.columns)
    copied[columns] = copied.groupby(cast("Any", grouper), observed=False)[columns].transform(
        lambda values: values - values.mean()
    )
    return copied


def print_table(table: pd.Series | pd.DataFrame, name: str | None = None, fmt: str | None = None) -> None:
    """Display one strict-source table lazily through IPython's display hook."""

    displayed: pd.DataFrame | pd.Series = table.to_frame() if isinstance(table, pd.Series) else table
    if isinstance(displayed, pd.DataFrame):
        displayed.columns.name = name
    previous_format = pd.get_option("display.float_format")
    if fmt is not None:
        pd.set_option("display.float_format", lambda value: fmt.format(value))
    try:
        display = importlib.import_module("IPython.display").display
    except ModuleNotFoundError as error:
        raise DependencyError(
            "print_table requires IPython. Install it with:\n    pip install fincore[alphalens]",
            dependency="IPython",
        ) from error
    try:
        display(displayed)
    finally:
        if fmt is not None:
            pd.set_option("display.float_format", previous_format)


def rate_of_return(period_ret: pd.Series | pd.DataFrame, base_period: object) -> pd.Series | pd.DataFrame:
    """Convert a named forward-return period to a named base-period rate."""

    conversion_factor = cast(
        "float",
        pd.Timedelta(cast("Any", base_period)) / pd.Timedelta(cast("Any", period_ret.name)),
    )
    return cast("pd.Series | pd.DataFrame", period_ret.add(1).pow(conversion_factor).sub(1))


def std_conversion(period_std: pd.Series | pd.DataFrame, base_period: object) -> pd.Series | pd.DataFrame:
    """Scale a named forward-period standard deviation to ``base_period``."""

    conversion_factor = cast(
        "float",
        pd.Timedelta(cast("Any", period_std.name)) / pd.Timedelta(cast("Any", base_period)),
    )
    return cast("pd.Series | pd.DataFrame", period_std / np.sqrt(conversion_factor))


def add_custom_calendar_timedelta(input: object, timedelta: object, freq: object) -> pd.Timestamp | pd.DatetimeIndex:
    return _calendar.add_custom_calendar_timedelta(input, timedelta, freq)  # type: ignore[arg-type]


def backshift_returns_series(series: pd.Series, N: int) -> pd.Series:
    return _calendar.backshift_returns_series(series, N)


def compute_forward_returns(
    factor: pd.Series,
    prices: pd.DataFrame,
    periods: Sequence[int] = (1, 5, 10),
    filter_zscore: float | None = None,
    cumulative_returns: bool = True,
) -> pd.DataFrame:
    try:
        return _data.compute_forward_returns(
            factor,
            _strict_prices_for_factor(factor, prices),  # type: ignore[arg-type]
            periods=periods,
            filter_zscore=filter_zscore,
            cumulative_returns=cumulative_returns,
        )
    except EnhancedNonMatchingTimezoneError as error:
        raise NonMatchingTimezoneError(str(error)) from None


def diff_custom_calendar_timedeltas(start: object, end: object, freq: object) -> pd.Timedelta:
    return _calendar.diff_custom_calendar_timedeltas(start, end, freq)  # type: ignore[arg-type]


def get_clean_factor(
    factor: pd.Series,
    forward_returns: pd.DataFrame,
    groupby: Mapping[object, object] | pd.Series | None = None,
    binning_by_group: bool = False,
    quantiles: int | Sequence[float] | None = 5,
    bins: int | Sequence[float] | None = None,
    groupby_labels: Mapping[object, object] | None = None,
    max_loss: float = 0.35,
    zero_aware: bool = False,
) -> pd.DataFrame:
    strict_empty = _strict_empty_factor_projection(
        factor,
        forward_returns,
        groupby=groupby,
        groupby_labels=groupby_labels,
        max_loss=max_loss,
    )
    if strict_empty is not None:
        return strict_empty
    try:
        prepared = _data.prepare_factor_data_from_forward_returns(
            factor,
            forward_returns,
            groupby=groupby,
            binning_by_group=binning_by_group,
            quantiles=quantiles,
            bins=bins,
            groupby_labels=groupby_labels,
            max_loss=max_loss,
            zero_aware=zero_aware,
        )
    except FactorLossExceededError as error:
        assert error.report is not None
        print(_legacy_loss_line(error.report))
        raise MaxLossExceededError(str(error), error.report) from None
    except ValueError as error:
        _raise_legacy_bin_edge_error(error)
    print(_legacy_loss_line(prepared.loss_report))
    print(f"max_loss is {float(max_loss) * 100:.1f}%, not exceeded: OK!")
    return prepared.data


def get_clean_factor_and_forward_returns(
    factor: pd.Series,
    prices: pd.DataFrame,
    groupby: Mapping[object, object] | pd.Series | None = None,
    binning_by_group: bool = False,
    quantiles: int | Sequence[float] | None = 5,
    bins: int | Sequence[float] | None = None,
    periods: Sequence[int] = (1, 5, 10),
    filter_zscore: float | None = 20,
    groupby_labels: Mapping[object, object] | None = None,
    max_loss: float = 0.35,
    zero_aware: bool = False,
    cumulative_returns: bool = True,
) -> pd.DataFrame:
    forward_returns = compute_forward_returns(
        factor,
        prices,
        periods=periods,
        filter_zscore=filter_zscore,
        cumulative_returns=cumulative_returns,
    )
    return get_clean_factor(
        factor,
        forward_returns,
        groupby=groupby,
        binning_by_group=binning_by_group,
        quantiles=quantiles,
        bins=bins,
        groupby_labels=groupby_labels,
        max_loss=max_loss,
        zero_aware=zero_aware,
    )


def get_forward_returns_columns(columns: pd.Index, require_exact_day_multiple: bool = False) -> pd.Index:
    return _calendar.get_forward_returns_columns(columns, require_exact_day_multiple=require_exact_day_multiple)


def infer_trading_calendar(factor_idx: pd.DatetimeIndex, prices_idx: pd.DatetimeIndex) -> object:
    return _calendar.infer_trading_calendar(factor_idx, prices_idx)


def quantize_factor(*args: Any, **kwargs: Any) -> pd.Series:
    """Bind the source signature while retaining the legacy decorator signature."""

    spec = _spec("quantize_factor")
    bound = spec.source_signature.bind(*args, **kwargs)
    try:
        return _data.quantize_factor(**bound.arguments)
    except ValueError as error:
        _raise_legacy_bin_edge_error(error)


_quantize_factor_compat = cast("Any", quantize_factor)
_quantize_factor_compat.__name__ = "quantize_factor"
_quantize_factor_compat.__qualname__ = "quantize_factor"
_quantize_factor_compat.__module__ = __name__
_quantize_factor_compat.__signature__ = _spec("quantize_factor").introspection_signature
_quantize_factor_compat.__fincore_source_signature__ = _spec("quantize_factor").source_signature
_quantize_factor_compat.__fincore_factor_spec__ = _spec("quantize_factor")


def timedelta_strings_to_integers(sequence: Sequence[str]) -> list[int]:
    return _calendar.timedelta_strings_to_integers(sequence)


def timedelta_to_string(timedelta: object) -> str:
    return _calendar.timedelta_to_string(timedelta)


# ``_attach_spec`` needs the name separately because it is also a useful
# signature fixture helper; assign it after definitions to avoid importing a
# runtime manifest or optional dependency.
for _name in (
    "add_custom_calendar_timedelta",
    "backshift_returns_series",
    "compute_forward_returns",
    "diff_custom_calendar_timedeltas",
    "get_clean_factor",
    "get_clean_factor_and_forward_returns",
    "get_forward_returns_columns",
    "infer_trading_calendar",
    "demean_forward_returns",
    "non_unique_bin_edges_error",
    "print_table",
    "rate_of_return",
    "rethrow",
    "std_conversion",
    "timedelta_strings_to_integers",
    "timedelta_to_string",
):
    _attach_spec(globals()[_name], _name)


__all__ = ("MaxLossExceededError", "NonMatchingTimezoneError", *_UTILITY_NAMES)
