"""Strict Alphalens utility facade backed by the Task 3 factor-data kernel."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    import pandas as pd

from fincore.alphalens._compat import export_deferred_functions
from fincore.contracts.factor_analysis import ALPHALENS_FUNCTION_SPECS, FactorFunctionSpec
from fincore.factor_analysis import calendar as _calendar
from fincore.factor_analysis import data as _data
from fincore.factor_analysis.exceptions import (
    FactorLossExceededError,
    MaxLossExceededError,
    NonMatchingTimezoneError,
)

_UTILITY_NAMES = export_deferred_functions(globals(), "utils")
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


def _deferred(name: str) -> None:
    raise NotImplementedError(
        f"Legacy Alphalens symbol '{name}' is available for C0/C1 compatibility, "
        "but its numerical or rendering kernel is not implemented yet."
    )


def _reject_opaque(name: str, *values: object) -> None:
    """Keep Task 2 C1 opaque-call grammar at the explicit implementation boundary."""

    if any(type(value) is object for value in values):
        _deferred(name)


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


def add_custom_calendar_timedelta(input: object, timedelta: object, freq: object) -> pd.Timestamp | pd.DatetimeIndex:
    _reject_opaque("add_custom_calendar_timedelta", input, timedelta, freq)
    return _calendar.add_custom_calendar_timedelta(input, timedelta, freq)  # type: ignore[arg-type]


def backshift_returns_series(series: pd.Series, N: int) -> pd.Series:
    _reject_opaque("backshift_returns_series", series, N)
    return _calendar.backshift_returns_series(series, N)


def compute_forward_returns(
    factor: pd.Series,
    prices: pd.DataFrame,
    periods: Sequence[int] = (1, 5, 10),
    filter_zscore: float | None = None,
    cumulative_returns: bool = True,
) -> pd.DataFrame:
    _reject_opaque("compute_forward_returns", factor, prices)
    return _data.compute_forward_returns(
        factor,
        prices,
        periods=periods,
        filter_zscore=filter_zscore,
        cumulative_returns=cumulative_returns,
    )


def diff_custom_calendar_timedeltas(start: object, end: object, freq: object) -> pd.Timedelta:
    _reject_opaque("diff_custom_calendar_timedeltas", start, end, freq)
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
    _reject_opaque("get_clean_factor", factor, forward_returns)
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
    _reject_opaque("get_clean_factor_and_forward_returns", factor, prices)
    try:
        prepared = _data.prepare_factor_data(
            factor,
            prices,
            groupby=groupby,
            binning_by_group=binning_by_group,
            quantiles=quantiles,
            bins=bins,
            periods=periods,
            filter_zscore=filter_zscore,
            groupby_labels=groupby_labels,
            max_loss=max_loss,
            zero_aware=zero_aware,
            cumulative_returns=cumulative_returns,
        )
    except FactorLossExceededError as error:
        assert error.report is not None
        print(_legacy_loss_line(error.report))
        raise MaxLossExceededError(str(error), error.report) from None
    print(_legacy_loss_line(prepared.loss_report))
    print(f"max_loss is {float(max_loss) * 100:.1f}%, not exceeded: OK!")
    return prepared.data


def get_forward_returns_columns(columns: pd.Index, require_exact_day_multiple: bool = False) -> pd.Index:
    _reject_opaque("get_forward_returns_columns", columns)
    return _calendar.get_forward_returns_columns(columns, require_exact_day_multiple=require_exact_day_multiple)


def infer_trading_calendar(factor_idx: pd.DatetimeIndex, prices_idx: pd.DatetimeIndex) -> object:
    _reject_opaque("infer_trading_calendar", factor_idx, prices_idx)
    return _calendar.infer_trading_calendar(factor_idx, prices_idx)


def quantize_factor(*args: Any, **kwargs: Any) -> pd.Series:
    """Bind the source signature while retaining the legacy decorator signature."""

    spec = _spec("quantize_factor")
    bound = spec.source_signature.bind(*args, **kwargs)
    _reject_opaque("quantize_factor", bound.arguments["factor_data"])
    try:
        return _data.quantize_factor(**bound.arguments)
    except ValueError as error:
        if "Bin edges must be unique" in str(error):
            raise ValueError(f"{error}{_NON_UNIQUE_BIN_EDGES_MESSAGE}") from None
        raise


_quantize_factor_compat = cast("Any", quantize_factor)
_quantize_factor_compat.__name__ = "quantize_factor"
_quantize_factor_compat.__qualname__ = "quantize_factor"
_quantize_factor_compat.__module__ = __name__
_quantize_factor_compat.__signature__ = _spec("quantize_factor").introspection_signature
_quantize_factor_compat.__fincore_source_signature__ = _spec("quantize_factor").source_signature
_quantize_factor_compat.__fincore_factor_spec__ = _spec("quantize_factor")


def timedelta_strings_to_integers(sequence: Sequence[str]) -> list[int]:
    _reject_opaque("timedelta_strings_to_integers", sequence)
    return _calendar.timedelta_strings_to_integers(sequence)


def timedelta_to_string(timedelta: object) -> str:
    _reject_opaque("timedelta_to_string", timedelta)
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
    "timedelta_strings_to_integers",
    "timedelta_to_string",
):
    _attach_spec(globals()[_name], _name)


__all__ = ("MaxLossExceededError", "NonMatchingTimezoneError", *_UTILITY_NAMES)

del export_deferred_functions
