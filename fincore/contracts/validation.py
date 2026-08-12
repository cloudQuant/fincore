"""Validation schemas shared by enhanced and context public surfaces.

The strict Empyrical/Pyfolio compatibility facades deliberately do not call
this module: their input and exception behaviour is part of the frozen
upstream oracle.  Enhanced callers bind a function signature once, validate
the bound public arguments here, and then enter an unvalidated metric kernel.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Mapping, cast

import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype

from fincore.contracts.time_series import normalize_time_series_timezone, validate_time_series_timezones
from fincore.exceptions import DataAlignmentError, MissingDataError, NumericalError, ValidationError

ValidationProfile = Literal["legacy_empyrical", "enhanced", "context"]

__all__ = [
    "ContextInputs",
    "ValidationProfile",
    "validate_context_inputs",
    "validate_factors_schema",
    "validate_market_data_schema",
    "validate_metric_arguments",
    "validate_positions_schema",
    "validate_returns_schema",
    "validate_transactions_schema",
]


@dataclass(frozen=True)
class ContextInputs:
    """Validated immutable-snapshot inputs for :class:`AnalysisContext`."""

    returns: pd.Series
    factor_returns: pd.Series | None
    positions: pd.DataFrame | None
    transactions: pd.DataFrame | None


def _validation_error(message: str, name: str, value: Any = None) -> ValidationError:
    return ValidationError(message, param_name=name, value=value)


def _copy_and_normalize_index(
    value: pd.Series | pd.DataFrame,
    *,
    name: str,
    normalize_tz: str | None,
    allow_duplicates: bool,
    require_datetime_index: bool,
) -> pd.Series | pd.DataFrame:
    result = value.copy(deep=True)
    if require_datetime_index and not isinstance(result.index, pd.DatetimeIndex):
        raise _validation_error(f"{name} must use a DatetimeIndex", name, type(result.index).__name__)
    if normalize_tz is not None:
        result = normalize_time_series_timezone(result, normalize_tz)
    if not allow_duplicates and result.index.has_duplicates:
        raise DataAlignmentError(f"{name} index contains duplicate labels")
    if not result.index.is_monotonic_increasing:
        raise DataAlignmentError(f"{name} index must be sorted in increasing order")
    return result


def _copy_and_normalize_multiindex(
    value: pd.Series | pd.DataFrame,
    *,
    name: str,
    normalize_tz: str | None,
) -> pd.Series | pd.DataFrame:
    """Copy a stacked panel and validate its ``(datetime, entity, ...)`` key."""

    result = value.copy(deep=True)
    index = result.index
    assert isinstance(index, pd.MultiIndex)
    if index.nlevels < 2 or any(index_name is None for index_name in index.names[:2]):
        raise _validation_error(f"{name} MultiIndex levels must be named", name, index.names)
    dates = index.get_level_values(0)
    if not isinstance(dates, pd.DatetimeIndex):
        raise _validation_error(f"{name} first MultiIndex level must be datetime", name, str(dates.dtype))
    if normalize_tz is not None:
        dates = normalize_time_series_timezone(
            pd.Series(np.arange(len(dates)), index=dates),
            normalize_tz,
        ).index
        result.index = pd.MultiIndex.from_arrays(
            [dates, *(index.get_level_values(level) for level in range(1, index.nlevels))],
            names=index.names,
        )
    if result.index.has_duplicates:
        raise DataAlignmentError(f"{name} index contains duplicate labels")
    if not result.index.is_monotonic_increasing:
        raise DataAlignmentError(f"{name} index must be sorted in increasing order")
    return result


def _require_finite(value: pd.Series | pd.DataFrame | np.ndarray, *, name: str) -> None:
    try:
        if isinstance(value, (pd.Series, pd.DataFrame)):
            array = value.to_numpy(dtype=float, na_value=np.nan)
        else:
            array = np.asarray(value)
        finite = np.isfinite(array)
    except TypeError as exc:
        raise _validation_error(f"{name} must contain numeric values", name, str(exc)) from exc
    if not bool(finite.all()):
        raise NumericalError(f"{name} must contain only finite numeric values", operation="validation")


def validate_returns_schema(
    value: Any,
    *,
    name: str = "returns",
    normalize_tz: str | None = None,
    allow_array: bool = False,
    allow_frame: bool = False,
    require_datetime_index: bool = True,
) -> pd.Series | pd.DataFrame | np.ndarray:
    """Validate a return/factor vector and return a defensive copy."""

    if value is None:
        raise MissingDataError(f"{name} is required", missing_field=name)
    if isinstance(value, pd.DataFrame):
        if not allow_frame:
            raise _validation_error(f"{name} must be one-dimensional", name, value.shape)
        if value.empty:
            raise _validation_error(f"{name} cannot be empty", name)
        if any(not is_numeric_dtype(dtype) for dtype in value.dtypes):
            raise _validation_error(f"{name} must contain numeric values", name, list(value.dtypes))
        result = _copy_and_normalize_index(
            value,
            name=name,
            normalize_tz=normalize_tz,
            allow_duplicates=False,
            require_datetime_index=require_datetime_index,
        )
        _require_finite(result, name=name)
        return result
    if isinstance(value, pd.Series):
        if value.empty:
            raise _validation_error(f"{name} cannot be empty", name)
        if not is_numeric_dtype(value.dtype):
            raise _validation_error(f"{name} must contain numeric values", name, str(value.dtype))
        result = _copy_and_normalize_index(
            value,
            name=name,
            normalize_tz=normalize_tz,
            allow_duplicates=False,
            require_datetime_index=require_datetime_index,
        )
        _require_finite(result, name=name)
        return result
    if allow_array and isinstance(value, (np.ndarray, list, tuple)):
        array_result = cast("np.ndarray[Any, Any]", np.array(value, copy=True))
        if array_result.ndim == 0 or (array_result.ndim > 1 and not allow_frame):
            raise _validation_error(f"{name} must be one-dimensional", name, array_result.shape)
        if array_result.size == 0:
            raise _validation_error(f"{name} cannot be empty", name)
        _require_finite(array_result, name=name)
        return array_result
    raise _validation_error(
        f"{name} must be a numeric pandas Series" + (" or array" if allow_array else ""),
        name,
        type(value).__name__,
    )


def validate_positions_schema(
    value: Any,
    *,
    name: str = "positions",
    normalize_tz: str | None = None,
    require_cash: bool = False,
) -> pd.Series | pd.DataFrame:
    """Validate position values, including the optional net-asset cash convention."""

    if not isinstance(value, (pd.Series, pd.DataFrame)):
        raise _validation_error(f"{name} must be a pandas Series or DataFrame", name, type(value).__name__)
    if isinstance(value, pd.DataFrame):
        if not value.columns.is_unique:
            raise _validation_error(f"{name} contains duplicate columns", name, list(value.columns))
        cash_columns = [column for column in value.columns if str(column).lower() == "cash"]
        if len(cash_columns) > 1:
            raise _validation_error(f"{name} contains duplicate cash columns", name, cash_columns)
        if require_cash and cash_columns != ["cash"]:
            raise _validation_error(f"{name} must contain one lowercase 'cash' column", name, list(value.columns))
        if any(not is_numeric_dtype(dtype) for dtype in value.dtypes):
            raise _validation_error(f"{name} must contain numeric values", name, list(value.dtypes))
    elif not is_numeric_dtype(value.dtype):
        raise _validation_error(f"{name} must contain numeric values", name, str(value.dtype))
    if isinstance(value.index, pd.MultiIndex):
        result = _copy_and_normalize_multiindex(value, name=name, normalize_tz=normalize_tz)
    else:
        result = _copy_and_normalize_index(
            value,
            name=name,
            normalize_tz=normalize_tz,
            allow_duplicates=False,
            require_datetime_index=True,
        )
    _require_finite(result, name=name)
    return result


def validate_transactions_schema(
    value: Any,
    *,
    name: str = "transactions",
    normalize_tz: str | None = None,
) -> pd.DataFrame:
    """Validate the canonical report transaction frame.

    Duplicate timestamps are valid because several executions may occur at the
    same instant; their row order is retained.
    """

    if not isinstance(value, pd.DataFrame):
        raise _validation_error(f"{name} must be a pandas DataFrame", name, type(value).__name__)
    if not value.columns.is_unique:
        raise _validation_error(f"{name} contains duplicate columns", name, list(value.columns))
    required = {"amount", "price", "symbol"}
    missing = sorted(required.difference(value.columns))
    if missing:
        raise _validation_error(f"{name} is missing required columns: {missing!r}", name, list(value.columns))
    for column in ("amount", "price"):
        if not is_numeric_dtype(value[column].dtype):
            raise _validation_error(f"{name}.{column} must be numeric", name, str(value[column].dtype))
        _require_finite(value[column], name=f"{name}.{column}")
    if value["symbol"].isna().any():
        raise MissingDataError(f"{name}.symbol cannot contain missing values", missing_field="symbol")
    return _copy_and_normalize_index(
        value,
        name=name,
        normalize_tz=normalize_tz,
        allow_duplicates=True,
        require_datetime_index=True,
    )  # type: ignore[return-value]


def validate_factors_schema(
    value: Any,
    *,
    name: str = "factors",
    normalize_tz: str | None = None,
) -> pd.Series | pd.DataFrame:
    """Validate factor returns or a numeric factor-loading table."""

    if isinstance(value, pd.Series):
        return validate_returns_schema(value, name=name, normalize_tz=normalize_tz)  # type: ignore[return-value]
    if not isinstance(value, pd.DataFrame):
        raise _validation_error(f"{name} must be a pandas Series or DataFrame", name, type(value).__name__)
    if value.empty or not value.columns.is_unique:
        raise _validation_error(f"{name} must have nonempty unique columns", name, list(value.columns))
    if any(not is_numeric_dtype(dtype) for dtype in value.dtypes):
        raise _validation_error(f"{name} must contain numeric values", name, list(value.dtypes))
    result = value.copy(deep=True)
    if isinstance(result.index, pd.MultiIndex):
        normalized = _copy_and_normalize_multiindex(result, name=name, normalize_tz=normalize_tz)
    else:
        normalized = _copy_and_normalize_index(
            result,
            name=name,
            normalize_tz=normalize_tz,
            allow_duplicates=False,
            require_datetime_index=True,
        )
    assert isinstance(normalized, pd.DataFrame)
    _require_finite(normalized, name=name)
    return normalized


def validate_market_data_schema(
    value: Any,
    *,
    name: str = "market_data",
    normalize_tz: str | None = None,
) -> dict[str, pd.DataFrame]:
    """Validate matched price/volume panels used by liquidity metrics."""

    if not isinstance(value, Mapping):
        raise _validation_error(f"{name} must be a mapping", name, type(value).__name__)
    missing = sorted({"price", "volume"}.difference(value))
    if missing:
        raise _validation_error(f"{name} is missing required entries: {missing!r}", name, list(value))
    panels: dict[str, pd.DataFrame] = {}
    for key in ("price", "volume"):
        panel = value[key]
        if not isinstance(panel, pd.DataFrame):
            raise _validation_error(f"{name}.{key} must be a DataFrame", name, type(panel).__name__)
        validated = validate_positions_schema(panel, name=f"{name}.{key}", normalize_tz=normalize_tz)
        panels[key] = validated  # type: ignore[assignment]
    if not panels["price"].index.equals(panels["volume"].index):
        raise DataAlignmentError(f"{name} price and volume indices must match")
    if not panels["price"].columns.equals(panels["volume"].columns):
        raise DataAlignmentError(f"{name} price and volume columns must match")
    if (panels["volume"] < 0).to_numpy().any():
        raise _validation_error(f"{name}.volume cannot be negative", name)
    return panels


def _validate_overlap(
    reference: pd.Series,
    value: pd.Series | pd.DataFrame,
    *,
    name: str,
    by_date: bool = False,
) -> None:
    if reference.empty or value.empty:
        raise DataAlignmentError(f"returns and {name} must have a nonempty overlap")
    reference_index: pd.Index
    value_index: pd.Index
    if by_date:
        if not isinstance(reference.index, pd.DatetimeIndex) or not isinstance(value.index, pd.DatetimeIndex):
            raise DataAlignmentError(f"returns and {name} must use datetime labels for calendar-day overlap")
        reference_index = reference.index.normalize()
        value_index = value.index.normalize()
    else:
        reference_index = reference.index
        value_index = value.index
    if not reference_index.isin(value_index).any():
        raise DataAlignmentError(f"returns and {name} must have a nonempty overlap")


def validate_context_inputs(
    *,
    returns: Any,
    factor_returns: Any = None,
    positions: Any = None,
    transactions: Any = None,
    normalize_tz: str | None = None,
) -> ContextInputs:
    """Validate a complete context snapshot atomically."""

    original = tuple(
        value
        for value in (returns, factor_returns, positions, transactions)
        if isinstance(value, (pd.Series, pd.DataFrame))
    )
    if normalize_tz is None:
        validate_time_series_timezones(*original)
    checked_returns = validate_returns_schema(returns, normalize_tz=normalize_tz)
    assert isinstance(checked_returns, pd.Series)
    checked_factor = (
        validate_returns_schema(factor_returns, name="factor_returns", normalize_tz=normalize_tz)
        if factor_returns is not None
        else None
    )
    checked_positions = (
        validate_positions_schema(positions, normalize_tz=normalize_tz, require_cash=True)
        if positions is not None
        else None
    )
    checked_transactions = (
        validate_transactions_schema(transactions, normalize_tz=normalize_tz) if transactions is not None else None
    )
    if checked_factor is not None:
        assert isinstance(checked_factor, pd.Series)
        _validate_overlap(checked_returns, checked_factor, name="factor_returns")
    if checked_positions is not None and (
        not isinstance(checked_positions, pd.DataFrame) or not isinstance(checked_positions.index, pd.DatetimeIndex)
    ):
        raise _validation_error(
            "positions must be a wide DataFrame with a DatetimeIndex for AnalysisContext",
            "positions",
            type(checked_positions).__name__,
        )
    if checked_positions is not None:
        _validate_overlap(checked_returns, checked_positions, name="positions")
    if checked_transactions is not None:
        _validate_overlap(checked_returns, checked_transactions, name="transactions", by_date=True)
    return ContextInputs(
        returns=checked_returns,
        factor_returns=checked_factor,
        positions=checked_positions,
        transactions=checked_transactions,
    )


def validate_metric_arguments(profile: ValidationProfile, arguments: Mapping[str, Any]) -> dict[str, Any]:
    """Validate bound public arguments for an enhanced metric invocation."""

    if profile == "legacy_empyrical":
        return dict(arguments)
    if profile not in {"enhanced", "context"}:
        raise ValueError(f"unknown metric validation profile: {profile!r}")
    checked = dict(arguments)
    variadic = checked.get("kwargs")
    options = variadic if isinstance(variadic, Mapping) else {}
    alignment = checked.get("alignment", options.get("alignment", "inner"))
    normalize_tz = checked.get("normalize_tz", options.get("normalize_tz"))

    # Alignment errors are more actionable than a finite-value error on rows
    # that the selected public policy would not retain.  Align binary inputs
    # before validating the resulting canonical vectors; kernels receive the
    # same values and may repeat only their inexpensive policy assertion.
    from fincore.contracts.time_series import AlignmentPolicy, align_binary_metric_inputs

    alignment_policy = cast("AlignmentPolicy", alignment)

    for left_name, right_name in (("returns", "factor_returns"), ("lhs", "rhs")):
        left = checked.get(left_name)
        right = checked.get(right_name)
        if isinstance(left, (pd.Series, pd.DataFrame, np.ndarray)) and isinstance(
            right, (pd.Series, pd.DataFrame, np.ndarray)
        ):
            checked[left_name], checked[right_name] = align_binary_metric_inputs(
                left,
                right,
                alignment=alignment_policy,
                normalize_tz=normalize_tz,
            )
            if isinstance(checked[left_name], (pd.Series, pd.DataFrame)) and checked[left_name].empty:
                raise DataAlignmentError(f"{left_name} and {right_name} have no common labels")

    for name, value in tuple(checked.items()):
        if value is None:
            continue
        if name in {"returns", "arr", "lhs", "prices", "factor_returns", "rhs"}:
            checked[name] = validate_returns_schema(
                value,
                name=name,
                allow_array=True,
                allow_frame=True,
                require_datetime_index=False,
                normalize_tz=normalize_tz,
            )
        elif name == "positions" and isinstance(value, (pd.Series, pd.DataFrame)):
            checked[name] = validate_positions_schema(value, require_cash=False, normalize_tz=normalize_tz)
        elif name == "transactions" and isinstance(value, pd.DataFrame):
            checked[name] = validate_transactions_schema(value, normalize_tz=normalize_tz)
        elif name == "factor_loadings":
            checked[name] = validate_factors_schema(value, name=name, normalize_tz=normalize_tz)
        elif name == "market_data":
            checked[name] = validate_market_data_schema(value, normalize_tz=normalize_tz)
    return checked
