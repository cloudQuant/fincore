"""Input validation utility tests."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from fincore.constants import DAILY, MONTHLY, QUARTERLY, WEEKLY, YEARLY
from fincore.exceptions import (
    DataAlignmentError,
    InsufficientDataError,
    InvalidPeriodError,
    MissingDataError,
    NumericalError,
    UnsupportedFormatError,
    ValidationError,
)
from fincore.validation import (
    validate_alignment,
    validate_input,
    validate_numeric_array,
    validate_percentage,
    validate_period,
    validate_positive,
    validate_returns,
    validate_risk_free,
    validate_window,
)

# ---------------------------------------------------------------------------
# validate_input decorator
# ---------------------------------------------------------------------------


def test_validate_input_passes_through_valid_args() -> None:
    @validate_input(lambda x: None)
    def fn(a: int) -> int:
        return a + 1

    assert fn(1) == 2


def test_validate_input_replaces_argument_with_validator_result() -> None:
    @validate_input(lambda x: x * 2)
    def fn(a: int) -> int:
        return a

    assert fn(3) == 6


def test_validate_input_raises_validation_error_on_validator_failure() -> None:
    def boom(_: int) -> None:
        raise ValueError("boom")

    @validate_input(boom, error_message="custom")
    def fn(a: int) -> int:
        return a

    with pytest.raises(ValidationError, match="custom"):
        fn(1)


def test_validate_input_swallows_error_when_raise_on_error_false() -> None:
    def boom(_: int) -> None:
        raise ValueError("boom")

    @validate_input(boom, raise_on_error=False)
    def fn(a: int) -> int:
        return a

    assert fn(1) == 1


def test_validate_input_skips_missing_default_args() -> None:
    seen: list[object] = []

    def record(x: object) -> None:
        seen.append(x)

    @validate_input(record, record)
    def fn(a: int, b: int = 10) -> int:
        return a + b

    # Only "a" is bound; the second validator maps to "b" which is absent.
    assert fn(1) == 11
    assert seen == [1]


def test_validate_input_breaks_when_more_validators_than_params() -> None:
    @validate_input(lambda x: None, lambda x: None, lambda x: None)
    def fn(a: int) -> int:
        return a

    assert fn(5) == 5


def test_validate_input_skips_varargs_and_kwargs() -> None:
    @validate_input(lambda x: None)
    def fn(a: int, *args: int, **kwargs: int) -> int:
        return a + sum(args) + sum(kwargs.values())

    assert fn(1, 2, 3, extra=4) == 10


# ---------------------------------------------------------------------------
# validate_returns
# ---------------------------------------------------------------------------


def test_validate_returns_none_raises() -> None:
    with pytest.raises(MissingDataError):
        validate_returns(None)


def test_validate_returns_unsupported_type_raises() -> None:
    with pytest.raises(UnsupportedFormatError):
        validate_returns({"a": 1})


def test_validate_returns_empty_raises() -> None:
    with pytest.raises(InsufficientDataError):
        validate_returns(pd.Series([], dtype=float))


def test_validate_returns_allow_empty_passes() -> None:
    empty = pd.Series([], index=pd.DatetimeIndex([]), dtype=float)
    result = validate_returns(empty, allow_empty=True)
    assert len(result) == 0


def test_validate_returns_too_short_raises() -> None:
    with pytest.raises(InsufficientDataError):
        validate_returns(pd.Series([0.01]), min_length=5)


def test_validate_returns_requires_datetime_index() -> None:
    with pytest.raises(ValidationError, match="DatetimeIndex"):
        validate_returns(pd.Series([0.01, -0.01]))


def test_validate_returns_accepts_datetime_index() -> None:
    idx = pd.date_range("2024-01-01", periods=2)
    result = validate_returns(pd.Series([0.01, -0.01], index=idx))
    assert len(result) == 2


def test_validate_returns_skips_index_check_for_ndarray() -> None:
    result = validate_returns(np.array([0.01, -0.01]))
    assert isinstance(result, np.ndarray)


def test_validate_returns_converts_list_to_array() -> None:
    result = validate_returns([0.01, -0.01])
    assert isinstance(result, np.ndarray)


def test_validate_returns_dataframe_allowed() -> None:
    idx = pd.date_range("2024-01-01", periods=2)
    df = pd.DataFrame({"a": [0.01, -0.01]}, index=idx)
    result = validate_returns(df)
    assert isinstance(result, pd.DataFrame)


# ---------------------------------------------------------------------------
# validate_period
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("period", [DAILY, WEEKLY, MONTHLY, QUARTERLY, YEARLY])
def test_validate_period_accepts_valid(period: str) -> None:
    assert validate_period(period) == period


def test_validate_period_rejects_invalid() -> None:
    with pytest.raises(InvalidPeriodError):
        validate_period("fortnightly")


# ---------------------------------------------------------------------------
# validate_positive
# ---------------------------------------------------------------------------


def test_validate_positive_accepts_positive() -> None:
    assert validate_positive(1.5) == 1.5


def test_validate_positive_accepts_zero_when_allowed() -> None:
    assert validate_positive(0.0, allow_zero=True) == 0.0


def test_validate_positive_rejects_below_min() -> None:
    with pytest.raises(ValidationError):
        validate_positive(-1.0)


def test_validate_positive_rejects_zero_when_not_allowed() -> None:
    with pytest.raises(ValidationError):
        validate_positive(0.0, allow_zero=False)


def test_validate_positive_rejects_equal_min_when_not_allowed() -> None:
    with pytest.raises(ValidationError):
        validate_positive(1.0, min_value=1.0, allow_zero=False)


def test_validate_positive_accepts_equal_min_when_allowed() -> None:
    assert validate_positive(1.0, min_value=1.0, allow_zero=True) == 1.0


# ---------------------------------------------------------------------------
# validate_alignment
# ---------------------------------------------------------------------------


def test_validate_alignment_matching_series() -> None:
    idx = pd.date_range("2024-01-01", periods=3)
    s1 = pd.Series([1.0, 2.0, 3.0], index=idx)
    s2 = pd.Series([4.0, 5.0, 6.0], index=idx)
    a, b = validate_alignment(s1, s2)
    assert a is s1 and b is s2


def test_validate_alignment_length_mismatch_raises() -> None:
    s1 = pd.Series([1.0, 2.0])
    s2 = pd.Series([1.0, 2.0, 3.0])
    with pytest.raises(DataAlignmentError):
        validate_alignment(s1, s2)


def test_validate_alignment_index_mismatch_raises() -> None:
    idx1 = pd.date_range("2024-01-01", periods=3)
    idx2 = pd.date_range("2024-01-02", periods=3)
    s1 = pd.Series([1.0, 2.0, 3.0], index=idx1)
    s2 = pd.Series([1.0, 2.0, 3.0], index=idx2)
    with pytest.raises(DataAlignmentError):
        validate_alignment(s1, s2)


def test_validate_alignment_non_pandas_passes_through() -> None:
    a, b = validate_alignment(np.array([1, 2]), np.array([1, 2, 3]))
    assert a is not None and b is not None


# ---------------------------------------------------------------------------
# validate_percentage
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("value", [0.0, 0.5, 1.0])
def test_validate_percentage_accepts_range(value: float) -> None:
    assert validate_percentage(value) == value


@pytest.mark.parametrize("value", [-0.1, 1.1])
def test_validate_percentage_rejects_out_of_range(value: float) -> None:
    with pytest.raises(ValidationError):
        validate_percentage(value)


# ---------------------------------------------------------------------------
# validate_numeric_array
# ---------------------------------------------------------------------------


def test_validate_numeric_array_none_raises() -> None:
    with pytest.raises(MissingDataError):
        validate_numeric_array(None)


def test_validate_numeric_array_unsupported_raises() -> None:
    with pytest.raises(UnsupportedFormatError):
        validate_numeric_array(pd.Series([1.0]))


def test_validate_numeric_array_too_short_raises() -> None:
    with pytest.raises(InsufficientDataError):
        validate_numeric_array(np.array([1.0]), min_length=3)


def test_validate_numeric_array_accepts_ndarray() -> None:
    result = validate_numeric_array(np.array([1.0, 2.0]))
    assert isinstance(result, np.ndarray)


def test_validate_numeric_array_converts_list() -> None:
    result = validate_numeric_array([1.0, 2.0])
    assert isinstance(result, np.ndarray)


def test_validate_numeric_array_rejects_nan_when_disallowed() -> None:
    with pytest.raises(NumericalError):
        validate_numeric_array(np.array([1.0, np.nan]), allow_nan=False)


def test_validate_numeric_array_allows_nan_by_default() -> None:
    result = validate_numeric_array(np.array([1.0, np.nan]))
    assert result.shape == (2,)


# ---------------------------------------------------------------------------
# validate_risk_free
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("value", [0.0, 0.02, 1])
def test_validate_risk_free_accepts_number(value: float) -> None:
    assert validate_risk_free(value) == value


def test_validate_risk_free_rejects_non_number() -> None:
    with pytest.raises(ValidationError):
        validate_risk_free("0.02")


# ---------------------------------------------------------------------------
# validate_window
# ---------------------------------------------------------------------------


def test_validate_window_accepts_valid() -> None:
    assert validate_window(10) == 10


def test_validate_window_rejects_too_small() -> None:
    with pytest.raises(ValidationError):
        validate_window(1, min_periods=2)
