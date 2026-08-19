"""Exception classes and error-handling utility tests."""

from __future__ import annotations

import numpy as np
import pytest

from fincore.exceptions import (
    DataAlignmentError,
    DependencyError,
    FincoreError,
    InsufficientDataError,
    InvalidPeriodError,
    MissingDataError,
    NumericalError,
    UnsupportedFormatError,
    ValidationError,
    check_dependencies,
    ensure_not_nan,
    handle_numerical_error,
    safe_divide,
    safe_sqrt,
)

# ---------------------------------------------------------------------------
# Exception hierarchy
# ---------------------------------------------------------------------------


def test_fincore_error_is_base() -> None:
    assert issubclass(ValidationError, FincoreError)


def test_dependency_error_also_is_importerror() -> None:
    assert issubclass(DependencyError, ImportError)


# ---------------------------------------------------------------------------
# ValidationError
# ---------------------------------------------------------------------------


def test_validation_error_str_with_all_fields() -> None:
    err = ValidationError("bad", param_name="x", value=3)
    text = str(err)
    assert "Parameter: x" in text
    assert "Value: 3" in text
    assert "Message: bad" in text


def test_validation_error_str_without_fields() -> None:
    err = ValidationError("bad")
    text = str(err)
    assert "unknown" in text


def test_validation_error_to_dict() -> None:
    err = ValidationError("bad", param_name="x", value=3)
    d = err.to_dict()
    assert d["error_type"] == "ValidationError"
    assert d["param_name"] == "x"
    assert d["value"] == "3"


def test_validation_error_from_dict_roundtrip() -> None:
    err = ValidationError.from_dict(
        {"error_type": "ValidationError", "message": "m", "param_name": "p", "value": "v"}
    )
    assert err.message == "m"
    assert err.param_name == "p"


def test_validation_error_from_dict_wrong_type() -> None:
    with pytest.raises(ValueError, match="error type"):
        ValidationError.from_dict({"error_type": "Other", "message": "m"})


# ---------------------------------------------------------------------------
# InsufficientDataError
# ---------------------------------------------------------------------------


def test_insufficient_data_error_str() -> None:
    err = InsufficientDataError("too short", required_length=5, actual_length=2)
    text = str(err)
    assert "required: 5" in text
    assert "actual: 2" in text


def test_insufficient_data_error_roundtrip() -> None:
    err = InsufficientDataError.from_dict(
        {"error_type": "InsufficientDataError", "message": "m", "required_length": 5, "actual_length": 2}
    )
    assert err.required_length == 5
    assert err.actual_length == 2
    assert err.to_dict()["error_type"] == "InsufficientDataError"


def test_insufficient_data_error_from_dict_wrong_type() -> None:
    with pytest.raises(ValueError):
        InsufficientDataError.from_dict({"error_type": "Other", "message": "m"})


# ---------------------------------------------------------------------------
# InvalidPeriodError
# ---------------------------------------------------------------------------


def test_invalid_period_error_str() -> None:
    err = InvalidPeriodError("fortnightly")
    assert str(err) == "InvalidPeriodError(fortnightly)"
    assert err.period == "fortnightly"


def test_invalid_period_error_roundtrip() -> None:
    err = InvalidPeriodError.from_dict({"error_type": "InvalidPeriodError", "period": "daily"})
    assert err.period == "daily"
    d = err.to_dict()
    assert d["error_type"] == "InvalidPeriodError"
    assert "valid_periods" in d


def test_invalid_period_error_from_dict_wrong_type() -> None:
    with pytest.raises(ValueError):
        InvalidPeriodError.from_dict({"error_type": "Other", "period": "daily"})


# ---------------------------------------------------------------------------
# DataAlignmentError
# ---------------------------------------------------------------------------


def test_data_alignment_error_str() -> None:
    err = DataAlignmentError("mismatch", returns_length=3, factor_length=4)
    text = str(err)
    assert "returns length: 3" in text
    assert "factor length: 4" in text


def test_data_alignment_error_roundtrip() -> None:
    err = DataAlignmentError.from_dict(
        {"error_type": "DataAlignmentError", "message": "m", "returns_length": 3, "factor_length": 4}
    )
    assert err.returns_length == 3
    assert err.factor_length == 4


# ---------------------------------------------------------------------------
# NumericalError
# ---------------------------------------------------------------------------


def test_numerical_error_str_with_operation() -> None:
    err = NumericalError("bad math", operation="div")
    assert "operation: div" in str(err)


def test_numerical_error_roundtrip() -> None:
    err = NumericalError.from_dict({"error_type": "NumericalError", "message": "m", "operation": "op"})
    assert err.operation == "op"


# ---------------------------------------------------------------------------
# MissingDataError
# ---------------------------------------------------------------------------


def test_missing_data_error_str() -> None:
    err = MissingDataError("missing", missing_field="returns")
    assert "missing field: returns" in str(err)


def test_missing_data_error_roundtrip() -> None:
    err = MissingDataError.from_dict({"error_type": "MissingDataError", "message": "m", "missing_field": "f"})
    assert err.missing_field == "f"


# ---------------------------------------------------------------------------
# UnsupportedFormatError
# ---------------------------------------------------------------------------


def test_unsupported_format_error_str() -> None:
    err = UnsupportedFormatError("bad", expected_format="Series", actual_format="dict")
    text = str(err)
    assert "expected: Series" in text
    assert "actual: dict" in text


def test_unsupported_format_error_roundtrip() -> None:
    err = UnsupportedFormatError.from_dict(
        {"error_type": "UnsupportedFormatError", "message": "m", "expected_format": "e", "actual_format": "a"}
    )
    assert err.expected_format == "e"
    assert err.actual_format == "a"


# ---------------------------------------------------------------------------
# DependencyError
# ---------------------------------------------------------------------------


def test_dependency_error_str() -> None:
    err = DependencyError("missing", dependency="yfinance", extra="data-yahoo")
    text = str(err)
    assert "dependency: yfinance" in text
    assert "install extra: data-yahoo" in text


def test_dependency_error_roundtrip() -> None:
    err = DependencyError.from_dict(
        {"error_type": "DependencyError", "message": "m", "dependency": "d", "extra": "e"}
    )
    assert err.dependency == "d"
    assert err.extra == "e"


def test_dependency_error_from_dict_wrong_type() -> None:
    with pytest.raises(ValueError):
        DependencyError.from_dict({"error_type": "Other", "message": "m"})


# ---------------------------------------------------------------------------
# handle_numerical_error
# ---------------------------------------------------------------------------


def test_handle_numerical_error_passes_through() -> None:
    @handle_numerical_error
    def fn(x: float) -> float:
        return x * 2

    assert fn(3.0) == 6.0


def test_handle_numerical_error_wraps_zero_division() -> None:
    @handle_numerical_error
    def fn() -> float:
        return 1.0 / 0.0

    with pytest.raises(NumericalError) as exc_info:
        fn()
    assert exc_info.value.operation == "ZeroDivisionError"


def test_handle_numerical_error_wraps_value_error() -> None:
    @handle_numerical_error
    def fn() -> float:
        raise ValueError("x")

    with pytest.raises(NumericalError) as exc_info:
        fn()
    assert exc_info.value.operation == "ValueError"


def test_handle_numerical_error_wraps_overflow() -> None:
    @handle_numerical_error
    def fn() -> float:
        raise OverflowError("x")

    with pytest.raises(NumericalError) as exc_info:
        fn()
    assert exc_info.value.operation == "OverflowError"


def test_handle_numerical_error_wraps_type_error() -> None:
    @handle_numerical_error
    def fn() -> float:
        raise TypeError("x")

    with pytest.raises(NumericalError) as exc_info:
        fn()
    assert exc_info.value.operation == "TypeError"


# ---------------------------------------------------------------------------
# check_dependencies
# ---------------------------------------------------------------------------


def test_check_dependencies_all_present() -> None:
    check_dependencies("sys", "os")


def test_check_dependencies_raises_when_missing() -> None:
    with pytest.raises(DependencyError, match="no_such_module_xyz"):
        check_dependencies("no_such_module_xyz")


def test_check_dependencies_raises_when_some_missing() -> None:
    with pytest.raises(DependencyError, match="no_such_module_xyz"):
        check_dependencies("sys", "no_such_module_xyz")


# ---------------------------------------------------------------------------
# ensure_not_nan
# ---------------------------------------------------------------------------


def test_ensure_not_nan_passes_through_number() -> None:
    assert ensure_not_nan(3.0, "x") == 3.0


def test_ensure_not_nan_passes_through_int() -> None:
    assert ensure_not_nan(3, "x") == 3


def test_ensure_not_nan_replaces_nan() -> None:
    assert ensure_not_nan(float("nan"), "x", replace_with=0.0) == 0.0


def test_ensure_not_nan_raises_without_replace() -> None:
    with pytest.raises(ValidationError):
        ensure_not_nan(float("nan"), "x")


# ---------------------------------------------------------------------------
# safe_divide
# ---------------------------------------------------------------------------


def test_safe_divide_normal() -> None:
    assert safe_divide(4.0, 2.0) == 2.0


def test_safe_divide_zero_scalar_denominator() -> None:
    assert safe_divide(4.0, 0.0, default=-1.0) == -1.0


def test_safe_divide_ndarray_with_zero() -> None:
    num = np.array([1.0, 2.0, 3.0])
    den = np.array([1.0, 0.0, 3.0])
    result = safe_divide(num, den, default=99.0)
    assert result[0] == 1.0
    assert result[1] == 99.0
    assert result[2] == 1.0


def test_safe_divide_default_is_nan() -> None:
    assert np.isnan(safe_divide(1.0, 0.0))


# ---------------------------------------------------------------------------
# safe_sqrt
# ---------------------------------------------------------------------------


def test_safe_sqrt_positive() -> None:
    assert safe_sqrt(4.0) == 2.0


def test_safe_sqrt_negative_scalar_returns_default() -> None:
    assert safe_sqrt(-4.0, default=-1.0) == -1.0


def test_safe_sqrt_ndarray() -> None:
    result = safe_sqrt(np.array([1.0, 4.0]))
    assert result[0] == 1.0
    assert result[1] == 2.0


def test_safe_sqrt_int() -> None:
    assert safe_sqrt(9) == 3.0


# ---------------------------------------------------------------------------
# __str__ with None/empty optional fields
# ---------------------------------------------------------------------------


def test_insufficient_data_error_str_without_fields() -> None:
    assert str(InsufficientDataError("msg")) == "msg"


def test_data_alignment_error_str_without_fields() -> None:
    assert str(DataAlignmentError("msg")) == "msg"


def test_numerical_error_str_without_operation() -> None:
    assert str(NumericalError("msg")) == "msg"


def test_numerical_error_str_empty_operation() -> None:
    assert str(NumericalError("msg", operation="")) == "msg"


def test_missing_data_error_str_without_field() -> None:
    assert str(MissingDataError("msg")) == "msg"


def test_unsupported_format_error_str_without_fields() -> None:
    assert str(UnsupportedFormatError("msg")) == "msg"


def test_dependency_error_str_without_fields() -> None:
    assert str(DependencyError("msg")) == "msg"


def test_dependency_error_str_empty_fields() -> None:
    assert str(DependencyError("msg", dependency="", extra="")) == "msg"


# ---------------------------------------------------------------------------
# to_dict coverage + from_dict wrong-type for the remaining exception classes
# ---------------------------------------------------------------------------


def test_data_alignment_error_to_dict_and_wrong_type() -> None:
    err = DataAlignmentError("m", returns_length=3, factor_length=4)
    assert err.to_dict()["error_type"] == "DataAlignmentError"
    with pytest.raises(ValueError):
        DataAlignmentError.from_dict({"error_type": "Other", "message": "m"})


def test_numerical_error_to_dict_and_wrong_type() -> None:
    err = NumericalError("m", operation="op")
    assert err.to_dict()["error_type"] == "NumericalError"
    with pytest.raises(ValueError):
        NumericalError.from_dict({"error_type": "Other", "message": "m"})


def test_missing_data_error_to_dict_and_wrong_type() -> None:
    err = MissingDataError("m", missing_field="f")
    assert err.to_dict()["error_type"] == "MissingDataError"
    with pytest.raises(ValueError):
        MissingDataError.from_dict({"error_type": "Other", "message": "m"})


def test_unsupported_format_error_to_dict_and_wrong_type() -> None:
    err = UnsupportedFormatError("m", expected_format="e", actual_format="a")
    assert err.to_dict()["error_type"] == "UnsupportedFormatError"
    with pytest.raises(ValueError):
        UnsupportedFormatError.from_dict({"error_type": "Other", "message": "m"})


def test_dependency_error_to_dict() -> None:
    err = DependencyError("m", dependency="d", extra="e")
    assert err.to_dict()["error_type"] == "DependencyError"
