from __future__ import annotations

import importlib

import numpy as np
import pandas as pd
import pytest

from fincore.exceptions import DataAlignmentError, NumericalError, ValidationError
from fincore.validation import validate_input


def _validation_module():
    return importlib.import_module("fincore.contracts.validation")


def _returns(values: list[float] | None = None) -> pd.Series:
    values = values or [0.01, -0.02, 0.03]
    return pd.Series(values, index=pd.date_range("2024-01-01", periods=len(values), tz="UTC"))


def test_returns_schema_returns_a_defensive_copy() -> None:
    source = _returns()

    actual = _validation_module().validate_returns_schema(source)

    assert actual is not source
    pd.testing.assert_series_equal(actual, source)


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
def test_returns_schema_rejects_non_finite_observations(bad: float) -> None:
    source = _returns([0.01, bad, -0.01])

    with pytest.raises(NumericalError, match="finite"):
        _validation_module().validate_returns_schema(source)


def test_returns_schema_classifies_nullable_numeric_missing_values_as_non_finite() -> None:
    source = pd.Series(
        [0.01, pd.NA, -0.01],
        dtype="Float64",
        index=pd.date_range("2024-01-01", periods=3, tz="UTC"),
    )

    with pytest.raises(NumericalError, match="finite"):
        _validation_module().validate_returns_schema(source)


def test_returns_schema_rejects_unsorted_labels() -> None:
    source = _returns().iloc[::-1]

    with pytest.raises(DataAlignmentError, match="sorted"):
        _validation_module().validate_returns_schema(source)


def test_returns_schema_rejects_duplicate_labels() -> None:
    source = _returns()
    source.index = pd.DatetimeIndex([source.index[0], source.index[0], source.index[2]])

    with pytest.raises(DataAlignmentError, match="duplicate"):
        _validation_module().validate_returns_schema(source)


def test_returns_schema_requires_a_numeric_one_dimensional_input() -> None:
    source = pd.DataFrame({"a": [0.01], "b": [0.02]}, index=pd.date_range("2024-01-01", periods=1))

    with pytest.raises(ValidationError, match="one-dimensional"):
        _validation_module().validate_returns_schema(source)


def test_returns_schema_normalizes_naive_index_to_utc_without_mutation() -> None:
    source = pd.Series([0.01, 0.02], index=pd.date_range("2024-01-01", periods=2))
    before = source.copy(deep=True)

    actual = _validation_module().validate_returns_schema(source, normalize_tz="UTC")

    assert str(actual.index.tz) == "UTC"
    pd.testing.assert_series_equal(source, before)


def test_factor_schema_requires_nonempty_overlap_with_returns() -> None:
    returns = _returns()
    factors = pd.Series([0.01, 0.02], index=pd.date_range("2025-01-01", periods=2, tz="UTC"))

    with pytest.raises(DataAlignmentError, match="overlap"):
        _validation_module().validate_context_inputs(returns=returns, factor_returns=factors)


def test_factor_schema_requires_an_exact_shared_label_not_only_overlapping_ranges() -> None:
    returns = pd.Series(
        [0.01, 0.02],
        index=pd.to_datetime(["2024-01-01", "2024-01-03"], utc=True),
    )
    factors = pd.Series(
        [0.005, 0.006],
        index=pd.to_datetime(["2024-01-02", "2024-01-04"], utc=True),
    )

    with pytest.raises(DataAlignmentError, match="overlap"):
        _validation_module().validate_context_inputs(returns=returns, factor_returns=factors)


def test_validate_input_binds_keyword_arguments_before_running_validators() -> None:
    @validate_input(lambda value: value + 1)
    def identity(value: int) -> int:
        return value

    assert identity(value=1) == 2
